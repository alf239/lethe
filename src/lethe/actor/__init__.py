"""Actor model for Lethe — subagents with lifecycles.

Actors are autonomous agents that can:
- Have their own goals, model, and tools
- Discover other actors in their group
- Communicate with each other via messages
- Spawn child actors for subtasks
- Terminate themselves (but not others)

The principal actor ("butler") is the only one that talks to the user.
All other actors communicate through the principal or with each other.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ActorState(str, Enum):
    """Lifecycle states for an actor."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING = "waiting"  # Waiting for response from another actor
    TERMINATED = "terminated"


@dataclass
class ActorMessage:
    """Message passed between actors."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str = ""       # Actor ID of sender
    recipient: str = ""    # Actor ID of recipient
    content: str = ""      # Message text
    reply_to: Optional[str] = None  # Message ID this replies to
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def format(self) -> str:
        """Format for inclusion in actor context."""
        ts = self.created_at.strftime("%H:%M:%S")
        reply = f" (reply to {self.reply_to})" if self.reply_to else ""
        return f"[{ts}] {self.sender}{reply}: {self.content}"


@dataclass
class ActorConfig:
    """Configuration for spawning an actor."""
    name: str                              # Human-readable name (e.g., "researcher")
    group: str = "default"                 # Actor group for discovery
    goals: str = ""                        # What this actor should accomplish
    model: str = ""                        # LLM model override (empty = use aux)
    tools: List[str] = field(default_factory=list)  # Tool names available to this actor
    max_turns: int = 20                    # Max LLM turns before forced termination
    max_messages: int = 50                 # Max inter-actor messages


@dataclass
class ActorInfo:
    """Public information about an actor, visible to other actors in the group."""
    id: str
    name: str
    group: str
    goals: str
    state: ActorState
    spawned_by: str  # Actor ID that created this one

    def format(self) -> str:
        """Format for inclusion in actor context."""
        return f"- {self.name} (id={self.id}, state={self.state.value}): {self.goals}"


class Actor:
    """An autonomous agent with a lifecycle.
    
    Each actor has its own LLM client, tools, goals, and message queue.
    The principal actor is special — it receives user messages and sends
    responses back to the user.
    """

    def __init__(
        self,
        config: ActorConfig,
        registry: "ActorRegistry",
        spawned_by: Optional[str] = None,
        is_principal: bool = False,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.config = config
        self.registry = registry
        self.spawned_by = spawned_by or ""
        self.is_principal = is_principal
        self.state = ActorState.INITIALIZING
        
        # Message queue (from other actors)
        self._inbox: asyncio.Queue[ActorMessage] = asyncio.Queue()
        # Conversation history (for this actor's LLM context)
        self._messages: List[ActorMessage] = []
        # Result (set when actor terminates)
        self._result: Optional[str] = None
        # Task handle (for async execution)
        self._task: Optional[asyncio.Task] = None
        # LLM client (set by registry on spawn)
        self._llm = None
        # Turn counter
        self._turns = 0
        
        self.created_at = datetime.now(timezone.utc)
        
        logger.info(f"Actor created: {self.config.name} (id={self.id}, group={self.config.group})")

    @property
    def info(self) -> ActorInfo:
        """Public info visible to other actors."""
        return ActorInfo(
            id=self.id,
            name=self.config.name,
            group=self.config.group,
            goals=self.config.goals,
            state=self.state,
            spawned_by=self.spawned_by,
        )

    async def send(self, message: ActorMessage):
        """Receive a message from another actor."""
        self._messages.append(message)
        await self._inbox.put(message)
        logger.debug(f"Actor {self.id} received message from {message.sender}: {message.content[:50]}...")

    async def send_to(self, recipient_id: str, content: str, reply_to: Optional[str] = None) -> ActorMessage:
        """Send a message to another actor."""
        msg = ActorMessage(
            sender=self.id,
            recipient=recipient_id,
            content=content,
            reply_to=reply_to,
        )
        recipient = self.registry.get(recipient_id)
        if recipient is None:
            raise ValueError(f"Actor {recipient_id} not found")
        await recipient.send(msg)
        self._messages.append(msg)
        return msg

    async def wait_for_reply(self, timeout: float = 120.0) -> Optional[ActorMessage]:
        """Wait for a message in the inbox."""
        try:
            msg = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
            return msg
        except asyncio.TimeoutError:
            logger.warning(f"Actor {self.id} timed out waiting for reply")
            return None

    def terminate(self, result: Optional[str] = None):
        """Terminate this actor. Only the actor itself should call this."""
        self._result = result or f"Actor {self.config.name} terminated"
        self.state = ActorState.TERMINATED
        logger.info(f"Actor terminated: {self.config.name} (id={self.id}), result: {self._result[:80]}...")
        # Notify registry
        self.registry._on_actor_terminated(self.id)

    def build_system_prompt(self) -> str:
        """Build the system prompt for this actor's LLM calls."""
        parts = []
        
        if self.is_principal:
            parts.append("You are the principal actor (butler) — the user's direct assistant.")
            parts.append("You are the ONLY actor that communicates with the user.")
            parts.append("You can spawn subagents to handle subtasks, then report results to the user.")
        else:
            parts.append(f"You are a subagent actor named '{self.config.name}'.")
            parts.append(f"You were spawned by actor '{self.spawned_by}' to accomplish a specific task.")
            parts.append("You CANNOT talk to the user directly. Report your results to the actor that spawned you.")
        
        parts.append(f"\n<goals>\n{self.config.goals}\n</goals>")
        
        # Group awareness
        group_actors = self.registry.discover(self.config.group)
        other_actors = [a for a in group_actors if a.id != self.id]
        if other_actors:
            parts.append("\n<group_actors>")
            parts.append(f"Other actors in group '{self.config.group}':")
            for actor_info in other_actors:
                parts.append(actor_info.format())
            parts.append("</group_actors>")
        
        # Recent messages from other actors
        inbox_messages = [m for m in self._messages if m.sender != self.id][-10:]
        if inbox_messages:
            parts.append("\n<inbox>")
            parts.append("Recent messages from other actors:")
            for m in inbox_messages:
                parts.append(m.format())
            parts.append("</inbox>")
        
        parts.append("\n<rules>")
        parts.append("- Use `send_message(actor_id, content)` to communicate with other actors")
        parts.append("- Use `discover_actors(group)` to find actors in your group")
        if self.is_principal:
            parts.append("- Use `spawn_subagent(name, group, goals, tools)` to create child actors for subtasks")
        parts.append("- Use `terminate(result)` when your task is complete — include a summary of what you accomplished")
        parts.append("- You can terminate yourself, but NOT other actors")
        parts.append("</rules>")
        
        return "\n".join(parts)

    def get_context_messages(self) -> List[Dict]:
        """Get conversation-formatted messages for LLM context."""
        result = []
        for msg in self._messages[-self.config.max_messages:]:
            if msg.sender == self.id:
                result.append({"role": "assistant", "content": msg.content})
            else:
                label = msg.sender
                actor = self.registry.get(msg.sender)
                if actor:
                    label = actor.config.name
                result.append({"role": "user", "content": f"[From {label}]: {msg.content}"})
        return result


class ActorRegistry:
    """Central registry for all actors. Manages lifecycle and discovery."""

    def __init__(self):
        self._actors: Dict[str, Actor] = {}
        self._principal_id: Optional[str] = None
        # Callbacks
        self._on_user_message: Optional[Callable] = None  # When principal needs to send to user
        self._llm_factory: Optional[Callable] = None  # Creates LLM client for actors

    def set_llm_factory(self, factory: Callable):
        """Set factory function that creates LLM clients for actors.
        
        Args:
            factory: Callable(actor: Actor) -> AsyncLLMClient
        """
        self._llm_factory = factory

    def set_user_callback(self, callback: Callable):
        """Set callback for when the principal actor sends messages to the user.
        
        Args:
            callback: async Callable(message: str) -> None
        """
        self._on_user_message = callback

    def spawn(
        self,
        config: ActorConfig,
        spawned_by: Optional[str] = None,
        is_principal: bool = False,
    ) -> Actor:
        """Spawn a new actor.
        
        Args:
            config: Actor configuration
            spawned_by: ID of the actor that spawned this one
            is_principal: Whether this is the principal (user-facing) actor
            
        Returns:
            The newly created Actor
        """
        actor = Actor(
            config=config,
            registry=self,
            spawned_by=spawned_by,
            is_principal=is_principal,
        )
        self._actors[actor.id] = actor
        
        if is_principal:
            self._principal_id = actor.id
        
        actor.state = ActorState.RUNNING
        logger.info(f"Registry: spawned {actor.config.name} (id={actor.id}, principal={is_principal})")
        return actor

    def get(self, actor_id: str) -> Optional[Actor]:
        """Get an actor by ID."""
        return self._actors.get(actor_id)

    def get_principal(self) -> Optional[Actor]:
        """Get the principal (user-facing) actor."""
        if self._principal_id:
            return self._actors.get(self._principal_id)
        return None

    def discover(self, group: str) -> List[ActorInfo]:
        """Discover all running actors in a group.
        
        Args:
            group: Group name to search
            
        Returns:
            List of ActorInfo for running actors in the group
        """
        return [
            actor.info
            for actor in self._actors.values()
            if actor.config.group == group and actor.state != ActorState.TERMINATED
        ]

    def get_children(self, parent_id: str) -> List[Actor]:
        """Get all actors spawned by a given parent."""
        return [
            actor for actor in self._actors.values()
            if actor.spawned_by == parent_id and actor.state != ActorState.TERMINATED
        ]

    def _on_actor_terminated(self, actor_id: str):
        """Called when an actor terminates itself."""
        actor = self._actors.get(actor_id)
        if not actor:
            return
        
        # Notify parent if exists
        parent = self._actors.get(actor.spawned_by) if actor.spawned_by else None
        if parent and parent.state == ActorState.RUNNING:
            msg = ActorMessage(
                sender=actor_id,
                recipient=actor.spawned_by,
                content=f"[TERMINATED] {actor.config.name} finished: {actor._result or 'no result'}",
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(parent.send(msg))
            except RuntimeError:
                # No running loop (sync context) — queue directly
                parent._messages.append(msg)
                parent._inbox.put_nowait(msg)

    @property
    def active_count(self) -> int:
        """Number of non-terminated actors."""
        return sum(1 for a in self._actors.values() if a.state != ActorState.TERMINATED)

    @property
    def all_actors(self) -> List[ActorInfo]:
        """Info for all actors (including terminated)."""
        return [a.info for a in self._actors.values()]

    def cleanup_terminated(self):
        """Remove terminated actors from registry."""
        terminated = [aid for aid, a in self._actors.items() if a.state == ActorState.TERMINATED]
        for aid in terminated:
            del self._actors[aid]
        if terminated:
            logger.info(f"Registry: cleaned up {len(terminated)} terminated actors")
