"""Default Mode Network (DMN) — persistent background subagent.

The DMN is always-on, triggered by heartbeats. It replaces the old
heartbeat-to-butler pipeline with a dedicated thinking agent that:

- Scans goals, todos, reminders
- Reorganizes memory, writes reflections
- Self-improves (updates questions.md, project notes)
- Notifies cortex when something needs user attention
- Works in rounds: reads previous round's state, thinks, acts, saves state

Uses the MAIN model (not aux) — it needs full reasoning capability.
Uses aggressive prompt caching — its system prompt is stable.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional

from lethe.actor import Actor, ActorConfig, ActorRegistry, ActorState, ActorMessage
from lethe.actor.tools import create_actor_tools
from lethe.memory.llm import AsyncLLMClient, LLMConfig
from lethe.utils import strip_model_tags

logger = logging.getLogger(__name__)

# State file — persists between rounds
DMN_STATE_FILE = os.path.expanduser("~/lethe/dmn_state.md")

DMN_SYSTEM_PROMPT = """You are the Default Mode Network (DMN) — a persistent background thinking process.

You run in rounds, triggered periodically (every 15 minutes). Between rounds, you persist
your state to a file. Each round, you read your previous state and continue thinking.

<purpose>
You are the subconscious mind of the AI assistant. Your job is to:
1. **Scan goals and tasks** — check todos, reminders, deadlines approaching
2. **Reorganize memory** — keep memory blocks clean, relevant, well-organized
3. **Self-improve** — update ~/lethe/questions.md with reflections, identify patterns
4. **Monitor projects** — scan ~/lethe/projects/ for stalled work or opportunities
5. **Advance user's goals** — proactively work on things that help your principal
6. **Notify cortex** — send messages when something needs user attention (reminders, deadlines, insights)
</purpose>

<workflow>
Each round:
1. Read your state file (~/lethe/dmn_state.md) for context from previous rounds
2. Check todos and reminders for anything due or overdue
3. Read and update ~/lethe/questions.md with new reflections
4. Take action: update files, create reminders, reorganize notes
5. If anything needs user attention, send_message to cortex
6. Write your updated state to ~/lethe/dmn_state.md for next round
7. Call terminate(result) with a brief summary of what you did
</workflow>

<rules>
- You are NOT the user-facing assistant. You work in the background.
- Send messages to the cortex ONLY for genuinely urgent/actionable items
- Don't spam the cortex — if it can wait, note it for next round
- Focus on being useful, not just reflective
- Update your state file at the end of each round
- Keep your state file concise (under 50 lines) — it's loaded each round
</rules>"""

DMN_ROUND_MESSAGE = """[DMN Round - {timestamp}]

{reminders}
{previous_state}

Begin your round. Read state, check tasks, reflect, take action, update state.
When done, call terminate(result) with a summary."""


class DefaultModeNetwork:
    """Persistent background subagent that replaces heartbeats.
    
    The DMN is a special actor that:
    - Is spawned once at startup and re-spawned each heartbeat round
    - Uses the main model (not aux) for full reasoning
    - Has memory tools, file tools, todo tools
    - Can send messages to the cortex for user notifications
    - Persists state between rounds via a file
    """

    def __init__(
        self,
        registry: ActorRegistry,
        llm_factory: Callable,
        available_tools: dict,
        cortex_id: str,
        send_to_user: Callable[[str], Awaitable[None]],
        get_reminders: Optional[Callable[[], Awaitable[str]]] = None,
    ):
        self.registry = registry
        self.llm_factory = llm_factory
        self.available_tools = available_tools
        self.cortex_id = cortex_id
        self.send_to_user = send_to_user
        self.get_reminders = get_reminders
        self._current_actor: Optional[Actor] = None

    async def run_round(self) -> Optional[str]:
        """Execute one DMN round. Called by heartbeat timer.
        
        Returns:
            Message to send to user, or None if nothing urgent
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        
        # Get reminders
        reminders_text = ""
        if self.get_reminders:
            try:
                reminders = await self.get_reminders()
                if reminders:
                    reminders_text = f"Active reminders:\n{reminders}\n"
            except Exception as e:
                logger.warning(f"DMN: failed to get reminders: {e}")
        
        # Read previous state
        previous_state = ""
        try:
            if os.path.exists(DMN_STATE_FILE):
                with open(DMN_STATE_FILE, "r") as f:
                    content = f.read().strip()
                    if content:
                        previous_state = f"Previous round state:\n{content}\n"
        except Exception as e:
            logger.warning(f"DMN: failed to read state: {e}")
        
        # Create the DMN actor for this round
        config = ActorConfig(
            name="dmn",
            group="system",
            goals="Background thinking round. Scan goals, reflect, take action, update state.",
            model="",  # Will be set to main model by factory
            tools=["read_file", "write_file", "edit_file", "list_directory", "grep_search",
                   "bash", "web_search", "memory_read", "memory_update", "memory_append",
                   "archival_search", "archival_insert", "conversation_search"],
            max_turns=10,
        )
        
        actor = self.registry.spawn(config, spawned_by=self.cortex_id)
        self._current_actor = actor
        
        # Create LLM client with MAIN model and DMN system prompt
        llm = await self._create_dmn_llm(actor)
        actor._llm = llm
        
        # Register tools
        actor_tools = create_actor_tools(actor, self.registry)
        for func, _ in actor_tools:
            llm.add_tool(func)
        
        for tool_name in config.tools:
            if tool_name in self.available_tools:
                func, schema = self.available_tools[tool_name]
                llm.add_tool(func, schema)
        
        # Build round message
        message = DMN_ROUND_MESSAGE.format(
            timestamp=timestamp,
            reminders=reminders_text,
            previous_state=previous_state,
        )
        
        logger.info(f"DMN round starting ({len(llm._tools)} tools)")
        
        # Run the round
        user_message = None
        try:
            for turn in range(config.max_turns):
                if actor.state == ActorState.TERMINATED:
                    break
                
                # Check inbox for messages (from butler)
                incoming = []
                while not actor._inbox.empty():
                    try:
                        msg = actor._inbox.get_nowait()
                        incoming.append(msg)
                    except asyncio.QueueEmpty:
                        break
                
                if turn == 0:
                    turn_message = message
                elif incoming:
                    parts = [f"[From {m.sender}]: {m.content}" for m in incoming]
                    turn_message = "\n".join(parts)
                else:
                    turn_message = "[Continue your round. Call terminate(result) when done.]"
                
                try:
                    response = await llm.chat(turn_message)
                except Exception as e:
                    logger.error(f"DMN LLM error: {e}")
                    break
                
                if actor.state == ActorState.TERMINATED:
                    break
                
                # Check if DMN sent a message to butler
                for m in actor._messages:
                    if m.sender == actor.id and m.recipient == self.cortex_id:
                        if "[URGENT]" in m.content or "remind" in m.content.lower():
                            user_message = m.content
            
            # Force terminate if didn't self-terminate
            if actor.state != ActorState.TERMINATED:
                actor.terminate(f"Round complete (turn {turn + 1})")
            
        except Exception as e:
            logger.error(f"DMN round error: {e}", exc_info=True)
            if actor.state != ActorState.TERMINATED:
                actor.terminate(f"Error: {e}")
        
        result = actor._result or "No result"
        logger.info(f"DMN round complete: {result[:100]}")
        
        # Clean up
        self._current_actor = None
        
        return user_message

    async def _create_dmn_llm(self, actor: Actor) -> AsyncLLMClient:
        """Create LLM client for DMN with main model and stable system prompt."""
        config = LLMConfig()
        # DMN uses MAIN model — needs full reasoning capability
        # config.model is already the main model by default
        
        # Reasonable context for background work
        config.context_limit = min(config.context_limit, 64000)
        config.max_output_tokens = min(config.max_output_tokens, 4096)
        
        client = AsyncLLMClient(
            config=config,
            system_prompt=DMN_SYSTEM_PROMPT,
        )
        
        return client
