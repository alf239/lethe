"""Tools available to actors for inter-actor communication and lifecycle management.

These tools are registered with each actor's LLM client, giving the model
the ability to spawn subagents, communicate with other actors, discover
group members, and terminate itself.
"""

import json
import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lethe.actor import Actor, ActorRegistry

logger = logging.getLogger(__name__)


def create_actor_tools(actor: "Actor", registry: "ActorRegistry") -> list:
    """Create tool functions bound to a specific actor.
    
    Returns list of (function, needs_approval) tuples.
    """
    
    async def send_message(actor_id: str, content: str, reply_to: str = "") -> str:
        """Send a message to another actor.
        
        Args:
            actor_id: ID of the actor to send to
            content: Message content
            reply_to: Optional message ID to reply to
            
        Returns:
            Confirmation with message ID
        """
        target = registry.get(actor_id)
        if target is None:
            return f"Error: actor {actor_id} not found. Use discover_actors() to find available actors."
        if target.state.value == "terminated":
            return f"Error: actor {actor_id} ({target.config.name}) is terminated."
        
        msg = await actor.send_to(actor_id, content, reply_to=reply_to or None)
        return f"Message sent (id={msg.id}) to {target.config.name} ({actor_id})"

    async def wait_for_response(timeout: int = 60) -> str:
        """Wait for a message from another actor.
        
        Blocks until a message arrives or timeout. Use this after sending
        a message when you need the response before continuing.
        
        Args:
            timeout: Seconds to wait (default 60)
            
        Returns:
            The message content, or timeout notice
        """
        msg = await actor.wait_for_reply(timeout=float(timeout))
        if msg is None:
            return "Timed out waiting for response."
        sender = registry.get(msg.sender)
        sender_name = sender.config.name if sender else msg.sender
        return f"[From {sender_name}] {msg.content}"

    def discover_actors(group: str = "") -> str:
        """Discover other actors in a group.
        
        Args:
            group: Group name to search. Empty = same group as you.
            
        Returns:
            List of actors with their IDs, names, goals, and state
        """
        search_group = group or actor.config.group
        actors = registry.discover(search_group)
        if not actors:
            return f"No active actors in group '{search_group}'."
        
        lines = [f"Actors in group '{search_group}':"]
        for info in actors:
            marker = " (you)" if info.id == actor.id else ""
            lines.append(f"  {info.name} (id={info.id}, state={info.state.value}){marker}: {info.goals}")
        return "\n".join(lines)

    def terminate(result: str = "") -> str:
        """Terminate this actor and report results.
        
        Call this when your task is complete. Include a summary of what
        you accomplished — this will be sent to the actor that spawned you.
        
        You can only terminate yourself, not other actors.
        
        Args:
            result: Summary of what was accomplished
            
        Returns:
            Confirmation
        """
        actor.terminate(result)
        return f"Terminated. Result sent to parent."

    # Tools available to all actors
    tools = [
        (send_message, False),
        (wait_for_response, False),
        (discover_actors, False),
        (terminate, False),
    ]

    # Only the principal (or actors explicitly allowed) can spawn subagents
    if actor.is_principal or "spawn" in actor.config.tools:
        async def spawn_subagent(
            name: str,
            goals: str,
            group: str = "",
            tools: str = "",
            model: str = "",
            max_turns: int = 20,
        ) -> str:
            """Spawn a new subagent actor to handle a subtask.
            
            The subagent will work autonomously toward its goals and report
            back when done. You'll receive a termination message with results.
            
            Args:
                name: Short name for the actor (e.g., "researcher", "coder")
                goals: What this actor should accomplish (be specific)
                group: Actor group for discovery (default: same as yours)
                tools: Comma-separated tool names available to this actor
                model: LLM model override (empty = use default aux model)
                max_turns: Max LLM turns before forced termination
                
            Returns:
                Actor ID and confirmation
            """
            from lethe.actor import ActorConfig
            
            tool_list = [t.strip() for t in tools.split(",") if t.strip()] if tools else []
            
            config = ActorConfig(
                name=name,
                group=group or actor.config.group,
                goals=goals,
                tools=tool_list,
                model=model,
                max_turns=max_turns,
            )
            
            child = registry.spawn(config, spawned_by=actor.id)
            
            return (
                f"Spawned actor '{name}' (id={child.id}, group={config.group}).\n"
                f"Goals: {goals}\n"
                f"It will send you a message when done."
            )
        
        tools.append((spawn_subagent, False))

    return tools
