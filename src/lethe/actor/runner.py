"""Actor runner — executes an actor's LLM loop.

The runner manages the lifecycle of a non-principal actor:
1. Build system prompt from actor config + group awareness
2. Run LLM tool loop until goals are met or max turns reached
3. Handle inter-actor message exchange
4. Terminate and report results to parent
"""

import asyncio
import json
import logging
from typing import Callable, Dict, List, Optional

from lethe.actor import Actor, ActorConfig, ActorMessage, ActorRegistry, ActorState
from lethe.actor.tools import create_actor_tools

logger = logging.getLogger(__name__)


class ActorRunner:
    """Runs a non-principal actor's LLM loop asynchronously."""

    def __init__(
        self,
        actor: Actor,
        registry: ActorRegistry,
        llm_factory: Callable,
        available_tools: Optional[Dict] = None,
    ):
        """
        Args:
            actor: The actor to run
            registry: Actor registry for inter-actor communication
            llm_factory: Factory to create LLM client for this actor
            available_tools: Dict of tool_name -> (func, schema) available for actors
        """
        self.actor = actor
        self.registry = registry
        self.llm_factory = llm_factory
        self.available_tools = available_tools or {}

    async def run(self) -> str:
        """Run the actor's LLM loop until completion or max turns.
        
        Returns:
            The actor's result string
        """
        actor = self.actor
        
        try:
            # Create LLM client for this actor
            llm = await self.llm_factory(actor)
            actor._llm = llm
            
            # Register actor-specific tools
            actor_tools = create_actor_tools(actor, self.registry)
            for func, _ in actor_tools:
                llm.add_tool(func)
            
            # Register requested tools from available pool
            for tool_name in actor.config.tools:
                if tool_name in self.available_tools:
                    func, schema = self.available_tools[tool_name]
                    llm.add_tool(func, schema)
                else:
                    logger.warning(f"Actor {actor.id}: requested tool '{tool_name}' not available")
            
            # Build initial prompt
            system_prompt = actor.build_system_prompt()
            llm.context.system_prompt = system_prompt
            
            # Initial message to kick off the actor
            initial_message = (
                f"You are actor '{actor.config.name}'. Your goals:\n\n"
                f"{actor.config.goals}\n\n"
                f"Begin working on your task. Use tools as needed. "
                f"When done, call terminate(result) with a summary."
            )
            
            logger.info(f"Actor {actor.id} ({actor.config.name}) starting execution")
            
            # Run the LLM loop
            for turn in range(actor.config.max_turns):
                actor._turns = turn + 1
                
                # Check if terminated (by self via tool call)
                if actor.state == ActorState.TERMINATED:
                    break
                
                # Check for incoming messages
                incoming = []
                while not actor._inbox.empty():
                    try:
                        msg = actor._inbox.get_nowait()
                        incoming.append(msg)
                    except asyncio.QueueEmpty:
                        break
                
                # Build the message for this turn
                if turn == 0:
                    message = initial_message
                elif incoming:
                    # Format incoming messages
                    parts = []
                    for msg in incoming:
                        sender = self.registry.get(msg.sender)
                        sender_name = sender.config.name if sender else msg.sender
                        parts.append(f"[Message from {sender_name}]: {msg.content}")
                    message = "\n".join(parts)
                else:
                    # No incoming messages — continue working
                    message = "[System: Continue working on your goals. Call terminate(result) when done.]"
                
                # Call LLM
                try:
                    response = await llm.chat(message)
                except Exception as e:
                    logger.error(f"Actor {actor.id} LLM error: {e}")
                    actor.terminate(f"Error: {e}")
                    break
                
                # Check if actor terminated during tool execution
                if actor.state == ActorState.TERMINATED:
                    break
                
                # If response is just acknowledgment, don't loop unnecessarily
                if response and response.strip().lower() in ("ok", "done", "understood"):
                    continue
                
                # Wait briefly for any incoming messages before next turn
                try:
                    await asyncio.wait_for(actor._inbox.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
            
            # Force terminate if max turns reached
            if actor.state != ActorState.TERMINATED:
                logger.warning(f"Actor {actor.id} hit max turns ({actor.config.max_turns})")
                actor.terminate(f"Max turns reached. Last response: {response[:200] if response else 'none'}")
            
        except Exception as e:
            logger.error(f"Actor {actor.id} runner error: {e}", exc_info=True)
            actor.terminate(f"Runner error: {e}")
        
        return actor._result or "No result"


async def run_actor_in_background(
    actor: Actor,
    registry: ActorRegistry,
    llm_factory: Callable,
    available_tools: Optional[Dict] = None,
) -> asyncio.Task:
    """Start an actor running in the background.
    
    Returns an asyncio.Task that can be awaited for the result.
    """
    runner = ActorRunner(actor, registry, llm_factory, available_tools)
    task = asyncio.create_task(runner.run(), name=f"actor-{actor.id}")
    actor._task = task
    return task
