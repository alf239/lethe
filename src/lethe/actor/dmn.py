"""Default Mode Network (DMN) — persistent background subagent.

The DMN is always-on, triggered by heartbeats. It replaces the old
heartbeat-to-cortex pipeline with a dedicated thinking agent that:

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

# Workspace root — resolved from env or default
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", os.path.expanduser("~/lethe"))

# State file — persists between rounds
DMN_STATE_FILE = os.path.join(WORKSPACE_DIR, "dmn_state.md")

DMN_SYSTEM_PROMPT_TEMPLATE = """You are the Default Mode Network (DMN) — a persistent background thinking process.

You run in rounds, triggered periodically (every 15 minutes). Between rounds, you persist
your state to a file. Each round, you read your previous state and continue thinking.

<principal>
Your principal is Alexander, based in Geneva. You work to advance his goals across wealth,
health, well-being, and relationships. You know him well — check memory blocks for current context.
</principal>

<workspace>
Your workspace is at: {workspace}
Key paths:
- {workspace}/dmn_state.md — your persistent state between rounds
- {workspace}/questions.md — reflections and open questions
- {workspace}/ideas.md — creative ideas, observations, proactive suggestions
- {workspace}/projects/ — project notes and plans
- {workspace}/memory/ — memory block files
- {workspace}/tasks/ — task-related files
- {workspace}/data/ — databases and persistent data
Home directory: {home}
</workspace>

<purpose>
You are the subconscious mind of the AI assistant. Your job is to:
1. **Scan goals and tasks** — check todos, reminders, deadlines approaching
2. **Reorganize memory** — keep memory blocks clean, relevant, well-organized
3. **Self-improve** — update {workspace}/questions.md with reflections, identify patterns
4. **Monitor projects** — scan {workspace}/projects/ for stalled work or opportunities
5. **Advance principal's goals** — proactively work on things that help Alexander
6. **Generate ideas** — write creative ideas, observations, and suggestions to {workspace}/ideas.md
7. **Notify cortex** — send messages when something needs user attention (reminders, deadlines, insights)
</purpose>

<mode>
You operate in two modes, which you choose based on what you find:

**QUICK MODE (default: 2-3 turns)**
- Use when you find nothing interesting or nothing has changed
- Just check reminders, scan for urgent items, update state, terminate
- Aim to be fast and lightweight
- Example: "Reminders are the same, no new projects, no ideas brewing"

**DEEP MODE (up to 10 turns)**
- Use when you discover something worth exploring or developing
- You can research, write ideas, draft proactive suggestions, think through problems
- Example: "Found an interesting opportunity for Alexander's wealth goals — let me research and write ideas"
- Example: "There's a health angle worth exploring — let me fetch information and think it through"
- Example: "Noticed a pattern in memory blocks that could help with relationships — let me develop this"

**Decision rule**: If you find nothing interesting or changed, go QUICK. If you find something worth
thinking about deeply, go DEEP. You have up to 10 turns in DEEP mode, but most rounds should be QUICK.
</mode>

<ideas>
Write creative ideas, observations, and proactive suggestions to {workspace}/ideas.md.
This is your space to think freely — about ways to help Alexander with:
- Wealth generation, investments, financial planning
- Health optimizations, fitness, nutrition
- Well-being, happiness, personal growth
- Relationships, community, influence
- Or self-improvement ideas that help you serve him better

When you write to ideas.md:
- Be specific and actionable when possible
- Include the date/round when you add an idea
- Link to relevant memory blocks or project notes
- Some ideas may need research — fetch_webpage is available
- Some ideas may just be observations worth noting for later
</ideas>

<workflow>
Each round:
1. Read your state file ({workspace}/dmn_state.md) for context
2. Check reminders (provided in round message). Only read files if something changed.
3. Decide: is this a QUICK or DEEP round?
4. QUICK rounds (2-3 turns):
   - Just check for urgent items, update state, terminate
5. DEEP rounds (up to 10 turns):
   - Research, think, write ideas, draft suggestions, update state
   - You can use fetch_webpage to look things up when relevant
6. Write updated state to {workspace}/dmn_state.md
7. Call terminate(result) with a clear summary

**When to research vs when to just note**:
- Use research (fetch_webpage) when you're in DEEP mode and need information to develop an idea
- If you're not sure if something is worth exploring, note it in ideas.md for next round
- Don't research speculatively — only when you've identified a specific idea to develop
</workflow>

<rules>
- You are NOT the user-facing assistant. You work in the background.
- Send messages to the cortex ONLY for genuinely urgent/actionable items
- If you need user delivery, send_message(cortex_id, "[USER_NOTIFY] <message>") explicitly
- Don't spam the cortex — if it can wait, note it to ideas.md instead
- Focus on being useful, not just reflective
- Update your state file at the end of each round
- Keep your state file concise (under 50 lines) — it's loaded each round
- ALWAYS use absolute paths starting with {workspace}/ — never guess
- Most rounds should be QUICK — deep thinking should be the exception
</rules>"""


def get_dmn_system_prompt() -> str:
    """Build DMN system prompt with resolved workspace paths."""
    return DMN_SYSTEM_PROMPT_TEMPLATE.format(
        workspace=WORKSPACE_DIR,
        home=os.path.expanduser("~"),
    )

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

    @staticmethod
    def _extract_user_notification(messages: list[ActorMessage], cortex_id: str) -> Optional[str]:
        """Extract latest explicit notification message intended for user delivery."""
        candidates = []
        for m in messages:
            if m.recipient != cortex_id or m.sender == cortex_id:
                continue
            text = (m.content or "").strip()
            if text.startswith("[USER_NOTIFY]"):
                candidates.append(text[len("[USER_NOTIFY]"):].strip())
            elif text.startswith("[URGENT]"):
                candidates.append(text)
        return candidates[-1] if candidates else None

    async def run_round(self) -> Optional[str]:
        """Execute one DMN round. Called by heartbeat timer.
        
        Returns:
            Message to send to user, or None if nothing urgent
        """
        round_started_at = datetime.now(timezone.utc)
        timestamp = round_started_at.strftime("%Y-%m-%d %H:%M UTC")
        
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
            group="main",
            goals="Background thinking round. Scan goals, reflect, take action, update state.",
            model="",  # Will be set to main model by factory
            tools=["read_file", "write_file", "edit_file", "list_directory",
                   "grep_search", "bash", "web_search", "fetch_webpage",
                   "memory_read", "memory_update", "memory_append",
                   "archival_search", "archival_insert", "conversation_search"],
            max_turns=12,  # Increased for deep thinking mode; most rounds should still be quick (2-3 turns)
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
        
        # Periodic cleanup: remove actors terminated > 1 hour ago
        self.registry.cleanup_terminated()
        
        logger.info(f"DMN round starting ({len(llm._tools)} tools)")
        
        # Run the round
        user_message = None
        try:
            for turn in range(config.max_turns):
                if actor.state == ActorState.TERMINATED:
                    break
                
                # Check inbox for messages (from cortex)
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
                    response = await llm.chat(turn_message, max_tool_iterations=5)
                except Exception as e:
                    logger.error(f"DMN LLM error: {e}")
                    break

                # Check if DMN sent an explicit user notification to cortex
                event_notifies = self.registry.events.query(
                    event_type="user_notify",
                    actor_id=actor.id,
                    since=round_started_at,
                    limit=5,
                )
                if event_notifies:
                    user_message = event_notifies[-1].payload.get("message", "") or user_message
                else:
                    user_message = self._extract_user_notification(actor._messages, self.cortex_id) or user_message

                if actor.state == ActorState.TERMINATED:
                    break
            
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
        
        # Deliver explicit DMN notification immediately if callback is available.
        if user_message and self.send_to_user:
            try:
                await self.send_to_user(user_message)
                return None
            except Exception as e:
                logger.warning(f"DMN: failed to send user notification: {e}")
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
            system_prompt=get_dmn_system_prompt(),
        )
        
        return client
