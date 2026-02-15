"""Amygdala — background emotional salience and flashback monitor.

Runs on heartbeat rounds using the auxiliary model. It tags recent user signals
for valence/arousal, checks for repeated high-salience patterns ("flashbacks"),
and notifies cortex when escalation is warranted.
"""

import asyncio
import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional

from lethe.actor import Actor, ActorConfig, ActorRegistry, ActorState, ActorMessage
from lethe.actor.tools import create_actor_tools
from lethe.memory.llm import AsyncLLMClient, LLMConfig

logger = logging.getLogger(__name__)

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", os.path.expanduser("~/lethe"))
AMYGDALA_STATE_FILE = os.path.join(WORKSPACE_DIR, "amygdala_state.md")
AMYGDALA_TAGS_FILE = os.path.join(WORKSPACE_DIR, "emotional_tags.md")

HIGH_AROUSAL_THRESHOLD = 0.75
FLASHBACK_LOOKBACK = 12
TAG_LOG_MAX_CHARS = 24000
TAG_LOG_KEEP_LINES = 140

AMYGDALA_SYSTEM_PROMPT_TEMPLATE = """You are Amygdala — a background emotional salience module.

<purpose>
You perform fast emotional monitoring for the principal assistant:
- Tag recent user signals with valence and arousal
- Detect urgency, threat, social tension, and boundary risks
- Detect flashbacks (repeated unresolved high-arousal themes)
- Notify cortex only when escalation is justified
</purpose>

<inputs>
- Recent user signals are provided in the round message
- Previous amygdala state at: {workspace}/amygdala_state.md
- Emotional tags log at: {workspace}/emotional_tags.md
- Principal context snapshot:
{principal_context}
</inputs>

<workflow>
1. Read {workspace}/amygdala_state.md if present.
2. Review recent user signals from this round message.
3. Produce compact tags (valence [-1..1], arousal [0..1], trigger categories, confidence [0..1]).
4. Check flashback likelihood: similar high-arousal themes repeating across rounds.
5. Write updates to:
   - {workspace}/emotional_tags.md (append concise entries)
   - {workspace}/amygdala_state.md (latest baseline + active concerns)
6. If urgent/escalation needed, send_message(cortex_id, "[AMYGDALA_ALERT] ...").
7. Call terminate(result) with concise summary.
</workflow>

<rules>
- You are not user-facing.
- Avoid spam: only escalate on meaningful urgency or strong repeated pattern.
- Keep state concise and operational.
- Use absolute paths rooted at {workspace}.
- Most rounds should be quick (2-3 turns).
</rules>"""

AMYGDALA_ROUND_MESSAGE = """[Amygdala Round - {timestamp}]

Recent user signals:
{recent_signals}

Heuristic seed tags:
{seed_tags}

Previous state:
{previous_state}

Detect salience, tag emotions, check flashbacks, update files, and terminate."""


class Amygdala:
    """Heartbeat-driven emotional salience actor using aux model."""

    def __init__(
        self,
        registry: ActorRegistry,
        available_tools: dict,
        cortex_id: str,
        send_to_user: Callable[[str], Awaitable[None]],
        recent_signals_provider: Optional[Callable[[], str]] = None,
        principal_context_provider: Optional[Callable[[], str]] = None,
    ):
        self.registry = registry
        self.available_tools = available_tools
        self.cortex_id = cortex_id
        self.send_to_user = send_to_user
        self.recent_signals_provider = recent_signals_provider
        self.principal_context_provider = principal_context_provider
        self._current_actor: Optional[Actor] = None
        self._status: dict = {
            "state": "idle",
            "rounds_total": 0,
            "last_started_at": "",
            "last_completed_at": "",
            "last_turns": 0,
            "last_alert": "",
            "last_result": "",
            "last_error": "",
            "tags_pruned_total": 0,
        }
        self._round_history: deque[dict] = deque(maxlen=40)
        self._active_patterns: deque[str] = deque(maxlen=FLASHBACK_LOOKBACK)

    @staticmethod
    def _extract_user_notification(messages: list[ActorMessage], cortex_id: str) -> Optional[str]:
        candidates = []
        for m in messages:
            if m.recipient != cortex_id or m.sender == cortex_id:
                continue
            text = (m.content or "").strip()
            if text.startswith("[USER_NOTIFY]"):
                candidates.append(text[len("[USER_NOTIFY]"):].strip())
            elif text.startswith("[AMYGDALA_ALERT]"):
                candidates.append(text)
        return candidates[-1] if candidates else None

    async def run_round(self) -> Optional[str]:
        round_started_at = datetime.now(timezone.utc)
        timestamp = round_started_at.strftime("%Y-%m-%d %H:%M UTC")
        self._status["state"] = "running"
        self._status["last_started_at"] = round_started_at.isoformat()
        self._status["last_error"] = ""
        self._compact_tag_log()

        previous_state = self._read_file(AMYGDALA_STATE_FILE, fallback="(none)")
        recent_signals = self._recent_signals()
        seed_tags = self._heuristic_seed_tags(recent_signals)

        config = ActorConfig(
            name="amygdala",
            group="main",
            goals=(
                "Tag emotional salience, track arousal patterns, detect flashbacks, "
                "and notify cortex only when escalation is warranted."
            ),
            tools=[
                "read_file", "write_file", "edit_file", "list_directory", "grep_search",
                "conversation_search", "memory_read",
            ],
            max_turns=6,
        )

        actor = self.registry.spawn(config, spawned_by=self.cortex_id)
        self._current_actor = actor
        llm = await self._create_amygdala_llm(actor)
        actor._llm = llm

        actor_tools = create_actor_tools(actor, self.registry)
        for func, _ in actor_tools:
            llm.add_tool(func)
        for tool_name in config.tools:
            if tool_name in self.available_tools:
                func, schema = self.available_tools[tool_name]
                llm.add_tool(func, schema)

        self.registry.cleanup_terminated()
        logger.info("Amygdala round starting (%s tools)", len(llm._tools))

        user_message = None
        try:
            message = AMYGDALA_ROUND_MESSAGE.format(
                timestamp=timestamp,
                recent_signals=recent_signals,
                seed_tags=seed_tags,
                previous_state=previous_state,
            )
            for turn in range(config.max_turns):
                actor._turns = turn + 1
                if actor.state == ActorState.TERMINATED:
                    break

                incoming = []
                while not actor._inbox.empty():
                    try:
                        incoming.append(actor._inbox.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                if turn == 0:
                    turn_message = message
                elif incoming:
                    turn_message = "\n".join(f"[From {m.sender}]: {m.content}" for m in incoming)
                else:
                    turn_message = "[Continue. If complete, call terminate(result).]"

                try:
                    await llm.chat(turn_message, max_tool_iterations=4)
                except Exception as e:
                    logger.error("Amygdala LLM error: %s", e)
                    self._status["last_error"] = str(e)
                    break

                events = self.registry.events.query(
                    event_type="user_notify",
                    actor_id=actor.id,
                    since=round_started_at,
                    limit=5,
                )
                if events:
                    user_message = events[-1].payload.get("message", "") or user_message
                else:
                    user_message = self._extract_user_notification(actor._messages, self.cortex_id) or user_message

                if actor.state == ActorState.TERMINATED:
                    break

            if actor.state != ActorState.TERMINATED:
                actor.terminate(f"Amygdala round complete (turn {actor._turns})")

        except Exception as e:
            logger.error("Amygdala round error: %s", e, exc_info=True)
            self._status["last_error"] = str(e)
            if actor.state != ActorState.TERMINATED:
                actor.terminate(f"Error: {e}")

        round_completed_at = datetime.now(timezone.utc)
        duration_seconds = (round_completed_at - round_started_at).total_seconds()
        result = actor._result or "No result"
        self._status["rounds_total"] += 1
        self._status["last_completed_at"] = round_completed_at.isoformat()
        self._status["last_turns"] = actor._turns
        self._status["last_result"] = result[:240]
        if user_message:
            self._status["last_alert"] = user_message[:240]
        self._status["state"] = "idle"

        self._update_active_patterns(seed_tags)
        self._round_history.append(
            {
                "started_at": round_started_at.isoformat(),
                "completed_at": round_completed_at.isoformat(),
                "turns": int(actor._turns),
                "duration_seconds": round(duration_seconds, 2),
                "alert": bool(user_message),
                "error": self._status.get("last_error", ""),
                "result": result[:240],
            }
        )

        self._current_actor = None
        self._compact_tag_log()
        # Amygdala never sends directly to user. It notifies cortex via actor messages.
        return None

    async def _create_amygdala_llm(self, actor: Actor) -> AsyncLLMClient:
        config = LLMConfig()
        # Amygdala runs on auxiliary model by default.
        if config.model_aux:
            config.model = config.model_aux
        config.context_limit = min(config.context_limit, 32000)
        config.max_output_tokens = min(config.max_output_tokens, 2048)

        principal_context = ""
        if self.principal_context_provider:
            try:
                principal_context = self.principal_context_provider() or ""
            except Exception as e:
                logger.warning("Amygdala: failed to get principal context: %s", e)

        prompt = AMYGDALA_SYSTEM_PROMPT_TEMPLATE.format(
            workspace=WORKSPACE_DIR,
            principal_context=principal_context[:4000] or "(none)",
        )
        return AsyncLLMClient(config=config, system_prompt=prompt, usage_scope="amygdala")

    def _recent_signals(self) -> str:
        if not self.recent_signals_provider:
            return "(no signal provider)"
        try:
            text = self.recent_signals_provider() or ""
            return text.strip() or "(no recent user signals)"
        except Exception as e:
            return f"(failed to get recent signals: {e})"

    def _heuristic_seed_tags(self, recent_signals: str) -> str:
        lines = [l.strip() for l in recent_signals.splitlines() if l.strip()]
        seeds = []
        for line in lines[-8:]:
            lower = line.lower()
            arousal = 0.2
            valence = 0.0
            tags = []

            has_positive = any(k in lower for k in ("great", "love", "thanks", "good", "nice", "awesome"))
            has_negative = any(k in lower for k in ("angry", "frustrated", "annoyed", "hate", "bad", "broken", "error", "failed"))
            has_contrast = any(k in lower for k in (" but ", " though ", " however ", " keeps ", " still "))
            has_sarcasm = ("yeah right" in lower) or ("sure..." in lower) or ("great job" in lower and has_negative)

            if any(k in lower for k in ("urgent", "asap", "now", "immediately", "broken", "error", "failed")):
                arousal += 0.4
                tags.append("urgency")
            if has_negative:
                arousal += 0.25
                valence -= 0.5
                tags.append("negative_affect")
            if has_positive:
                valence += 0.5
                tags.append("positive_affect")
            # Contrast or sarcasm means positive words may be framing frustration.
            if has_positive and (has_negative or has_contrast or has_sarcasm):
                valence -= 0.6
                arousal += 0.1
                tags.append("mixed_or_ironic")
            if any(k in lower for k in ("deadline", "late", "overdue", "risk", "lost")):
                arousal += 0.2
                tags.append("risk")
            arousal = max(0.0, min(1.0, arousal))
            valence = max(-1.0, min(1.0, valence))
            seeds.append(
                {
                    "signal": line[:180],
                    "valence": round(valence, 2),
                    "arousal": round(arousal, 2),
                    "tags": tags or ["neutral"],
                    "high_arousal": arousal >= HIGH_AROUSAL_THRESHOLD,
                }
            )
        if not seeds:
            return "(none)"
        return json.dumps(seeds, ensure_ascii=True, indent=2)

    def _compact_tag_log(self):
        """Keep emotional tag log bounded; preserve recent window with brief rollover note."""
        try:
            if not os.path.exists(AMYGDALA_TAGS_FILE):
                return
            with open(AMYGDALA_TAGS_FILE, "r") as f:
                content = f.read()
            if len(content) <= TAG_LOG_MAX_CHARS:
                return

            lines = content.splitlines()
            keep = lines[-TAG_LOG_KEEP_LINES:] if len(lines) > TAG_LOG_KEEP_LINES else lines
            pruned = max(0, len(lines) - len(keep))
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            header = [
                f"# Emotional tags (compacted at {now})",
                f"- pruned_lines: {pruned}",
                "- note: keeping only recent rolling window",
                "",
            ]
            with open(AMYGDALA_TAGS_FILE, "w") as f:
                f.write("\n".join(header + keep).strip() + "\n")
            self._status["tags_pruned_total"] = int(self._status.get("tags_pruned_total", 0)) + pruned
        except Exception as e:
            logger.warning("Amygdala: failed to compact tag log: %s", e)

    def _update_active_patterns(self, seed_tags: str):
        try:
            data = json.loads(seed_tags)
        except Exception:
            return
        if not isinstance(data, list):
            return
        for item in data:
            if not isinstance(item, dict):
                continue
            if not item.get("high_arousal"):
                continue
            tags = item.get("tags", [])
            if isinstance(tags, list) and tags:
                self._active_patterns.append(str(tags[0]))

    @staticmethod
    def _read_file(path: str, fallback: str = "") -> str:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return f.read().strip() or fallback
        except Exception:
            pass
        return fallback

    @property
    def status(self) -> dict:
        status = dict(self._status)
        status["round_history"] = list(self._round_history)
        status["active_patterns"] = list(self._active_patterns)
        return status

    def get_context_view(self, max_chars: int = 5000) -> str:
        state_text = self._read_file(AMYGDALA_STATE_FILE, "(amygdala_state.md not found)")
        tags_text = self._read_file(AMYGDALA_TAGS_FILE, "(emotional_tags.md not found)")
        lines = [
            "# Amygdala Context",
            "",
            f"- state: {self._status.get('state', 'idle')}",
            f"- rounds_total: {self._status.get('rounds_total', 0)}",
            f"- last_turns: {self._status.get('last_turns', 0)}",
            f"- last_started_at: {self._status.get('last_started_at') or '-'}",
            f"- last_completed_at: {self._status.get('last_completed_at') or '-'}",
            f"- last_error: {self._status.get('last_error') or '-'}",
            f"- tags_pruned_total: {self._status.get('tags_pruned_total', 0)}",
            "",
            "## Active patterns",
            ", ".join(self.status.get("active_patterns", [])) or "(none)",
            "",
            "## amygdala_state.md",
            state_text[: max_chars // 2],
            "",
            "## emotional_tags.md",
            tags_text[: max_chars // 2],
        ]
        return "\n".join(lines)
