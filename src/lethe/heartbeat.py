"""Heartbeat - Periodic check-ins for proactive agent behavior.

Sends periodic internal messages to the agent, allowing it to:
- Review pending tasks and reminders
- Surface important information proactively
- Maintain continuity even when user is away
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

# Default interval in seconds (15 minutes)
DEFAULT_HEARTBEAT_INTERVAL = 15 * 60

HEARTBEAT_MESSAGE = """[System Heartbeat - {timestamp}]

Periodic check-in. You may use tools if needed to check tasks, reminders, or gather information.

After your work, provide a final summary. The summary will be evaluated to decide if it's worth sending to the user.

Guidelines:
- Check pending tasks, reminders, calendar if relevant
- Gather any time-sensitive information
- Your tool outputs and ponderings are captured but NOT sent directly to user
- Only your final summary is evaluated

End with either:
- "ok" if nothing urgent
- A brief, actionable message if something needs user attention NOW
"""


SUMMARIZE_HEARTBEAT_PROMPT = """Evaluate this AI assistant's heartbeat check-in:

{response}

---
RESPOND WITH EXACTLY ONE OF:
1. The word "ok" (if nothing urgent)
2. A brief 1-2 sentence message (only if genuinely time-sensitive)

Criteria: Would you interrupt someone's work for this? If not → "ok"

YOUR RESPONSE (no reasoning, just ok or the message):"""


class Heartbeat:
    """Manages periodic heartbeat check-ins."""
    
    def __init__(
        self,
        process_callback: Callable[[str], Awaitable[Optional[str]]],
        send_callback: Callable[[str], Awaitable[None]],
        summarize_callback: Optional[Callable[[str], Awaitable[str]]] = None,
        interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        enabled: bool = True,
    ):
        """Initialize heartbeat.
        
        Args:
            process_callback: Async function to process heartbeat message through agent
                             Returns response string or None
            send_callback: Async function to send response to user (e.g., Telegram)
            summarize_callback: Async function to summarize/evaluate response before sending
            interval: Seconds between heartbeats
            enabled: Whether heartbeats are enabled
        """
        self.process_callback = process_callback
        self.send_callback = send_callback
        self.summarize_callback = summarize_callback
        self.interval = interval
        self.enabled = enabled
        self._task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"Heartbeat initialized (interval={interval}s, enabled={enabled})")
    
    async def start(self):
        """Start the heartbeat loop."""
        if not self.enabled:
            logger.info("Heartbeat disabled, not starting")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Heartbeat started")
    
    async def stop(self):
        """Stop the heartbeat loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Heartbeat stopped")
    
    async def _heartbeat_loop(self):
        """Main heartbeat loop."""
        # Initial heartbeat on startup
        await self._send_heartbeat()
        
        while self._running:
            try:
                # Wait for interval
                await asyncio.sleep(self.interval)
                
                if not self._running:
                    break
                
                # Send heartbeat
                await self._send_heartbeat()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Heartbeat error: {e}")
                # Continue running despite errors
                await asyncio.sleep(60)  # Brief pause before retry
    
    async def _send_heartbeat(self):
        """Send a heartbeat message to the agent."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        message = HEARTBEAT_MESSAGE.format(timestamp=timestamp)
        
        logger.info(f"Sending heartbeat at {timestamp}")
        
        try:
            # Process through agent (may use tools, full context)
            response = await self.process_callback(message)
            
            if not response or not response.strip():
                logger.debug("No heartbeat response")
                return
            
            # Summarize/evaluate response before deciding to send
            if self.summarize_callback:
                prompt = SUMMARIZE_HEARTBEAT_PROMPT.format(response=response)
                evaluated = await self.summarize_callback(prompt)
                
                # Strip model reasoning (some models output <think>...</think> blocks)
                final_response = evaluated.strip() if evaluated else "ok"
                if "</think>" in final_response:
                    final_response = final_response.split("</think>")[-1].strip()
                # Also handle reasoning_content that some models include
                if final_response.startswith("The user is asking") or final_response.startswith("Analysis:"):
                    # Model outputted reasoning, try to find actual answer
                    lines = final_response.split("\n")
                    # Take last non-empty line as the actual response
                    for line in reversed(lines):
                        if line.strip() and not line.startswith("So I"):
                            final_response = line.strip()
                            break
            else:
                final_response = response.strip()
            
            # "ok" means nothing urgent
            if final_response.lower() == "ok":
                logger.info("Heartbeat: nothing urgent (work saved to history)")
            else:
                logger.info(f"Heartbeat sending: {final_response[:100]}...")
                await self.send_callback(final_response)
                
        except Exception as e:
            logger.exception(f"Heartbeat processing failed: {e}")
    
    async def trigger(self):
        """Manually trigger a heartbeat (for testing)."""
        await self._send_heartbeat()
