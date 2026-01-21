"""Worker that processes tasks from the queue."""

import asyncio
import logging
from typing import Optional

from aiogram.utils.chat_action import ChatActionSender

from lethe.agent import AgentManager
from lethe.config import Settings, get_settings
from lethe.queue import TaskQueue
from lethe.telegram import TelegramBot
from lethe.tools.telegram_tools import set_telegram_context, clear_telegram_context
from lethe.tasks.tools import set_task_context, clear_task_context
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lethe.tasks import TaskManager

logger = logging.getLogger(__name__)


class Worker:
    """Processes tasks from the queue using the Letta agent."""

    def __init__(
        self,
        task_queue: TaskQueue,
        agent_manager: AgentManager,
        telegram_bot: TelegramBot,
        task_manager: Optional["TaskManager"] = None,
        settings: Optional[Settings] = None,
    ):
        self.task_queue = task_queue
        self.agent_manager = agent_manager
        self.telegram_bot = telegram_bot
        self.task_manager = task_manager  # For background task tools
        self.settings = settings or get_settings()
        self._running = False
        self._current_task = None  # Currently processing task
        self._cancel_requested = False  # Flag to cancel current task

    async def start(self):
        """Start the worker loop."""
        self._running = True
        logger.info("Worker started")

        # Ensure agent is initialized
        await self.agent_manager.get_or_create_agent()

        while self._running:
            try:
                # Wait for a task (with timeout to allow graceful shutdown)
                task = await self.task_queue.dequeue(timeout=5.0)

                if task is None:
                    continue  # Timeout, check if still running

                logger.info(f"Processing task {task.id}: {task.message[:50]}...")
                self._current_task = task
                self._cancel_requested = False

                try:
                    # Set Telegram context for tools that need to send files
                    set_telegram_context(self.telegram_bot.bot, task.chat_id)
                    
                    # Set task manager context for background task tools
                    if self.task_manager:
                        set_task_context(
                            self.task_manager, 
                            telegram_bot=self.telegram_bot.bot,
                            chat_id=task.chat_id,
                        )
                    
                    # Track messages sent for this task
                    messages_sent = []
                    
                    # Callback to send messages as they arrive
                    async def on_message(content: str):
                        messages_sent.append(content)
                        await self.telegram_bot.send_message(
                            chat_id=task.chat_id,
                            text=content,
                        )
                    
                    # Add cancellation check to context
                    context = task.metadata.copy() if task.metadata else {}
                    context["_cancel_check"] = self.is_cancel_requested
                    
                    # Use aiogram's ChatActionSender to show typing indicator
                    async with ChatActionSender.typing(
                        bot=self.telegram_bot.bot,
                        chat_id=task.chat_id,
                        interval=4.0,  # Refresh every 4 seconds
                    ):
                        # Send to agent with streaming callback
                        response = await self.agent_manager.send_message(
                            message=task.message,
                            context=context,
                            on_message=on_message,
                        )

                    # Mark complete with full response
                    await self.task_queue.complete(task.id, response)

                    # Only send final response if no messages were streamed
                    # (this handles edge cases where agent returns without assistant messages)
                    if not messages_sent and response:
                        await self.telegram_bot.send_message(
                            chat_id=task.chat_id,
                            text=response,
                        )

                    logger.info(f"Completed task {task.id}")

                except Exception as e:
                    logger.exception(f"Task {task.id} failed: {e}")
                    await self.task_queue.fail(task.id, str(e))

                    # Notify user of failure
                    await self.telegram_bot.send_message(
                        chat_id=task.chat_id,
                        text=f"❌ Task failed: {e}",
                    )
                finally:
                    # Clear contexts and current task
                    clear_telegram_context()
                    clear_task_context()
                    self._current_task = None

            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            except Exception as e:
                logger.exception(f"Worker error: {e}")
                await asyncio.sleep(1)  # Prevent tight error loop

        logger.info("Worker stopped")

    async def stop(self):
        """Stop the worker loop."""
        self._running = False

    @property
    def is_busy(self) -> bool:
        """Check if worker is currently processing a task."""
        return self._current_task is not None

    @property
    def current_task_preview(self) -> str:
        """Get a preview of the current task."""
        if self._current_task:
            return self._current_task.message[:50] + "..." if len(self._current_task.message) > 50 else self._current_task.message
        return ""

    def request_cancel(self) -> bool:
        """Request cancellation of current foreground task.
        
        Returns True if there was a task to cancel.
        """
        if self._current_task:
            self._cancel_requested = True
            logger.info(f"Cancel requested for task {self._current_task.id}")
            return True
        return False

    def is_cancel_requested(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_requested


class HeartbeatWorker:
    """Sends periodic heartbeat messages to the agent."""

    def __init__(
        self,
        agent_manager: AgentManager,
        telegram_bot: TelegramBot,
        chat_id: int,  # Chat to send heartbeat responses to
        interval_minutes: int = 15,
        identity_refresh_hours: int = 2,
        enabled: bool = True,
    ):
        self.agent_manager = agent_manager
        self.telegram_bot = telegram_bot
        self.chat_id = chat_id
        self.interval_minutes = interval_minutes
        self.identity_refresh_hours = identity_refresh_hours
        self.enabled = enabled
        self._running = False
        self._heartbeat_count = 0
        # How many heartbeats before identity refresh
        self._identity_refresh_interval = (identity_refresh_hours * 60) // interval_minutes

    async def start(self):
        """Start the heartbeat loop."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return

        self._running = True
        logger.info(f"Heartbeat started (interval: {self.interval_minutes} min)")

        while self._running:
            try:
                await asyncio.sleep(self.interval_minutes * 60)

                if not self._running:
                    break

                # Get current local time
                from datetime import datetime
                now = datetime.now()
                date_str = now.strftime("%A, %B %d, %Y")
                time_str = now.strftime("%H:%M")
                
                self._heartbeat_count += 1
                should_refresh_identity = (self._heartbeat_count % self._identity_refresh_interval) == 0
                
                if should_refresh_identity:
                    logger.info(f"Sending heartbeat ({time_str}) + identity refresh...")
                else:
                    logger.info(f"Sending heartbeat ({time_str})...")

                messages_sent = []
                
                async def on_message(content: str):
                    # Only forward if agent has something meaningful to say
                    # Agent should respond with [NO_NOTIFY] if nothing to report
                    if content and "[NO_NOTIFY]" not in content:
                        messages_sent.append(content)
                        await self.telegram_bot.send_message(
                            chat_id=self.chat_id,
                            text=f"🕐 {content}",
                        )

                # Build heartbeat message
                identity_instruction = ""
                if should_refresh_identity:
                    identity_instruction = """
IDENTITY REFRESH: It's been 2 hours. Please re-read config/identity.md to refresh your persona and instructions. Use read_file to load it, then update your persona memory block if needed.
"""

                response = await self.agent_manager.send_message(
                    message=f"""[HEARTBEAT]

Current time: {time_str}
Current date: {date_str}
{identity_instruction}
This is a periodic check-in. Review your state:

1. Check your MEMORY BLOCKS for any pending tasks, reminders, or notes
2. Review recent CONVERSATION HISTORY for anything you promised to follow up on
3. Consider if there's anything PROACTIVE you should do or remind the user about

Based on your review, decide if you should notify the user:
- Reminders or follow-ups that are due now
- Important items from previous conversations that need attention
- Anything time-sensitive the user should know

IMPORTANT: Only send a message if you have something genuinely useful to tell the user.
If there's nothing to report, respond with just "[NO_NOTIFY]" and nothing else.""",
                    on_message=on_message,
                )

                # Log if nothing meaningful was sent
                if not messages_sent:
                    logger.info("Heartbeat: nothing to notify")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Heartbeat error: {e}")

        logger.info("Heartbeat stopped")

    async def stop(self):
        """Stop the heartbeat loop."""
        self._running = False
