"""Telegram bot interface."""

import asyncio
import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.enums import ChatAction
from aiogram.types import Message

from lethe.config import Settings, get_settings
from lethe.queue import TaskQueue

logger = logging.getLogger(__name__)

# Type hint for TaskManager without importing (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lethe.tasks import TaskManager


class TelegramBot:
    """Async Telegram bot that queues messages for agent processing."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        task_queue: Optional[TaskQueue] = None,
        bg_task_manager: Optional["TaskManager"] = None,
        on_task_stopped: Optional[Callable] = None,  # Callback when task stopped via /stop
        worker: Optional[Any] = None,  # Reference to foreground worker for cancellation
    ):
        self.settings = settings or get_settings()
        self.task_queue = task_queue
        self.bg_task_manager = bg_task_manager
        self.on_task_stopped = on_task_stopped  # async callback(task_ids: list[str])
        self.worker = worker  # Foreground task worker

        self.bot = Bot(
            token=self.settings.telegram_bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
        )
        self.dp = Dispatcher()

        self._setup_handlers()

    @property
    def downloads_dir(self) -> Path:
        """Get the downloads directory, creating if needed."""
        downloads = self.settings.workspace_dir / "Downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        return downloads

    async def save_file_locally(self, file_path: str, original_name: str) -> Path:
        """Download a file from Telegram and save to workspace/Downloads.
        
        Args:
            file_path: Telegram file path from get_file()
            original_name: Original filename to preserve
            
        Returns:
            Path to the saved file
        """
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(original_name).stem
        suffix = Path(original_name).suffix or ""
        local_name = f"{timestamp}_{stem}{suffix}"
        local_path = self.downloads_dir / local_name
        
        # Download from Telegram
        await self.bot.download_file(file_path, local_path)
        logger.info(f"Saved file to {local_path}")
        return local_path

    def _setup_handlers(self):
        """Set up message handlers."""

        @self.dp.message(CommandStart())
        async def handle_start(message: Message):
            """Handle /start command."""
            if not self._is_authorized(message.from_user.id):
                await message.answer("Unauthorized.")
                return

            await message.answer(
                "Hello! I'm Lethe, your autonomous assistant.\n\n"
                "Send me any message and I'll process it asynchronously. "
                "I'll reply when I'm done.\n\n"
                "Commands:\n"
                "/status - Check message queue status\n"
                "/list - Show background tasks\n"
                "/stop - Stop current foreground task\n"
                "/stop all - Stop all background tasks\n"
                "/stop N - Stop specific background task (N from /list)"
            )

        @self.dp.message(Command("status"))
        async def handle_status(message: Message):
            """Handle /status command."""
            if not self._is_authorized(message.from_user.id):
                return

            if self.task_queue:
                pending = await self.task_queue.get_pending_count()
                await message.answer(f"Pending tasks: {pending}")
            else:
                await message.answer("Queue not initialized.")

        @self.dp.message(Command("list"))
        async def handle_list(message: Message):
            """Handle /list command - show background tasks."""
            if not self._is_authorized(message.from_user.id):
                return

            if not self.bg_task_manager:
                await message.answer("Background task manager not initialized.")
                return

            from lethe.tasks import TaskStatus
            
            # Get all non-completed tasks
            tasks = await self.bg_task_manager.list_tasks(limit=20)
            active_tasks = [t for t in tasks if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]
            
            if not active_tasks:
                await message.answer("No active background tasks.")
                return

            lines = ["📋 *Background Tasks:*\n"]
            for i, task in enumerate(active_tasks, 1):
                status_emoji = "⏳" if task.status == TaskStatus.PENDING else "🔄"
                short_desc = task.description[:50] + "..." if len(task.description) > 50 else task.description
                progress = ""
                if task.progress is not None:
                    progress = f" ({task.progress * 100:.0f}%)"
                lines.append(f"{i}. {status_emoji} {short_desc}{progress}")
                lines.append(f"   ID: `{task.id[:8]}` | Mode: {task.mode.value}")
            
            lines.append(f"\nUse /stop to stop all, or /stop N to stop one.")
            
            await message.answer("\n".join(lines), parse_mode="Markdown")

        @self.dp.message(Command("stop"))
        async def handle_stop(message: Message):
            """Handle /stop command.
            
            /stop - Stop current foreground task
            /stop all - Stop all background tasks
            /stop N - Stop specific background task (N from /list)
            """
            if not self._is_authorized(message.from_user.id):
                return

            args = message.text.split(maxsplit=1)
            arg = args[1].strip().lower() if len(args) > 1 else ""
            
            # /stop (no args) - stop foreground task
            if not arg:
                if self.worker and self.worker.is_busy:
                    self.worker.request_cancel()
                    await message.answer(f"🛑 Stopping current task: {self.worker.current_task_preview}")
                else:
                    await message.answer("No foreground task running.")
                return
            
            # /stop all - stop all background tasks
            if arg == "all":
                if not self.bg_task_manager:
                    await message.answer("Background task manager not initialized.")
                    return

                from lethe.tasks import TaskStatus
                tasks = await self.bg_task_manager.list_tasks(limit=20)
                active_tasks = [t for t in tasks if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]
                
                if not active_tasks:
                    await message.answer("No active background tasks.")
                    return

                stopped_ids = []
                for task in active_tasks:
                    if await self.bg_task_manager.cancel_task(task.id):
                        stopped_ids.append(task.id)

                if stopped_ids:
                    await message.answer(f"🛑 Stopped {len(stopped_ids)} background task(s).")
                    if self.on_task_stopped:
                        try:
                            await self.on_task_stopped(stopped_ids, message.chat.id)
                        except Exception as e:
                            logger.warning(f"Error notifying agent: {e}")
                else:
                    await message.answer("Failed to stop tasks.")
                return
            
            # /stop N - stop specific background task
            try:
                task_number = int(arg)
            except ValueError:
                await message.answer("Usage: /stop (foreground), /stop all (background), /stop N (specific)")
                return

            if not self.bg_task_manager:
                await message.answer("Background task manager not initialized.")
                return

            from lethe.tasks import TaskStatus
            tasks = await self.bg_task_manager.list_tasks(limit=20)
            active_tasks = [t for t in tasks if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]

            if task_number < 1 or task_number > len(active_tasks):
                await message.answer(f"Invalid task number. Use 1-{len(active_tasks)}.")
                return

            task = active_tasks[task_number - 1]
            if await self.bg_task_manager.cancel_task(task.id):
                await message.answer(f"🛑 Stopped background task: {task.description[:50]}...")
                if self.on_task_stopped:
                    try:
                        await self.on_task_stopped([task.id], message.chat.id)
                    except Exception as e:
                        logger.warning(f"Error notifying agent: {e}")
            else:
                await message.answer("Failed to stop task.")

        @self.dp.message(F.photo)
        async def handle_photo(message: Message):
            """Handle photo messages."""
            if not self._is_authorized(message.from_user.id):
                logger.warning(f"Unauthorized photo from user {message.from_user.id}")
                return

            if not self.task_queue:
                await message.answer("System not ready. Please try again later.")
                return

            try:
                # Get the largest photo (last in the array)
                photo = message.photo[-1]
                file = await self.bot.get_file(photo.file_id)
                file_name = file.file_path.split('/')[-1]
                
                # Save locally to workspace/Downloads
                local_path = await self.save_file_locally(file.file_path, file_name)
                
                # Read file and encode as base64 for Letta multimodal
                image_data = local_path.read_bytes()
                base64_data = base64.standard_b64encode(image_data).decode("utf-8")
                
                # Determine media type from extension
                ext = local_path.suffix.lower()
                media_type = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg", 
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }.get(ext, "image/jpeg")
                
                # Queue the task with base64 image data
                caption = message.caption or f"[Image: {local_path.name}]"
                task = await self.task_queue.enqueue(
                    chat_id=message.chat.id,
                    user_id=message.from_user.id,
                    message=caption,
                    metadata={
                        "username": message.from_user.username,
                        "first_name": message.from_user.first_name,
                        "message_id": message.message_id,
                        "attachments": [{
                            "type": "image",
                            "local_path": str(local_path),
                            "base64_data": base64_data,
                            "media_type": media_type,
                        }],
                    },
                )
            except Exception as e:
                logger.error(f"Failed to process image: {e}", exc_info=True)
                await message.answer("Sorry, failed to process the image. Please try again.")
                return

            # Only show queue position if there are multiple tasks
            pending = await self.task_queue.get_pending_count()
            if pending > 1:
                await message.answer(f"📋 Queued (position: {pending})")

            logger.info(f"Queued photo task {task.id} from user {message.from_user.id}")

        @self.dp.message(F.document)
        async def handle_document(message: Message):
            """Handle document messages."""
            if not self._is_authorized(message.from_user.id):
                logger.warning(f"Unauthorized document from user {message.from_user.id}")
                return

            if not self.task_queue:
                await message.answer("System not ready. Please try again later.")
                return

            try:
                file = await self.bot.get_file(message.document.file_id)
                file_name = message.document.file_name or f"document_{message.document.file_id}"
                
                # Save locally to workspace/Downloads
                local_path = await self.save_file_locally(file.file_path, file_name)

                # Queue the task with local path
                caption = message.caption or f"[Document: {file_name}]"
                task = await self.task_queue.enqueue(
                    chat_id=message.chat.id,
                    user_id=message.from_user.id,
                    message=caption,
                    metadata={
                        "username": message.from_user.username,
                        "first_name": message.from_user.first_name,
                        "message_id": message.message_id,
                        "attachments": [{
                            "type": "document",
                            "local_path": str(local_path),
                            "file_name": file_name,
                            "mime_type": message.document.mime_type,
                        }],
                    },
                )
            except Exception as e:
                logger.error(f"Failed to process document: {e}", exc_info=True)
                await message.answer("Sorry, failed to process the document. Please try again.")
                return

            # Only show queue position if there are multiple tasks
            pending = await self.task_queue.get_pending_count()
            if pending > 1:
                await message.answer(f"📋 Queued (position: {pending})")

            logger.info(f"Queued document task {task.id} from user {message.from_user.id}")

        @self.dp.message(F.audio)
        async def handle_audio(message: Message):
            """Handle audio messages."""
            if not self._is_authorized(message.from_user.id):
                return

            if not self.task_queue:
                await message.answer("System not ready. Please try again later.")
                return

            try:
                file = await self.bot.get_file(message.audio.file_id)
                file_name = message.audio.file_name or f"audio_{message.audio.file_id}.mp3"
                
                local_path = await self.save_file_locally(file.file_path, file_name)

                caption = message.caption or f"[Audio: {file_name}]"
                task = await self.task_queue.enqueue(
                    chat_id=message.chat.id,
                    user_id=message.from_user.id,
                    message=caption,
                    metadata={
                        "username": message.from_user.username,
                        "first_name": message.from_user.first_name,
                        "message_id": message.message_id,
                        "attachments": [{
                            "type": "audio",
                            "local_path": str(local_path),
                            "file_name": file_name,
                            "mime_type": message.audio.mime_type,
                            "duration": message.audio.duration,
                        }],
                    },
                )
            except Exception as e:
                logger.error(f"Failed to process audio: {e}", exc_info=True)
                await message.answer("Sorry, failed to process the audio. Please try again.")
                return

            pending = await self.task_queue.get_pending_count()
            if pending > 1:
                await message.answer(f"📋 Queued (position: {pending})")
            logger.info(f"Queued audio task {task.id} from user {message.from_user.id}")

        @self.dp.message(F.voice)
        async def handle_voice(message: Message):
            """Handle voice messages."""
            if not self._is_authorized(message.from_user.id):
                return

            if not self.task_queue:
                await message.answer("System not ready. Please try again later.")
                return

            try:
                file = await self.bot.get_file(message.voice.file_id)
                file_name = f"voice_{message.voice.file_id}.ogg"
                
                local_path = await self.save_file_locally(file.file_path, file_name)

                task = await self.task_queue.enqueue(
                    chat_id=message.chat.id,
                    user_id=message.from_user.id,
                    message=f"[Voice message: {message.voice.duration}s]",
                    metadata={
                        "username": message.from_user.username,
                        "first_name": message.from_user.first_name,
                        "message_id": message.message_id,
                        "attachments": [{
                            "type": "voice",
                            "local_path": str(local_path),
                            "duration": message.voice.duration,
                        }],
                    },
                )
            except Exception as e:
                logger.error(f"Failed to process voice: {e}", exc_info=True)
                await message.answer("Sorry, failed to process the voice message. Please try again.")
                return

            pending = await self.task_queue.get_pending_count()
            if pending > 1:
                await message.answer(f"📋 Queued (position: {pending})")
            logger.info(f"Queued voice task {task.id} from user {message.from_user.id}")

        @self.dp.message(F.video)
        async def handle_video(message: Message):
            """Handle video messages."""
            if not self._is_authorized(message.from_user.id):
                return

            if not self.task_queue:
                await message.answer("System not ready. Please try again later.")
                return

            try:
                file = await self.bot.get_file(message.video.file_id)
                file_name = message.video.file_name or f"video_{message.video.file_id}.mp4"
                
                local_path = await self.save_file_locally(file.file_path, file_name)

                caption = message.caption or f"[Video: {file_name}]"
                task = await self.task_queue.enqueue(
                    chat_id=message.chat.id,
                    user_id=message.from_user.id,
                    message=caption,
                    metadata={
                        "username": message.from_user.username,
                        "first_name": message.from_user.first_name,
                        "message_id": message.message_id,
                        "attachments": [{
                            "type": "video",
                            "local_path": str(local_path),
                            "file_name": file_name,
                            "mime_type": message.video.mime_type,
                            "duration": message.video.duration,
                        }],
                    },
                )
            except Exception as e:
                logger.error(f"Failed to process video: {e}", exc_info=True)
                await message.answer("Sorry, failed to process the video. Please try again.")
                return

            pending = await self.task_queue.get_pending_count()
            if pending > 1:
                await message.answer(f"📋 Queued (position: {pending})")
            logger.info(f"Queued video task {task.id} from user {message.from_user.id}")

        @self.dp.message(F.video_note)
        async def handle_video_note(message: Message):
            """Handle video note (round video) messages."""
            if not self._is_authorized(message.from_user.id):
                return

            if not self.task_queue:
                await message.answer("System not ready. Please try again later.")
                return

            try:
                file = await self.bot.get_file(message.video_note.file_id)
                file_name = f"video_note_{message.video_note.file_id}.mp4"
                
                local_path = await self.save_file_locally(file.file_path, file_name)

                task = await self.task_queue.enqueue(
                    chat_id=message.chat.id,
                    user_id=message.from_user.id,
                    message=f"[Video note: {message.video_note.duration}s]",
                    metadata={
                        "username": message.from_user.username,
                        "first_name": message.from_user.first_name,
                        "message_id": message.message_id,
                        "attachments": [{
                            "type": "video_note",
                            "local_path": str(local_path),
                            "duration": message.video_note.duration,
                        }],
                    },
                )
            except Exception as e:
                logger.error(f"Failed to process video note: {e}", exc_info=True)
                await message.answer("Sorry, failed to process the video note. Please try again.")
                return

            pending = await self.task_queue.get_pending_count()
            if pending > 1:
                await message.answer(f"📋 Queued (position: {pending})")
            logger.info(f"Queued video_note task {task.id} from user {message.from_user.id}")

        @self.dp.message(F.text)
        async def handle_message(message: Message):
            """Handle regular text messages."""
            if not self._is_authorized(message.from_user.id):
                logger.warning(f"Unauthorized message from user {message.from_user.id}")
                return

            if not self.task_queue:
                await message.answer("System not ready. Please try again later.")
                return

            # Queue the task
            task = await self.task_queue.enqueue(
                chat_id=message.chat.id,
                user_id=message.from_user.id,
                message=message.text,
                metadata={
                    "username": message.from_user.username,
                    "first_name": message.from_user.first_name,
                    "message_id": message.message_id,
                },
            )

            # Only show queue position if there are multiple tasks
            pending = await self.task_queue.get_pending_count()
            if pending > 1:
                await message.answer(f"📋 Queued (position: {pending})")

            logger.info(f"Queued task {task.id} from user {message.from_user.id}")

    def _is_authorized(self, user_id: int) -> bool:
        """Check if a user is authorized to use the bot."""
        if not self.settings.allowed_user_ids:
            return True  # No restrictions
        return user_id in self.settings.allowed_user_ids

    async def send_typing(self, chat_id: int):
        """Send typing indicator to a chat."""
        try:
            await self.bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception as e:
            logger.warning(f"Failed to send typing to {chat_id}: {e}")

    async def send_message(self, chat_id: int, text: str, parse_mode: Optional[str] = None, **kwargs):
        """Send a message to a chat (can be called from anywhere).
        
        Args:
            chat_id: Telegram chat ID
            text: Message text
            parse_mode: Optional parse mode. If None, sends as plain text (safest for agent responses)
        """
        # Split long messages
        max_len = 4096
        chunks = [text] if len(text) <= max_len else self._split_message(text, max_len)
        
        for chunk in chunks:
            try:
                await self.bot.send_message(chat_id, chunk, parse_mode=parse_mode, **kwargs)
            except Exception as e:
                # If markdown parsing fails, try without parse_mode
                if "parse entities" in str(e).lower():
                    await self.bot.send_message(chat_id, chunk, parse_mode=None, **kwargs)
                else:
                    raise
            if len(chunks) > 1:
                await asyncio.sleep(0.1)  # Rate limiting

    def _split_message(self, text: str, max_len: int) -> list[str]:
        """Split a long message into chunks."""
        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break

            # Try to split at a newline
            split_idx = text.rfind("\n", 0, max_len)
            if split_idx == -1 or split_idx < max_len // 2:
                # No good newline, split at space
                split_idx = text.rfind(" ", 0, max_len)
            if split_idx == -1 or split_idx < max_len // 2:
                # No good space, hard split
                split_idx = max_len

            chunks.append(text[:split_idx])
            text = text[split_idx:].lstrip()

        return chunks

    async def start(self):
        """Start the bot polling."""
        logger.info("Starting Telegram bot...")
        await self.dp.start_polling(self.bot, handle_signals=False)

    async def stop(self):
        """Stop the bot."""
        logger.info("Stopping Telegram bot...")
        await self.dp.stop_polling()
        await self.bot.session.close()


async def create_bot(
    settings: Optional[Settings] = None,
    task_queue: Optional[TaskQueue] = None,
) -> TelegramBot:
    """Create and return a TelegramBot instance."""
    return TelegramBot(settings=settings, task_queue=task_queue)
