"""Telegram bot interface."""

import asyncio
import io
import logging
from typing import Callable, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.enums import ChatAction
from aiogram.types import Message
from letta_client import AsyncLetta

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
    ):
        self.settings = settings or get_settings()
        self.task_queue = task_queue
        self.bg_task_manager = bg_task_manager
        self.on_task_stopped = on_task_stopped  # async callback(task_ids: list[str])
        
        self._letta_client: Optional[AsyncLetta] = None
        self._letta_folder_id: Optional[str] = None

        self.bot = Bot(
            token=self.settings.telegram_bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
        )
        self.dp = Dispatcher()

        self._setup_handlers()

    @property
    def letta_client(self) -> AsyncLetta:
        """Get or create the Letta client for file uploads."""
        if self._letta_client is None:
            if self.settings.letta_api_key:
                self._letta_client = AsyncLetta(
                    base_url=self.settings.letta_base_url,
                    api_key=self.settings.letta_api_key,
                )
            else:
                self._letta_client = AsyncLetta(base_url=self.settings.letta_base_url)
        return self._letta_client
    
    async def get_or_create_folder(self) -> str:
        """Get or create a folder for storing Telegram images."""
        if self._letta_folder_id is None:
            # Try to find existing folder named "telegram_images"
            folders = await self.letta_client.folders.list()
            for folder in folders:
                if getattr(folder, "name", None) == "telegram_images":
                    self._letta_folder_id = folder.id
                    break
            
            # Create if it doesn't exist
            if self._letta_folder_id is None:
                folder = await self.letta_client.folders.create(
                    name="telegram_images",
                    description="Images uploaded from Telegram"
                )
                self._letta_folder_id = folder.id
                logger.info(f"Created Letta folder for images: {self._letta_folder_id}")
        
        return self._letta_folder_id

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
                "/stop - Stop all background tasks\n"
                "/stop N - Stop specific task (N from /list)"
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
            """Handle /stop command - stop background tasks."""
            if not self._is_authorized(message.from_user.id):
                return

            if not self.bg_task_manager:
                await message.answer("Background task manager not initialized.")
                return

            from lethe.tasks import TaskStatus
            
            # Parse argument (optional task number)
            args = message.text.split(maxsplit=1)
            task_number = None
            if len(args) > 1:
                try:
                    task_number = int(args[1])
                except ValueError:
                    await message.answer("Usage: /stop or /stop N (where N is task number from /list)")
                    return

            # Get active tasks
            tasks = await self.bg_task_manager.list_tasks(limit=20)
            active_tasks = [t for t in tasks if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]
            
            if not active_tasks:
                await message.answer("No active background tasks to stop.")
                return

            # Determine which tasks to stop
            if task_number is not None:
                if task_number < 1 or task_number > len(active_tasks):
                    await message.answer(f"Invalid task number. Use 1-{len(active_tasks)}.")
                    return
                tasks_to_stop = [active_tasks[task_number - 1]]
            else:
                tasks_to_stop = active_tasks

            # Stop the tasks
            stopped_ids = []
            for task in tasks_to_stop:
                success = await self.bg_task_manager.cancel_task(task.id)
                if success:
                    stopped_ids.append(task.id)

            if stopped_ids:
                await message.answer(f"🛑 Stopped {len(stopped_ids)} task(s).")
                
                # Notify agent about stopped tasks
                if self.on_task_stopped:
                    try:
                        await self.on_task_stopped(stopped_ids, message.chat.id)
                    except Exception as e:
                        logger.warning(f"Error notifying agent about stopped tasks: {e}")
            else:
                await message.answer("Failed to stop tasks.")

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
                
                # Download the file from Telegram
                file_bytes = io.BytesIO()
                await self.bot.download_file(file.file_path, file_bytes)
                file_bytes.seek(0)
                
                # Upload to Letta
                folder_id = await self.get_or_create_folder()
                file_name = file.file_path.split('/')[-1]  # Get original filename
                
                upload_response = await self.letta_client.folders.files.upload(
                    folder_id=folder_id,
                    file=file_bytes,
                    name=file_name,
                    duplicate_handling="suffix"  # Add suffix if duplicate
                )
                
                letta_file_id = upload_response.id
                logger.info(f"Uploaded image to Letta: {letta_file_id}")
                
                # Queue the task with Letta file_id
                caption = message.caption or ""
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
                            "letta_file_id": letta_file_id,
                        }],
                    },
                )
            except Exception as e:
                logger.error(f"Failed to upload image to Letta: {e}", exc_info=True)
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

            file = await self.bot.get_file(message.document.file_id)
            file_url = f"https://api.telegram.org/file/bot{self.settings.telegram_bot_token}/{file.file_path}"

            # Queue the task with document URL
            caption = message.caption or f"Sent file: {message.document.file_name}"
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
                        "url": file_url,
                        "file_id": message.document.file_id,
                        "file_name": message.document.file_name,
                        "mime_type": message.document.mime_type,
                    }],
                },
            )

            # Only show queue position if there are multiple tasks
            pending = await self.task_queue.get_pending_count()
            if pending > 1:
                await message.answer(f"📋 Queued (position: {pending})")

            logger.info(f"Queued document task {task.id} from user {message.from_user.id}")

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
