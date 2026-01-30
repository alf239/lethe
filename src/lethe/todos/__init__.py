"""Todo management system with SQLite backend.

Provides persistent task tracking with smart reminder logic:
- Tasks persist until explicitly completed
- Tracks when user was last reminded about each task
- Prevents spam while ensuring nothing is forgotten
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List
import aiosqlite

logger = logging.getLogger(__name__)


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class TodoPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# Minimum time between reminders per priority
REMIND_INTERVALS = {
    TodoPriority.LOW: timedelta(days=7),
    TodoPriority.NORMAL: timedelta(days=1),
    TodoPriority.HIGH: timedelta(hours=4),
    TodoPriority.URGENT: timedelta(hours=1),
}


class TodoManager:
    """Manages todos in SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialized = False

    async def _ensure_initialized(self):
        """Create tables if they don't exist."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS todos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    priority TEXT DEFAULT 'normal',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    due_date TEXT,
                    last_reminded_at TEXT,
                    remind_count INTEGER DEFAULT 0,
                    tags TEXT,
                    source TEXT
                )
            """)
            await db.commit()

        self._initialized = True
        logger.info(f"TodoManager initialized with database: {self.db_path}")

    async def create(
        self,
        title: str,
        description: Optional[str] = None,
        priority: str = "normal",
        due_date: Optional[str] = None,
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
    ) -> int:
        """Create a new todo.
        
        Args:
            title: Short task title
            description: Detailed description
            priority: low, normal, high, urgent
            due_date: Optional due date (YYYY-MM-DD)
            tags: Optional list of tags
            source: Where this task came from (e.g., "user request", "memory recall")
            
        Returns:
            ID of created todo
        """
        await self._ensure_initialized()

        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags) if tags else None

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO todos (title, description, priority, due_date, tags, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (title, description, priority, due_date, tags_json, source, now, now),
            )
            await db.commit()
            todo_id = cursor.lastrowid

        logger.info(f"Created todo #{todo_id}: {title}")
        return todo_id

    async def list(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        include_completed: bool = False,
        limit: int = 50,
    ) -> List[dict]:
        """List todos with optional filters.
        
        Args:
            status: Filter by status
            priority: Filter by priority
            include_completed: Include completed/cancelled tasks
            limit: Maximum results
            
        Returns:
            List of todo dicts
        """
        await self._ensure_initialized()

        query = "SELECT * FROM todos WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        elif not include_completed:
            query += " AND status NOT IN ('completed', 'cancelled')"

        if priority:
            query += " AND priority = ?"
            params.append(priority)

        # Order by priority (urgent first), then due date, then created
        query += """
            ORDER BY 
                CASE priority 
                    WHEN 'urgent' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'normal' THEN 3 
                    WHEN 'low' THEN 4 
                END,
                due_date NULLS LAST,
                created_at DESC
            LIMIT ?
        """
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    async def get(self, todo_id: int) -> Optional[dict]:
        """Get a single todo by ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM todos WHERE id = ?", (todo_id,))
            row = await cursor.fetchone()

        return dict(row) if row else None

    async def update(
        self,
        todo_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        due_date: Optional[str] = None,
    ) -> bool:
        """Update a todo.
        
        Returns:
            True if updated, False if not found
        """
        await self._ensure_initialized()

        updates = []
        params = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status == "completed":
                updates.append("completed_at = ?")
                params.append(datetime.now(timezone.utc).isoformat())
        if priority is not None:
            updates.append("priority = ?")
            params.append(priority)
        if due_date is not None:
            updates.append("due_date = ?")
            params.append(due_date)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(todo_id)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"UPDATE todos SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            await db.commit()
            updated = cursor.rowcount > 0

        if updated:
            logger.info(f"Updated todo #{todo_id}")
        return updated

    async def complete(self, todo_id: int) -> bool:
        """Mark a todo as completed."""
        return await self.update(todo_id, status="completed")

    async def mark_reminded(self, todo_id: int) -> bool:
        """Mark that we reminded user about this todo."""
        await self._ensure_initialized()

        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE todos 
                SET last_reminded_at = ?, remind_count = remind_count + 1, updated_at = ?
                WHERE id = ?
                """,
                (now, now, todo_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get_due_reminders(self) -> List[dict]:
        """Get todos that are due for a reminder.
        
        Returns todos where enough time has passed since last reminder
        based on their priority.
        """
        await self._ensure_initialized()

        now = datetime.now(timezone.utc)
        results = []

        # Get all active todos
        todos = await self.list(include_completed=False)

        for todo in todos:
            priority = TodoPriority(todo.get("priority", "normal"))
            interval = REMIND_INTERVALS[priority]

            last_reminded = todo.get("last_reminded_at")
            if last_reminded:
                last_reminded_dt = datetime.fromisoformat(last_reminded.replace("Z", "+00:00"))
                if now - last_reminded_dt < interval:
                    continue  # Too soon to remind again

            results.append(todo)

        return results

    async def search(self, query: str, limit: int = 20) -> List[dict]:
        """Search todos by title or description."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM todos 
                WHERE (title LIKE ? OR description LIKE ?)
                AND status NOT IN ('completed', 'cancelled')
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            )
            rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    async def delete(self, todo_id: int) -> bool:
        """Delete a todo (use complete/cancel instead normally)."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
            await db.commit()
            return cursor.rowcount > 0
