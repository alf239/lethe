"""Todo management tools for the agent.

These tools allow the agent to track tasks persistently:
- Create todos from user requests or recalled memories
- List and search existing todos
- Update status, complete tasks
- Smart reminders prevent spam while ensuring follow-through
"""

from typing import Optional
from lethe.todos import TodoManager


def create_todo_tools(todo_manager: TodoManager) -> list[dict]:
    """Create todo tool functions bound to a TodoManager instance.
    
    Returns list of (function, requires_approval) tuples.
    """
    
    async def todo_create(
        title: str,
        description: str = "",
        priority: str = "normal",
        due_date: str = "",
    ) -> str:
        """Create a new todo/task to track.
        
        Use this when:
        - User requests something that will take time
        - You recall an unfinished task from memory
        - You identify something that needs follow-up
        
        Args:
            title: Short task title (e.g., "Deploy to production")
            description: Details about what needs to be done
            priority: low, normal, high, or urgent
            due_date: Optional due date in YYYY-MM-DD format
            
        Returns:
            Confirmation with todo ID
        """
        todo_id = await todo_manager.create(
            title=title,
            description=description if description else None,
            priority=priority,
            due_date=due_date if due_date else None,
        )
        return f"Created todo #{todo_id}: {title}"

    async def todo_list(
        status: str = "",
        priority: str = "",
        include_completed: bool = False,
    ) -> str:
        """List todos/tasks.
        
        Use this to:
        - See what tasks are pending
        - Check if a recalled task is already tracked
        - Review completed work
        
        Args:
            status: Filter by status (pending, in_progress, completed, deferred, cancelled)
            priority: Filter by priority (low, normal, high, urgent)
            include_completed: Whether to include completed/cancelled tasks
            
        Returns:
            Formatted list of todos
        """
        todos = await todo_manager.list(
            status=status if status else None,
            priority=priority if priority else None,
            include_completed=include_completed,
        )
        
        if not todos:
            return "No todos found matching criteria."
        
        lines = [f"Found {len(todos)} todo(s):\n"]
        for todo in todos:
            status_icon = {
                "pending": "⬜",
                "in_progress": "🔄",
                "completed": "✅",
                "deferred": "⏸️",
                "cancelled": "❌",
            }.get(todo["status"], "•")
            
            priority_icon = {
                "urgent": "🔴",
                "high": "🟠",
                "normal": "",
                "low": "⚪",
            }.get(todo["priority"], "")
            
            due = f" (due: {todo['due_date']})" if todo.get("due_date") else ""
            reminded = f" [reminded {todo['remind_count']}x]" if todo.get("remind_count", 0) > 0 else ""
            
            lines.append(f"{status_icon} #{todo['id']} {priority_icon}{todo['title']}{due}{reminded}")
            if todo.get("description"):
                lines.append(f"   {todo['description'][:100]}")
        
        return "\n".join(lines)

    async def todo_update(
        todo_id: int,
        status: str = "",
        priority: str = "",
        description: str = "",
        due_date: str = "",
    ) -> str:
        """Update an existing todo.
        
        Args:
            todo_id: ID of the todo to update
            status: New status (pending, in_progress, completed, deferred, cancelled)
            priority: New priority (low, normal, high, urgent)
            description: New description
            due_date: New due date (YYYY-MM-DD)
            
        Returns:
            Confirmation or error
        """
        updated = await todo_manager.update(
            todo_id=todo_id,
            status=status if status else None,
            priority=priority if priority else None,
            description=description if description else None,
            due_date=due_date if due_date else None,
        )
        
        if updated:
            return f"Updated todo #{todo_id}"
        return f"Todo #{todo_id} not found"

    async def todo_complete(todo_id: int) -> str:
        """Mark a todo as completed.
        
        Use this when a task is finished.
        
        Args:
            todo_id: ID of the todo to complete
            
        Returns:
            Confirmation or error
        """
        completed = await todo_manager.complete(todo_id)
        if completed:
            return f"✅ Completed todo #{todo_id}"
        return f"Todo #{todo_id} not found"

    async def todo_search(query: str) -> str:
        """Search todos by title or description.
        
        Use this to check if a task is already being tracked
        before creating a duplicate.
        
        Args:
            query: Search text
            
        Returns:
            Matching todos or "no matches"
        """
        todos = await todo_manager.search(query)
        
        if not todos:
            return f"No todos matching '{query}'"
        
        lines = [f"Found {len(todos)} matching todo(s):"]
        for todo in todos:
            lines.append(f"#{todo['id']} [{todo['status']}] {todo['title']}")
        
        return "\n".join(lines)

    async def todo_remind_check() -> str:
        """Check which todos are due for a reminder.
        
        Returns todos where enough time has passed since the last reminder.
        Reminder intervals vary by priority:
        - Urgent: 1 hour
        - High: 4 hours  
        - Normal: 1 day
        - Low: 1 week
        
        After reminding user, call todo_reminded(id) to prevent spam.
        
        Returns:
            List of todos due for reminder, or "nothing due"
        """
        todos = await todo_manager.get_due_reminders()
        
        if not todos:
            return "No todos due for reminder."
        
        lines = [f"{len(todos)} todo(s) due for reminder:"]
        for todo in todos:
            priority_icon = {"urgent": "🔴", "high": "🟠", "normal": "", "low": "⚪"}.get(todo["priority"], "")
            lines.append(f"#{todo['id']} {priority_icon}{todo['title']}")
            if todo.get("description"):
                lines.append(f"   {todo['description'][:100]}")
        
        return "\n".join(lines)

    async def todo_reminded(todo_id: int) -> str:
        """Mark that you reminded the user about a todo.
        
        Call this AFTER telling the user about a pending task.
        This prevents spamming them with the same reminder.
        
        Args:
            todo_id: ID of the todo you reminded about
            
        Returns:
            Confirmation
        """
        marked = await todo_manager.mark_reminded(todo_id)
        if marked:
            return f"Marked todo #{todo_id} as reminded"
        return f"Todo #{todo_id} not found"

    # Return tools with approval requirements
    return [
        (todo_create, False),
        (todo_list, False),
        (todo_update, False),
        (todo_complete, False),
        (todo_search, False),
        (todo_remind_check, False),
        (todo_reminded, False),
    ]
