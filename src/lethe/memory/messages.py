"""Message history - Conversation storage.

Stores conversation messages for context and retrieval.
"""

import json
from datetime import datetime, timezone
from typing import Optional, List
import uuid

import lancedb
import logging

logger = logging.getLogger(__name__)


class MessageHistory:
    """Conversation message storage.
    
    Stores messages with role, content, and metadata.
    Supports retrieval by time range or search.
    """
    
    TABLE_NAME = "message_history"
    
    def __init__(self, db: lancedb.DBConnection):
        """Initialize message history.
        
        Args:
            db: LanceDB connection
        """
        self.db = db
        self._ensure_table()
    
    def _ensure_table(self):
        """Create table if it doesn't exist."""
        if self.TABLE_NAME not in self.db.table_names():
            self.db.create_table(
                self.TABLE_NAME,
                data=[{
                    "id": "_init_",
                    "role": "system",
                    "content": "",
                    "metadata": "{}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }]
            )
            logger.info(f"Created table {self.TABLE_NAME}")
            
            # Create FTS index for search
            table = self._get_table()
            table.create_fts_index("content", replace=True)
    
    def _get_table(self):
        """Get the messages table."""
        return self.db.open_table(self.TABLE_NAME)
    
    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a message to history.
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional metadata (tool_call_id, etc.)
            
        Returns:
            Message ID
        """
        message_id = f"msg-{uuid.uuid4()}"
        now = datetime.now(timezone.utc).isoformat()
        
        table = self._get_table()
        table.add([{
            "id": message_id,
            "role": role,
            "content": content,
            "metadata": json.dumps(metadata or {}),
            "created_at": now,
        }])
        
        logger.debug(f"Added message {message_id} ({role})")
        return message_id
    
    def get(self, message_id: str) -> Optional[dict]:
        """Get a message by ID.
        
        Args:
            message_id: Message ID
            
        Returns:
            Message dict or None
        """
        table = self._get_table()
        results = table.search().where(f"id = '{message_id}'").limit(1).to_list()
        
        if not results:
            return None
        
        r = results[0]
        return {
            "id": r["id"],
            "role": r["role"],
            "content": r["content"],
            "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
            "created_at": r["created_at"],
        }
    
    def get_recent(self, limit: int = 20) -> List[dict]:
        """Get recent messages.
        
        Args:
            limit: Max messages to return
            
        Returns:
            List of messages (oldest first)
        """
        table = self._get_table()
        # Get more than needed to filter out init
        results = table.search().limit(limit + 10).to_list()
        
        messages = []
        for r in results:
            if r["id"] == "_init_":
                continue
            messages.append({
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "created_at": r["created_at"],
            })
        
        # Sort by created_at ascending (oldest first for context)
        messages.sort(key=lambda m: m["created_at"])
        return messages[-limit:]  # Return most recent `limit` messages
    
    def search(self, query: str, limit: int = 20) -> List[dict]:
        """Search messages by content.
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            List of matching messages
        """
        table = self._get_table()
        results = table.search(query, query_type="fts").limit(limit).to_list()
        
        messages = []
        for r in results:
            if r["id"] == "_init_":
                continue
            messages.append({
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "created_at": r["created_at"],
                "score": r.get("_score", 0),
            })
        
        return messages
    
    def get_by_role(self, role: str, limit: int = 50) -> List[dict]:
        """Get messages by role.
        
        Args:
            role: Message role
            limit: Max results
            
        Returns:
            List of messages
        """
        table = self._get_table()
        results = table.search().where(f"role = '{role}'").limit(limit).to_list()
        
        messages = []
        for r in results:
            messages.append({
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "created_at": r["created_at"],
            })
        
        messages.sort(key=lambda m: m["created_at"])
        return messages
    
    def delete(self, message_id: str) -> bool:
        """Delete a message.
        
        Args:
            message_id: Message ID
            
        Returns:
            True if deleted
        """
        table = self._get_table()
        table.delete(f"id = '{message_id}'")
        return True
    
    def count(self) -> int:
        """Get total message count.
        
        Returns:
            Number of messages
        """
        table = self._get_table()
        return table.count_rows() - 1  # Exclude init row
    
    def clear(self) -> int:
        """Clear all messages.
        
        Returns:
            Number of messages deleted
        """
        count = self.count()
        table = self._get_table()
        table.delete("id != '_init_'")
        logger.info(f"Cleared {count} messages")
        return count
    
    def get_context_window(
        self,
        max_messages: int = 50,
        max_chars: int = 50000,
    ) -> List[dict]:
        """Get messages for LLM context window.
        
        Args:
            max_messages: Max number of messages
            max_chars: Max total characters
            
        Returns:
            List of messages fitting within limits
        """
        messages = self.get_recent(limit=max_messages)
        
        # Trim to fit within max_chars
        total_chars = 0
        result = []
        
        for msg in reversed(messages):  # Start from most recent
            msg_chars = len(msg["content"])
            if total_chars + msg_chars > max_chars:
                break
            result.insert(0, msg)
            total_chars += msg_chars
        
        return result
