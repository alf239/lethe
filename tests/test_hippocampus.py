"""Tests for hippocampus memory recall."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lethe.memory.hippocampus import Hippocampus


class MockMemoryStore:
    """Mock memory store for testing hippocampus."""
    
    def __init__(self):
        self.archival = MagicMock()
        self.messages = MagicMock()


class TestHippocampus:
    """Tests for Hippocampus memory retrieval."""
    
    @pytest.fixture
    def memory_store(self):
        """Create mock memory store."""
        return MockMemoryStore()
    
    @pytest.fixture
    def hippocampus(self, memory_store):
        """Create hippocampus with mock store."""
        return Hippocampus(memory_store, enabled=True)
    
    def test_disabled_returns_none(self, memory_store):
        """Should return None when disabled."""
        hippo = Hippocampus(memory_store, enabled=False)
        
        result = hippo.recall("test message")
        
        assert result is None
        memory_store.archival.search.assert_not_called()
    
    def test_recall_searches_archival(self, hippocampus, memory_store):
        """Should search archival memory."""
        memory_store.archival.search.return_value = [
            {"text": "relevant memory", "score": 0.8, "created_at": "2024-01-01"}
        ]
        memory_store.messages.search.return_value = []
        
        result = hippocampus.recall("test query")
        
        memory_store.archival.search.assert_called_once()
        assert "relevant memory" in result
    
    def test_recall_searches_conversations(self, hippocampus, memory_store):
        """Should search conversation history."""
        memory_store.archival.search.return_value = []
        memory_store.messages.search.return_value = [
            # First 5 are skipped (recent)
            {"role": "user", "content": "old1", "created_at": "2024-01-01T10:00"},
            {"role": "user", "content": "old2", "created_at": "2024-01-01T10:00"},
            {"role": "user", "content": "old3", "created_at": "2024-01-01T10:00"},
            {"role": "user", "content": "old4", "created_at": "2024-01-01T10:00"},
            {"role": "user", "content": "old5", "created_at": "2024-01-01T10:00"},
            # These should be included
            {"role": "user", "content": "past conversation", "created_at": "2024-01-01T10:00"},
        ]
        
        result = hippocampus.recall("test query")
        
        memory_store.messages.search.assert_called_once()
        assert "past conversation" in result
    
    def test_recall_filters_low_scores(self, hippocampus, memory_store):
        """Should filter out low-score archival results."""
        memory_store.archival.search.return_value = [
            {"text": "high score", "score": 0.8, "created_at": "2024-01-01"},
            {"text": "low score", "score": 0.1, "created_at": "2024-01-01"},
        ]
        memory_store.messages.search.return_value = []
        
        result = hippocampus.recall("test query")
        
        assert "high score" in result
        assert "low score" not in result
    
    def test_recall_returns_none_when_nothing_found(self, hippocampus, memory_store):
        """Should return None when no relevant memories found."""
        memory_store.archival.search.return_value = []
        memory_store.messages.search.return_value = []
        
        result = hippocampus.recall("test query")
        
        assert result is None
    
    def test_augment_message_adds_recall(self, hippocampus, memory_store):
        """Should augment message with recalled context."""
        memory_store.archival.search.return_value = [
            {"text": "context info", "score": 0.9, "created_at": "2024-01-01"}
        ]
        memory_store.messages.search.return_value = []
        
        result = hippocampus.augment_message("user question")
        
        assert "user question" in result
        assert "context info" in result
        assert "[Associative memory recall]" in result
    
    def test_augment_message_unchanged_when_no_recall(self, hippocampus, memory_store):
        """Should return original message when no recall found."""
        memory_store.archival.search.return_value = []
        memory_store.messages.search.return_value = []
        
        result = hippocampus.augment_message("user question")
        
        assert result == "user question"
    
    def test_recall_respects_max_lines(self, hippocampus, memory_store):
        """Should limit recalled memories to max_lines."""
        memory_store.archival.search.return_value = [
            {"text": f"memory {i}", "score": 0.9, "created_at": "2024-01-01"}
            for i in range(20)
        ]
        memory_store.messages.search.return_value = []
        
        result = hippocampus.recall("test", max_lines=5)
        
        # Should have at most 5 memory lines (plus header/footer)
        assert result.count("- [") <= 5
    
    def test_build_query_uses_recent_context(self, hippocampus, memory_store):
        """Should include recent user messages in query."""
        recent = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow up"},
        ]
        
        query = hippocampus._build_query("new question", recent)
        
        assert "new question" in query
        assert "previous question" in query
        assert "follow up" in query
    
    def test_handles_search_errors_gracefully(self, hippocampus, memory_store):
        """Should handle search errors without crashing."""
        memory_store.archival.search.side_effect = Exception("Search failed")
        memory_store.messages.search.return_value = []
        
        result = hippocampus.recall("test query")
        
        # Should not raise, returns None
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
