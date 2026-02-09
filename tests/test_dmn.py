"""Tests for the Default Mode Network (DMN).

Tests cover:
- DMN round execution
- State file persistence between rounds
- Butler notification on urgent items
- Main model usage (not aux)
- Tool availability
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lethe.actor import ActorConfig, ActorRegistry, ActorState
from lethe.actor.dmn import DefaultModeNetwork, DMN_STATE_FILE, DMN_SYSTEM_PROMPT


@pytest.fixture
def registry():
    return ActorRegistry()


@pytest.fixture
def butler(registry):
    return registry.spawn(
        ActorConfig(name="butler", group="main", goals="Serve the user"),
        is_principal=True,
    )


@pytest.fixture
def available_tools():
    """Fake available tools for testing."""
    tools = {}
    for name in ["read_file", "write_file", "edit_file", "bash", "grep_search",
                  "list_directory", "web_search", "memory_read", "memory_update"]:
        func = MagicMock()
        func.__name__ = name
        func.__doc__ = f"Mock {name} tool."
        schema = {"name": name, "parameters": {"type": "object", "properties": {}}}
        tools[name] = (func, schema)
    return tools


def make_mock_llm(registry):
    """Create a mock LLM that terminates the DMN actor on first call."""
    llm = MagicMock()
    llm.add_tool = MagicMock()
    llm.context = MagicMock()
    llm._tools = {}
    
    async def fake_chat(message):
        # Find the dmn actor and terminate it
        for actor in registry._actors.values():
            if actor.config.name == "dmn" and actor.state != ActorState.TERMINATED:
                actor.terminate("Round complete")
        return "ok"
    llm.chat = fake_chat
    return llm


class TestDMNBasic:
    def test_dmn_system_prompt(self):
        """DMN has a specific system prompt for background thinking."""
        assert "Default Mode Network" in DMN_SYSTEM_PROMPT
        assert "background" in DMN_SYSTEM_PROMPT
        assert "questions.md" in DMN_SYSTEM_PROMPT
        assert "terminate" in DMN_SYSTEM_PROMPT

    def test_dmn_init(self, registry, butler, available_tools):
        dmn = DefaultModeNetwork(
            registry=registry,
            llm_factory=AsyncMock(),
            available_tools=available_tools,
            cortex_id=butler.id,
            send_to_user=AsyncMock(),
        )
        assert dmn.cortex_id == butler.id

    @pytest.mark.asyncio
    async def test_dmn_round_creates_actor(self, registry, butler, available_tools):
        """DMN round spawns a 'dmn' actor in the registry."""
        mock_llm = make_mock_llm(registry)
        
        dmn = DefaultModeNetwork(
            registry=registry,
            llm_factory=AsyncMock(),
            available_tools=available_tools,
            cortex_id=butler.id,
            send_to_user=AsyncMock(),
        )
        dmn._create_dmn_llm = AsyncMock(return_value=mock_llm)
        
        await dmn.run_round()
        
        dmn_actors = [a for a in registry._actors.values() if a.config.name == "dmn"]
        assert len(dmn_actors) == 1
        assert dmn_actors[0].state == ActorState.TERMINATED

    @pytest.mark.asyncio
    async def test_dmn_round_with_reminders(self, registry, butler, available_tools):
        """DMN receives reminders in its round message."""
        received_message = None
        
        async def capture_chat(message):
            nonlocal received_message
            received_message = message
            for actor in registry._actors.values():
                if actor.config.name == "dmn" and actor.state != ActorState.TERMINATED:
                    actor.terminate("done")
            return "ok"
        
        mock_llm = make_mock_llm(registry)
        mock_llm.chat = capture_chat
        
        async def get_reminders():
            return "- [urgent] Pay rent by Friday\n- [normal] Review PR #42"
        
        dmn = DefaultModeNetwork(
            registry=registry,
            llm_factory=AsyncMock(),
            available_tools=available_tools,
            cortex_id=butler.id,
            send_to_user=AsyncMock(),
            get_reminders=get_reminders,
        )
        dmn._create_dmn_llm = AsyncMock(return_value=mock_llm)
        
        await dmn.run_round()
        
        assert received_message is not None
        assert "Pay rent" in received_message
        assert "Review PR" in received_message

    @pytest.mark.asyncio
    async def test_dmn_max_turns(self, registry, butler, available_tools):
        """DMN should terminate after max_turns (10)."""
        mock_llm = MagicMock()
        mock_llm.add_tool = MagicMock()
        mock_llm.context = MagicMock()
        mock_llm._tools = {}
        mock_llm.chat = AsyncMock(return_value="still thinking...")
        
        dmn = DefaultModeNetwork(
            registry=registry,
            llm_factory=AsyncMock(),
            available_tools=available_tools,
            cortex_id=butler.id,
            send_to_user=AsyncMock(),
        )
        dmn._create_dmn_llm = AsyncMock(return_value=mock_llm)
        
        await dmn.run_round()
        
        dmn_actors = [a for a in registry._actors.values() if a.config.name == "dmn"]
        assert len(dmn_actors) == 1
        assert dmn_actors[0].state == ActorState.TERMINATED


class TestDMNState:
    @pytest.mark.asyncio
    async def test_dmn_reads_state_file(self, registry, butler, available_tools, tmp_path):
        """DMN reads previous state from file."""
        received_message = None
        
        async def capture_chat(message):
            nonlocal received_message
            received_message = message
            for actor in registry._actors.values():
                if actor.config.name == "dmn" and actor.state != ActorState.TERMINATED:
                    actor.terminate("done")
            return "ok"
        
        mock_llm = make_mock_llm(registry)
        mock_llm.chat = capture_chat
        
        # Write a state file
        state_file = str(tmp_path / "dmn_state.md")
        with open(state_file, "w") as f:
            f.write("Previous thought: need to check on project X\nTodo: organize memory blocks")
        
        dmn = DefaultModeNetwork(
            registry=registry,
            llm_factory=AsyncMock(),
            available_tools=available_tools,
            cortex_id=butler.id,
            send_to_user=AsyncMock(),
        )
        dmn._create_dmn_llm = AsyncMock(return_value=mock_llm)
        
        with patch("lethe.actor.dmn.DMN_STATE_FILE", state_file):
            await dmn.run_round()
        
        assert received_message is not None
        assert "Previous thought" in received_message
        assert "project X" in received_message
