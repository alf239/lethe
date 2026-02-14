"""Architecture-level integration tests for actor/runtime behavior."""

import asyncio
from pathlib import Path
import pytest
from unittest.mock import AsyncMock

from lethe.actor import ActorConfig, ActorRegistry
from lethe.actor.integration import ActorSystem
from lethe.memory.llm import AsyncLLMClient, LLMConfig


@pytest.mark.asyncio
async def test_user_notify_event_emitted():
    registry = ActorRegistry()
    cortex = registry.spawn(ActorConfig(name="cortex", group="main", goals="serve"), is_principal=True)
    worker = registry.spawn(ActorConfig(name="worker", group="main", goals="work"), spawned_by=cortex.id)

    await worker.send_to(cortex.id, "[USER_NOTIFY] Check the deadline now")

    events = registry.events.query(event_type="user_notify", actor_id=worker.id)
    assert events
    assert events[-1].payload["message"] == "Check the deadline now"


@pytest.mark.asyncio
async def test_llm_circuit_breaker_forces_final_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = LLMConfig(
        provider="openrouter",
        model="openrouter/moonshotai/kimi-k2.5-0127",
        model_aux="openrouter/qwen/qwen3-coder-next",
    )
    client = AsyncLLMClient(config=config, system_prompt="test")

    call_counts = {"api": 0, "no_tools": 0}

    async def fake_call_api():
        call_counts["api"] += 1
        return {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": f"call-{call_counts['api']}",
                        "function": {"name": "noop", "arguments": "{}"},
                    }],
                }
            }]
        }

    async def fake_call_api_no_tools():
        call_counts["no_tools"] += 1
        return {"choices": [{"message": {"content": "partial report"}}]}

    client._call_api = fake_call_api
    client._call_api_no_tools = fake_call_api_no_tools

    result = await client.chat("Do the task", max_tool_iterations=10)
    assert result == "partial report"
    assert call_counts["no_tools"] == 1
    assert call_counts["api"] < 10  # Circuit breaker should stop early.


@pytest.mark.asyncio
async def test_principal_monitor_forwards_done_and_failed_updates(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class DummyLLM:
        def __init__(self):
            self._tools = {}
            self.tools = []
            self.context = type("Ctx", (), {"_tool_reference": "", "_build_tool_reference": lambda self, tools: ""})()
        def add_tool(self, func, schema=None):
            self._tools[func.__name__] = (func, schema or {})
        def _update_tool_budget(self):
            return None

    class DummyAgent:
        def __init__(self):
            self.llm = DummyLLM()
        def add_tool(self, func):
            self.llm.add_tool(func)

    agent = DummyAgent()
    actor_system = ActorSystem(agent)
    await actor_system.setup()
    send_to_user = AsyncMock()
    actor_system.set_callbacks(send_to_user=send_to_user)

    principal = actor_system.principal
    worker = actor_system.registry.spawn(
        ActorConfig(name="worker", group="main", goals="Do work"),
        spawned_by=principal.id,
    )

    worker.terminate("finished result")
    await asyncio.sleep(1.2)
    send_to_user.assert_awaited()
    assert any("[DONE]" in c.args[0] for c in send_to_user.await_args_list)

    send_to_user.reset_mock()
    failed = actor_system.registry.spawn(
        ActorConfig(name="failed-worker", group="main", goals="Fail"),
        spawned_by=principal.id,
    )
    failed.terminate("Error: boom")
    await asyncio.sleep(1.2)
    assert any("[FAILED]" in c.args[0] for c in send_to_user.await_args_list)

    await actor_system.shutdown()


@pytest.mark.asyncio
async def test_view_image_tool_registered_and_available_to_cortex(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class DummyBlocks:
        def get(self, label):
            if label == "identity":
                return {"value": "You are a test assistant."}
            return None
        def list_blocks(self):
            return []

    class DummyMessages:
        def get_recent(self, n):
            return []
        def add(self, role, content, metadata=None):
            return None
        def count(self):
            return 0
        def search(self, query, limit=10):
            return []
        def search_by_role(self, query, role, limit=10):
            return []

    class DummyArchival:
        def add(self, text):
            return "id"
        def search(self, query, limit=10, search_type="hybrid"):
            return []
        def count(self):
            return 0

    class DummyMemory:
        def __init__(self):
            self.blocks = DummyBlocks()
            self.messages = DummyMessages()
            self.archival = DummyArchival()
        def get_context_for_prompt(self):
            return ""

    class DummySettings:
        llm_model = ""
        llm_model_aux = ""
        llm_api_base = ""
        llm_context_limit = 8000
        memory_dir = Path(".")
        workspace_dir = Path(".")
        lethe_config_dir = Path(".")
        llm_messages_load = 0
        llm_messages_summarize = 0

    from lethe.agent import Agent
    from lethe.actor.integration import ActorSystem
    from unittest.mock import patch

    with patch("lethe.agent.get_settings", return_value=DummySettings()), patch("lethe.agent.MemoryStore", return_value=DummyMemory()):
        agent = Agent()
        assert "view_image" in agent.llm._tools
        actor_system = ActorSystem(agent)
        await actor_system.setup()
        assert "view_image" in agent.llm._tools  # Kept on cortex in hybrid mode.
