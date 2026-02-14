"""Architecture-level integration tests for actor/runtime behavior."""

import pytest

from lethe.actor import ActorConfig, ActorRegistry
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
