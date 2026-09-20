from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _make_fake_completion(content: str = "Hello!", prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build a fake ChatCompletion-shaped object."""
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_mock_client(content: str = "Hello!", prompt_tokens: int = 10, completion_tokens: int = 5):
    fake = _make_fake_completion(content, prompt_tokens, completion_tokens)
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=fake)
    return mock_client


@pytest.fixture()
async def test_client(tmp_path):
    from llm_expert_router.app import app
    from llm_expert_router.telemetry import init_db
    from llm_expert_router.registry import load_experts

    import pathlib
    experts_path = pathlib.Path(__file__).parent.parent / "experts.yaml"
    db_path = str(tmp_path / "test_telemetry.db")

    experts = load_experts(experts_path)
    engine = await init_db(db_path)
    mock_client = _make_mock_client()

    app.state.experts = experts
    app.state.engine = engine
    app.state.client = mock_client

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    await engine.dispose()


@pytest.fixture()
async def test_client_no_key(tmp_path):
    from llm_expert_router.app import app
    from llm_expert_router.telemetry import init_db
    from llm_expert_router.registry import load_experts

    import pathlib
    experts_path = pathlib.Path(__file__).parent.parent / "experts.yaml"
    db_path = str(tmp_path / "test_telemetry_nokey.db")

    experts = load_experts(experts_path)
    engine = await init_db(db_path)

    app.state.experts = experts
    app.state.engine = engine
    app.state.client = None  # no API key

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    await engine.dispose()


async def test_chat_returns_response_shape(test_client):
    resp = await test_client.post("/chat", json={"message": "Hello there"})
    assert resp.status_code == 200
    body = resp.json()
    assert "response" in body
    assert isinstance(body["response"], str)
    meta = body["routing_metadata"]
    for key in ("task_category", "expert_name", "model", "classification_method",
                "latency_ms", "prompt_tokens", "completion_tokens", "estimated_cost_usd"):
        assert key in meta, f"Missing key: {key}"


async def test_chat_routing_metadata_types(test_client):
    resp = await test_client.post("/chat", json={"message": "Summarize this article"})
    assert resp.status_code == 200
    meta = resp.json()["routing_metadata"]
    assert isinstance(meta["latency_ms"], float)
    assert meta["latency_ms"] >= 0
    assert isinstance(meta["prompt_tokens"], int)
    assert meta["prompt_tokens"] >= 0
    assert isinstance(meta["completion_tokens"], int)
    assert meta["completion_tokens"] >= 0
    assert isinstance(meta["estimated_cost_usd"], float)
    assert meta["estimated_cost_usd"] >= 0.0


async def test_chat_expert_selected_matches_category(tmp_path):
    from llm_expert_router.app import app
    from llm_expert_router.telemetry import init_db
    from llm_expert_router.registry import load_experts
    import pathlib

    experts_path = pathlib.Path(__file__).parent.parent / "experts.yaml"
    db_path = str(tmp_path / "test_code.db")
    experts = load_experts(experts_path)
    engine = await init_db(db_path)

    # mock client: classifier returns "code" via LLM path
    fake_classifier_result = _make_fake_completion(content="code")
    fake_chat_result = _make_fake_completion(content="Here is your code")

    call_count = 0

    async def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return fake_classifier_result  # classifier call
        return fake_chat_result  # expert call

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=side_effect)

    app.state.experts = experts
    app.state.engine = engine
    app.state.client = mock_client

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/chat", json={"message": "write a python function"})

    await engine.dispose()

    assert resp.status_code == 200
    meta = resp.json()["routing_metadata"]
    assert meta["expert_name"] == "code"


async def test_stats_empty_db(tmp_path):
    from llm_expert_router.app import app
    from llm_expert_router.telemetry import init_db
    from llm_expert_router.registry import load_experts
    import pathlib

    experts_path = pathlib.Path(__file__).parent.parent / "experts.yaml"
    db_path = str(tmp_path / "test_empty.db")
    experts = load_experts(experts_path)
    engine = await init_db(db_path)
    mock_client = _make_mock_client()

    app.state.experts = experts
    app.state.engine = engine
    app.state.client = mock_client

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/stats")

    await engine.dispose()

    assert resp.status_code == 200
    assert resp.json() == []


async def test_stats_after_request(test_client):
    await test_client.post("/chat", json={"message": "Hello"})

    resp = await test_client.get("/stats")
    assert resp.status_code == 200
    stats = resp.json()
    assert len(stats) == 1
    entry = stats[0]
    assert "expert_name" in entry
    assert entry["calls"] == 1
    assert "avg_latency_ms" in entry
    assert "total_cost_usd" in entry


async def test_chat_no_api_key_returns_503(test_client_no_key):
    resp = await test_client_no_key.post("/chat", json={"message": "Hello"})
    assert resp.status_code == 503
