from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from llm_expert_router.demo import _truncate, build_table, call_chat


def _fake_response_payload(prompt: str = "hello") -> dict:
    return {
        "_prompt": prompt,
        "response": "Fake response",
        "routing_metadata": {
            "task_category": "general",
            "expert_name": "general",
            "model": "gpt-4o-mini",
            "classification_method": "keyword",
            "latency_ms": 123.4,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "estimated_cost_usd": 0.000015,
        },
    }


def test_call_chat_returns_dict(monkeypatch):
    payload = _fake_response_payload("hello")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=payload)

    mock_post = MagicMock(return_value=mock_response)

    monkeypatch.setattr("httpx.Client.post", mock_post)

    result = call_chat("hello")
    assert isinstance(result, dict)
    assert "routing_metadata" in result
    assert result["routing_metadata"]["expert_name"] == "general"


def test_build_table_has_correct_columns():
    results = [_fake_response_payload("prompt one"), _fake_response_payload("prompt two")]
    table = build_table(results)

    col_names = [col.header for col in table.columns]
    assert col_names == ["#", "Prompt", "Category", "Expert", "Model", "Method", "Latency(ms)", "Cost(USD)"]
    assert len(table.rows) == 2


def test_truncate_at_limit():
    text = "a" * 48
    assert _truncate(text, 48) == text


def test_truncate_over_limit():
    text = "a" * 49
    result = _truncate(text, 48)
    assert len(result) == 48
    assert result.endswith("\u2026")


def test_truncate_under_limit():
    text = "short"
    assert _truncate(text, 48) == "short"
