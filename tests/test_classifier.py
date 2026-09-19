"""Tests for the prompt classifier."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from llm_expert_router.classifier import _heuristic_classify, classify
from llm_expert_router.registry import load_experts

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture()
def experts():
    return load_experts(REPO_ROOT / "experts.yaml")


# ---------------------------------------------------------------------------
# Heuristic path (no LLM call)
# ---------------------------------------------------------------------------


def test_heuristic_code():
    assert _heuristic_classify("Write a Python function to sort a list") == "code"


def test_heuristic_translation():
    assert _heuristic_classify("Translate this to French") == "translation"


def test_heuristic_fallback_general():
    assert _heuristic_classify("Hello there") == "general"


# ---------------------------------------------------------------------------
# classify() — heuristic path when no client
# ---------------------------------------------------------------------------


async def test_classify_uses_heuristic_when_no_client(experts):
    category, method = await classify("Write a Python function", experts, client=None)
    assert method == "heuristic"
    assert category in experts


# ---------------------------------------------------------------------------
# classify() — LLM path (mocked)
# ---------------------------------------------------------------------------


def _make_mock_client(content: str) -> MagicMock:
    """Build a mock AsyncOpenAI client that returns `content` as the message."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_create = AsyncMock(return_value=mock_response)

    mock_client = MagicMock(spec=openai.AsyncOpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = mock_create

    return mock_client


async def test_classify_llm_path(experts):
    client = _make_mock_client("code")
    category, method = await classify("Write a sorting algorithm", experts, client=client)
    assert category == "code"
    assert method == "llm"


async def test_classify_llm_fallback_on_exception(experts):
    mock_client = MagicMock(spec=openai.AsyncOpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=openai.APIConnectionError(request=MagicMock())
    )

    _, method = await classify("Write a Python function", experts, client=mock_client)
    assert method == "heuristic"


async def test_classify_unknown_expert_falls_back(experts):
    client = _make_mock_client("foobar")
    category, _ = await classify("some prompt", experts, client=client)
    assert category == "general"
