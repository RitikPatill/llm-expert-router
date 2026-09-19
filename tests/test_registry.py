"""Tests for the expert registry loader."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llm_expert_router.registry import load_experts

REPO_ROOT = Path(__file__).parent.parent


def test_load_valid_yaml():
    experts = load_experts(REPO_ROOT / "experts.yaml")
    assert len(experts) == 6
    assert set(experts.keys()) == {"code", "reasoning", "summarisation", "translation", "creative", "general"}
    assert experts["code"].model == "gpt-4o-mini"
    assert experts["creative"].temperature == 0.9


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_experts("nonexistent.yaml")


def test_invalid_extra_field_raises(tmp_path: Path):
    bad_yaml = {
        "experts": {
            "code": {
                "model": "gpt-4o-mini",
                "system_prompt": "You are a coder.",
                "unknown_field": "oops",
            }
        }
    }
    p = tmp_path / "experts.yaml"
    p.write_text(yaml.dump(bad_yaml))
    with pytest.raises(ValidationError):
        load_experts(p)


def test_defaults_applied(tmp_path: Path):
    minimal_yaml = {
        "experts": {
            "simple": {
                "model": "gpt-4o-mini",
                "system_prompt": "You are helpful.",
            }
        }
    }
    p = tmp_path / "experts.yaml"
    p.write_text(yaml.dump(minimal_yaml))
    experts = load_experts(p)
    assert experts["simple"].temperature == 0.7
    assert experts["simple"].max_tokens == 1024
    assert experts["simple"].description == ""
