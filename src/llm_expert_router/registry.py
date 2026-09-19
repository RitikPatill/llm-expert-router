from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError  # noqa: F401 (re-exported for callers)

DEFAULT_EXPERTS_PATH = Path("experts.yaml")


class ExpertConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 1024
    description: str = ""


class ExpertRegistry(BaseModel):
    experts: dict[str, ExpertConfig]


def load_experts(path: str | Path) -> dict[str, ExpertConfig]:
    """Load and validate experts.yaml.

    Raises:
        FileNotFoundError: if the file does not exist.
        pydantic.ValidationError: if the YAML does not match the schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experts file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    registry = ExpertRegistry.model_validate(raw)
    return registry.experts
