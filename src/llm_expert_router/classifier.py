from __future__ import annotations

import re
from typing import TYPE_CHECKING

import openai

from llm_expert_router.registry import ExpertConfig

if TYPE_CHECKING:
    pass

CATEGORY_LABELS: list[str] = [
    "code",
    "reasoning",
    "summarisation",
    "translation",
    "creative",
    "general",
]

KEYWORD_PATTERNS: dict[str, list[str]] = {
    "code": [
        r"\bcode\b",
        r"\bfunction\b",
        r"\bpython\b",
        r"\bbug\b",
        r"\bdebug\b",
        r"\bscript\b",
    ],
    "reasoning": [
        r"\bprove\b",
        r"\breason\b",
        r"\bmath\b",
        r"\bsolve\b",
        r"\blogic\b",
        r"\bstep.by.step\b",
    ],
    "summarisation": [
        r"\bsumm(ari[sz]e|ary)\b",
        r"\btldr\b",
        r"\bshorten\b",
        r"\bcondense\b",
    ],
    "translation": [
        r"\btranslat",
        r"\bin (french|spanish|german|japanese|chinese|arabic)\b",
    ],
    "creative": [
        r"\bwrite a (poem|story|essay)\b",
        r"\bcreative\b",
        r"\bbrainstorm\b",
    ],
}

_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a text classifier. Reply with exactly one word from this list: "
    "code, reasoning, summarisation, translation, creative, general."
)


def _heuristic_classify(prompt: str) -> str:
    """Pure-regex fallback. Returns a category name, defaulting to 'general'."""
    for category, patterns in KEYWORD_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return category
    return "general"


async def classify(
    prompt: str,
    experts: dict[str, ExpertConfig],
    client: openai.AsyncOpenAI | None = None,
) -> tuple[str, str]:
    """Classify a prompt into an expert category.

    Returns:
        A tuple of (category, method) where method is "llm" or "heuristic".
        Falls back to heuristic if client is None or the LLM call raises.
        Falls back to "general" if no heuristic matches or the chosen category
        is not present in the experts registry.
    """
    if client is not None:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content or ""
            category = raw.strip().lower()
            if category in CATEGORY_LABELS and category in experts:
                return category, "llm"
            # Unknown label — fall through to heuristic
        except Exception:
            pass  # Any API error → fall through to heuristic

    category = _heuristic_classify(prompt)
    if category not in experts:
        category = "general"
    return category, "heuristic"
