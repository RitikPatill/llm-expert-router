from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_expert_router.classifier import classify
from llm_expert_router.registry import load_experts
from llm_expert_router.telemetry import get_stats, init_db, log_request

logger = logging.getLogger(__name__)

# USD per 1 K tokens
_PRICE: dict[str, dict[str, float]] = {
    "gpt-4o":      {"prompt": 0.005,    "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.000150, "completion": 0.000600},
}


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = _PRICE.get(model)
    if prices is None:
        return 0.0
    return (prompt_tokens * prices["prompt"] + completion_tokens * prices["completion"]) / 1000.0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class RoutingMetadata(BaseModel):
    task_category: str
    expert_name: str
    model: str
    classification_method: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


class ChatResponse(BaseModel):
    response: str
    routing_metadata: RoutingMetadata


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = os.getenv("ROUTER_DB_PATH", "./telemetry.db")
    experts_file = os.getenv("ROUTER_EXPERTS_FILE", "./experts.yaml")
    api_key = os.getenv("OPENAI_API_KEY")

    app.state.experts = load_experts(experts_file)
    app.state.engine = await init_db(db_path)
    app.state.client = openai.AsyncOpenAI(api_key=api_key) if api_key else None
    yield
    await app.state.engine.dispose()


app = FastAPI(title="LLM Expert Router", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    client = app.state.client
    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    experts = app.state.experts
    engine = app.state.engine

    t0 = time.monotonic()

    category, method = await classify(req.message, experts, client)

    expert = experts.get(category) or experts["general"]
    expert_name = category if category in experts else "general"

    messages = [
        {"role": "system", "content": expert.system_prompt},
        {"role": "user", "content": req.message},
    ]
    llm_response = await client.chat.completions.create(
        model=expert.model,
        messages=messages,
        temperature=expert.temperature,
        max_tokens=expert.max_tokens,
    )

    latency_ms = (time.monotonic() - t0) * 1000.0

    usage = getattr(llm_response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    estimated_cost_usd = _compute_cost(expert.model, prompt_tokens, completion_tokens)

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        await log_request(
            engine,
            timestamp=timestamp,
            task_category=category,
            expert_name=expert_name,
            model=expert.model,
            classification_method=method,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
    except Exception:
        logger.warning("Failed to log request to telemetry DB", exc_info=True)

    response_text = llm_response.choices[0].message.content or ""

    return ChatResponse(
        response=response_text,
        routing_metadata=RoutingMetadata(
            task_category=category,
            expert_name=expert_name,
            model=expert.model,
            classification_method=method,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
        ),
    )


@app.get("/stats")
async def stats() -> list[dict]:
    return await get_stats(app.state.engine)
