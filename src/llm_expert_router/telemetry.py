from __future__ import annotations

from sqlalchemy import Column, Float, Integer, MetaData, Table, Text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_METADATA = MetaData()

requests_table = Table(
    "requests",
    _METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Text, nullable=False),
    Column("task_category", Text, nullable=False),
    Column("expert_name", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("classification_method", Text, nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("prompt_tokens", Integer, nullable=False),
    Column("completion_tokens", Integer, nullable=False),
    Column("estimated_cost_usd", Float, nullable=False),
)


async def init_db(db_path: str) -> AsyncEngine:
    """Create the requests table if it doesn't exist and return the engine."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(_METADATA.create_all)
    return engine


async def log_request(
    engine: AsyncEngine,
    *,
    timestamp: str,
    task_category: str,
    expert_name: str,
    model: str,
    classification_method: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost_usd: float,
) -> None:
    """Insert one request row into the requests table."""
    async with engine.begin() as conn:
        await conn.execute(
            requests_table.insert().values(
                timestamp=timestamp,
                task_category=task_category,
                expert_name=expert_name,
                model=model,
                classification_method=classification_method,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost_usd,
            )
        )


async def get_stats(engine: AsyncEngine) -> list[dict]:
    """Return per-expert aggregate stats."""
    from sqlalchemy import func, select

    stmt = select(
        requests_table.c.expert_name,
        func.count().label("calls"),
        func.avg(requests_table.c.latency_ms).label("avg_latency_ms"),
        func.sum(requests_table.c.estimated_cost_usd).label("total_cost_usd"),
    ).group_by(requests_table.c.expert_name)

    async with engine.connect() as conn:
        result = await conn.execute(stmt)
        rows = result.fetchall()

    return [
        {
            "expert_name": row.expert_name,
            "calls": row.calls,
            "avg_latency_ms": row.avg_latency_ms,
            "total_cost_usd": row.total_cost_usd,
        }
        for row in rows
    ]
