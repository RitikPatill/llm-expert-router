import os
import sqlite3
import sys
import tempfile

import pandas as pd

sys.path.insert(0, ".")


def _make_db(path: str) -> None:
    """Create a minimal telemetry.db with one row."""
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE requests (
        id INTEGER PRIMARY KEY, timestamp TEXT, task_category TEXT,
        expert_name TEXT, model TEXT, classification_method TEXT,
        latency_ms REAL, prompt_tokens INTEGER, completion_tokens INTEGER,
        estimated_cost_usd REAL)""")
    conn.execute(
        "INSERT INTO requests VALUES "
        "(1,'2024-01-01T00:00:00','code','code','gpt-4o-mini','keyword',123.4,10,20,0.000015)"
    )
    conn.commit()
    conn.close()


def test_load_data_returns_dataframe():
    import dashboard

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        _make_db(db_path)
        df = dashboard.load_data(db_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "expert_name" in df.columns
    finally:
        os.unlink(db_path)


def test_load_data_missing_db_returns_empty():
    import dashboard

    df = dashboard.load_data("/tmp/nonexistent_router_xyzzy.db")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
