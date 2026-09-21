import os
import sqlite3
import time

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = os.getenv("ROUTER_DB_PATH", "./telemetry.db")
REFRESH_INTERVAL_S = int(os.getenv("DASHBOARD_REFRESH_S", "5"))


def load_data(db_path: str) -> pd.DataFrame:
    """Return all rows from the requests table as a DataFrame, or empty DataFrame."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM requests", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def main():
    st.set_page_config(page_title="LLM Expert Router", layout="wide")
    st.title("LLM Expert Router — Live Dashboard")

    df = load_data(DB_PATH)

    if df.empty:
        st.info("No data yet. Run `python demo.py` to fire some prompts.")
    else:
        # Section 1 — KPI: total cost counter
        total_cost = df["estimated_cost_usd"].sum()
        total_calls = len(df)
        col1, col2 = st.columns(2)
        col1.metric("Total Calls", total_calls)
        col2.metric("Total Cost (USD)", f"${total_cost:.6f}")

        # Section 2 — Pie chart: routing distribution
        st.subheader("Routing Distribution by Expert")
        pie_df = df.groupby("expert_name").size().reset_index(name="calls")
        fig_pie = px.pie(pie_df, names="expert_name", values="calls", hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Section 3 — Bar chart: avg latency per expert
        st.subheader("Average Latency per Expert (ms)")
        lat_df = df.groupby("expert_name")["latency_ms"].mean().reset_index()
        lat_df.columns = ["expert_name", "avg_latency_ms"]
        fig_bar = px.bar(lat_df, x="expert_name", y="avg_latency_ms", text_auto=".0f")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Section 4 — Raw request log
        st.subheader("Request Log")
        display_cols = [
            "timestamp", "expert_name", "task_category", "model",
            "classification_method", "latency_ms", "estimated_cost_usd",
        ]
        st.dataframe(
            df[display_cols].sort_values("timestamp", ascending=False),
            use_container_width=True,
            height=300,
        )

    time.sleep(REFRESH_INTERVAL_S)
    st.rerun()


if __name__ == "__main__":
    main()
