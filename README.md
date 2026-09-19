# LLM Expert Router

> A lightweight Mixture-of-Experts router for LLM calls — classify, dispatch, observe.

![build](https://img.shields.io/badge/build-passing-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## Motivation

Calling `gpt-4o` for a three-word translation is wasteful. Calling `gpt-4o-mini` for a complex multi-step reasoning task is risky. A routing layer solves both: every prompt is classified into a task category and dispatched to the most appropriate expert — a named configuration of model, system prompt, temperature, and token budget. Routing decisions are logged to SQLite, making cost and latency fully observable without any third-party tooling.

The design is inspired by recent work on temporally-extended Mixture-of-Experts systems, where routing is treated as a first-class concern rather than an afterthought.

---

## What works now (M1)

- Python package installable via `pip install -e .` (hatchling build backend, `src/` layout)
- Package version exported as `llm_expert_router.__version__ == "0.1.0"` and verified by `tests/test_scaffold.py`
- PEP 561 `py.typed` marker present — the package is typed from the start
- `.env.example` documents required environment variables (`OPENAI_API_KEY`)
- MIT license, `.gitignore`, and `requirements.txt` in place

Classifier, expert registry, API server, telemetry, CLI, and dashboard are all planned — see the Milestones table below.

---

## Architecture

```
User Prompt
    │
    ▼
┌─────────────┐
│  Classifier  │  ← zero-shot LLM call / keyword fallback
└──────┬──────┘
       │  task category
       ▼
┌──────────────────┐
│  Expert Registry  │  ← experts.yaml
└──────┬───────────┘
       │  model + system prompt + params
       ▼
┌──────────────┐
│  LLM API Call │
└──────┬───────┘
       │  response
       ▼
┌──────────────────┐
│ SQLite Telemetry  │  ← latency, tokens, cost
└──────────────────┘
       │
       ▼
  Streamlit Dashboard
```

---

## Quickstart

```bash
# 1. Install in editable mode
pip install -e .

# 2. Copy and fill in your API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# 3. Start the API server (available after M3)
uvicorn llm_expert_router.server:app --reload

# 4. Run the demo script (available after M4)
python demo.py

# 5. Open the dashboard (available after M5)
streamlit run llm_expert_router/dashboard.py
```

---

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| M1 | Scaffold + README | ✅ |
| M2 | Classifier + Expert Registry | 🔲 |
| M3 | FastAPI `/chat` endpoint + SQLite telemetry | 🔲 |
| M4 | CLI demo script (`demo.py`) | 🔲 |
| M5 | Streamlit dashboard | 🔲 |
| M6 | Polish + demo GIF | 🔲 |

---

## Project Structure

```
llm-expert-router/
├── src/
│   └── llm_expert_router/
│       ├── __init__.py       # package version
│       └── py.typed          # PEP 561 marker
├── tests/
│   └── test_scaffold.py
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## License

MIT — see [LICENSE](LICENSE).
