# LLM Expert Router


> **Video walkthrough:** https://youtu.be/N55yENl_3g8
> **60-second overview:** https://youtu.be/bIt0JY9Xcbk

> MoE-inspired prompt router that classifies tasks and dispatches to specialised LLM expert configs, with cost & latency telemetry.

<!-- TODO: replace with a 5-10 second demo gif. Record with ScreenToGif on
     Windows or peek on macOS. Save to docs/demo.gif and update path here. -->
![demo](docs/demo.gif)

## What it is

LLM Expert Router is a locally-runnable routing layer for OpenAI API calls. Every incoming prompt is classified into a task category — code, reasoning, summarisation, translation, creative, or general — and dispatched to the matching *expert*: a named configuration of model, system prompt, temperature, and token budget defined in a plain YAML file. The response is returned alongside routing metadata showing which expert was chosen and why.

Routing decisions are logged to SQLite on every request. A Streamlit dashboard reads that database and renders a live routing distribution, per-expert latency, and cumulative cost — no third-party observability tooling required. The design is directly inspired by Mixture-of-Experts research, where routing is treated as a first-class concern rather than an afterthought.

## Quickstart

```bash
git clone https://github.com/RitikPatill/llm-expert-router.git
cd llm-expert-router
pip install -e .
cp .env.example .env          # fill in OPENAI_API_KEY
make serve                     # terminal 1: start API server
python demo.py                 # terminal 2: fire 10 sample prompts
streamlit run dashboard.py     # terminal 3: open live dashboard
```

`make run` runs server + demo in a single command (requires bash — Git Bash or WSL on Windows).

## Usage

**API**

```bash
curl -s http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the difference between a mutex and a semaphore"}' \
  | python -m json.tool
```

The response includes `response` (the LLM answer) and `routing_metadata` (expert name, model, task category, classification method, latency, and token counts).

**Stats endpoint**

```bash
curl http://localhost:8000/stats
```

Returns per-expert aggregates: call count, average latency (ms), and total estimated cost (USD).

**Dashboard**

Run `streamlit run dashboard.py` while the server has processed at least one request. The dashboard auto-refreshes every 5 seconds and shows a routing distribution pie chart, per-expert latency bar chart, KPI tiles, and a raw request log.

**Adding a custom expert**

Append a new key to `experts.yaml` and restart the server:

```yaml
data_analysis:
  model: gpt-4o
  system_prompt: "You are an expert data analyst. Be precise and show your working."
  temperature: 0.2
  max_tokens: 2048
  description: "Data analysis and statistical reasoning"
```

Pydantic validation (`extra="forbid"`) rejects unknown fields at startup.

## Architecture

```
User Prompt
    │
    ▼
POST /chat  (FastAPI)
    │
    ▼
Classifier ──────── LLM path: gpt-4o-mini zero-shot, temperature=0
    │               Heuristic fallback: regex keyword patterns
    ▼
Expert Registry  ── experts.yaml (model, system_prompt, temperature, max_tokens)
    │
    ▼
OpenAI API Call
    │
    ├──► Response + routing_metadata  ──► caller
    │
    └──► SQLite (telemetry.db)  ──► Streamlit Dashboard
```

## Project structure

```
llm-expert-router/
├── src/llm_expert_router/   # installable package
│   ├── app.py               # FastAPI server — /chat and /stats endpoints
│   ├── classifier.py        # LLM + heuristic prompt classifier
│   ├── registry.py          # Pydantic expert loader (reads experts.yaml)
│   ├── telemetry.py         # SQLAlchemy async SQLite telemetry writer
│   └── demo.py              # CLI demo logic — rich table output
├── tests/                   # pytest suite (27 tests across 6 modules)
├── dashboard.py             # Streamlit live dashboard
├── demo.py                  # thin shim — entry point from repo root
├── experts.yaml             # six built-in expert definitions
├── Makefile                 # make serve / demo / run
├── pyproject.toml           # hatchling build, Python 3.10+
└── .env.example             # env var template
```

## Roadmap

- [ ] Streaming responses via Server-Sent Events on `/chat`
- [ ] Anthropic Claude support (one additional YAML field: `provider`)
- [ ] Confidence threshold: re-route to `general` when classifier certainty is low
- [ ] Docker Compose file bundling server + dashboard for one-command deployment
- [ ] Export telemetry to Parquet for offline cost analysis

## License

MIT — see LICENSE.

---

Built autonomously by [autodev](https://github.com/RitikPatill/autodev),
a multi-agent orchestrator I designed. Each commit in this repo was
authored by me; the implementation work was performed by Sonnet under
the orchestrator's control. Read the orchestrator's README to see how.
