# Contributing

Thanks for your interest in LLM Expert Router!

## Development setup

```bash
git clone https://github.com/<your-handle>/llm-expert-router.git
cd llm-expert-router
pip install -e ".[dev]"
pytest
```

VHS is required only to regenerate `docs/demo.gif`:

```
vhs >= 0.7.2  # https://github.com/charmbracelet/vhs
# Windows: winget install charmbracelet.vhs  (pulls in ffmpeg + ttyd)
# macOS:   brew install vhs
```

## Adding an expert

1. Open `experts.yaml` and add a new top-level key:

```yaml
data_analysis:
  model: gpt-4o
  system_prompt: "You are an expert data analyst. Be precise and show your working."
  temperature: 0.2
  max_tokens: 2048
  description: "Data analysis and statistical reasoning"
```

2. Pydantic validation (`extra="forbid"`) will catch typos on startup — unknown fields raise a clear error.

## Running the full stack

```bash
make run                    # server + demo in one command
streamlit run dashboard.py  # open live dashboard at http://localhost:8501
```

## Submitting a PR

- Branch naming: `feat/`, `fix/`, `docs/`, `chore/`
- Run `pytest` locally and make sure it exits 0 before pushing
- Open a PR against `main` — no CLA or DCO required
