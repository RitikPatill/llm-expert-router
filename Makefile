# Requires bash (Git Bash or WSL on Windows)
.PHONY: serve demo run

UVICORN_PID_FILE := .uvicorn.pid
BASE_URL        ?= http://localhost:8000

serve:
	uvicorn src.llm_expert_router.app:app --reload

demo:
	python demo.py --url $(BASE_URL)

run:
	@uvicorn src.llm_expert_router.app:app --port 8000 & echo $$! > $(UVICORN_PID_FILE)
	@echo "Waiting for server..."
	@until curl -sf $(BASE_URL)/docs > /dev/null 2>&1; do sleep 1; done
	@echo "Server ready."
	python demo.py --url $(BASE_URL) || true
	@kill $$(cat $(UVICORN_PID_FILE)) && rm -f $(UVICORN_PID_FILE)
