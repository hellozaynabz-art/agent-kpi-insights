.PHONY: setup data brief brief-rules test lint all clean

setup:  ## install dependencies into .venv
	uv sync

data:  ## generate daily KPI data with injected incidents
	uv run python -m kpi_agent.data

brief:  ## detect, investigate with Claude, write reports/brief_<date>.md
	uv run kpi-agent

brief-rules:  ## same pipeline with a rule-based brief (no API key)
	uv run kpi-agent --no-llm

test:  ## offline tests (no API key needed)
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

all: setup data test brief-rules

clean:
	rm -rf data/*.parquet data/*.json .pytest_cache .ruff_cache
