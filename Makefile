.PHONY: install lint format test download train evaluate simulate drift report figures all serve docker

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q

download:          ## fetch C-MAPSS from pinned mirrors and verify SHA-256 hashes
	uv run rul download

train: download    ## fit the main model per subset: models/FD00x.joblib (+ out-of-fold predictions)
	uv run rul train

evaluate: train    ## benchmark all models on the official test sets -> results/metrics.json
	uv run rul evaluate

simulate: train    ## maintenance policy comparison -> results/maintenance.json
	uv run rul simulate

drift: train       ## out-of-distribution guard experiment -> results/drift.json
	uv run rul drift

report:            ## rewrite the README result tables from results/*.json
	uv run rul report

figures:           ## regenerate docs/figs/*.svg from results/ and models/
	uv run --group report python scripts/figures.py

all: evaluate simulate drift report figures

serve:
	uv run rul serve

docker:
	docker build -t rul-api .
