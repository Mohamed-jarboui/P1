# Predictive maintenance: turbofan RUL

Predicts the remaining useful life of NASA C-MAPSS turbofan engines with calibrated intervals,
then turns predictions into maintenance decisions and compares policies by cost. Design
decisions live in `docs/design.md`: read it before changing the architecture.

## Rules for all sessions

### Commits and PRs
- No AI attribution anywhere: no co-author lines, no "generated with" text, in commits or PR
  descriptions.
- Conventional commit style (`feat:`, `fix:`, `chore:`, `test:`, `ci:`, `docs:`).
- Work on a branch, never directly on `main`; open a PR into `main`.

### Numbers and claims
- Never invent metrics. Every number in the README comes from `results/*.json` through
  `rul report`; never edit the generated tables by hand. If a number is missing, write
  "not measured yet".

### Validation
- Split by engine, never by row. Preprocessing is fitted inside each fold.
- Never tune anything on the official test sets.

### Dependencies
- Keep dependencies light. Prefer the standard library and scikit-learn; justify every new
  package. `matplotlib` only in the `report` group.

## Commands

```bash
make install   # uv sync
make lint      # ruff check + ruff format --check
make test      # pytest (synthetic engines, no network)
make all       # download, train, evaluate, simulate, drift, README tables, figures
```

## Layout

- `src/rul/` - package (`data`, `preprocess`, `features`, `models`, `pipeline`, `conformal`,
  `maintenance`, `monitoring`, `experiments`, `report`, `api`, `cli`)
- `tests/` - pytest on synthetic engines from `tests/conftest.py`
- `results/` - JSON outputs of the experiments (committed)
- `scripts/figures.py` - `docs/figs/*.svg`
