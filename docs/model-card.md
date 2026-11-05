# Model card: `hgb_window` RUL model (one per C-MAPSS subset)

**Task.** Predict the remaining useful life (cycles before failure, capped at 125) of a
turbofan engine from its sensor history, with a 90% interval, and recommend an action.

**Model.** scikit-learn `HistGradientBoostingRegressor` (400 iterations, learning rate 0.05,
31 leaves, at least 40 samples per leaf, L2 1.0, no early stopping) on 30-cycle window
features of per-condition z-scored sensors. Intervals: Mondrian split conformal on out-of-fold
residuals, five bands of predicted RUL, widened where needed so they contain the prediction.

**Training data.** The training engines of one subset (see `docs/data-card.md`). Validation:
5-fold cross-validation by engine. No hyperparameter search on the test sets.

**Evaluation.** Official test sets, scored once per engine at its last observed cycle. The
README tables are generated from `results/*.json`; `make all` reproduces them.

**Intended use.** Portfolio and teaching: how to go from sensor data to calibrated predictions
and cost-based maintenance decisions, with honest validation. **Not** for real maintenance
decisions: the data is simulated and the cost model is an assumption.

**Outputs of the API.**

- `rul`: point prediction, cycles, in [0, 125].
- `interval`: 90% conformal interval; its lower end is the value the maintenance policy uses.
- `action`: `continue_operation`, `schedule_maintenance` (lower bound at or below the tuned
  threshold), or `inspect_manually` (the input guards fired).
- `guards`: share of cycles at an unknown operating condition, share with sensors out of the
  training range, and the reasons in words.

**Failure modes to know.**

- The cap: an engine predicted at 125 may have far more life left; the model cannot tell.
- Short histories: with fewer than 30 cycles the window features use what is there and the
  error grows.
- A different fleet (other conditions, other fault modes) breaks the model; the guards catch
  unknown operating conditions reliably, out-of-range sensors only partly (see the cross-subset
  table in the README).
- Interval coverage is checked on 100 to 259 test engines per subset; the 95% binomial range
  around a true 0.90 is about ±0.06 at 100 engines and ±0.04 at 250. The lower bound came out
  slightly below its 0.95 target in three of four subsets (see `results/conformal.json`).
