# Design notes

Why the system is built the way it is. Read this before changing the architecture.

## The question the project answers

Not "how low can the RMSE go on C-MAPSS" (hundreds of notebooks do that), but: **given a
stream of sensor readings from a fleet, when should each engine go to maintenance, and how much
does a model save compared with a fixed schedule?** RUL accuracy is a means; the decision and
its cost are the result.

## Pipeline

1. **Data** (`data.py`). C-MAPSS from two GitHub mirrors pinned to a commit; every file is
   checked against a SHA-256 hash, so a changed mirror cannot silently change the results.
   The two mirrors were compared and hold byte-identical files.
2. **Operating conditions** (`preprocess.py`). FD002/FD004 mix six flight conditions. Rounding
   the three settings recovers them exactly (largest deviation within a condition in training:
   < 0.01). Each sensor is z-scored with its own condition's mean and standard deviation.
   Sensors with fewer than 5 distinct values in some condition are dropped (constant or
   quantisation noise); the list is computed from the training data, not hard-coded.
3. **Features** (`features.py`). Per sensor: current value, 30-cycle mean, standard deviation,
   least-squares slope, and an exponentially weighted mean; plus the cycle count. A test checks
   that features at cycle *t* do not change when later cycles are removed (no look-ahead).
4. **Labels.** RUL capped at 125 cycles (piecewise linear), the usual convention: early in life
   an engine looks healthy and its exact remaining life is not observable.
5. **Validation by engine** (`pipeline.py`). 5-fold `GroupKFold` on engines; preprocessing is
   refitted inside each fold. Splitting by row would put cycles of the same engine on both
   sides and give optimistic numbers. Gradient boosting runs without early stopping for the
   same reason (its internal split is by row).
6. **Conformal intervals** (`conformal.py`). Split conformal on the out-of-fold residuals, one
   set of quantiles per band of predicted RUL (Mondrian), because the error spread depends on
   where the engine is in its life. Coverage is measured on the official test sets, not
   assumed: calibration residuals within an engine are correlated, so the textbook guarantee
   does not strictly apply.
7. **Decisions** (`maintenance.py`). Plan maintenance when the signal (point prediction or
   conformal lower bound) drops to a threshold; the work happens `lead_time` cycles later.
   Policies are compared by cost per cycle flown (renewal-reward rate), on the out-of-fold
   predictions of the training engines, because only they run to failure. Thresholds and the
   fixed interval are tuned on 4 folds and scored on the 5th.
8. **Guards** (`monitoring.py`). Every API request is checked for settings far from every known
   condition and for sensors outside the training range. Flagged inputs get the action
   `inspect_manually` instead of a confident number.
9. **Serving** (`api.py`). FastAPI; one artifact per subset (`models/FD00x.joblib`) holding the
   preprocessor, model, calibrator, guards, policy and metadata, so the API cannot pair a model
   with the wrong normalisation.

## Choices and the alternatives not taken

| Choice | Alternative | Why |
|---|---|---|
| Gradient boosting on window features | 1D-CNN / LSTM / transformer | Trains on a laptop CPU in seconds, no GPU and no 2 GB of PyTorch wheels; accuracy is in the range of published deep models (see README). A deep model is a possible next step, not a requirement for the decision layer. |
| One model per subset | One global model | Matches the benchmark protocol and keeps guards meaningful (a FD001 model should refuse FD002 inputs). |
| Conformal intervals | Quantile regression, Bayesian models | Model-agnostic, cheap, and the coverage can be checked directly. |
| Cost rate (cost / cycles flown) | Fixed cost per wasted cycle | No arbitrary price for "unused life": replacing early costs more simply because it happens more often. |
| Cross-fitted thresholds | Tune and score on the same engines | Tuning on the scored engines would flatter every tuned policy. |
| JSON results in git | MLflow / W&B | The results are small, reviewable in a diff, and the README tables are generated from them. |

## What the numbers do and do not mean

- C-MAPSS is **simulated** data. Costs come from an assumed cost ratio and lead time; they show
  how to compare policies, not what an airline would save.
- The maintenance simulation reuses the training engines (out of fold). The test sets cannot be
  used for it: they stop before failure, so the outcome of a late decision is unknown.
- Test-set metrics use the true RUL capped at 125, like the labels; uncapped RMSE is reported too.
