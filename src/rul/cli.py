"""Command line: `rul download | train | evaluate | simulate | drift | report | predict | serve`."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rul import __version__
from rul.config import SUBSETS, data_dir, models_dir, results_dir


def _round(obj: object, digits: int = 4) -> object:
    if isinstance(obj, float | np.floating):
        return round(float(obj), digits)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_round(v, digits) for v in obj]
    return obj


def _meta(started: float) -> dict[str, object]:
    import sklearn

    from rul.data import verify

    hashes = verify(data_dir())
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rul_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cpus": os.cpu_count(),
        "seconds": round(time.perf_counter() - started, 1),
        "data_sha256": {k: v[:12] for k, v in hashes.items()},
    }


def write_result(name: str, payload: dict[str, object], started: float) -> Path:
    out = results_dir() / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {"_meta": _meta(started), **payload}
    out.write_text(json.dumps(_round(body), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return out


def _subsets(args: argparse.Namespace) -> list[str]:
    return args.subsets or list(SUBSETS)


def _load(subset: str):
    from rul.pipeline import RULModel

    path = models_dir() / f"{subset}.joblib"
    if not path.exists():
        sys.exit(f"{path} not found: run `rul train` first")
    oof = pd.read_csv(models_dir() / f"oof_{subset}.csv.gz")
    return RULModel.load(path), oof


def cmd_download(args: argparse.Namespace) -> None:
    from rul.data import download

    fetched = download(data_dir(), force=args.force)
    print(f"{len(fetched)} file(s) downloaded, all 12 verified in {data_dir()}")


def cmd_train(args: argparse.Namespace) -> None:
    from rul.data import load_subset
    from rul.maintenance import Costs, engines_from_oof, tune_threshold
    from rul.pipeline import Policy, train_model

    for subset in _subsets(args):
        t0 = time.perf_counter()
        sub = load_subset(subset, data_dir())
        model, oof = train_model(subset, sub.train, args.model, folds=args.folds, seed=args.seed)
        costs, lead = Costs(1.0, 5.0), 10
        threshold, _ = tune_threshold(engines_from_oof(oof), "lo", costs, lead)
        model.policy = Policy(threshold, lead, costs.planned, costs.failure)
        model.save(models_dir() / f"{subset}.joblib")
        oof.to_csv(models_dir() / f"oof_{subset}.csv.gz", index=False)
        print(f"{subset}: trained in {time.perf_counter() - t0:.0f}s, policy threshold {threshold}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from rul.data import load_subset
    from rul.experiments import benchmark, conformal_check, importance

    started = time.perf_counter()
    metrics, conformal, imp = {}, {}, {}
    for subset in _subsets(args):
        model, oof = _load(subset)
        sub = load_subset(subset, data_dir())
        metrics[subset] = benchmark(sub, model, oof)
        conformal[subset] = conformal_check(sub, model)
        imp[subset] = importance(sub, model.model_name)
        print(f"{subset}: test RMSE {metrics[subset][model.model_name]['rmse']:.2f}")
    write_result("metrics", {"subsets": metrics}, started)
    write_result("conformal", {"subsets": conformal}, started)
    write_result("importance", {"subsets": imp}, started)


def cmd_simulate(args: argparse.Namespace) -> None:
    from rul.experiments import simulate

    started = time.perf_counter()
    out = {}
    for subset in _subsets(args):
        _, oof = _load(subset)
        out[subset] = simulate(oof)
        p = out[subset]["default"]["policies"]
        print(
            f"{subset}: predictive x{p['predictive_lower_bound']['vs_oracle']:.3f} of oracle, "
            f"fixed age x{p['fixed_age']['vs_oracle']:.3f}"
        )
    write_result("maintenance", {"subsets": out}, started)


def cmd_drift(args: argparse.Namespace) -> None:
    from rul.data import load_subset
    from rul.experiments import cross_subset

    started = time.perf_counter()
    names = _subsets(args)
    models = {s: _load(s)[0] for s in names}
    subsets = {s: load_subset(s, data_dir()) for s in SUBSETS}
    write_result("cross_subset", cross_subset(models, subsets), started)


def cmd_report(args: argparse.Namespace) -> None:
    from rul.report import update_readme

    path = Path(args.readme)
    changed = update_readme(path, results_dir())
    print(f"{path}: {'updated' if changed else 'already up to date'}")


def cmd_predict(args: argparse.Namespace) -> None:
    from rul.data import read_split

    model, _ = _load(args.subset)
    raw = read_split(Path(args.file))
    raw = raw[raw["unit"] == args.unit]
    if raw.empty:
        sys.exit(f"unit {args.unit} not in {args.file}")
    last = model.predict_last(raw).iloc[0]
    guard = model.guard_check(raw)
    action, reasons = model.decide(float(last["lo"]), guard)
    print(
        json.dumps(
            _round(
                {
                    "unit": args.unit,
                    "cycles_seen": len(raw),
                    "rul": last["pred"],
                    "interval": [last["lo"], last["hi"]],
                    "action": action,
                    "reasons": reasons,
                    "guards": guard.as_dict(),
                },
                2,
            ),
            indent=2,
        )
    )


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("rul.api:app", host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rul", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="fetch and verify the C-MAPSS files")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    def with_subsets(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sp.add_argument("--subsets", nargs="+", choices=SUBSETS)
        return sp

    t = with_subsets(sub.add_parser("train", help="fit and save the model per subset"))
    t.add_argument("--model", default="hgb_window")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)

    with_subsets(sub.add_parser("evaluate", help="benchmark on the test sets")).set_defaults(
        func=cmd_evaluate
    )
    with_subsets(sub.add_parser("simulate", help="maintenance policy comparison")).set_defaults(
        func=cmd_simulate
    )
    with_subsets(sub.add_parser("drift", help="cross-subset guard experiment")).set_defaults(
        func=cmd_drift
    )

    r = sub.add_parser("report", help="rewrite README result tables from results/")
    r.add_argument("--readme", default="README.md")
    r.set_defaults(func=cmd_report)

    pr = sub.add_parser("predict", help="score one engine from a C-MAPSS file")
    pr.add_argument("--subset", required=True, choices=SUBSETS)
    pr.add_argument("--file", required=True)
    pr.add_argument("--unit", type=int, required=True)
    pr.set_defaults(func=cmd_predict)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
