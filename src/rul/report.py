"""Render the README result tables from results/*.json, so no number is typed by hand.

Each table lives between `<!-- results:NAME -->` and `<!-- /results:NAME -->` in the README.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from rul.config import SUBSETS
from rul.models import MODELS

POLICY_LABELS = {
    "reactive": "Run to failure",
    "fixed_age": "Fixed interval (best age, tuned)",
    "predictive_point": "Predictive, point RUL (tuned threshold)",
    "predictive_lower_bound": "Predictive, conformal lower bound (tuned threshold)",
    "oracle": "Oracle (knows the failure cycle)",
}


def _load(results: Path, name: str) -> dict | None:
    path = results / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _subsets(data: dict) -> list[str]:
    return [s for s in SUBSETS if s in data["subsets"]]


def benchmark_table(results: Path) -> str | None:
    data = _load(results, "metrics")
    if not data:
        return None
    subs = _subsets(data)
    rows = []
    for name, spec in MODELS.items():
        cells = [f"{data['subsets'][s][name]['rmse']:.2f}" for s in subs]
        rows.append([f"`{name}` ({spec.description})", *cells])
    n = " / ".join(str(data["subsets"][s]["hgb_window"]["n"]) for s in subs)
    return _table(["Test RMSE, cycles (lower is better)", *subs], rows) + (
        f"\n\nTest engines per subset: {n}. True RUL capped at 125 like the labels."
    )


def main_table(results: Path) -> str | None:
    data = _load(results, "metrics")
    if not data:
        return None
    rows = []
    for s in _subsets(data):
        m = data["subsets"][s]
        main = m["cv_rmse_all_cycles"]["model"]
        r = m[main]
        rows.append(
            [
                s,
                f"{r['rmse']:.2f}",
                f"{r['rmse_uncapped']:.2f}",
                f"{r['mae']:.2f}",
                f"{r['nasa_score']:.0f}",
                f"{r['late_share']:.0%}",
                str(r["n"]),
                f"{m['cv_rmse_all_cycles']['rmse']:.2f}",
                f"{m['test_truth_above_cap']['share']:.0%}",
            ]
        )
    return _table(
        [
            "Subset",
            "RMSE",
            "RMSE (uncapped truth)",
            "MAE",
            "NASA score",
            "Late predictions",
            "Test engines",
            "CV RMSE (all cycles, by engine)",
            "Test engines with > 125 cycles left",
        ],
        rows,
    )


def conformal_table(results: Path) -> str | None:
    data = _load(results, "conformal")
    if not data:
        return None
    rows = []
    for s in _subsets(data):
        t = data["subsets"][s]["test"]
        rows.append(
            [
                s,
                f"{t['coverage']:.2f}",
                f"{t['lower_bound_coverage']:.2f}",
                f"{t['mean_width']:.1f}",
                str(t["n"]),
            ]
        )
    target = data["subsets"][_subsets(data)[0]]["target_coverage"]
    return _table(
        [
            "Subset",
            f"Interval coverage (target {target:.2f})",
            "Lower bound holds (target 0.95)",
            "Mean width, cycles",
            "Test engines",
        ],
        rows,
    )


def maintenance_table(results: Path) -> str | None:
    data = _load(results, "maintenance")
    if not data:
        return None
    subs = _subsets(data)
    first = data["subsets"][subs[0]]["default"]
    rows = []
    for key, label in POLICY_LABELS.items():
        cells = []
        for s in subs:
            p = data["subsets"][s]["default"]["policies"][key]
            cells.append(
                f"{p['vs_oracle']:.2f}x · {p['failures']}/{p['n']} failed · "
                f"{p['mean_unused_life']:.0f} unused"
            )
        rows.append([label, *cells])
    note = (
        f"\n\nCost of a failure = {first['costs']['failure']:g} x a planned removal; "
        f"lead time {first['lead_time']} cycles. Cell: cost per cycle flown relative to "
        "the oracle · engines that failed before maintenance · mean cycles of life thrown "
        "away per planned removal. Thresholds tuned on 4 folds of engines, scored on the 5th."
    )
    return _table(["Policy", *subs], rows) + note


def cost_ratio_table(results: Path) -> str | None:
    data = _load(results, "maintenance")
    if not data:
        return None
    subs = _subsets(data)
    ratios = list(data["subsets"][subs[0]]["vs_oracle_by_cost_ratio"])
    rows = []
    for s in subs:
        by = data["subsets"][s]["vs_oracle_by_cost_ratio"]
        rows.append(
            [
                s,
                *(
                    f"{by[r]['fixed_age']:.2f}x / {by[r]['predictive_point']:.2f}x / "
                    f"{by[r]['predictive_lower_bound']:.2f}x"
                    for r in ratios
                ),
            ]
        )
    return _table(["Subset", *(f"failure = {r} x planned" for r in ratios)], rows) + (
        "\n\nCell: fixed interval / predictive (point) / predictive (lower bound), cost per "
        "cycle flown relative to the oracle; lead time 10 cycles."
    )


def cross_table(results: Path) -> str | None:
    data = _load(results, "cross_subset")
    if not data:
        return None
    matrix = data["train_on_rows_test_on_columns"]
    tests = data["subsets"]
    rows = []
    for train, row in matrix.items():
        rows.append(
            [
                f"model {train}",
                *(
                    f"{row[t]['rmse']:.1f} · {row[t]['engines_flagged']:.0%} · "
                    f"{row[t]['max_sensor_psi']:.2f}"
                    for t in tests
                ),
            ]
        )
    return _table(["", *(f"test {t}" for t in tests)], rows) + (
        "\n\nCell: test RMSE · share of test engines the input guards flag as unreliable · "
        "largest population stability index (PSI) over the sensors, test set vs training data."
    )


def importance_table(results: Path) -> str | None:
    data = _load(results, "importance")
    if not data:
        return None
    rows = []
    for s in _subsets(data):
        top = data["subsets"][s]["groups"][:5]
        rows.append([s, ", ".join(f"{g['group']} (+{g['rmse_increase']:.1f})" for g in top)])
    return _table(["Subset", "Top 5 inputs by RMSE increase when shuffled"], rows)


def headline(results: Path) -> str | None:
    metrics, conformal, maint = (_load(results, n) for n in ("metrics", "conformal", "maintenance"))
    if not (metrics and conformal and maint):
        return None
    subs = _subsets(metrics)
    lines = []
    rmse_cells = ", ".join(f"{s} {metrics['subsets'][s]['hgb_window']['rmse']:.1f}" for s in subs)
    lines.append(f"- **Test RMSE (cycles):** {rmse_cells}.")
    cov = ", ".join(f"{s} {conformal['subsets'][s]['test']['coverage']:.2f}" for s in subs)
    lines.append(f"- **90% intervals, measured coverage on the test sets:** {cov}.")
    parts = []
    for s in subs:
        d = maint["subsets"][s]["default"]
        p = d["policies"]
        parts.append(
            f"{s}: +{100 * (p['predictive_lower_bound']['vs_oracle'] - 1):.0f}% "
            f"vs +{100 * (p['fixed_age']['vs_oracle'] - 1):.0f}% "
            f"({p['predictive_lower_bound']['failures']} vs {p['fixed_age']['failures']} "
            f"failures in {p['fixed_age']['n']} engines)"
        )
    d0 = maint["subsets"][subs[0]]["default"]
    lines.append(
        "- **Maintenance cost above the oracle, predictive policy vs best fixed interval** "
        f"(failure = {d0['costs']['failure']:g}x planned, lead time {d0['lead_time']} cycles): "
        + "; ".join(parts)
        + "."
    )
    return "\n".join(lines)


SECTIONS: dict[str, Callable[[Path], str | None]] = {
    "headline": headline,
    "benchmark": benchmark_table,
    "main": main_table,
    "conformal": conformal_table,
    "maintenance": maintenance_table,
    "cost_ratio": cost_ratio_table,
    "cross": cross_table,
    "importance": importance_table,
}


def update_readme(readme: Path, results: Path) -> bool:
    text = readme.read_text(encoding="utf-8")
    new = text
    for name, render in SECTIONS.items():
        body = render(results)
        if body is None:
            continue
        pattern = re.compile(rf"(<!-- results:{name} -->).*?(<!-- /results:{name} -->)", re.S)
        new = pattern.sub(lambda m, b=body: f"{m.group(1)}\n{b}\n{m.group(2)}", new)
    if new != text:
        readme.write_text(new, encoding="utf-8")
    return new != text
