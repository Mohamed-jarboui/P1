"""Regenerate docs/figs/*.svg (light and dark variants) from results/ and models/.

Run with `make figures` (needs the `report` dependency group for matplotlib).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rul.config import RUL_CAP, data_dir, models_dir, results_dir  # noqa: E402
from rul.data import load_subset  # noqa: E402
from rul.pipeline import RULModel  # noqa: E402

OUT = Path("docs/figs")

# Colour roles; categorical slots validated for both modes (adjacent CVD and contrast checks).
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "critical": "#d03b3b",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "critical": "#d03b3b",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}


def _style(ax: plt.Axes, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


def _figure(t: dict, size: tuple[float, float], ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=size, facecolor=t["surface"])
    for ax in np.atleast_1d(axes):
        _style(ax, t)
    return fig, axes


def _title(fig: plt.Figure, t: dict, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.97, title, color=t["ink"], fontsize=12, weight="bold", va="top")
    fig.text(0.01, 0.905, subtitle, color=t["ink2"], fontsize=9, va="top")


def _save(fig: plt.Figure, name: str, mode: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def engine_trajectory(mode: str, model: RULModel, oof: pd.DataFrame) -> None:
    """One held-out training engine: prediction, interval and the maintenance decision."""
    t = THEMES[mode]
    life = oof.groupby("unit")["cycle"].max().sort_values()
    unit = int(life.index[len(life) // 2])  # the engine with the median life
    e = oof[oof["unit"] == unit]
    threshold, lead = model.policy.threshold, model.policy.lead_time
    trig = e.loc[e["lo"] <= threshold, "cycle"]

    fig, ax = _figure(t, (8, 4.2))
    blue = t["series"][0]
    ax.fill_between(e["cycle"], e["lo"], e["hi"], color=blue, alpha=0.18, linewidth=0)
    ax.plot(e["cycle"], e["rul"], color=t["ink2"], linewidth=2, label="True RUL (capped at 125)")
    ax.plot(e["cycle"], e["pred"], color=blue, linewidth=2, label="Predicted RUL")
    last = e.iloc[-1]
    if len(trig):
        c = int(trig.iloc[0])
        lo_c = float(e.loc[e["cycle"] == c, "lo"].iloc[0])
        ax.scatter([c], [lo_c], s=40, color=t["ink"], zorder=4)
        ax.annotate(
            f"cycle {c}: lower bound reaches {threshold:.0f}, plan maintenance\n"
            f"done {lead} cycles later, {int(last['cycle']) - c - lead} cycles before failure",
            xy=(c, lo_c),
            xytext=(8, 40),
            textcoords="data",
            ha="left",
            color=t["ink"],
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 1},
        )
    ax.set_xlabel("cycle")
    ax.set_ylabel("remaining useful life (cycles)")
    ax.set_ylim(0, RUL_CAP + 12)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=t["ink2"])
    _title(
        fig,
        t,
        f"FD001 engine {unit}, never seen by the model that scored it",
        "Out-of-fold predictions at every cycle; band = 90% conformal interval",
    )
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.13)
    _save(fig, "engine", mode)


def test_set(mode: str, model: RULModel) -> None:
    t = THEMES[mode]
    sub = load_subset("FD001", data_dir())
    p = model.predict_last(sub.test)
    p["truth"] = np.minimum(sub.test_rul, RUL_CAP)
    p = p.sort_values("truth", kind="stable").reset_index(drop=True)
    x = np.arange(len(p))
    miss = (p["truth"] < p["lo"]) | (p["truth"] > p["hi"])

    fig, ax = _figure(t, (8, 4.2))
    blue = t["series"][0]
    ax.vlines(x, p["lo"], p["hi"], color=blue, alpha=0.35, linewidth=2)
    ax.plot(x, p["truth"], color=t["ink2"], linewidth=2, label="True RUL (capped at 125)")
    ax.scatter(
        x[~miss],
        p["pred"][~miss],
        s=16,
        color=blue,
        zorder=3,
        edgecolors=t["surface"],
        linewidths=0.8,
        label="Prediction, 90% interval",
    )
    ax.scatter(
        x[miss],
        p["truth"][miss],
        s=26,
        facecolors="none",
        edgecolors=t["critical"],
        linewidths=1.5,
        zorder=4,
        label=f"Truth outside interval ({int(miss.sum())})",
    )
    ax.scatter(
        x[miss],
        p["pred"][miss],
        s=16,
        color=blue,
        zorder=3,
        edgecolors=t["surface"],
        linewidths=0.8,
    )
    ax.set_xlabel("test engines, sorted by true RUL")
    ax.set_ylabel("RUL at last observed cycle")
    ax.set_xticks([])
    ax.set_ylim(0, RUL_CAP + 12)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=t["ink2"])
    _title(
        fig,
        t,
        f"FD001 test set: {len(p)} engines cut off before failure",
        "Each engine scored once, at its last cycle; whiskers = 90% conformal interval",
    )
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.08)
    _save(fig, "test_fd001", mode)


def policies(mode: str, results: dict) -> None:
    t = THEMES[mode]
    subsets = [s for s in ("FD001", "FD002", "FD003", "FD004") if s in results["subsets"]]
    names = [
        ("fixed_age", "Fixed interval"),
        ("predictive_point", "Predictive, point"),
        ("predictive_lower_bound", "Predictive, lower bound"),
    ]
    fig, ax = _figure(t, (8, 4.2))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    h = 0.24
    y0 = np.arange(len(subsets))[::-1]
    for i, (key, label) in enumerate(names):
        vals = [
            100 * (results["subsets"][s]["default"]["policies"][key]["vs_oracle"] - 1)
            for s in subsets
        ]
        y = y0 + (1 - i) * h
        ax.barh(y, vals, height=h - 0.04, color=t["series"][i], label=label)
        if key == "predictive_lower_bound":
            for yy, v in zip(y, vals, strict=True):
                ax.text(v + 1, yy, f"+{v:.0f}%", va="center", fontsize=8, color=t["ink2"])
    ax.set_yticks(y0, subsets)
    ax.set_xlabel("extra cost per cycle flown, compared with the oracle (%)")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=t["ink2"],
        handlelength=1.2,
    )
    costs = results["subsets"][subsets[0]]["default"]["costs"]
    lead = results["subsets"][subsets[0]]["default"]["lead_time"]
    reactive = results["subsets"][subsets[0]]["default"]["policies"]["reactive"]["vs_oracle"]
    _title(
        fig,
        t,
        "Maintenance cost by policy (lower is better)",
        f"Failure = {costs['failure']:g}x a planned removal, lead time {lead} cycles; "
        f"run to failure would be +{100 * (reactive - 1):.0f}%",
    )
    fig.subplots_adjust(top=0.76, left=0.08, right=0.98, bottom=0.13)
    _save(fig, "policies", mode)


def importance(mode: str, results: dict) -> None:
    t = THEMES[mode]
    picks = [s for s in ("FD001", "FD004") if s in results["subsets"]]
    fig, axes = _figure(t, (8, 4.2), ncols=len(picks))
    for ax, s in zip(np.atleast_1d(axes), picks, strict=True):
        groups = results["subsets"][s]["groups"][:8][::-1]
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
        ax.barh(
            ["age (cycle)" if g["group"] == "cycle" else g["group"] for g in groups],
            [g["rmse_increase"] for g in groups],
            height=0.6,
            color=t["series"][0],
        )
        ax.set_title(s, color=t["ink"], fontsize=10, loc="left")
        ax.set_xlabel("RMSE increase when shuffled (cycles)")
    _title(
        fig,
        t,
        "Which sensors the model relies on",
        "Grouped permutation importance on held-out engines (all features of a sensor "
        "shuffled together)",
    )
    fig.subplots_adjust(top=0.78, left=0.1, right=0.98, bottom=0.14, wspace=0.35)
    _save(fig, "importance", mode)


def main() -> None:
    model = RULModel.load(models_dir() / "FD001.joblib")
    oof = pd.read_csv(models_dir() / "oof_FD001.csv.gz")
    maint = json.loads((results_dir() / "maintenance.json").read_text())
    imp = json.loads((results_dir() / "importance.json").read_text())
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["svg.hashsalt"] = "rul"  # stable ids, so unchanged figures give no diff
    plt.rcParams["font.family"] = "DejaVu Sans"
    for mode in THEMES:
        engine_trajectory(mode, model, oof)
        test_set(mode, model)
        policies(mode, maint)
        importance(mode, imp)
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")


if __name__ == "__main__":
    main()
