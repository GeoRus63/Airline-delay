"""P0 development-only EDA from the shared, eligible RQ2 feature frame."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

try:
    from .build_features import DEVELOPMENT_END, DEVELOPMENT_START, G0_FEATURES
except ImportError:
    from build_features import DEVELOPMENT_END, DEVELOPMENT_START, G0_FEATURES

BLUE = "#356A91"
GOLD = "#B77D2D"
UP_BINS = [0, 5, 15, 30, 60, 120, 240, np.inf]
UP_LABELS = ["0-<5", "5-<15", "15-<30", "30-<60", "60-<120", "120-<240", "240+"]
TURN_BINS = [0, 30, 60, 90, 120, 180, 360, 720, 1441]
TURN_LABELS = ["0-<30", "30-<60", "60-<90", "90-<120", "120-<180", "180-<360", "360-<720", "720-1440"]


def plot_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlepad": 12, "figure.facecolor": "white",
                         "axes.labelcolor": "#292929", "text.color": "#292929", "savefig.facecolor": "white"})


def save_figure(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def turnaround_heatmap(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit count and delta for every predeclared upstream x turnaround cell."""
    data = pd.DataFrame({
        "upstream_delay_bin": pd.cut(frame.upstream_arr_delay_minutes, UP_BINS, labels=UP_LABELS, right=False),
        "scheduled_turnaround_bin": pd.cut(frame.scheduled_turnaround_minutes, TURN_BINS, labels=TURN_LABELS, right=False),
        "delta": frame.target_delay - frame.upstream_arr_delay_minutes,
    })
    table = data.groupby(["upstream_delay_bin", "scheduled_turnaround_bin"], observed=False).delta.agg(
        n="size", mean_delta="mean", median_delta="median").reset_index()
    if int(table.n.sum()) != len(frame):
        raise ValueError("Heatmap bins do not cover the supplied cohort")
    return table


def draw_heatmaps(table, figure_dir, prefix, population):
    count = table.pivot(index="upstream_delay_bin", columns="scheduled_turnaround_bin", values="n").to_numpy()
    means = table.pivot(index="upstream_delay_bin", columns="scheduled_turnaround_bin", values="mean_delta").to_numpy()
    for suffix, values, label, title in [
        ("count", count, "Connections (log color scale)", "Connection support"),
        ("mean_delta", means, "Mean downstream - upstream delay (min)", "Average delay change"),
    ]:
        fig, ax = plt.subplots(figsize=(11.2, 5.5))
        if suffix == "count":
            maximum = max(float(np.max(values)), 2)
            image = ax.imshow(np.ma.masked_less(values, 1), cmap="Blues", norm=LogNorm(vmin=1, vmax=maximum), aspect="auto")
        else:
            finite = values[np.isfinite(values)]
            bound = max(float(np.max(np.abs(finite))) if len(finite) else 1, 1)
            image = ax.imshow(np.ma.masked_invalid(values), cmap="PuOr_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
        for i in range(len(UP_LABELS)):
            for j in range(len(TURN_LABELS)):
                value = values[i, j]
                text = f"{int(value):,}" if suffix == "count" else (f"{value:.1f}" if np.isfinite(value) else "—")
                # Backplate keeps large and small cells readable in either palette.
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        bbox={"facecolor": "white", "alpha": .76, "edgecolor": "none", "pad": 1})
        ax.set(xticks=range(len(TURN_LABELS)), xticklabels=TURN_LABELS,
               yticks=range(len(UP_LABELS)), yticklabels=UP_LABELS,
               xlabel="Scheduled turnaround (min)", ylabel="Upstream arrival delay (min)",
               title=f"{title} | {population}")
        ax.tick_params(axis="x", labelrotation=35)
        fig.colorbar(image, ax=ax, label=label, shrink=.9)
        save_figure(fig, figure_dir / f"{prefix}_{suffix}.png")


def run(frame: pd.DataFrame, run_dir: str | Path) -> dict:
    """Write development EDA tables/figures; no model fitting or August outcome use."""
    prediction = pd.to_datetime(frame.prediction_time, utc=True, errors="raise")
    data = frame.loc[prediction.ge(DEVELOPMENT_START) & prediction.lt(DEVELOPMENT_END)].copy()
    if data.empty:
        raise ValueError("No development observations for EDA")
    if not (np.isfinite(data.target_delay) & data.target_delay.ge(0)).all():
        raise ValueError("EDA expects the shared eligible, nonnegative RQ2 target")
    data["month_utc"] = prediction.loc[data.index].dt.strftime("%Y-%m")
    output = Path(run_dir)
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    plot_style()
    monthly = data.groupby("month_utc", observed=True).target_delay.agg(
        n="size", mean="mean", median="median", p10=lambda x: x.quantile(.1),
        p90=lambda x: x.quantile(.9), p95=lambda x: x.quantile(.95), p99=lambda x: x.quantile(.99), maximum="max")
    monthly["target_ge15_share"] = data.target_delay.ge(15).groupby(data.month_utc).mean()
    monthly["target_ge60_share"] = data.target_delay.ge(60).groupby(data.month_utc).mean()
    monthly.reset_index().to_csv(tables / "eda_monthly_target.csv", index=False)
    bin_edges = [0, 15, 30, 60, 120, 240, 480, 1440, np.inf]
    labels = ["0-<15", "15-<30", "30-<60", "60-<120", "120-<240", "240-<480", "480-<1440", "1440+"]
    bins = pd.cut(data.target_delay, bin_edges, labels=labels, right=False).value_counts(sort=False).rename_axis("target_delay_bin").reset_index(name="n")
    bins["share"] = bins.n / len(data)
    bins.to_csv(tables / "eda_target_bins.csv", index=False)
    for column, name in [("airport_id", "airport"), ("route", "route"), ("downstream_airline", "airline")]:
        support = data.groupby(column, observed=True, dropna=False).agg(n=("connection_id", "size"), months=("month_utc", "nunique")).reset_index()
        support.sort_values("n", ascending=False).to_csv(tables / f"eda_support_{name}.csv", index=False)
    missing = data[G0_FEATURES].isna().sum().rename_axis("feature").reset_index(name="missing_n")
    missing["missing_share"] = missing.missing_n / len(data)
    missing.to_csv(tables / "eda_feature_missing.csv", index=False)
    heatmap = turnaround_heatmap(data)
    heatmap.to_csv(tables / "eda_turnaround_heatmap.csv", index=False)
    draw_heatmaps(heatmap, figures, "eda_turnaround", f"RQ2 development, n={len(data):,}")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(bins.target_delay_bin.astype(str), bins.share * 100, color=BLUE, width=.95)
    ax.set(xlabel="Downstream nonnegative departure delay (min)", ylabel="Connections (%)",
           title=f"Target distribution | UTC Apr-Jul 2026, n={len(data):,}")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)
    save_figure(fig, figures / "eda_target_distribution.png")
    support = pd.read_csv(tables / "eda_support_airport.csv", dtype={"airport_id": str}).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(support.airport_id.astype(str), support.n, color=BLUE)
    ax.set(xlabel="Eligible RQ2 connections", ylabel="Connection AirportID",
           title=f"20 largest airport cohorts | UTC Apr-Jul, total n={len(data):,}")
    ax.ticklabel_format(axis="x", style="plain")
    save_figure(fig, figures / "eda_airport_support.png")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    positions = np.arange(len(monthly))
    ax.bar(positions - .2, monthly["mean"], width=.4, label="Mean", color=BLUE)
    ax.bar(positions + .2, monthly["median"], width=.4, label="Median", color=GOLD)
    ax.set(xticks=positions, xticklabels=monthly.index, xlabel="Prediction month (UTC)", ylabel="Delay (min)",
           title="Monthly target location | same eligible RQ2 definition")
    ax.legend(frameon=False)
    save_figure(fig, figures / "eda_monthly_target.png")
    summary = {"scope": "2026-04-01 <= prediction_time < 2026-08-01 UTC", "population": "eligible RQ2 connections",
               "n": len(data), "airport_n": int(data.airport_id.nunique()), "route_n": int(data.route.nunique()),
               "overnight_n": int(data.overnight_scheduled.sum()), "main_target": "nonnegative downstream departure delay",
               "feature_audit": frame.attrs.get("feature_audit", {}),
               "heatmap_note": "Delta is downstream departure delay minus upstream arrival delay; no causal interpretation."}
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
