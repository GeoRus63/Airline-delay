"""P0 unadjusted RQ1 summaries; standardization and uncertainty are P5 work."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .analyze_eda import BLUE, GOLD, draw_heatmaps, plot_style, save_figure, turnaround_heatmap
except ImportError:
    from analyze_eda import BLUE, GOLD, draw_heatmaps, plot_style, save_figure, turnaround_heatmap

ROOT = Path(__file__).resolve().parents[1]
RQ1_COLUMNS = ["connection_id", "airport_id", "airport_code", "upstream_actual_arr_utc",
               "downstream_dep_delay_minutes", "upstream_arr_delay_minutes", "scheduled_turnaround_minutes",
               "actual_ground_minutes", "overnight_scheduled"]


def select_rq1(source: pd.DataFrame, development: bool = True):
    """Filter UTC scope before reading outcomes, independently of RQ2 eligibility."""
    prediction = pd.to_datetime(source.upstream_actual_arr_utc, utc=True, errors="coerce")
    end = pd.Timestamp("2026-08-01" if development else "2026-09-01", tz="UTC")
    scope = prediction.ge(pd.Timestamp("2026-04-01", tz="UTC")) & prediction.lt(end)
    data = source.loc[scope].copy()
    data["prediction_time"] = prediction.loc[scope]
    data["target_delay"] = pd.to_numeric(data.downstream_dep_delay_minutes, errors="coerce")
    data["upstream_arr_delay_minutes"] = pd.to_numeric(data.upstream_arr_delay_minutes, errors="coerce")
    data["actual_ground_minutes"] = pd.to_numeric(data.actual_ground_minutes, errors="coerce")
    if data.connection_id.isna().any() or not data.connection_id.is_unique:
        raise ValueError("RQ1 connections must have nonmissing unique IDs")
    overnight = data.overnight_scheduled.astype("string").str.lower()
    if not overnight.isin(["true", "false"]).all():
        raise ValueError("Invalid overnight flags in RQ1 scope")
    data["overnight_scheduled"] = overnight.eq("true")
    checks = {
        "invalid_delay": np.isfinite(data.target_delay) & data.target_delay.ge(0)
            & np.isfinite(data.upstream_arr_delay_minutes) & data.upstream_arr_delay_minutes.ge(0),
        "invalid_actual_ground": np.isfinite(data.actual_ground_minutes) & data.actual_ground_minutes.ge(0),
        "upstream_below_15": data.upstream_arr_delay_minutes.ge(15),
        "scheduled_overnight": ~data.overnight_scheduled,
    }
    audit = {"input_accepted_connections": len(source), "outside_prediction_utc_scope": int((~scope).sum()),
             "scoped_accepted_connections": len(data)}
    keep = pd.Series(True, index=data.index)
    for reason, passed in checks.items():
        passed = passed.fillna(False)
        audit[reason] = int((keep & ~passed).sum())
        keep &= passed
    data = data.loc[keep].copy()
    data["delta"] = data.target_delay - data.upstream_arr_delay_minutes
    data["increase"] = data.delta.clip(lower=0)
    data["reduction"] = -data.delta.clip(upper=0)
    data["is_reduction"] = data.delta.lt(0)
    data["is_amplification"] = data.delta.gt(0)
    data["is_unchanged"] = data.delta.eq(0)
    data["month_utc"] = data.prediction_time.dt.strftime("%Y-%m")
    if not np.allclose(data.delta, data.increase - data.reduction, atol=1e-9, rtol=0):
        raise AssertionError("Per-connection delta identity failed")
    audit["main_connections"] = len(data)
    audit["main_ground_zero_n"] = int(data.actual_ground_minutes.eq(0).sum())
    return data, audit


def summarize_airports(data: pd.DataFrame, monthly: bool = False):
    # Stable AirportID is the comparison unit: a PBI/DJT code change must not
    # split the same airport into two supposedly independent airport estimates.
    keys = (["month_utc"] if monthly else []) + ["airport_id"]
    summary = data.groupby(keys, observed=True, dropna=False).agg(
        airport_code=("airport_code", lambda x: '|'.join(sorted(set(x.dropna().astype(str))))),
        n=("connection_id", "size"), mean_delta=("delta", "mean"), median_delta=("delta", "median"),
        p10_delta=("delta", lambda x: x.quantile(.1)), p90_delta=("delta", lambda x: x.quantile(.9)),
        reduction_share=("is_reduction", "mean"), amplification_share=("is_amplification", "mean"),
        unchanged_share=("is_unchanged", "mean"), s_plus=("increase", "sum"), s_minus=("reduction", "sum"),
        s_net=("delta", "sum"), upstream_mean=("upstream_arr_delay_minutes", "mean"),
        downstream_mean=("target_delay", "mean"), scheduled_turnaround_mean=("scheduled_turnaround_minutes", "mean"),
        supported_months=("month_utc", "nunique")).reset_index()
    if not np.allclose(summary.s_net, summary.s_plus - summary.s_minus, atol=1e-6, rtol=0):
        raise AssertionError("Airport cumulative delta identity failed")
    if not np.allclose(summary.reduction_share + summary.amplification_share + summary.unchanged_share, 1):
        raise AssertionError("Airport outcome proportions do not sum to one")
    return summary


def run(run_dir: str | Path, development: bool = True) -> dict:
    """Read accepted connections, retaining actual ground zero for RQ1.

    The repository root is fixed by this script; use select_rq1/summarize_airports
    for lightweight synthetic verification. Final scope is explicit, never default.
    """
    config_path = Path(run_dir) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
    connection_path = Path(config.get("connection_path", "processed/flight_connections.csv.gz"))
    if not connection_path.is_absolute():
        connection_path = ROOT / connection_path
    cache = connection_path.with_name('flight_connections.pkl')
    source = pd.read_pickle(cache)[RQ1_COLUMNS] if cache.is_file() else pd.read_csv(connection_path, usecols=RQ1_COLUMNS, low_memory=False)
    data, audit = select_rq1(source, development=development)
    if data.empty:
        raise ValueError("No main-sample RQ1 observations in the requested scope")
    tables, figures = Path(run_dir) / "tables", Path(run_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    plot_style()
    raw = summarize_airports(data).sort_values("n", ascending=False)
    monthly = summarize_airports(data, monthly=True)
    raw.to_csv(tables / "rq1_airport_raw.csv", index=False)
    monthly.to_csv(tables / "rq1_airport_monthly.csv", index=False)
    heatmap = turnaround_heatmap(data)
    heatmap.to_csv(tables / "rq1_turnaround_heatmap.csv", index=False)
    draw_heatmaps(heatmap, figures, "rq1_turnaround", f"RQ1 upstream >=15, non-overnight, n={len(data):,}")
    # Select by support, not by observed performance; raw values are not rankings.
    supported = raw.head(20).sort_values("mean_delta")
    labels = supported.airport_code.astype(str) + " (n=" + supported.n.map(lambda x: f"{x:,}") + ")"
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(labels, supported.mean_delta, color=np.where(supported.mean_delta.lt(0), BLUE, GOLD))
    ax.axvline(0, color="#444444", linewidth=.8)
    ax.set(xlabel="Unadjusted mean downstream - upstream delay (min)", ylabel="Connection airport",
           title="Raw delay change | 20 largest RQ1 airport cohorts")
    save_figure(fig, figures / "rq1_raw_airport_mean.png")
    cumulative = raw.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    positions = np.arange(len(cumulative))
    ax.barh(positions - .18, cumulative.s_plus / 1000, height=.36, label="Added delay", color=GOLD)
    ax.barh(positions + .18, cumulative.s_minus / 1000, height=.36, label="Reduced delay", color=BLUE)
    ax.set(yticks=positions, yticklabels=cumulative.airport_code, xlabel="Sample cumulative delay (thousand min)",
           ylabel="Connection airport", title="Cumulative added and reduced delay | 20 largest cohorts")
    ax.legend(frameon=False)
    save_figure(fig, figures / "rq1_cumulative_delay.png")
    try:
        source_name = connection_path.relative_to(ROOT).as_posix()
    except ValueError:
        source_name = str(connection_path)
    summary = {"development_only": development, "connection_path": source_name,
               "sample_rule": "accepted, upstream delay >=15 min, non-overnight; ground >=0 allowed",
               "counts": audit, "airports": len(raw), "mean_delta": float(data.delta.mean()),
               "s_plus": float(data.increase.sum()), "s_minus": float(data.reduction.sum()), "s_net": float(data.delta.sum()),
               "delta_identity_checked": True, "standardized": False, "uncertainty_intervals": False,
               "interpretation": "Descriptive raw differences; airport composition is unadjusted and results are not causal."}
    (Path(run_dir) / "rq1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
