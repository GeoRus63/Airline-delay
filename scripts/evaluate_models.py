"""Nonnegative prediction metrics and fixed, outcome-defined error strata."""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_COLUMNS = {"severity": "severity", "overnight": "overnight_scheduled", "airport": "airport_id",
                 "airline": "downstream_airline", "day_of_week": "downstream_day_of_week",
                 "scheduled_hour": "downstream_sched_hour", "aircraft_family": "aircraft_family"}
SUM_COLUMNS = ["n", "absolute_error_sum", "squared_error_sum", "target_ge15_n"]


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return additive sufficient statistics, allowing overall pooling without rereads."""
    frame = predictions.copy()
    target = frame.target_delay.to_numpy(dtype=float)
    predicted = np.maximum(frame.predicted_delay.to_numpy(dtype=float), 0)
    if not np.isfinite(target).all() or not np.isfinite(predicted).all() or (target < 0).any():
        raise ValueError("Evaluation requires finite nonnegative targets and finite predictions")
    error = predicted - target
    frame["absolute_error_sum"] = np.abs(error)
    frame["squared_error_sum"] = error ** 2
    frame["target_ge15_n"] = target >= 15
    frame["n"] = 1
    frame["severity"] = np.where(target < 60, "ordinary_lt60", "severe_ge60")
    results = []
    for group_name, column in {"all": None, **GROUP_COLUMNS}.items():
        if column is not None and column not in frame.columns:
            if column == "aircraft_family":
                continue
            raise ValueError(f"Missing required evaluation metadata: {column}")
        keys = ["model", "fold"] + ([column] if column else [])
        grouped = frame.groupby(keys, observed=True, dropna=False)[SUM_COLUMNS].sum().reset_index()
        grouped["group"] = group_name
        grouped["value"] = grouped.pop(column).astype(str) if column else "all"
        results.append(grouped)
    return pd.concat(results, ignore_index=True)


def metrics_from_summaries(summaries: pd.DataFrame, include_overall: bool = True) -> pd.DataFrame:
    """Pool exact sums rather than averaging fold MAEs/RMSEs without weights."""
    if include_overall:
        overall = summaries.groupby(["model", "group", "value"], observed=True, dropna=False)[SUM_COLUMNS].sum().reset_index()
        overall["fold"] = "overall"
        summaries = pd.concat([summaries, overall], ignore_index=True)
    result = summaries.copy()
    result["mae"] = result.absolute_error_sum / result.n
    result["rmse"] = np.sqrt(result.squared_error_sum / result.n)
    result["target_ge15_share"] = result.target_ge15_n / result.n
    result["unit"] = "minutes"
    return result[["model", "fold", "group", "value", "n", "mae", "rmse", "target_ge15_share", "unit",
                   "absolute_error_sum", "squared_error_sum", "target_ge15_n"]]


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return metrics_from_summaries(summarize_predictions(predictions))
