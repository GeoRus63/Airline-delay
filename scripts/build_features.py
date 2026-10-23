"""Shared P0 feature construction; outcome fields are metadata, never predictors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_START = pd.Timestamp("2026-04-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2026-08-01", tz="UTC")
G0_CATEGORICAL = ["airport_id", "route", "downstream_airline"]
G0_NUMERIC = ["downstream_distance", "downstream_crs_elapsed_time", "downstream_day_of_week",
              "downstream_sched_hour", "scheduled_turnaround_minutes"]
G0_FEATURES = G0_CATEGORICAL + G0_NUMERIC
SOURCE_COLUMNS = [
    "connection_id", "upstream_flight_id", "downstream_flight_id", "tail_number", "airport_id",
    "airport_code", "downstream_origin_id", "downstream_dest_id", "downstream_airline",
    "upstream_scheduled_arr_utc", "downstream_scheduled_dep_utc", "upstream_actual_arr_utc",
    "downstream_actual_dep_utc", "upstream_arr_delay_minutes", "downstream_dep_delay_minutes",
    "downstream_flight_date", "overnight_scheduled", *G0_NUMERIC,
]


def build_base_features(source: pd.DataFrame, development_only: bool = True) -> pd.DataFrame:
    """Construct the common eligible RQ2 cohort and retain a sequential exclusion audit.

    Development filtering precedes any use of targets. Set development_only=False
    only after the final-test protocol has been frozen. Missing outcomes are excluded,
    never imputed. The accepted-connection file may contain zero ground time; RQ2
    excludes these because the downstream departure has already occurred.
    """
    missing = set(SOURCE_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f"Connection fields missing: {sorted(missing)}")
    if source.connection_id.isna().any() or not source.connection_id.is_unique:
        raise ValueError("connection_id must be nonmissing and unique")
    prediction_time = pd.to_datetime(source.upstream_actual_arr_utc, utc=True, errors="coerce")
    in_scope = prediction_time.ge(DEVELOPMENT_START)
    if development_only:
        in_scope &= prediction_time.lt(DEVELOPMENT_END)
    else:
        in_scope &= prediction_time.lt(pd.Timestamp("2026-09-01", tz="UTC"))
    # Do not calculate outcome distributions or eligibility counts outside the UTC scope.
    frame = source.loc[in_scope, SOURCE_COLUMNS].copy()
    frame["prediction_time"] = prediction_time.loc[in_scope]
    frame["label_time"] = pd.to_datetime(frame.downstream_actual_dep_utc, utc=True, errors="coerce")
    frame["target_delay"] = pd.to_numeric(frame.downstream_dep_delay_minutes, errors="coerce")
    frame["upstream_arr_delay_minutes"] = pd.to_numeric(frame.upstream_arr_delay_minutes, errors="coerce")
    counts = {"input_connections": len(source), "outside_prediction_utc_scope": int((~in_scope).sum()),
              "scoped_connections": len(frame)}
    checks = {
        "missing_or_nonfinite_target": np.isfinite(frame.target_delay),
        "negative_target": frame.target_delay.ge(0),
        "missing_label_time": frame.label_time.notna(),
        "prediction_not_before_departure": frame.prediction_time.lt(frame.label_time),
        "missing_or_invalid_upstream_delay": np.isfinite(frame.upstream_arr_delay_minutes)
            & frame.upstream_arr_delay_minutes.ge(0),
    }
    keep = pd.Series(True, index=frame.index)
    for reason, passed in checks.items():
        passed = passed.fillna(False)
        counts[reason] = int((keep & ~passed).sum())
        keep &= passed
    frame = frame.loc[keep].copy()
    counts["eligible_connections"] = len(frame)
    frame["route"] = (frame.downstream_origin_id.astype("Int64").astype("string") + "->"
                      + frame.downstream_dest_id.astype("Int64").astype("string"))
    for column in G0_CATEGORICAL:
        # Object strings + np.nan are supported consistently by sklearn imputers.
        frame[column] = frame[column].astype("string").astype(object)
        frame.loc[frame[column].isna(), column] = np.nan
    for column in G0_NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if frame.overnight_scheduled.dtype != bool:
        overnight = frame.overnight_scheduled.astype("string").str.lower()
        if not overnight.isin(["true", "false"]).all():
            raise ValueError("overnight_scheduled contains missing or invalid flags")
        frame["overnight_scheduled"] = overnight.eq("true")
    frame["prediction_date_utc"] = frame.prediction_time.dt.strftime("%Y-%m-%d")
    frame["prediction_month_utc"] = frame.prediction_time.dt.strftime("%Y-%m")
    frame = frame.drop(columns=["downstream_dep_delay_minutes", "downstream_actual_dep_utc",
                                "upstream_actual_arr_utc"])
    frame = frame.sort_values(["prediction_time", "connection_id"], kind="stable").reset_index(drop=True)
    frame.attrs["feature_audit"] = {"development_only": development_only, "counts": counts,
                                    "categorical_features": G0_CATEGORICAL, "numeric_features": G0_NUMERIC,
                                    "outcome_rule": "finite target >= 0 and prediction_time < label_time"}
    return frame


def load_base_features(path: str | Path = ROOT / "processed" / "flight_connections.csv.gz",
                       development_only: bool = True) -> pd.DataFrame:
    """Read only declared fields from prepared connections, preserving raw inputs."""
    cache = Path(path).with_name('flight_connections.pkl')
    source = pd.read_pickle(cache)[SOURCE_COLUMNS] if cache.is_file() else pd.read_csv(path, usecols=SOURCE_COLUMNS, low_memory=False)
    return build_base_features(source, development_only=development_only)
