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
G1_ADDED_NUMERIC = ["upstream_arr_delay_minutes", "tail_completed_24h_n", "tail_completed_24h_arr_delay_mean",
                    "history_window_incomplete"]
G1_CATEGORICAL = G0_CATEGORICAL.copy()
G1_NUMERIC = G0_NUMERIC + G1_ADDED_NUMERIC
G1_FEATURES = G1_CATEGORICAL + G1_NUMERIC
HISTORY_SOURCE_COLUMNS = ["flight_id", "Tail_Number", "actual_arr_utc", "ArrDelayMinutes",
                          "normal_operation", "time_valid"]
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


def _tail_key(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.upper().replace("", pd.NA)


def augment_history(frame: pd.DataFrame, flights: pd.DataFrame, *, include_current_upstream: bool = False,
                    source_start_utc=DEVELOPMENT_START) -> pd.DataFrame:
    """Add G1 from completed events in [prediction_time - 24h, prediction_time).

    The plan says 'previously completed legs', so the current upstream leg is
    excluded by default and its known arrival delay is already a separate feature.
    An explicit include_current_upstream=True adds only its exact matching flight ID
    at prediction_time; other simultaneous arrivals remain excluded. Events are
    ordered by actual arrival, never scheduled order. A validation query can use any
    temporally available history, regardless of the model's training split.

    Missing tails produce missing count/mean; a known tail with no valid history has
    count zero and missing mean. history_window_incomplete is a separate predictor
    flag for known left-truncation by the declared source start, not a guarantee
    of complete coverage when zero: prior-month carry-in flights remain absent. Source quality
    uses only completed flight record fields, never full-tail future QA flags.
    """
    required = {"prediction_time", "tail_number", "upstream_flight_id", "upstream_arr_delay_minutes"}
    if required - set(frame):
        raise ValueError(f"History query fields missing: {sorted(required - set(frame))}")
    if set(HISTORY_SOURCE_COLUMNS) - set(flights):
        raise ValueError(f"History flight fields missing: {sorted(set(HISTORY_SOURCE_COLUMNS) - set(flights))}")
    if not isinstance(include_current_upstream, bool):
        raise ValueError("include_current_upstream must be a boolean")
    prediction = pd.to_datetime(frame.prediction_time, utc=True, errors="raise").astype("datetime64[ns, UTC]")
    if prediction.isna().any():
        raise ValueError("History queries require valid prediction_time")
    source_start = pd.Timestamp(source_start_utc)
    if source_start.tzinfo is None:
        raise ValueError("source_start_utc must have a timezone")
    source_start = source_start.tz_convert("UTC")
    result = frame.copy()
    tails = _tail_key(frame.tail_number)
    counts = np.full(len(frame), np.nan)
    means = np.full(len(frame), np.nan)
    latest = np.full(len(frame), np.iinfo(np.int64).min, dtype=np.int64)
    incomplete = (prediction - pd.Timedelta(hours=24)).lt(source_start).to_numpy(dtype=np.int8)
    arrival = pd.to_datetime(flights.actual_arr_utc, utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    # Global temporal prefilter comes BEFORE delay/quality checks. An event newer
    # than an earlier query remains safe because each query has its own strict cut.
    if len(frame):
        temporal = arrival.ge(prediction.min() - pd.Timedelta(hours=24)) & arrival.le(prediction.max())
    else:
        temporal = pd.Series(False, index=flights.index)
    events = flights.loc[temporal, HISTORY_SOURCE_COLUMNS].copy()
    events["arrival"] = arrival.loc[temporal]
    events["tail"] = _tail_key(events.Tail_Number)
    events["flight_id"] = events.flight_id.astype("string").str.strip().replace("", pd.NA)
    events["delay"] = pd.to_numeric(events.ArrDelayMinutes, errors="coerce")
    duplicate_id = events.flight_id.notna() & events.flight_id.duplicated(keep=False)
    if duplicate_id.any():
        # Do not retrospectively delete a past event because a duplicate occurs
        # later. Prepared source IDs must be unique; stop on corrupted identity.
        raise ValueError(f"History source has {int(duplicate_id.sum())} duplicate flight_id rows in temporal scope")
    normal = events.normal_operation.astype("string").str.lower().isin(["true", "1"])
    valid_time = events.time_valid.astype("string").str.lower().isin(["true", "1"])
    audit_counts = {"source_rows": len(flights), "missing_actual_arrival_rows": int(arrival.isna().sum()),
                    "outside_global_history_time_scope": int((~temporal).sum()),
                    "temporally_scoped_rows": len(events)}
    checks = {
        "missing_tail_rows": events["tail"].notna(),
        "missing_flight_id_rows": events.flight_id.notna(),
        "non_normal_rows": normal,
        "invalid_record_time_rows": valid_time,
        "invalid_arrival_delay_rows": np.isfinite(events.delay) & events.delay.ge(0),
    }
    keep = pd.Series(True, index=events.index)
    for reason, passed in checks.items():
        passed = passed.fillna(False)
        audit_counts[reason] = int((keep & ~passed).sum())
        keep &= passed
    events = events.loc[keep, ["flight_id", "tail", "arrival", "delay"]].sort_values(
        ["tail", "arrival", "flight_id"], kind="stable").reset_index(drop=True)
    audit_counts["valid_history_event_rows"] = len(events)
    audit_counts["duplicate_flight_id_rows"] = 0
    audit_counts["unique_history_flight_ids"] = int(events.flight_id.nunique())
    audit_counts["unique_history_tails"] = int(events["tail"].nunique())
    audit_counts["same_tail_arrival_tie_rows_retained"] = int(events.duplicated(["tail", "arrival"], keep=False).sum())
    query = pd.DataFrame({"tail": tails.to_numpy(), "prediction_ns": prediction.array.asi8,
                          "position": np.arange(len(frame))})
    event_groups = events.groupby("tail", sort=False, observed=True).indices
    event_ns = events.arrival.array.asi8
    delay = events.delay.to_numpy(dtype=float)
    day_ns = int(pd.Timedelta(hours=24).value)
    for tail, positions in query.groupby("tail", sort=False, observed=True).indices.items():
        query_ns = query.prediction_ns.to_numpy()[positions]
        event_positions = event_groups.get(tail)
        counts[positions] = 0
        if event_positions is None:
            continue
        times = event_ns[event_positions]
        values = delay[event_positions]
        left = np.searchsorted(times, query_ns - day_ns, side="left")
        right = np.searchsorted(times, query_ns, side="left")
        n = right - left
        prefix = np.r_[0., np.cumsum(values, dtype=np.float64)]
        sums = prefix[right] - prefix[left]
        counts[positions] = n
        means[positions] = np.divide(sums, n, out=np.full(len(n), np.nan), where=n > 0)
        present = n > 0
        latest[positions[present]] = times[right[present] - 1]
    added_current = np.zeros(len(frame), dtype=bool)
    if include_current_upstream and len(events):
        # Uniqueness was checked before indexing; never add arbitrary same-time legs.
        by_id = events.set_index("flight_id")
        upstream = frame.upstream_flight_id.astype("string").str.strip()
        matched_tail = upstream.map(by_id["tail"]).reset_index(drop=True)
        matched_time = pd.to_datetime(upstream.map(by_id.arrival), utc=True).reset_index(drop=True)
        same_tail = matched_tail.eq(tails.reset_index(drop=True)).fillna(False).to_numpy(dtype=bool)
        added_current = same_tail & matched_time.eq(prediction.reset_index(drop=True)).to_numpy(dtype=bool)
        values = pd.to_numeric(upstream.map(by_id.delay), errors="coerce").to_numpy(dtype=float)
        selected = np.flatnonzero(added_current)
        sums = np.nan_to_num(means[selected], nan=0) * counts[selected] + values[selected]
        counts[selected] += 1
        means[selected] = sums / counts[selected]
        latest[selected] = prediction.array.asi8[selected]
    result["tail_completed_24h_n"] = counts
    result["tail_completed_24h_arr_delay_mean"] = means
    result["history_window_incomplete"] = incomplete
    result["history_missing_tail"] = tails.isna().to_numpy()
    result["history_latest_completion_time"] = pd.to_datetime(latest, utc=True)
    result.attrs["history_audit"] = {
        "history_window": "[prediction_time - 24 hours, prediction_time)",
        "include_current_upstream": include_current_upstream, "source_start_utc": source_start.isoformat(),
        "current_upstream_rule": "Excluded by default per plan: previously completed history; current upstream delay is separate",
        "event_quality_rule": "unique flight_id, present tail, normal_operation, record-local time_valid, finite arrival delay >=0",
        "ignored_future_qa_columns": ["tail_schedule_incomplete", "schedule_tie", "actual_order_conflict"],
        "quality_exclusions_are_sequential": True, "counts": audit_counts,
        "queries": {"rows": len(frame), "missing_tail": int(tails.isna().sum()),
                    "known_tail_zero_history": int(np.sum(counts == 0)),
                    "history_window_incomplete": int(incomplete.sum()),
                    "current_upstream_added": int(added_current.sum()),
                    "events_present": int(np.sum(counts > 0))},
        "missingness_rule": "missing tail: n and mean missing; known tail with no valid events: n=0 and mean missing; incomplete window flagged separately",
        "left_truncation_rule": "Known-left-truncation indicator, not a completeness guarantee when zero",
        "carry_in_limitation": "Flights with prior-local-month departure dates can complete in the window but are absent from the source; no empirical maximum delay is used to claim completeness",
    }
    result.attrs["feature_audit"] = {**result.attrs.get("feature_audit", {}), "feature_group": "G1",
                                     "categorical_features": G1_CATEGORICAL, "numeric_features": G1_NUMERIC}
    return result


def load_base_features(path: str | Path = ROOT / "processed" / "flight_connections.csv.gz",
                       development_only: bool = True, *, feature_group: str = "G0",
                       flights_path: str | Path = ROOT / "processed" / "flights_clean.csv.gz") -> pd.DataFrame:
    """Read only declared fields from prepared connections, preserving raw inputs."""
    cache = Path(path).with_name('flight_connections.pkl')
    source = pd.read_pickle(cache)[SOURCE_COLUMNS] if cache.is_file() else pd.read_csv(path, usecols=SOURCE_COLUMNS, low_memory=False)
    frame = build_base_features(source, development_only=development_only)
    if feature_group == "G0":
        return frame
    if feature_group != "G1":
        raise ValueError("feature_group must be G0 or G1")
    flights = pd.read_csv(flights_path, usecols=HISTORY_SOURCE_COLUMNS, low_memory=False)
    try:
        from .operational_features import source_calendar_starts
    except ImportError:
        from operational_features import source_calendar_starts
    starts = source_calendar_starts(pd.read_csv(ROOT / 'references/airports.csv'))
    return augment_history(frame, flights, source_start_utc=starts.max())


def load_configured_features(config, development_only=True):
    """Compose the fixed cumulative information groups; keep one eligible cohort."""
    try:
        from .operational_features import augment_aircraft, augment_airport_operations, AIRPORT_SOURCE_COLUMNS, AIRPORT_NUMERIC, AIRCRAFT_COLUMNS, source_calendar_starts
    except ImportError:
        from operational_features import augment_aircraft, augment_airport_operations, AIRPORT_SOURCE_COLUMNS, AIRPORT_NUMERIC, AIRCRAFT_COLUMNS, source_calendar_starts
    group = str(config.get('feature_group', 'G0')).upper()
    if group not in ['G0', 'G1', 'G2', 'G2_FAMILY', 'G2_VARIANT', 'G3', 'G3_VARIANT', 'G4A', 'G4B']:
        raise ValueError(f'Unknown information group: {group}')
    stage = {'G2_FAMILY': 'G2', 'G2_VARIANT': 'G2', 'G3_VARIANT': 'G3'}.get(group, group)

    def audited(result):
        try:
            from .train_models import resolve_features
        except ImportError:
            from train_models import resolve_features
        result.attrs['feature_audit'] = {**result.attrs.get('feature_audit', {}), **resolve_features(config)}
        return result
    frame = load_base_features(ROOT / config.get('connection_path', 'processed/flight_connections.csv.gz'),
                               development_only=development_only)
    if group == 'G0':
        return frame
    source_columns = list(dict.fromkeys(HISTORY_SOURCE_COLUMNS +
                         (AIRPORT_SOURCE_COLUMNS if stage in ['G3', 'G4A', 'G4B'] else [])))
    flights = pd.read_csv(ROOT / 'processed/flights_clean.csv.gz', usecols=source_columns, low_memory=False)
    starts = source_calendar_starts(pd.read_csv(ROOT / 'references/airports.csv'))
    frame = augment_history(frame, flights, source_start_utc=starts.max())
    if stage == 'G1':
        return audited(frame)
    reference = pd.read_csv(ROOT / 'references/aircraft_reference_2024.csv', dtype=str)
    aliases = pd.read_csv(ROOT / 'references/aircraft_type_aliases.csv', dtype=str)
    frame = augment_aircraft(frame, reference, aliases, inventory_year=2024)
    if stage == 'G2':
        return audited(frame)
    frame = augment_airport_operations(frame, flights, source_start_utc=starts)
    if stage == 'G3':
        return audited(frame)
    try:
        from .weather_pipeline_features import augment_weather
    except ImportError:
        from weather_pipeline_features import augment_weather
    return audited(augment_weather(frame, config))
