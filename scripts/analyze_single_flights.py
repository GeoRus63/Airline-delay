"""Development-period descriptive EDA for individual completed flights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FLIGHT_COLUMNS = ["FlightDate", "Reporting_Airline", "Tail_Number", "OriginAirportID", "DestAirportID",
                  "DayOfWeek", "CRSDepTime", "scheduled_dep_hour", "DepDelay", "ArrDelay",
                  "DepDelayMinutes", "ArrDelayMinutes", "Cancelled", "Diverted"]
TARGETS = ["DepDelay", "ArrDelay", "DepDelayMinutes", "ArrDelayMinutes"]
SIGNED_TARGETS = ["DepDelay", "ArrDelay"]
DIMENSIONS = ["month", "OriginAirportID", "DestAirportID", "DayOfWeek", "scheduled_dep_hour",
              "Reporting_Airline"]


def _path(value, default):
    path = Path(value if value is not None else default)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must be inside project root: {path}") from exc
    return path


def _month_scope(mode):
    if mode == "development":
        return [4, 5, 6, 7], "2026-04 through 2026-07 local FlightDate; August excluded before target parsing"
    if mode == "final":
        return [8], "2026-08 local FlightDate; explicit post-freeze final mode"
    raise ValueError("mode must be explicitly 'development' or 'final'")


def _read_scoped_flights(path, months):
    missing = set(FLIGHT_COLUMNS) - {"scheduled_dep_hour"}
    try:
        frame = pd.read_csv(path, compression="gzip", usecols=lambda column: column in FLIGHT_COLUMNS,
                            dtype={column: "string" for column in
                                   ["FlightDate", "Reporting_Airline", "Tail_Number", "DepDelay", "ArrDelay",
                                    "DepDelayMinutes", "ArrDelayMinutes", "Cancelled", "Diverted"]},
                            low_memory=False)
    except ValueError as exc:
        raise ValueError(f"Could not read required columns from {path}: {exc}") from exc
    absent = missing - set(frame.columns)
    if absent:
        raise ValueError(f"Cleaned flight table missing required columns: {sorted(absent)}")
    frame["FlightDate"] = pd.to_datetime(frame.FlightDate, errors="coerce")
    in_scope = frame.FlightDate.dt.year.eq(2026) & frame.FlightDate.dt.month.isin(months)
    frame = frame.loc[in_scope].copy()
    # Outcome and status parsing deliberately begins only after local-date scope.
    for column in TARGETS + ["Cancelled", "Diverted"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("OriginAirportID", "DestAirportID", "DayOfWeek"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    if "scheduled_dep_hour" not in frame:
        hhmm = pd.to_numeric(frame.CRSDepTime, errors="coerce")
        frame["scheduled_dep_hour"] = (hhmm // 100).where(hhmm.between(0, 2359) & hhmm.mod(100).lt(60))
        frame.loc[hhmm.eq(2400), "scheduled_dep_hour"] = 0
    frame["scheduled_dep_hour"] = pd.to_numeric(frame.scheduled_dep_hour, errors="coerce").astype("Int64")
    frame["month"] = frame.FlightDate.dt.strftime("%Y-%m")
    frame["operation_status"] = np.select(
        [frame.Cancelled.eq(0) & frame.Diverted.eq(0), frame.Cancelled.eq(1), frame.Diverted.eq(1)],
        ["completed", "cancelled", "diverted"], default="unknown")
    return frame


def _as_group_value(value):
    return "Unknown" if pd.isna(value) else str(value)


def _group_summaries(frame, source_rows):
    rows = []
    grouped = frame.groupby("operation_status", dropna=False).size().to_dict()
    completed = frame.loc[frame.operation_status.eq("completed")]
    for dimension in DIMENSIONS:
        for group_value, group in completed.groupby(dimension, dropna=False, sort=True):
            for target in TARGETS:
                values = pd.to_numeric(group[target], errors="coerce")
                valid = values.dropna()
                rows.append({
                    "dimension": dimension, "group_value": _as_group_value(group_value), "outcome": target,
                    "normal_completed_flights": len(group), "n": int(valid.size),
                    "missing": int(values.isna().sum()), "negative": int(values.lt(0).sum()),
                    "zero": int(values.eq(0).sum()), "mean": valid.mean() if len(valid) else np.nan,
                    "q50": valid.quantile(.50) if len(valid) else np.nan,
                    "q90": valid.quantile(.90) if len(valid) else np.nan,
                    "q99": valid.quantile(.99) if len(valid) else np.nan,
                    "max": valid.max() if len(valid) else np.nan,
                })

    operations = []
    for dimension in DIMENSIONS:
        for group_value, group in frame.groupby(dimension, dropna=False, sort=True):
            operations.append({
                "dimension": dimension, "group_value": _as_group_value(group_value), "flights": len(group),
                "completed": int(group.operation_status.eq("completed").sum()),
                "cancelled": int(group.operation_status.eq("cancelled").sum()),
                "diverted": int(group.operation_status.eq("diverted").sum()),
                "status_unknown": int(group.operation_status.eq("unknown").sum()),
            })
    summary = {"input_rows": int(source_rows), "scoped_flight_rows": len(frame),
               "normal_completed_rows": len(completed),
               "cancelled_rows": grouped.get("cancelled", 0), "diverted_rows": grouped.get("diverted", 0),
               "unknown_operation_status_rows": grouped.get("unknown", 0),
               "distribution_population": "normal completed flights only; cancellations and diversions summarized separately"}
    return pd.DataFrame(rows), pd.DataFrame(operations), summary


def _aircraft_tables(flights):
    ref_dir = ROOT / "references"
    alias_path = ref_dir / "aircraft_model_mapping.csv"
    aliases = pd.read_csv(alias_path, dtype=str).fillna("")
    alias_lookup = aliases.drop_duplicates("aircraft_model").set_index("aircraft_model")
    coverage_parts, model_parts = [], []
    flight_keys = flights[["month", "Reporting_Airline", "Tail_Number"]].copy()
    for year in (2024, 2025):
        reference = pd.read_csv(ref_dir / f"aircraft_reference_{year}.csv", dtype=str).fillna("")
        reference = reference.drop_duplicates(["airline", "tail_number"], keep="last")
        reference = reference[["airline", "tail_number", "aircraft_model", "manufacturer"]]
        joined = flight_keys.merge(reference, left_on=["Reporting_Airline", "Tail_Number"],
                                   right_on=["airline", "tail_number"], how="left", validate="many_to_one")
        joined["aircraft_model"] = joined.aircraft_model.astype("string").str.strip()
        joined["aircraft_model"] = joined.aircraft_model.mask(joined.aircraft_model.eq(""), pd.NA)
        joined["model_status"] = np.where(joined.aircraft_model.notna(), "matched", "unknown")
        coverage = joined.groupby(["month", "Reporting_Airline"], dropna=False, sort=True).agg(
            flight_rows=("month", "size"), rows_with_tail=("Tail_Number", lambda x: int(x.notna().sum())),
            mapped_model_rows=("aircraft_model", lambda x: int(x.notna().sum())),
            missing_tail_rows=("Tail_Number", lambda x: int(x.isna().sum()))).reset_index()
        coverage["inventory_year"] = year
        coverage["unknown_model_rows"] = coverage.flight_rows - coverage.mapped_model_rows
        coverage["model_match_rate"] = coverage.mapped_model_rows / coverage.flight_rows.replace(0, np.nan)
        coverage_parts.append(coverage)

        frequencies = joined.assign(aircraft_model=joined.aircraft_model.fillna("unknown")).groupby(
            ["month", "Reporting_Airline", "aircraft_model"], dropna=False, sort=True).size().rename(
                "flight_rows").reset_index()
        frequencies["inventory_year"] = year
        frequencies = frequencies.merge(aliases, on="aircraft_model", how="left", validate="many_to_one")
        frequencies[["aircraft_family", "aircraft_variant"]] = frequencies[
            ["aircraft_family", "aircraft_variant"]].fillna("unknown")
        model_parts.append(frequencies)
    return pd.concat(coverage_parts, ignore_index=True), pd.concat(model_parts, ignore_index=True)


def _airport_activity(flights, connections_path, airports_path, months):
    departures = flights.groupby("OriginAirportID", dropna=False).size().rename("origin_flights")
    arrivals = flights.groupby("DestAirportID", dropna=False).size().rename("destination_flights")
    activity = pd.concat([departures, arrivals], axis=1).fillna(0).astype("int64").reset_index()
    activity = activity.rename(columns={"index": "airport_id", "OriginAirportID": "airport_id"})
    activity["total_flight_movements"] = activity.origin_flights + activity.destination_flights

    connection_parts = []
    columns = ["airport_id", "qa_cohort_month", "main_analysis_eligible", "rq2_eligible"]
    for chunk in pd.read_csv(connections_path, compression="gzip", usecols=columns, chunksize=300_000,
                             low_memory=False):
        chunk = chunk.loc[chunk.qa_cohort_month.astype("string").str[-2:].astype("Int64").isin(months)]
        if chunk.empty:
            continue
        for flag in ("main_analysis_eligible", "rq2_eligible"):
            chunk[flag] = chunk[flag].astype("string").str.lower().isin(["true", "1", "yes"])
        chunk["airport_id"] = pd.to_numeric(chunk.airport_id, errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["airport_id"])
        grouped = chunk.groupby("airport_id").agg(
            accepted_connection_count=("airport_id", "size"),
            main_analysis_eligible_connection_count=("main_analysis_eligible", "sum"),
            rq2_eligible_connection_count=("rq2_eligible", "sum"))
        connection_parts.append(grouped)
    if connection_parts:
        counts = pd.concat(connection_parts).groupby(level=0).sum().reset_index()
        activity = activity.merge(counts, on="airport_id", how="outer", validate="one_to_one")
    else:
        for column in ("accepted_connection_count", "main_analysis_eligible_connection_count",
                       "rq2_eligible_connection_count"):
            activity[column] = 0
    count_cols = ["origin_flights", "destination_flights", "total_flight_movements", "accepted_connection_count",
                  "main_analysis_eligible_connection_count", "rq2_eligible_connection_count"]
    activity[count_cols] = activity[count_cols].fillna(0).astype("int64")
    airports = pd.read_csv(airports_path)
    needed = {"airport_id", "airport_code", "latitude", "longitude"}
    if needed - set(airports.columns):
        raise ValueError(f"Airport reference missing columns: {sorted(needed - set(airports.columns))}")
    activity = activity.merge(airports[["airport_id", "airport_code", "latitude", "longitude"]].drop_duplicates(
        "airport_id"), on="airport_id", how="left", validate="one_to_one")
    return activity


def _rq2_weather_coverage(path, mode):
    if path is None:
        return None
    frame = pd.read_pickle(path)
    required = {"prediction_time", "target_delay", "metar_missing", "taf_missing"}
    if required - set(frame.columns):
        raise ValueError(f"Feature archive missing weather EDA fields: {sorted(required - set(frame.columns))}")
    prediction = pd.to_datetime(frame.prediction_time, utc=True, errors="coerce")
    if mode == "development":
        scoped = prediction.ge(pd.Timestamp("2026-04-01", tz="UTC")) & prediction.lt(
            pd.Timestamp("2026-08-01", tz="UTC"))
    else:
        scoped = prediction.ge(pd.Timestamp("2026-08-01", tz="UTC")) & prediction.lt(
            pd.Timestamp("2026-09-01", tz="UTC"))
    frame = frame.loc[scoped].copy()
    # Target conversion happens only after the RQ2 prediction-time scope filter.
    frame["target_delay"] = pd.to_numeric(frame.target_delay, errors="coerce")
    for column in ("metar_missing", "taf_missing"):
        frame[column] = frame[column].astype("string").str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    target = frame.target_delay
    rows = []
    for (metar_missing, taf_missing), group in frame.groupby(["metar_missing", "taf_missing"], dropna=False,
                                                             sort=True):
        values = group.target_delay.dropna()
        rows.append({"analysis_population": "RQ2 prediction connections, not individual flights",
                     "metar_missing": metar_missing, "taf_missing": taf_missing,
                     "connections": len(group), "target_n": len(values),
                     "target_missing": int(group.target_delay.isna().sum()),
                     "target_mean_minutes": values.mean() if len(values) else np.nan,
                     "target_q50_minutes": values.quantile(.5) if len(values) else np.nan,
                     "target_q90_minutes": values.quantile(.9) if len(values) else np.nan,
                     "target_q99_minutes": values.quantile(.99) if len(values) else np.nan,
                     "target_max_minutes": values.max() if len(values) else np.nan})
    result = pd.DataFrame(rows)
    result.attrs["scoped_connections"] = len(frame)
    return result


def _figures(monthly, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months = sorted(monthly.group_value.dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, target, title in zip(axes, SIGNED_TARGETS, ("Departure delay", "Arrival delay")):
        subset = monthly.loc[monthly.outcome.eq(target)].set_index("group_value").reindex(months)
        for quantile in ("q50", "q90", "q99"):
            axis.plot(months, subset[quantile], marker="o", label=quantile.upper())
        axis.set_title(title)
        axis.set_ylabel("Minutes (signed)")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=.25)
    axes[0].legend(frameon=False)
    quantile_path = output_dir / "monthly_delay_quantiles.png"
    fig.savefig(quantile_path, dpi=150)
    plt.close(fig)

    categories = ["negative", "zero", "positive"]
    colors = ["#4C78A8", "#72B7B2", "#F58518"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, target in zip(axes.flat, TARGETS):
        subset = monthly.loc[monthly.outcome.eq(target)].set_index("group_value").reindex(months)
        category_counts = pd.DataFrame({
            "negative": subset.negative.fillna(0), "zero": subset.zero.fillna(0),
            "positive": (subset.n - subset.negative - subset.zero).clip(lower=0).fillna(0),
        }, index=months)
        category_counts.plot(kind="bar", stacked=True, color=colors, ax=axis, width=.75)
        axis.set_title(target)
        axis.set_ylabel("Flight count")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=30)
        axis.legend(title="Value", frameon=False, fontsize=8)
    count_path = output_dir / "delay_sign_counts.png"
    fig.savefig(count_path, dpi=150)
    plt.close(fig)
    return [quantile_path.name, count_path.name]


def run(config, run_dir):
    """Write development flight EDA artifacts without fitting models."""
    config = dict(config or {})
    mode = config.get("mode", "development")
    months, scope_description = _month_scope(mode)
    flight_path = _path(config.get("flight_path"), "processed/flights_clean.csv.gz")
    connection_path = _path(config.get("connections_path"), "processed/D0003/flight_connections.csv.gz")
    airport_path = _path(config.get("airport_reference_path"), "references/airports.csv")
    run_path = _path(run_dir, "experiments/runs/EDA")
    table_dir, figure_dir = run_path / "tables", run_path / "figures"
    feature_path = (_path(config["feature_archive_path"], config["feature_archive_path"])
                    if config.get("feature_archive_path") else None)
    table_names = ["single_flight_delay_stats.csv", "single_flight_operations.csv",
                   "airline_aircraft_coverage.csv", "airline_aircraft_models.csv", "airport_activity.csv"]
    if feature_path is not None:
        table_names.append("rq2_weather_missing_coverage.csv")
    figure_names = ["monthly_delay_quantiles.png", "delay_sign_counts.png"]
    summary_path = table_dir / "eda_complete_summary.json"
    for target in [summary_path, *(table_dir / name for name in table_names),
                   *(figure_dir / name for name in figure_names)]:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing EDA artifact: {target}")

    # Count source rows without parsing outcome values, for an honest audit denominator.
    source_rows = sum(len(chunk) for chunk in pd.read_csv(flight_path, compression="gzip", usecols=["FlightDate"],
                                                          chunksize=500_000))
    flights = _read_scoped_flights(flight_path, months)
    monthly, operations, single_summary = _group_summaries(flights, source_rows)
    aircraft_coverage, aircraft_models = _aircraft_tables(flights)
    airport_activity = _airport_activity(flights, connection_path, airport_path, months)
    weather_coverage = _rq2_weather_coverage(feature_path, mode) if feature_path is not None else None

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(table_dir / table_names[0], index=False)
    operations.to_csv(table_dir / table_names[1], index=False)
    aircraft_coverage.to_csv(table_dir / table_names[2], index=False)
    aircraft_models.to_csv(table_dir / table_names[3], index=False)
    airport_activity.to_csv(table_dir / table_names[4], index=False)
    if weather_coverage is not None:
        weather_coverage.to_csv(table_dir / "rq2_weather_missing_coverage.csv", index=False)
    figure_list = _figures(monthly.loc[monthly.dimension.eq("month")], figure_dir)
    summary = {
        "mode": mode, "scope": scope_description, "flight_source": flight_path.relative_to(ROOT).as_posix(),
        "connection_source": connection_path.relative_to(ROOT).as_posix(),
        "single_flight_summary": single_summary,
        "single_flight_target_fields": TARGETS,
        "aircraft_inventory_years": [2024, 2025],
        "aircraft_unknown_policy": "unknown tail/model retained as an explicit category",
        "airport_activity_note": "Flight movements and eligible connection counts are separate columns; connections use D0003 QA cohort month.",
        "rq2_weather_coverage": ("Optional feature archive strata describe RQ2 eligible prediction connections, not single-flight EDA."
                                 if weather_coverage is not None else "Not requested; no feature archive was read."),
        "tables": table_names, "figures": figure_list,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return summary
