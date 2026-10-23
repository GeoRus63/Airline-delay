"""Prepare BTS flight rows and same-aircraft connections for the rebuild package."""
from __future__ import annotations

import hashlib
import argparse
import json
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "flights"
REF = ROOT / "references"
OUT = ROOT / "processed"
TABLES = ROOT / "tables"
MONTHS = [f"2026_{month:02d}" for month in range(4, 9)]
MAX_SCHEDULED_TURN_MIN = 1440
CLOCK_TOLERANCE_MIN = 1
TEXT_COLUMNS = ["FlightDate", "Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline",
                "Origin", "Dest", "OriginState", "DestState", "CRSDepTime", "CRSArrTime",
                "DepTime", "ArrTime", "CancellationCode"]
NUMERIC_COLUMNS = ["DayOfWeek", "OriginAirportID", "DestAirportID", "CRSElapsedTime", "ActualElapsedTime",
                   "DepDelay", "ArrDelay", "DepDelayMinutes", "ArrDelayMinutes", "DepDel15", "ArrDel15",
                   "Cancelled", "Diverted", "Distance", "CarrierDelay", "WeatherDelay", "NASDelay",
                   "SecurityDelay", "LateAircraftDelay"]
USECOLS = TEXT_COLUMNS + NUMERIC_COLUMNS


def hhmm_minutes(values):
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    regular = numbers.between(0, 2359) & numbers.mod(1).eq(0) & numbers.mod(100).lt(60)
    valid = regular | numbers.eq(2400)
    return (numbers.floordiv(100) * 60 + numbers.mod(100)).where(valid)


def local_schedule_to_utc(dates, hhmm, timezones):
    local = pd.to_datetime(dates, errors="coerce").astype("datetime64[ns]") + pd.to_timedelta(hhmm_minutes(hhmm), unit="m")
    utc = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns, UTC]")
    for tz in timezones.dropna().unique():
        ix = timezones.eq(tz).fillna(False)
        utc.loc[ix] = local.loc[ix].dt.tz_localize(str(tz), ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return utc


def local_clock_error(utc, hhmm, timezones):
    expected = hhmm_minutes(hhmm).mod(1440)
    observed = pd.Series(np.nan, index=utc.index)
    for tz in timezones.dropna().unique():
        ix = timezones.eq(tz).fillna(False)
        local = utc.loc[ix].dt.tz_convert(str(tz))
        observed.loc[ix] = local.dt.hour * 60 + local.dt.minute
    difference = (observed - expected).abs()
    return np.minimum(difference, 1440 - difference)


def crosses_local_date(arrival_utc, departure_utc, timezones):
    result = pd.Series(False, index=arrival_utc.index)
    for tz in timezones.dropna().unique():
        ix = timezones.eq(tz).fillna(False)
        arrival_date = arrival_utc.loc[ix].dt.tz_convert(str(tz)).dt.date
        departure_date = departure_utc.loc[ix].dt.tz_convert(str(tz)).dt.date
        result.loc[ix] = arrival_utc.loc[ix].notna() & departure_utc.loc[ix].notna() & arrival_date.ne(departure_date)
    return result


def read_flights():
    frames, manifest = [], []
    for stem in MONTHS:
        path = RAW / f"{stem}.zip"
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            assert len(names) == 1, f"{path.name}: expected exactly one CSV"
            with archive.open(names[0]) as stream:
                header = pd.read_csv(stream, nrows=0).columns.tolist()
            assert set(USECOLS).issubset(header), f"{path.name}: required columns missing"
            with archive.open(names[0]) as stream:
                part = pd.read_csv(stream, usecols=USECOLS, dtype={c: "string" for c in TEXT_COLUMNS}, low_memory=False)
        part["source_file"] = path.name
        part["source_row"] = np.arange(2, len(part) + 2)
        part["flight_id"] = path.stem + ":" + part.source_row.astype(str)
        frames.append(part)
        manifest.append({"file": path.name, "member": names[0], "rows": len(part),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "original_column_count": len(header)})
    flights = pd.concat(frames, ignore_index=True)
    assert flights.flight_id.is_unique
    for column in TEXT_COLUMNS:
        flights[column] = flights[column].str.strip().replace("", pd.NA)
    for column in NUMERIC_COLUMNS:
        flights[column] = pd.to_numeric(flights[column], errors="raise")
    flights["FlightDate"] = pd.to_datetime(flights.FlightDate, format="%Y-%m-%d", errors="coerce")
    key = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime"]
    flights["duplicate_key"] = flights.duplicated(key, keep=False)
    flights["normal_operation"] = flights.Cancelled.eq(0) & flights.Diverted.eq(0)
    for column in ["Cancelled", "Diverted", "DepDel15", "ArrDel15"]:
        assert flights[column].dropna().isin([0, 1]).all(), f"{column}: values outside 0/1"
    assert flights[["Cancelled", "Diverted"]].notna().all().all()
    return flights, manifest


def build_connections(frame, max_turn=MAX_SCHEDULED_TURN_MIN):
    ordered = frame.loc[frame.Tail_Number.notna()].copy()
    ordered = ordered.sort_values(["Tail_Number", "scheduled_dep_utc", "flight_id"], kind="stable").reset_index(drop=True)
    ordered["tail_schedule_incomplete"] = ordered.scheduled_dep_utc.isna().groupby(ordered.Tail_Number).transform("any")
    ordered["schedule_tie"] = ordered.duplicated(["Tail_Number", "scheduled_dep_utc"], keep=False)
    known = ordered.loc[ordered.actual_dep_utc.notna()]
    prev_max = known.groupby("Tail_Number").actual_dep_utc.cummax().groupby(known.Tail_Number).shift()
    reverse = known.iloc[::-1]
    next_min = reverse.groupby("Tail_Number").actual_dep_utc.cummin().groupby(reverse.Tail_Number).shift().reindex(known.index)
    ordered["actual_order_conflict"] = False
    ordered.loc[known.index, "actual_order_conflict"] = known.actual_dep_utc.le(prev_max) | known.actual_dep_utc.ge(next_min)
    next_rows = ordered.groupby("Tail_Number", sort=False).shift(-1)
    has_next = next_rows.flight_id.notna()
    up, down = ordered.loc[has_next], next_rows.loc[has_next]
    pairs = pd.DataFrame(index=up.index)
    pairs["connection_id"] = up.flight_id + "->" + down.flight_id
    for label, src in [("upstream", up), ("downstream", down)]:
        for original, suffix in [("flight_id", "flight_id"), ("Reporting_Airline", "airline"), ("FlightDate", "flight_date"),
                                 ("scheduled_dep_utc", "scheduled_dep_utc"), ("scheduled_arr_utc", "scheduled_arr_utc"),
                                 ("actual_dep_utc", "actual_dep_utc"), ("actual_arr_utc", "actual_arr_utc")]:
            pairs[f"{label}_{suffix}"] = src[original]
    pairs["tail_number"] = up.Tail_Number
    pairs["airport_id"] = up.DestAirportID
    pairs["airport_code"] = up.Dest
    pairs["upstream_origin_id"] = up.OriginAirportID
    pairs["downstream_origin_id"] = down.OriginAirportID
    pairs["downstream_dest_id"] = down.DestAirportID
    pairs["upstream_arr_delay"] = up.ArrDelay
    pairs["upstream_arr_delay_minutes"] = up.ArrDelayMinutes
    pairs["downstream_dep_delay"] = down.DepDelay
    pairs["downstream_dep_delay_minutes"] = down.DepDelayMinutes
    pairs["downstream_sched_hour"] = down.scheduled_dep_hour
    pairs["downstream_day_of_week"] = down.DayOfWeek
    pairs["downstream_distance"] = down.Distance
    pairs["downstream_crs_elapsed_time"] = down.CRSElapsedTime
    pairs["scheduled_turnaround_minutes"] = (down.scheduled_dep_utc - up.scheduled_arr_utc).dt.total_seconds() / 60
    pairs["actual_ground_minutes"] = (down.actual_dep_utc - up.actual_arr_utc).dt.total_seconds() / 60
    pairs["delta_delay_minutes"] = down.DepDelayMinutes - up.ArrDelayMinutes
    pairs["overnight_scheduled"] = crosses_local_date(up.scheduled_arr_utc, down.scheduled_dep_utc, up.dest_tz)
    checks = {
        "tail_schedule_complete": ~up.tail_schedule_incomplete,
        "unique_schedule": ~(up.schedule_tie | down.schedule_tie.fillna(True) | up.duplicate_key | down.duplicate_key.fillna(True)),
        "normal_both": up.normal_operation & down.normal_operation.fillna(False),
        "time_valid_both": up.time_valid & down.time_valid.fillna(False),
        "actual_order_consistent": ~(up.actual_order_conflict | down.actual_order_conflict.fillna(True)),
        "airport_continuity": up.DestAirportID.eq(down.OriginAirportID),
        "scheduled_gap_nonnegative": pairs.scheduled_turnaround_minutes.ge(0),
        "scheduled_gap_in_scope": pairs.scheduled_turnaround_minutes.le(max_turn),
        "actual_ground_nonnegative": pairs.actual_ground_minutes.ge(0),
    }
    pairs["first_exclusion_reason"] = "accepted"
    for label, passed in checks.items():
        passed = passed.fillna(False).astype(bool)
        pairs[f"check_{label}"] = passed
        pairs.loc[pairs.first_exclusion_reason.eq("accepted") & ~passed, "first_exclusion_reason"] = label
    pairs["main_analysis_eligible"] = pairs.first_exclusion_reason.eq("accepted")
    return pairs.reset_index(drop=True), ordered[["flight_id", "tail_schedule_incomplete", "schedule_tie", "actual_order_conflict"]]


def self_check():
    dep = pd.to_datetime(["2026-04-01 09:00Z", "2026-04-01 11:00Z", "2026-04-01 13:00Z"])
    frame = pd.DataFrame({
        "flight_id": ["a", "b", "c"], "Tail_Number": ["TEST"] * 3,
        "Reporting_Airline": ["XX"] * 3, "FlightDate": pd.to_datetime(["2026-04-01"] * 3),
        "Origin": ["AAA", "BBB", "CCC"], "Dest": ["BBB", "CCC", "DDD"],
        "DestAirportID": [1, 2, 3], "OriginAirportID": [0, 1, 2], "dest_tz": ["UTC"] * 3,
        "scheduled_dep_utc": dep, "scheduled_arr_utc": dep + pd.Timedelta(hours=1),
        "actual_dep_utc": dep, "actual_arr_utc": dep + pd.Timedelta(hours=1),
        "scheduled_dep_hour": [9, 11, 13], "ArrDelay": [0.] * 3, "ArrDelayMinutes": [0.] * 3,
        "DepDelay": [0.] * 3, "DepDelayMinutes": [0.] * 3, "DayOfWeek": [3] * 3,
        "Distance": [100.] * 3, "CRSElapsedTime": [60.] * 3, "normal_operation": [True] * 3,
        "time_valid": [True] * 3, "duplicate_key": [False] * 3,
    })
    pairs, _ = build_connections(frame)
    assert len(pairs) == 2 and pairs.main_analysis_eligible.all()
    same_time = frame.copy()
    same_time.loc[1, "actual_dep_utc"] = same_time.loc[0, "actual_arr_utc"]
    assert build_connections(same_time)[0].iloc[0].main_analysis_eligible
    same_time.loc[1, "actual_dep_utc"] -= pd.Timedelta(minutes=1)
    assert not build_connections(same_time)[0].iloc[0].main_analysis_eligible
    frame.loc[1, "normal_operation"] = False
    pairs, _ = build_connections(frame)
    assert not pairs.main_analysis_eligible.any() and "a->c" not in pairs.connection_id.tolist()
    frame.loc[1, "normal_operation"] = True
    tied = frame.copy()
    tied.loc[1, "scheduled_dep_utc"] = tied.loc[0, "scheduled_dep_utc"]
    assert not build_connections(tied)[0].main_analysis_eligible.any()
    no_tail = frame.copy()
    no_tail["Tail_Number"] = pd.NA
    assert build_connections(no_tail)[0].empty
    for minutes, accepted in [(1440, True), (1441, False), (-1, False)]:
        sample = frame.iloc[:2].copy()
        sample.loc[1, "scheduled_dep_utc"] = sample.loc[0, "scheduled_arr_utc"] + pd.Timedelta(minutes=minutes)
        sample.loc[1, "actual_dep_utc"] = sample.loc[1, "scheduled_dep_utc"]
        assert bool(build_connections(sample)[0].iloc[0].main_analysis_eligible) is accepted
    frame.loc[1, "OriginAirportID"] = 99
    assert build_connections(frame)[0].main_analysis_eligible.tolist() == [False, True]
    early_upstream = frame.copy()
    early_upstream.loc[1, "OriginAirportID"] = 1
    early_upstream.loc[0, ["ArrDelay", "ArrDelayMinutes"]] = [-30, 0]
    early_upstream.loc[1, ["DepDelay", "DepDelayMinutes"]] = [-5, 0]
    assert build_connections(early_upstream)[0].iloc[0].delta_delay_minutes == 0
    assert hhmm_minutes(pd.Series(["0005", "2400", "1260"])).iloc[:2].tolist() == [5, 1440]
    assert pd.isna(hhmm_minutes(pd.Series(["1260"])).iloc[0])
    dates = pd.Series(pd.to_datetime(["2026-03-08", "2026-11-01"]))
    ambiguous = local_schedule_to_utc(dates, pd.Series(["0230", "0130"]), pd.Series(["America/New_York"] * 2))
    assert ambiguous.isna().all()


def run():
    for path in [REF / "airports.csv", REF / "airport_codes.csv"]:
        assert path.is_file(), f"Missing reference file: {path}"
    airports = pd.read_csv(REF / "airports.csv")
    code_history = pd.read_csv(REF / "airport_codes.csv", parse_dates=["first_date", "last_date"])
    assert {"airport_id", "airport_code", "icao", "timezone", "latitude", "longitude"}.issubset(airports.columns)
    assert {"airport_id", "airport_code", "first_date", "last_date"}.issubset(code_history.columns)
    assert airports.airport_id.is_unique and not airports.airport_code.duplicated().any()
    assert set(code_history.airport_id).issubset(set(airports.airport_id))
    assert not code_history.airport_code.duplicated().any(), "Airport code lookup is not many-to-one for this period"
    assert code_history.loc[code_history.airport_code.isin(["PBI", "DJT"]), "airport_id"].nunique() == 1
    for tz in airports.timezone.dropna().unique():
        ZoneInfo(tz)

    flights, manifest = read_flights()
    code_to_id = code_history.set_index("airport_code").airport_id
    assert flights.Origin.map(code_to_id).eq(flights.OriginAirportID).all(), "Origin code/ID mismatch"
    assert flights.Dest.map(code_to_id).eq(flights.DestAirportID).all(), "Destination code/ID mismatch"
    assert flights[["OriginAirportID", "DestAirportID"]].notna().all().all(), "Missing source airport ID"
    origin_map = airports.set_index("airport_id").timezone
    flights["origin_tz"] = flights.OriginAirportID.map(origin_map)
    flights["dest_tz"] = flights.DestAirportID.map(origin_map)
    assert flights[["origin_tz", "dest_tz"]].notna().all().all(), "Airport timezone missing"
    flights["scheduled_dep_utc"] = local_schedule_to_utc(flights.FlightDate, flights.CRSDepTime, flights.origin_tz)
    flights["scheduled_arr_utc"] = flights.scheduled_dep_utc + pd.to_timedelta(flights.CRSElapsedTime, unit="m")
    flights["actual_dep_utc"] = flights.scheduled_dep_utc + pd.to_timedelta(flights.DepDelay, unit="m")
    flights["actual_arr_utc"] = flights.scheduled_arr_utc + pd.to_timedelta(flights.ArrDelay, unit="m")
    flights["scheduled_dep_hour"] = hhmm_minutes(flights.CRSDepTime).mod(1440).floordiv(60).astype("Int64")
    flights["scheduled_arr_clock_error"] = local_clock_error(flights.scheduled_arr_utc, flights.CRSArrTime, flights.dest_tz)
    flights["actual_dep_clock_error"] = local_clock_error(flights.actual_dep_utc, flights.DepTime, flights.origin_tz)
    flights["actual_arr_clock_error"] = local_clock_error(flights.actual_arr_utc, flights.ArrTime, flights.dest_tz)
    elapsed = (flights.actual_arr_utc - flights.actual_dep_utc).dt.total_seconds() / 60
    flights["actual_elapsed_error"] = (elapsed - flights.ActualElapsedTime).abs()
    flights["delay_values_valid"] = ((flights.DepDelay.clip(lower=0) - flights.DepDelayMinutes).abs().le(CLOCK_TOLERANCE_MIN)
        & (flights.ArrDelay.clip(lower=0) - flights.ArrDelayMinutes).abs().le(CLOCK_TOLERANCE_MIN)
        & flights.DepDel15.eq(flights.DepDelay.ge(15).astype(int)) & flights.ArrDel15.eq(flights.ArrDelay.ge(15).astype(int))
        & flights.DepDelayMinutes.ge(0) & flights.ArrDelayMinutes.ge(0))
    flights["time_valid"] = (flights.scheduled_dep_utc.notna() & flights.CRSElapsedTime.gt(0) & flights.ActualElapsedTime.gt(0)
        & elapsed.gt(0) & flights.scheduled_arr_clock_error.le(CLOCK_TOLERANCE_MIN)
        & flights.actual_dep_clock_error.le(CLOCK_TOLERANCE_MIN) & flights.actual_arr_clock_error.le(CLOCK_TOLERANCE_MIN)
        & flights.actual_elapsed_error.le(CLOCK_TOLERANCE_MIN) & flights.delay_values_valid)
    assert flights.FlightDate.notna().all()
    assert flights.DayOfWeek.eq(flights.FlightDate.dt.dayofweek + 1).all()

    candidates, flags = build_connections(flights)
    flights = flights.merge(flags, on="flight_id", how="left", validate="one_to_one")
    connections = candidates.loc[candidates.main_analysis_eligible].copy()
    assert candidates.connection_id.is_unique and connections.connection_id.is_unique
    assert connections.upstream_flight_id.is_unique and connections.downstream_flight_id.is_unique
    assert connections.airport_id.eq(connections.downstream_origin_id).all()
    assert connections.scheduled_turnaround_minutes.between(0, MAX_SCHEDULED_TURN_MIN).all()
    assert connections.actual_ground_minutes.ge(0).all()

    OUT.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    for name, frame in [("flights_clean", flights), ("connection_candidates", candidates), ("flight_connections", connections)]:
        frame.to_csv(OUT / f"{name}.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 3, "mtime": 0})
    counts = candidates.first_exclusion_reason.value_counts().rename_axis("reason").reset_index(name="pairs")
    counts["share_of_candidates"] = counts.pairs / len(candidates)
    counts.to_csv(TABLES / "connection_exclusions.csv", index=False)
    summary = {"raw_flight_rows": len(flights), "duplicate_key_rows": int(flights.duplicate_key.sum()),
        "missing_tail_rows": int(flights.Tail_Number.isna().sum()), "cancelled_rows": int(flights.Cancelled.sum()),
        "diverted_rows": int(flights.Diverted.sum()), "normal_operation_rows": int(flights.normal_operation.sum()),
        "normal_time_valid_rows": int((flights.normal_operation & flights.time_valid).sum()),
        "airport_count": int(pd.concat([flights.OriginAirportID, flights.DestAirportID]).nunique()),
        "candidate_connections": len(candidates), "accepted_connections": len(connections),
        "accepted_share_of_candidates": len(connections) / len(candidates),
        "accepted_overnight_connections": int(connections.overnight_scheduled.sum()),
        "accepted_cross_month_connections": int(connections.upstream_flight_date.dt.month.ne(connections.downstream_flight_date.dt.month).sum())}
    (TABLES / "cleaning_summary.json").write_text(json.dumps({"inputs": manifest, "parameters": {
        "max_scheduled_turn_minutes": MAX_SCHEDULED_TURN_MIN, "clock_tolerance_minutes": CLOCK_TOLERANCE_MIN}, "summary": summary}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run small synthetic boundary assertions only")
    args = parser.parse_args()
    if args.check:
        self_check()
        print("Synthetic boundary checks passed.")
    else:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
