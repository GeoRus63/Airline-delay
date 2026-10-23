"""Prepare weather observations and forecasts without fitting flight models.

The as-of helpers are shared by later flight-weather joins. METAR availability
is assumed to be observation time plus five minutes. TAF body issue time is an
availability proxy only; the archive does not contain a verified receive time.
"""
from __future__ import annotations

import ast
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METAR_ASSUMED_LAG = pd.Timedelta(minutes=5)
METAR_MAX_AGE = pd.Timedelta(minutes=120)


def _utc_ns(values):
    """Parse timestamps as UTC and normalize pandas 3 datetime resolution."""
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(parsed, pd.Series):
        return parsed.dt.as_unit("ns")
    if isinstance(parsed, pd.DatetimeIndex):
        return parsed.as_unit("ns")
    if isinstance(parsed, pd.Timestamp) and pd.notna(parsed):
        return parsed.as_unit("ns")
    return parsed

# The GHCNh documentation assigns different legacy QC codes to different
# source families. This whitelist accepts only explicitly documented full-pass
# codes. Blank codes are usable only for sources outside those legacy schemes;
# any unrecognized nonblank code is retained but marked ineligible.
NOAA_LEGACY_FULL_PASS = {
    **{source: {"1", "5"} for source in ("313", "314", "315", "322", "335", "343", "344", "346")},
    **{source: {"1"} for source in ("220", "221", "222", "223", "347", "348")},
}
NOAA_LEGACY_QC_SOURCES = set(NOAA_LEGACY_FULL_PASS) | {"345", "382"}
NOAA_PRECIPITATION_TRACE_CODE_2_SOURCES = set(NOAA_LEGACY_FULL_PASS)

NOAA_ELEMENTS = (
    "temperature", "dew_point_temperature", "wind_direction", "wind_speed",
    "wind_gust", "visibility", "ceiling_height", "relative_humidity", "precipitation", "snow_depth",
    "pres_wx_MW1", "pres_wx_MW2", "pres_wx_MW3",
    "pres_wx_AU1", "pres_wx_AU2", "pres_wx_AU3",
    "pres_wx_AW1", "pres_wx_AW2", "pres_wx_AW3",
)
NOAA_META_SUFFIXES = (
    "Measurement_Code", "Quality_Code", "Report_Type", "Source_Code", "Source_Station_ID",
)
NOAA_COLUMNS = ["STATION", "Station_name", "DATE", "LATITUDE", "LONGITUDE"] + [
    column
    for element in NOAA_ELEMENTS
    for column in (element, *(f"{element}_{suffix}" for suffix in NOAA_META_SUFFIXES))
]


def station_lookup():
    airports = pd.read_csv(ROOT / "references/airports.csv")
    lookup = {}
    for row in airports.itertuples():
        aliases = str(row.airport_aliases).split("|")
        for station in [row.metar_station, row.icao, *aliases]:
            if pd.isna(station) or not station:
                continue
            lookup[station] = int(row.airport_id)
            if len(station) == 3:
                lookup["K" + station] = int(row.airport_id)
    return lookup


def read_batches(kind):
    files = sorted((ROOT / "raw/weather" / kind).glob("batch_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No {kind} files in raw/weather/{kind}")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    d["airport_id"] = d.station.map(station_lookup())
    assert d.airport_id.notna().all(), "A weather station has no airport ID"
    d["airport_id"] = d.airport_id.astype("int64")
    return d


def ceiling(codes, heights):
    values = [float(h) for c, h in zip(codes, heights)
              if c in {"BKN", "OVC", "VV"} and h is not None and pd.notna(h)]
    return min(values) if values else np.nan


def parsed_list(value):
    if pd.isna(value) or value == "":
        return []
    return ast.literal_eval(value)


def _weather_tokens(value):
    if pd.isna(value):
        return []
    text = str(value).strip().upper()
    if not text or text == "M":
        return []
    return text.split()


def _has_precip_type(tokens, patterns):
    return any(any(pattern in token for pattern in patterns) for token in tokens)


def _nullable_weather_flag(tokens, reported, patterns):
    """Return NA when the source omitted weather codes; explicit codes may be zero."""
    if not reported:
        return pd.NA
    return int(_has_precip_type(tokens, patterns))


def prepare_metar():
    d = read_batches("metar")
    d["weather_time"] = _utc_ns(d.valid)
    if d.weather_time.isna().any():
        raise ValueError("METAR rows contain invalid observation times")
    d["is_correction"] = d.metar.fillna("").str.contains(r"\bCOR\b", case=False, regex=True)
    # COR messages are excluded: the archive has no correction receive time.
    d = d.loc[~d.is_correction].drop_duplicates().copy()
    keys = ["airport_id", "station", "weather_time"]
    d["metar_ambiguous"] = d.groupby(keys, dropna=False).metar.transform("size").gt(1)
    numeric = ["sknt", "gust", "vsby", "tmpf", "dwpf", "relh", "drct", "p01i",
               "skyl1", "skyl2", "skyl3", "skyl4"]
    for col in numeric:
        d[col + "_num"] = pd.to_numeric(d[col].replace("M", np.nan), errors="coerce")
    d["ceiling_ft"] = [ceiling(c, h) for c, h in zip(
        d[["skyc1", "skyc2", "skyc3", "skyc4"]].values,
        d[["skyl1_num", "skyl2_num", "skyl3_num", "skyl4_num"]].values)]
    sky_codes = d[["skyc1", "skyc2", "skyc3", "skyc4"]]
    d["sky_reported"] = (sky_codes.notna() & sky_codes.ne("") & sky_codes.ne("M")).any(axis=1).astype("int8")
    d["ceiling_reported"] = d.ceiling_ft.notna().astype("int8")

    tokens = d.wxcodes.map(_weather_tokens)
    codes_reported = d.wxcodes.notna() & d.wxcodes.astype(str).str.strip().ne("") & d.wxcodes.astype(str).str.upper().ne("M")
    d["weather_codes_reported"] = codes_reported.astype("int8")
    d["thunder_flag"] = pd.array(tokens.map(
        lambda xs: int(any("TS" in x for x in xs)) if xs else pd.NA), dtype="Int8")
    d["rain_flag"] = pd.array(tokens.map(lambda xs: int(_has_precip_type(xs, ("RA", "DZ")))
                                         if xs else pd.NA), dtype="Int8")
    d["snow_flag"] = pd.array(tokens.map(lambda xs: int(_has_precip_type(xs, ("SN", "SG", "PL", "GR", "GS")))
                                         if xs else pd.NA), dtype="Int8")

    p01_raw = d.p01i.astype("string").str.strip().str.upper()
    d["precip_amount_in"] = pd.to_numeric(p01_raw.mask(p01_raw.isin(["M", "T", ""])), errors="coerce")
    p01_trace = p01_raw.eq("T").fillna(False)
    p01_missing = (p01_raw.eq("M") | p01_raw.eq("") | p01_raw.isna()).fillna(False)
    d["precip_measurement_status"] = np.select(
        [p01_trace.to_numpy(), d.precip_amount_in.notna().to_numpy(), p01_missing.to_numpy()],
        ["trace", "measured", "missing"], default="invalid")
    wx_precip = tokens.map(lambda xs: _has_precip_type(xs, ("RA", "DZ", "SN", "SG", "PL", "GR", "GS", "UP")))
    precip_known = (d.precip_amount_in.notna() | p01_trace | codes_reported).fillna(False)
    precip_positive = (d.precip_amount_in.gt(0) | p01_trace | wx_precip).fillna(False)
    d["precip_flag"] = pd.array(np.where(precip_positive, 1, 0), dtype="Int8")
    d.loc[~precip_known & ~wx_precip, "precip_flag"] = pd.NA

    d = d.rename(columns={
        "sknt_num": "wind_knots", "gust_num": "gust_knots", "vsby_num": "visibility_miles",
        "tmpf_num": "temperature_f", "dwpf_num": "dewpoint_f", "relh_num": "relative_humidity",
        "drct_num": "wind_direction_deg",
    })
    d["available_at"] = d.weather_time + METAR_ASSUMED_LAG
    d["availability_assumption"] = "weather_time_plus_5_minutes"
    # Multiple distinct non-COR reports at the same station/time have no
    # version/receive timestamp in this archive; mark the whole group unusable.
    keep = ["airport_id", "station", "weather_time", "available_at", "availability_assumption",
            "wind_knots", "wind_direction_deg", "gust_knots", "visibility_miles", "ceiling_ft",
            "ceiling_reported", "sky_reported", "precip_amount_in", "precip_measurement_status",
            "precip_flag", "rain_flag", "snow_flag", "weather_codes_reported", "thunder_flag",
            "temperature_f", "dewpoint_f", "relative_humidity", "metar_ambiguous", "metar"]
    feature_cols = [column for column in keep if column not in {*keys, "available_at", "availability_assumption",
                                                                  "metar_ambiguous", "metar"}]
    d.loc[d.metar_ambiguous, feature_cols + ["metar"]] = pd.NA
    d = d.drop_duplicates(keys, keep="first")
    return d[keep].sort_values(["weather_time", "airport_id"])


def _airport_join_key(values):
    def normalize(value):
        if pd.isna(value):
            return pd.NA
        text = str(value).strip()
        try:
            number = float(text)
            if np.isfinite(number) and number.is_integer():
                return str(int(number))
        except (TypeError, ValueError):
            pass
        return text
    return values.map(normalize).astype("string")


def match_metar_asof(flights, observations, *, airport_col="airport_id",
                     prediction_col="prediction_time", max_age=METAR_MAX_AGE):
    """Attach latest METAR available by prediction time; retain all flight rows."""
    left = flights.copy()
    left["_metar_input_order"] = np.arange(len(left), dtype="int64")
    left["_metar_original_index"] = list(flights.index)
    left[prediction_col] = _utc_ns(left[prediction_col])
    left["_metar_airport_key"] = _airport_join_key(left[airport_col])
    right = observations.copy()
    right["available_at"] = _utc_ns(right.available_at)
    right["weather_time"] = _utc_ns(right.weather_time)
    right["_metar_airport_key"] = _airport_join_key(right[airport_col])
    right = right.dropna(subset=["available_at", "weather_time"])
    valid_mask = left[prediction_col].notna() & left._metar_airport_key.notna()
    valid_left = left.loc[valid_mask]
    invalid_left = left.loc[~valid_mask].copy()
    if not right.empty and not valid_left.empty:
        observed_lag = (right.available_at - right.weather_time).max()
        if pd.isna(observed_lag):
            observed_lag = METAR_ASSUMED_LAG
        tolerance = pd.Timedelta(max_age) + observed_lag
        result = pd.merge_asof(valid_left.sort_values(prediction_col), right.sort_values("available_at"),
                               left_on=prediction_col, right_on="available_at", by="_metar_airport_key",
                               direction="backward", tolerance=tolerance, suffixes=("", "_metar"))
    else:
        result = valid_left.copy()
        for column in observations.columns:
            output_column = column if column not in result.columns else f"{column}_metar"
            if column != airport_col and output_column not in result:
                if column in {"available_at", "weather_time"}:
                    result[output_column] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
                else:
                    result[output_column] = pd.NA
    if not invalid_left.empty:
        for column in observations.columns:
            output_column = column if column not in invalid_left.columns else f"{column}_metar"
            if column != airport_col and output_column not in invalid_left:
                if column in {"available_at", "weather_time"}:
                    invalid_left[output_column] = pd.Series(pd.NaT, index=invalid_left.index,
                                                            dtype="datetime64[ns, UTC]")
                else:
                    invalid_left[output_column] = pd.NA
        result = pd.concat([result, invalid_left], ignore_index=False, sort=False)
    result["metar_age_minutes"] = (result[prediction_col] - result.weather_time).dt.total_seconds() / 60
    stale = result.metar_age_minutes.gt(pd.Timedelta(max_age).total_seconds() / 60)
    result["metar_candidate_weather_time"] = result.weather_time
    ambiguous = result.get("metar_ambiguous", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    feature_columns = [column for column in ("wind_knots", "wind_direction_deg", "gust_knots",
        "visibility_miles", "ceiling_ft", "precip_amount_in", "precip_measurement_status",
        "precip_flag", "rain_flag", "snow_flag", "thunder_flag", "temperature_f", "dewpoint_f",
        "relative_humidity") if column in result]
    result.loc[stale, feature_columns] = np.nan
    result["metar_missing"] = (result.weather_time.isna() | stale | ambiguous).astype("int8")
    result["metar_missing_reason"] = np.select(
        [ambiguous, stale, result.weather_time.isna()],
        ["conflicting_reports", "stale_observation", "no_observation"],
        default="available",
    )
    result = result.sort_values("_metar_input_order")
    original_index = result.pop("_metar_original_index")
    result = result.drop(columns=["_metar_input_order", "_metar_airport_key"])
    result.index = pd.Index(original_index.tolist(), name=flights.index.name)
    return result


def _body_issue_time(raw_header, archive_issue_time):
    if not raw_header or pd.isna(archive_issue_time):
        return pd.NaT
    match = re.search(r"^\s*(?:TAF(?:\s+(?:AMD|COR))?\s+)?[A-Z0-9]{4}\s+(\d{6})Z",
                      str(raw_header), re.I | re.M)
    if not match:
        return pd.NaT
    day, hour, minute = int(match.group(1)[:2]), int(match.group(1)[2:4]), int(match.group(1)[4:6])
    archive = pd.Timestamp(archive_issue_time)
    choices = []
    for offset in (-1, 0, 1):
        base = archive + pd.DateOffset(months=offset)
        try:
            choices.append(pd.Timestamp(year=base.year, month=base.month, day=day,
                                         hour=hour, minute=minute, tz="UTC"))
        except ValueError:
            continue
    return min(choices, key=lambda x: abs(x - archive)) if choices else pd.NaT


def _taf_message_type(product_id, raw_header):
    header = str(raw_header or "").upper()
    pid = str(product_id or "").upper()
    if re.search(r"\bTAF\s+COR\b", header) or re.search(r"-(?:CC[A-X])(?:-|$)", pid):
        return "corrected"
    if re.search(r"\bTAF\s+AMD\b", header) or re.search(r"-AA[A-X](?:-|$)", pid):
        return "amended"
    if re.search(r"-RR[A-X](?:-|$)", pid):
        return "delayed"
    return "routine"


def taf_availability_proxy(body_issue_time, archive_issue_time):
    """Return max(body, archive issue) only when both timestamps are known."""
    if pd.isna(body_issue_time) or pd.isna(archive_issue_time):
        return pd.NaT
    return max(pd.Timestamp(body_issue_time), pd.Timestamp(archive_issue_time))


def prepare_taf():
    d = read_batches("taf").drop_duplicates(
        ["airport_id", "station", "product_id", "valid", "fx_valid", "ftype", "raw"])
    validity_path = ROOT / "references/taf_validity.csv"
    if not validity_path.exists():
        raise FileNotFoundError("Run scripts/download_taf_text.py --prepare-validity before preparing TAF")
    bulletins = pd.read_csv(validity_path)
    # A parsed segment proves that a product existed even when its full text
    # could not be recovered. Keep that state in the bulletin stream so a newer
    # unknown product cannot silently revive an older forecast.
    keys = ["product_id", "station"]
    missing = d.merge(bulletins[keys].drop_duplicates(), on=keys,
                      how="left", indicator=True)
    missing = missing.loc[missing._merge.eq("left_only"), keys + ["valid"]].copy()
    if not missing.empty:
        missing["issue_time"] = _utc_ns(missing.pop("valid"))
        # First archived evidence is the conservative blocking boundary; this
        # is never used as a substitute for the missing body issue timestamp.
        missing = missing.groupby(keys, as_index=False, dropna=False).issue_time.min()
        missing["valid_from"] = pd.NaT
        missing["valid_to"] = pd.NaT
        missing["raw_header"] = ""
        missing["status"] = "unverified"
        bulletins = pd.concat([bulletins, missing], ignore_index=True)
    for col in ["issue_time", "valid_from", "valid_to"]:
        bulletins[col] = _utc_ns(bulletins[col])
    bulletins["airport_id"] = bulletins.station.map(station_lookup())
    bulletins = bulletins[bulletins.airport_id.notna()].copy()
    bulletins["airport_id"] = bulletins.airport_id.astype("int64")
    bulletins["archive_issue_time"] = bulletins.issue_time
    bulletins["body_issue_time"] = _utc_ns([
        _body_issue_time(raw, archive) for raw, archive in zip(bulletins.raw_header, bulletins.archive_issue_time)])
    bulletins["issue_time_mismatch_minutes"] = (
        (bulletins.body_issue_time - bulletins.archive_issue_time).dt.total_seconds() / 60)
    bulletins["message_type"] = [
        _taf_message_type(pid, header) for pid, header in zip(bulletins.product_id, bulletins.raw_header)]
    bulletins["availability_time"] = _utc_ns([taf_availability_proxy(body, archive)
                                      for body, archive in zip(bulletins.body_issue_time,
                                                               bulletins.archive_issue_time)])
    bulletins["availability_verified"] = False
    bulletins["availability_assumption"] = "max_body_and_archive_issue_proxy; receipt_time_not_archived"
    # Product IDs identify a unique bulletin version. Preserve every version;
    # never keep only the final row for a station/product pair.
    bulletins = bulletins.drop_duplicates(["product_id", "station"], keep="first")
    merge_cols = ["product_id", "station", "issue_time", "archive_issue_time", "body_issue_time",
                  "issue_time_mismatch_minutes", "availability_time", "availability_verified",
                  "availability_assumption", "message_type", "valid_from", "valid_to", "status"]
    d = d.merge(bulletins[merge_cols], on=["product_id", "station"], how="left", validate="many_to_one")
    d["segment_start"] = _utc_ns(d.fx_valid)
    d["segment_end"] = _utc_ns(d.fx_valid_end)
    initial = d.ftype.eq("Observation")
    # IEM labels the opening prevailing forecast "Observation"; it remains a forecast.
    d.loc[initial, "segment_start"] = d.loc[initial, "valid_from"]
    prevailing = d.ftype.isin(["Observation", "Forecast"])
    base = d.loc[prevailing].sort_values(["product_id", "station", "segment_start"])
    next_start = base.groupby(["product_id", "station"]).segment_start.shift(-1)
    d.loc[base.index, "segment_end"] = next_start.fillna(base.valid_to)
    d["segment_start"] = d[["segment_start", "valid_from"]].max(axis=1)
    d["segment_end"] = d[["segment_end", "valid_to"]].min(axis=1)
    d["validity_matched"] = d.valid_to.notna() & d.valid_from.notna() & d.body_issue_time.notna()
    d["interval_usable"] = d.validity_matched & d.segment_end.gt(d.segment_start) & d.status.eq("normal")
    if "is_tempo" in d:
        d["is_tempo"] = d.is_tempo.astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_tempo"] = False
    d["is_probability"] = d.ftype.astype(str).str.lower().str.contains("prob", regex=False)
    d["ceiling_ft"] = [ceiling(parsed_list(c), parsed_list(h)) for c, h in zip(d.skyc, d.skyl)]
    def taf_tokens(value):
        try:
            items = parsed_list(value)
        except (ValueError, SyntaxError):
            items = [value]
        return _weather_tokens(" ".join(map(str, items)))
    tokens = d.presentwx.map(taf_tokens)
    wx_reported = (d.presentwx.notna() & d.presentwx.astype("string").str.strip().ne("") &
                   d.presentwx.astype("string").str.upper().ne("M"))
    d["thunder_flag"] = pd.array([_nullable_weather_flag(xs, known, ("TS",))
                                   for xs, known in zip(tokens, wx_reported)], dtype="Int8")
    d["rain_flag"] = pd.array([_nullable_weather_flag(xs, known, ("RA", "DZ"))
                                for xs, known in zip(tokens, wx_reported)], dtype="Int8")
    d["snow_flag"] = pd.array([_nullable_weather_flag(xs, known, ("SN", "SG", "PL", "GR", "GS"))
                                for xs, known in zip(tokens, wx_reported)], dtype="Int8")
    d["precip_flag"] = pd.array([_nullable_weather_flag(
        xs, known, ("RA", "DZ", "SN", "SG", "PL", "GR", "GS", "UP"))
        for xs, known in zip(tokens, wx_reported)], dtype="Int8")
    d = d.rename(columns={"sknt": "wind_knots", "drct": "wind_direction_deg",
                          "gust": "gust_knots", "visibility": "visibility_miles"})
    keep = ["airport_id", "station", "product_id", "issue_time", "archive_issue_time", "body_issue_time",
            "issue_time_mismatch_minutes", "availability_time", "availability_verified",
            "availability_assumption", "message_type", "valid_from", "valid_to", "segment_start",
            "segment_end", "ftype", "is_amendment", "status", "validity_matched", "interval_usable",
            "is_tempo", "is_probability",
            "wind_knots", "wind_direction_deg", "gust_knots", "visibility_miles", "ceiling_ft",
            "precip_flag", "rain_flag", "snow_flag", "thunder_flag", "raw"]
    return d[keep], bulletins


def select_taf_asof(segments, bulletins, airport_id, decision_time, query_time, *,
                    allow_unverified_revisions=False):
    """Select latest as-of bulletin first, then its segments at the query time.

    max(body issue time, archive issue time) is an explicit availability proxy.
    Missing body times block fallback when their archive stamp may be newer.
    AMD/COR/delayed versions are disabled by default because no receive time is
    archived; a disabled latest version never falls back to an older bulletin.
    """
    decision_time = _utc_ns(pd.Timestamp(decision_time))
    query_time = _utc_ns(pd.Timestamp(query_time))
    if decision_time.tzinfo is None or query_time.tzinfo is None:
        raise ValueError("decision_time and query_time must be timezone-aware")
    d = segments.loc[segments.airport_id.eq(airport_id)].copy()
    b = bulletins.loc[bulletins.airport_id.eq(airport_id)].copy()
    for frame, columns in ((d, ("segment_start", "segment_end")),
                           (b, ("availability_time", "archive_issue_time", "body_issue_time"))):
        for column in columns:
            if column in frame:
                frame[column] = _utc_ns(frame[column])
    if b.empty:
        return {"status": "no_bulletin", "bulletin": None, "segments": d.iloc[0:0]}
    bulletin_cols = ["product_id", "station", "status", "message_type", "availability_time",
                     "archive_issue_time", "body_issue_time", "availability_verified"]
    b = b[bulletin_cols].drop_duplicates(["product_id", "station"])
    untimed = b[b.availability_time.isna() & b.archive_issue_time.notna() &
                b.archive_issue_time.le(decision_time)]
    b = b.dropna(subset=["availability_time"])
    b = b[b.availability_time.le(decision_time)]
    latest_timed = b.availability_time.max() if not b.empty else pd.NaT
    if not untimed.empty and (pd.isna(latest_timed) or untimed.archive_issue_time.max() >= latest_timed):
        row = untimed.sort_values("archive_issue_time").iloc[-1].to_dict()
        return {"status": "unverified_issue_time", "bulletin": row, "segments": d.iloc[0:0]}
    if b.empty:
        return {"status": "no_available_bulletin", "bulletin": None, "segments": d.iloc[0:0]}
    latest_time = b.availability_time.max()
    latest_rows = b.loc[b.availability_time.eq(latest_time)]
    if latest_rows.product_id.nunique() > 1:
        return {"status": "ambiguous_bulletin", "bulletin": None, "segments": d.iloc[0:0]}
    latest = b.sort_values(["availability_time", "product_id"]).iloc[-1]
    bulletin = latest.to_dict()
    status = str(latest.status).lower()
    if status in {"cancelled", "canceled", "nil"}:
        return {"status": status, "bulletin": bulletin, "segments": d.iloc[0:0]}
    if status == "ambiguous":
        return {"status": "ambiguous_bulletin", "bulletin": bulletin, "segments": d.iloc[0:0]}
    if latest.message_type in {"amended", "corrected", "delayed"} and not allow_unverified_revisions:
        return {"status": "disabled_revision", "bulletin": bulletin, "segments": d.iloc[0:0]}
    selected = d.loc[d.product_id.eq(latest.product_id)]
    selected = selected.loc[selected.interval_usable & selected.segment_start.le(query_time) &
                            selected.segment_end.gt(query_time)]
    return {"status": "matched" if not selected.empty else "outside_validity",
            "bulletin": bulletin, "segments": selected}


def taf_query_time(flights, *, scheduled_dep_col="scheduled_dep_utc",
                   prediction_col="prediction_time"):
    """Return max(scheduled downstream departure, prediction time), in UTC."""
    scheduled = _utc_ns(flights[scheduled_dep_col])
    prediction = _utc_ns(flights[prediction_col])
    return scheduled.where(scheduled.ge(prediction), prediction).where(scheduled.notna() | prediction.notna())


def noaa_qc_is_eligible(source_code, quality_code, element):
    source = "" if pd.isna(source_code) else str(source_code).strip()
    qc = "" if pd.isna(quality_code) else str(quality_code).strip()
    if not qc:
        # Blank QC is accepted only where GHCNh explicitly documents its
        # meaning. For unknown source codes (including 413) blank is not proof
        # of a pass and the value remains ineligible pending source-list review.
        if source == "382" and element == "precipitation":
            return True  # GHCNh defines blank source-382 precipitation QC as no failed QA check.
        if source == "345":
            return False
        return False
    if source in NOAA_LEGACY_FULL_PASS:
        return qc in NOAA_LEGACY_FULL_PASS[source]
    if source == "345" and element in {"wind_speed", "relative_humidity"}:
        return qc == "0"  # Source 345 documents 0=Good, 1=Issue, 3=Error.
    if source == "382":
        return False  # Every nonblank source-382 QC value is a documented failure/flag.
    # Unknown source/QC combinations are preserved but are not eligible.
    return False


def noaa_precip_status(value, measurement_code, source_code):
    if pd.isna(value):
        return "missing"
    code = "" if pd.isna(measurement_code) else str(measurement_code).strip().upper()
    source = "" if pd.isna(source_code) else str(source_code).strip()
    if (code == "2" and source in NOAA_PRECIPITATION_TRACE_CODE_2_SOURCES) or (
            code == "T" and source == "382"):
        return "trace"
    # The documentation assigns measurement code 1 to impossible/inaccurate
    # amounts for the listed legacy source families. Do not relabel such a
    # numeric payload as a normal measurement; the original code remains intact.
    if code == "1" and source in NOAA_PRECIPITATION_TRACE_CODE_2_SOURCES:
        return "invalid_measurement_code"
    # Other nonblank codes are source-specific (e.g. accumulation boundaries).
    # Keep the raw code and avoid implying a plain hourly measured amount.
    if code:
        return "source_specific_measurement_code"
    return "measured"


def prepare_noaa_frame(raw, source_file=""):
    """Normalize one GHCNh chunk, retaining native units and field provenance."""
    d = raw.copy()
    required = {"STATION", "DATE"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"GHCNh chunk missing required columns: {sorted(missing)}")
    d["observation_time"] = _utc_ns(d.DATE)
    if d.observation_time.isna().any():
        raise ValueError("GHCNh rows contain invalid DATE values")
    d["source_file"] = source_file
    for element in NOAA_ELEMENTS:
        if element not in d:
            continue
        d[element] = pd.to_numeric(d[element], errors="coerce").mask(lambda values: values.eq(-9999))
        qc_col = f"{element}_Quality_Code"
        source_col = f"{element}_Source_Code"
        d[f"{element}_qc_eligible"] = [
            bool(pd.notna(value) and noaa_qc_is_eligible(src, qc, element))
            for value, src, qc in zip(d[element], d.get(source_col, pd.Series(index=d.index, dtype=object)),
                                      d.get(qc_col, pd.Series(index=d.index, dtype=object)))
        ]
    if "precipitation" in d:
        d["precipitation_measurement_status"] = [
            noaa_precip_status(value, code, source)
            for value, code, source in zip(d.precipitation,
                d.get("precipitation_Measurement_Code", pd.Series(index=d.index, dtype=object)),
                d.get("precipitation_Source_Code", pd.Series(index=d.index, dtype=object)))
        ]
        d["precipitation_period_note"] = (
            "nominal_hour; source may include intermediate or accumulated reports; preserve report/source metadata")
    return d


def iter_noaa_batches(chunksize=250_000):
    files = sorted((ROOT / "raw/weather/noaa").glob("GHCNh_*.psv.gz"))
    if not files:
        return
    for path in files:
        available = pd.read_csv(path, sep="|", compression="gzip", nrows=0).columns
        usecols = [column for column in NOAA_COLUMNS if column in available]
        if not {"STATION", "DATE"}.issubset(usecols):
            raise ValueError(f"{path.name}: missing GHCNh identifier/time columns")
        metadata_cols = [column for column in usecols if column in {"STATION", "Station_name", "DATE"}
                         or column.endswith(("Measurement_Code", "Quality_Code", "Report_Type",
                                             "Source_Code", "Source_Station_ID"))]
        for chunk in pd.read_csv(path, sep="|", compression="gzip", usecols=usecols,
                                 dtype={column: str for column in metadata_cols},
                                 chunksize=chunksize, low_memory=False):
            yield prepare_noaa_frame(chunk, source_file=path.name)


def run(output_dir=None, summary_path=None):
    """Prepare weather tables into an optional isolated run directory.

    With no arguments, retain the original processed/ and tables/ locations.
    An explicitly supplied output directory must be new or empty to avoid
    silently mixing artifacts from different runs.
    """
    explicit_output = output_dir is not None
    out = Path(output_dir) if explicit_output else ROOT / "processed"
    if explicit_output and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty weather output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    summary_file = Path(summary_path) if summary_path is not None else ROOT / "tables/weather_preparation_summary.json"
    if explicit_output and summary_path is not None and summary_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing weather summary: {summary_file}")
    metar = prepare_metar()
    taf, bulletins = prepare_taf()
    compression = {"method": "gzip", "compresslevel": 3, "mtime": 0}
    for name, d in [("metar_observations", metar), ("taf_segments", taf), ("taf_bulletins", bulletins)]:
        d.to_csv(out / f"{name}.csv.gz", index=False, compression=compression)
        d.to_pickle(out / f"{name}.pkl")

    noaa_path = out / "noaa_observations.csv.gz"
    noaa_rows = 0
    noaa_stations = set()
    first = True
    # Chunked streaming keeps the 25-station source archive out of memory.
    with gzip.open(noaa_path, "wt", encoding="utf-8", newline="") as stream:
        for chunk in iter_noaa_batches():
            chunk.to_csv(stream, index=False, header=first)
            first = False
            noaa_rows += len(chunk)
            noaa_stations.update(chunk.STATION.dropna().astype(str).unique())

    summary = {
        "metar_rows": len(metar), "metar_airports": int(metar.airport_id.nunique()),
        "taf_segment_rows": len(taf), "taf_airports": int(taf.airport_id.nunique()),
        "taf_validity_matched_rows": int(taf.validity_matched.sum()),
        "taf_usable_interval_rows": int(taf.interval_usable.sum()), "taf_bulletins": len(bulletins),
        "metar_assumed_lag_minutes": 5, "metar_max_observation_age_minutes": 120,
        "taf_body_issue_time_used_as_availability_proxy": True,
        "taf_receive_time_verified": False,
        "noaa_rows": noaa_rows, "noaa_stations": len(noaa_stations),
        "noaa_quality_policy": "strict exact source/QC whitelist; original element codes retained; sensitivity only",
        "noaa_postintegration_qc_serialization_verified": False,
        "noaa_qc_limitation": (
            "GHCNh publishes one Quality_Code string per element and documents post-integration QC plus "
            "component-source QC, but does not specify precedence/serialization when both apply. "
            "Only exact documented source/QC pairs are accepted; this is not full-QC certification."
        ),
        "noaa_historical_availability_verified": False,
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
