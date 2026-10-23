"""从原始 ZIP 复算衔接样本 / Recalculate selected connections from source ZIP rows.

使用标准库，直接读取原始 ZIP 中的样本行。 / Uses the standard library to read sample rows directly from the source ZIPs.
运行命令 / Run: python _build/verify_samples.py
"""

import csv
import json
import io
import math
import re
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


OUT = Path(__file__).resolve().parents[1]
RAW = OUT.parent
GROUPS = (
    "accepted",
    "overnight",
    "cross_month",
    "early_upstream_arrival",
    "upstream_delay_180plus",
)
TIME_FIELDS = (
    "FlightDate", "Tail_Number", "Origin", "Dest", "CRSDepTime", "CRSArrTime",
    "DepTime", "ArrTime", "DepDelay", "ArrDelay", "DepDelayMinutes",
    "ArrDelayMinutes", "Cancelled", "Diverted", "CRSElapsedTime", "ActualElapsedTime",
)
ID_RE = re.compile(r"^(2026_0[45]):([0-9]+)$")


def number(value):
    return None if value is None or value.strip() == "" else float(value)


def hhmm(value):
    raw = number(value)
    if raw is None or not raw.is_integer():
        raise AssertionError(f"无效 HHMM 值 / Invalid HHMM value: {value!r}")
    value = int(raw)
    if value == 2400:
        return 1440
    hour, minute = divmod(value, 100)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise AssertionError(f"无效 HHMM 值 / Invalid HHMM value: {value!r}")
    return hour * 60 + minute


def source_id(value):
    match = ID_RE.fullmatch(value)
    if not match:
        raise AssertionError(f"航班 ID 格式不匹配 / Unexpected flight_id: {value!r}")
    return match.group(1), int(match.group(2))


def parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    return parsed.astimezone(timezone.utc)


def close_minutes(actual, expected, label):
    error = abs((actual - expected).total_seconds() / 60)
    assert error < 1e-7, f"{label} 差异 / Difference: {error} min"


def load_original_rows(ids):
    wanted = {}
    for flight_id in ids:
        month, row = source_id(flight_id)
        wanted.setdefault(month, set()).add(row)
    originals = {}
    for month, row_numbers in wanted.items():
        zip_path = RAW / f"{month}.zip"
        assert zip_path.is_file(), f"缺少原始 ZIP / Missing input ZIP: {zip_path}"
        with zipfile.ZipFile(zip_path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            assert len(csv_names) == 1, f"预期一个 CSV / Expected one CSV in {zip_path.name}"
            with archive.open(csv_names[0]) as binary, io.TextIOWrapper(
                binary, encoding="utf-8-sig", newline=""
            ) as text:
                reader = csv.DictReader(text)
                assert set(TIME_FIELDS).issubset(reader.fieldnames or [])
                for row_number, row in enumerate(reader, start=2):
                    if row_number in row_numbers:
                        flight_id = f"{month}:{row_number}"
                        originals[flight_id] = {field: row[field] for field in TIME_FIELDS}
                    if row_number >= max(row_numbers):
                        break
        assert {source_id(fid)[1] for fid in originals if fid.startswith(month + ":")} == row_numbers
    return originals


def scheduled_utc(row, airport_tz):
    flight_day = date.fromisoformat(row["FlightDate"])
    minute = hhmm(row["CRSDepTime"])
    local = datetime.combine(flight_day, datetime.min.time()) + timedelta(minutes=minute)
    # 检查夏令时切换造成的重复或不存在时刻。 / Check repeated or nonexistent local times at DST transitions.
    first = local.replace(tzinfo=airport_tz, fold=0)
    roundtrip = first.astimezone(timezone.utc).astimezone(airport_tz).replace(tzinfo=None)
    second = local.replace(tzinfo=airport_tz, fold=1)
    roundtrip2 = second.astimezone(timezone.utc).astimezone(airport_tz).replace(tzinfo=None)
    assert roundtrip == local and roundtrip2 == local, f"当地计划时间不存在 / Nonexistent local schedule time: {local} {airport_tz}"
    assert first.utcoffset() == second.utcoffset(), f"当地计划时间重复 / Ambiguous local schedule time: {local} {airport_tz}"
    return first.astimezone(timezone.utc)


def verify_pair(sample, originals, tz_lookup):
    up_id, down_id = sample["upstream_flight_id"], sample["downstream_flight_id"]
    up, down = originals[up_id], originals[down_id]
    assert up["Tail_Number"].strip() == down["Tail_Number"].strip() == sample["tail_number"]
    assert up["Dest"].strip() == down["Origin"].strip() == sample["airport"]
    assert up["Origin"].strip() == sample["upstream_origin"]
    assert down["Origin"].strip() == sample["downstream_origin"]
    assert float(up["Cancelled"]) == float(down["Cancelled"]) == 0
    assert float(up["Diverted"]) == float(down["Diverted"]) == 0

    up_dep = scheduled_utc(up, ZoneInfo(tz_lookup[up["Origin"].strip()]))
    down_dep = scheduled_utc(down, ZoneInfo(tz_lookup[down["Origin"].strip()]))
    up_arr = up_dep + timedelta(minutes=float(up["CRSElapsedTime"]))
    down_arr = down_dep + timedelta(minutes=float(down["CRSElapsedTime"]))
    up_actual_dep = up_dep + timedelta(minutes=float(up["DepDelay"]))
    up_actual_arr = up_arr + timedelta(minutes=float(up["ArrDelay"]))
    down_actual_dep = down_dep + timedelta(minutes=float(down["DepDelay"]))
    down_actual_arr = down_arr + timedelta(minutes=float(down["ArrDelay"]))

    for side, scheduled_dep, scheduled_arr, actual_dep, actual_arr in (
        ("upstream", up_dep, up_arr, up_actual_dep, up_actual_arr),
        ("downstream", down_dep, down_arr, down_actual_dep, down_actual_arr),
    ):
        close_minutes(scheduled_dep, parse_utc(sample[f"{side}_scheduled_dep_utc"]), f"{side} scheduled dep")
        close_minutes(scheduled_arr, parse_utc(sample[f"{side}_scheduled_arr_utc"]), f"{side} scheduled arr")
        close_minutes(actual_dep, parse_utc(sample[f"{side}_actual_dep_utc"]), f"{side} actual dep")
        close_minutes(actual_arr, parse_utc(sample[f"{side}_actual_arr_utc"]), f"{side} actual arr")

    scheduled_turn = (down_dep - up_arr).total_seconds() / 60
    actual_ground = (down_actual_dep - up_actual_arr).total_seconds() / 60
    assert math.isclose(scheduled_turn, float(sample["scheduled_turnaround_minutes"]), abs_tol=1e-7)
    assert math.isclose(actual_ground, float(sample["actual_ground_minutes"]), abs_tol=1e-7)
    assert actual_ground >= 0, f"实际地面间隔为负 / Negative actual ground interval: {up_id}->{down_id}"

    # 按项目口径使用非负延误计算差值。 / Calculate the delta from nonnegative delays, as defined for this project.
    expected_delta = float(down["DepDelayMinutes"]) - float(up["ArrDelayMinutes"])
    assert math.isclose(expected_delta, float(sample["delta_delay_minutes"]), abs_tol=1e-7)

    airport_zone = ZoneInfo(tz_lookup[up["Dest"].strip()])
    overnight = up_arr.astimezone(airport_zone).date() != down_dep.astimezone(airport_zone).date()
    assert overnight == (sample["overnight_scheduled"].lower() == "true"), (
        f"当地日期隔夜标记不一致 / Local-date overnight mismatch for {up_id}->{down_id}: {overnight}"
    )
    assert 0 <= scheduled_turn <= 1440, f"计划周转超出区间 / Scheduled turnaround outside window: {scheduled_turn}"
    return scheduled_turn, actual_ground, expected_delta, overnight


def main():
    with (OUT / "references" / "airport_timezones_used.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        tz_lookup = {row["iata"]: row["tz"] for row in csv.DictReader(handle)}

    with (OUT / "tables" / "connection_audit_samples.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        all_samples = list(csv.DictReader(handle))
    selected = []
    for group in GROUPS:
        group_rows = [row for row in all_samples if row["sample_group"] == group]
        assert group_rows, f"缺少样本组 / Missing sample group: {group}"
        assert len(group_rows) <= 5, f"样本超过五条 / More than five samples for {group}"
        selected.extend(group_rows)
    assert all(row["main_analysis_eligible"].lower() == "true" for row in selected)

    ids = {row[key] for row in selected for key in ("upstream_flight_id", "downstream_flight_id")}
    originals = load_original_rows(ids)
    verified = []
    for row in selected:
        result = verify_pair(row, originals, tz_lookup)
        up = originals[row["upstream_flight_id"]]
        down = originals[row["downstream_flight_id"]]
        assert up["FlightDate"] == row["upstream_flight_date"][:10]
        assert down["FlightDate"] == row["downstream_flight_date"][:10]
        group = row["sample_group"]
        if group == "accepted":
            assert row["first_exclusion_reason"] == "accepted"
        elif group == "overnight":
            assert result[3]
        elif group == "cross_month":
            assert up["FlightDate"][:7] != down["FlightDate"][:7]
        elif group == "early_upstream_arrival":
            assert float(up["ArrDelay"]) < 0
        elif group == "upstream_delay_180plus":
            assert float(up["ArrDelayMinutes"]) >= 180
        verified.append((row, result))

    with (OUT / "tables" / "connection_exclusions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        exclusion_rows = list(csv.DictReader(handle))
    reason_total = sum(int(row["pairs"]) for row in exclusion_rows)
    accepted_count = next(int(row["pairs"]) for row in exclusion_rows if row["reason"] == "accepted")
    with (OUT / "tables" / "quality_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert reason_total == summary["candidate_connections"]
    assert accepted_count == summary["accepted_connections"]

    print(f"通过 / PASS: {len(selected)} 对衔接 / pairs; {len(GROUPS)} 组 / groups")
    print(f"通过 / PASS: 从原始 ZIP 复读 {len(ids)} 班航班 / {len(ids)} distinct source flights re-read from the two ZIPs")
    print(
        f"通过：时间、尾号、机场、间隔、延误变化及隔夜标记一致 / PASS: timestamps, route/tail continuity, intervals, "
        f"delay delta, and local-date overnight agree for every selected pair"
    )
    print(f"通过 / PASS: 候选 / Candidates {reason_total:,}; 有效 / Eligible {accepted_count:,}; 与质量汇总一致 / Matches quality summary")
    for group in GROUPS:
        print(f"  {group}: {sum(row['sample_group'] == group for row in selected)} 对 / pairs")


if __name__ == "__main__":
    main()
