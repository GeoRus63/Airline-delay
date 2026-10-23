#!/usr/bin/env python3
"""Download original NOAA GHCNh 2026 PSV files for METAR gap stations."""

from __future__ import annotations

import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEATHER_COVERAGE = ROOT / "tables" / "weather_coverage.csv"
AIRPORTS = ROOT / "references" / "airports.csv"
OUTPUT = ROOT / "raw" / "weather" / "noaa"
MANIFEST = ROOT / "tables" / "noaa_download_manifest.json"
DOC_URL = "https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly"
BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/2026/psv"
MONTH_DAYS = {"2026-04": 30, "2026-05": 31, "2026-06": 30, "2026-07": 31, "2026-08": 31}
OBSERVATION_FIELDS = {
    "temperature", "dew_point_temperature", "station_level_pressure",
    "sea_level_pressure", "wind_direction", "wind_speed", "wind_gust",
    "precipitation", "visibility", "ceiling_height", "sky_condition",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def targets() -> tuple[list[dict], list[dict]]:
    coverage = read_csv(WEATHER_COVERAGE)
    airports = read_csv(AIRPORTS)
    day_totals: dict[str, int] = {}
    for row in coverage:
        if row["kind"] == "metar" and row["month"] in MONTH_DAYS:
            day_totals[row["airport_id"]] = day_totals.get(row["airport_id"], 0) + int(row["days"])

    stations: dict[str, dict] = {}
    blocked: list[dict] = []
    for airport in airports:
        airport_id = airport["airport_id"]
        days = day_totals.get(airport_id)
        if days is None or days >= sum(MONTH_DAYS.values()):
            continue
        station = (airport.get("ghcnh_id") or "").strip()
        record = {
            "airport_id": airport_id,
            "airport_code": airport.get("airport_code", ""),
            "ghcnh_id": station,
            "coverage_days": days,
            "expected_days": sum(MONTH_DAYS.values()),
            "missing_days": sum(MONTH_DAYS.values()) - days,
        }
        if not station:
            blocked.append({**record, "status": "missing_ghcnh_id", "error": "No GHCNh ID in airports.csv"})
            continue
        if station not in stations:
            stations[station] = {
                "ghcnh_id": station,
                "airports": [],
                "airport_ids": [],
                "coverage_days": days,
                "expected_days": sum(MONTH_DAYS.values()),
                "missing_days": sum(MONTH_DAYS.values()) - days,
            }
        stations[station]["airports"].append(record["airport_code"])
        stations[station]["airport_ids"].append(airport_id)
        stations[station]["coverage_days"] = min(stations[station]["coverage_days"], days)
        stations[station]["missing_days"] = max(stations[station]["missing_days"], sum(MONTH_DAYS.values()) - days)
    return sorted(stations.values(), key=lambda x: x["ghcnh_id"]), blocked


def inspect_psv(stream) -> dict:
    stream.seek(0)
    digest = hashlib.sha256()
    row_count = 0
    station_ids: set[str] = set()
    date_days: set[str] = set()
    observed_month_days: dict[str, set[str]] = {month: set() for month in MONTH_DAYS}
    header: list[str] | None = None
    reader = csv.reader(stream, delimiter="|", quotechar='"')
    for row in reader:
        if header is None:
            if row:
                row[0] = row[0].lstrip("\ufeff")
            header = row
            if "STATION" not in header or "DATE" not in header or len(header) < 3:
                raise ValueError(f"Unexpected PSV header: {'|'.join(header[:5])}")
            if not OBSERVATION_FIELDS.intersection(header):
                raise ValueError("No recognized observation field in PSV header")
            continue
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"PSV row {row_count + 2} has {len(row)} fields; expected {len(header)}")
        row_count += 1
        station_ids.add(row[header.index("STATION")])
        day = row[header.index("DATE")][:10]
        if len(day) == 10 and day[4] == "-" and day[7] == "-":
            date_days.add(day)
            month = day[:7]
            if month in observed_month_days:
                observed_month_days[month].add(day)
    if header is None:
        raise ValueError("Empty PSV")
    return {
        "header": header,
        "row_count": row_count,
        "station_ids": sorted(station_ids),
        "date_first": min(date_days) if date_days else None,
        "date_last": max(date_days) if date_days else None,
        "unique_dates_2026": len(date_days),
        "available_days_2026_04_08": sum(len(days) for days in observed_month_days.values()),
        "available_days_by_month": {month: len(days) for month, days in observed_month_days.items()},
    }


def verify_compressed(path: Path, expected_station: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        details = inspect_psv(stream)
    if details["station_ids"] != [expected_station]:
        raise ValueError(f"Station IDs in file {details['station_ids']} do not equal {expected_station}")
    details["compressed_bytes"] = path.stat().st_size
    with path.open("rb") as source:
        digest = hashlib.sha256()
        while block := source.read(1024 * 1024):
            digest.update(block)
        details["compressed_sha256"] = digest.hexdigest()
    with gzip.open(path, "rb") as source:
        digest = hashlib.sha256()
        source_bytes = 0
        while block := source.read(1024 * 1024):
            digest.update(block)
            source_bytes += len(block)
        details["source_sha256"] = digest.hexdigest()
        details["source_bytes"] = source_bytes
    return details


def remote_size(url: str) -> tuple[int, str]:
    response = subprocess.run(
        [
            "curl", "--head", "--location", "--silent", "--show-error",
            "--retry", "4", "--retry-all-errors", "--retry-delay", "1",
            "--connect-timeout", "20", "--max-time", "90",
            "--write-out", "\n__HTTP_CODE__%{http_code}", url,
        ],
        text=True,
        capture_output=True,
    )
    status_line = next((line for line in response.stdout.splitlines() if line.startswith("__HTTP_CODE__")), "")
    status = status_line.removeprefix("__HTTP_CODE__")
    lengths = [line.split(":", 1)[1].strip() for line in response.stdout.splitlines() if line.lower().startswith("content-length:")]
    if response.returncode != 0 or status != "200" or not lengths:
        raise RuntimeError(f"HEAD failed: HTTP {status or 'unknown'}; {response.stderr.strip()[-1000:]}")
    return int(lengths[-1]), status


def download_one(station: dict) -> dict:
    station_id = station["ghcnh_id"]
    url = f"{BASE_URL}/GHCNh_{station_id}_2026.psv"
    target = OUTPUT / f"GHCNh_{station_id}_2026.psv.gz"
    try:
        expected_bytes, http_status = remote_size(url)
    except Exception as error:
        return {**station, "status": "head_failed", "source_url": url, "error": str(error)}
    if target.exists():
        try:
            details = verify_compressed(target, station_id)
            if details["source_bytes"] != expected_bytes:
                raise ValueError(f"Cached source has {details['source_bytes']} bytes; server reports {expected_bytes}")
            return {**station, "status": "verified_cached", "source_url": url, "http_status": http_status, "expected_source_bytes": expected_bytes, **details}
        except Exception:
            target.unlink()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / f".GHCNh_{station_id}_2026.psv.download"
    part_path = OUTPUT / f".GHCNh_{station_id}_2026.psv.download.part"
    gz_path = None
    try:
        if not raw_path.exists():
            if part_path.exists():
                part_path.unlink()
            response = subprocess.run(
                [
                    "curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "4", "--retry-all-errors", "--retry-delay", "1",
                    "--connect-timeout", "20", "--max-time", "900",
                    "--output", str(part_path), "--write-out", "%{http_code}", url,
                ],
                text=True,
                capture_output=True,
            )
            http_status = response.stdout.strip()[-3:] if response.stdout.strip() else None
            if response.returncode != 0 or http_status != "200":
                return {
                    **station,
                    "status": "download_failed",
                    "source_url": url,
                    "http_status": http_status,
                    "curl_exit_code": response.returncode,
                    "temporary_file": str(part_path) if part_path.exists() else None,
                    "error": response.stderr.strip()[-2000:],
                }
            os.replace(part_path, raw_path)
        with raw_path.open("r", encoding="utf-8-sig", newline="") as source:
            details = inspect_psv(source)
        if details["station_ids"] != [station_id]:
            raise ValueError(f"Station IDs in downloaded file {details['station_ids']} do not equal {station_id}")
        with raw_path.open("rb") as source:
            digest = hashlib.sha256()
            while block := source.read(1024 * 1024):
                digest.update(block)
            source_hash = digest.hexdigest()
            source_bytes = raw_path.stat().st_size
            if source_bytes != expected_bytes:
                raise ValueError(f"Downloaded source has {source_bytes} bytes; server reports {expected_bytes}")
            with tempfile.NamedTemporaryFile(prefix=f".{station_id}.", suffix=".psv.gz.part", dir=OUTPUT, delete=False) as temp_gz:
                gz_path = Path(temp_gz.name)
            source.seek(0)
            with gzip.open(gz_path, "wb", compresslevel=6) as compressed:
                while block := source.read(1024 * 1024):
                    compressed.write(block)
        compressed_details = verify_compressed(gz_path, station_id)
        if compressed_details["source_bytes"] != source_bytes or compressed_details["source_sha256"] != source_hash:
            raise ValueError("gzip round-trip did not preserve the downloaded PSV bytes")
        os.replace(gz_path, target)
        raw_path.unlink()
        return {
            **station,
            "status": "downloaded_verified",
            "source_url": url,
            "http_status": http_status,
            "expected_source_bytes": expected_bytes,
            "source_bytes": source_bytes,
            "source_sha256": source_hash,
            **compressed_details,
        }
    except Exception as error:
        return {
            **station, "status": "validation_failed", "source_url": url,
            "temporary_file": str(raw_path) if raw_path.exists() else None,
            "error": str(error),
        }
    finally:
        if gz_path and gz_path.exists():
            gz_path.unlink()


def write_manifest(records: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "NOAA NCEI Global Historical Climatology Network hourly (GHCNh)",
        "year": 2026,
        "source_product_page": DOC_URL,
        "source_bulk_download_pattern": f"{BASE_URL}/GHCNh_{{ghcnh_id}}_2026.psv",
        "compression": "gzip compression of downloaded PSV bytes; source fields and quality codes are not modified",
        "coverage_window": {"start": "2026-04-01", "end": "2026-08-31", "expected_calendar_days": sum(MONTH_DAYS.values())},
        "stations": records,
    }
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, MANIFEST)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stations, blocked = targets()
    records = [*blocked]
    by_id = {station["ghcnh_id"]: {**station, "status": "pending"} for station in stations}
    records.extend(by_id.values())
    write_manifest(records)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download_one, station): station["ghcnh_id"] for station in stations}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            by_id[result["ghcnh_id"]] = result
            records = [*blocked, *by_id.values()]
            write_manifest(records)
            print(f"{result['ghcnh_id']}: {result['status']} {result.get('available_days_2026_04_08', 0)}/153 days")
    print(f"manifest={MANIFEST} stations={len(stations)} blocked={len(blocked)}")


if __name__ == "__main__":
    main()
