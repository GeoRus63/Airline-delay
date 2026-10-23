"""Create an isolated, auditable weather-preparation dataset from local archives.

No network access is used here. TAF validity is parsed directly from the ZIP
archives already named by ``tables/taf_text_manifest.json``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

try:  # package import (tests / notebooks)
    from . import download_taf_text, prepare_weather
except ImportError:  # run_pipeline.py imports modules from the scripts directory
    import download_taf_text
    import prepare_weather


ROOT = Path(__file__).resolve().parents[1]
TAF_VALIDITY_FIELDS = ["product_id", "station", "issue_time", "valid_from", "valid_to",
                       "status", "raw_header", "raw_name", "source_archives", "raw_sha256s",
                       "occurrence_count", "ambiguity_reason"]
MANIFEST_NAME = "data_manifest.json"


def _root_relative(value, *, label):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must resolve inside project root: {value}") from exc
    return path


def _relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path, block_size=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path):
    return {"path": _relative(path), "bytes": Path(path).stat().st_size, "sha256": _sha256(path)}


def _read_config(config):
    if isinstance(config, (str, os.PathLike)):
        config_path = _root_relative(config, label="config path")
        return json.loads(config_path.read_text(encoding="utf-8")), config_path
    if config is None:
        return {}, None
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping or a project-relative JSON path")
    return dict(config), None


def _taf_archives(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requests = manifest.get("requests", [])
    if not requests:
        raise ValueError(f"TAF text manifest has no local archive requests: {_relative(manifest_path)}")
    archives = []
    for request in requests:
        item = request.get("file")
        if not item:
            continue
        archive_path = _root_relative(item, label="TAF archive")
        if not archive_path.is_file():
            raise FileNotFoundError(f"TAF archive listed by manifest is missing: {_relative(archive_path)}")
        archives.append((archive_path, request))
    if not archives:
        raise ValueError("TAF text manifest contains no local archive paths")
    return manifest, archives


def _taf_science_fingerprint(payload):
    """Normalize ZIP wrapper noise while preserving every bulletin token."""
    text = payload.decode("latin-1", "replace")
    lines = text.splitlines()
    if lines and re.fullmatch(r"\s*\d+\s*", lines[0]):
        lines = lines[1:]
    return " ".join(" ".join(lines).split())


def _taf_body_sha256(payload):
    """Hash normalized TAF body tokens, excluding a ZIP member sequence line."""
    text = payload.decode("latin-1", "replace")
    lines = text.splitlines()
    if lines and re.fullmatch(r"\s*\d+\s*", lines[0]):
        lines = lines[1:]
    text = "\n".join(lines)
    match = re.search(r"(?m)^[A-Z0-9]{4}\s+\d{6}Z\b", text)
    body = text[match.start():] if match else text
    normalized = " ".join(body.split())
    return hashlib.sha256(normalized.encode("latin-1", "replace")).hexdigest()


def _common_body_issue_prefix(rows):
    """Keep only an original station/issue prefix shared by all copies."""
    prefixes = []
    for row in rows:
        match = re.search(r"(?m)^\s*([A-Z0-9]{4}\s+\d{6}Z)\b", row.get("raw_header", ""))
        if not match:
            return ""
        prefixes.append(match.group(1).strip())
    return prefixes[0] if prefixes and len(set(prefixes)) == 1 else ""


def _merge_taf_occurrences(key, occurrences):
    """Coalesce byte-content-equivalent science; mark real conflicts unusable."""
    rows = [item["row"] for item in occurrences]
    science_fields = ("product_id", "station", "issue_time", "valid_from", "valid_to", "status")
    same_parsed_science = all(
        all(row.get(field, "") == rows[0].get(field, "") for field in science_fields)
        for row in rows[1:])
    same_bulletin = len({item["science_fingerprint"] for item in occurrences}) == 1
    merged = dict(rows[0])
    merged["raw_name"] = json.dumps([item["raw_name"] for item in occurrences], ensure_ascii=False)
    merged["source_archives"] = json.dumps([item["source_archive"] for item in occurrences], ensure_ascii=False)
    merged["raw_sha256s"] = json.dumps([item["raw_sha256"] for item in occurrences], ensure_ascii=False)
    merged["occurrence_count"] = len(occurrences)
    if same_parsed_science and same_bulletin:
        return merged

    # A bulletin ID/station pair with conflicting body tokens is not a valid
    # forecast choice. Retain the common issue prefix so the status can enter
    # the as-of stream, but expose no interval and never select either body.
    merged["status"] = "ambiguous"
    merged["valid_from"] = ""
    merged["valid_to"] = ""
    merged["raw_header"] = _common_body_issue_prefix(rows)
    merged["ambiguity_reason"] = "conflicting_scientific_content_or_parsed_validity"
    return merged


def _prepare_validity(manifest_path, output_path):
    manifest, archives = _taf_archives(manifest_path)
    occurrences_by_product = {}
    archive_diagnostics = []
    for archive_path, request in archives:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".txt"):
                    continue
                payload = archive.read(member)
                product_id = download_taf_text.identify_member(member.filename, payload)
                if not product_id:
                    continue
                row = download_taf_text.validity_row(member.filename, payload)
                if not row or row["product_id"] != product_id:
                    continue
                key = (row["product_id"], row["station"])
                occurrences_by_product.setdefault(key, []).append({
                    "row": row,
                    "raw_name": member.filename,
                    "source_archive": _relative(archive_path),
                    "raw_sha256": hashlib.sha256(payload).hexdigest(),
                    "science_fingerprint": _taf_science_fingerprint(payload),
                    "body_sha256": _taf_body_sha256(payload),
                })
        archive_diagnostics.append({"path": _relative(archive_path),
                                    "manifest_sha256": request.get("sha256"),
                                    "members": request.get("product_count")})

    rows = [_merge_taf_occurrences(key, occurrences)
            for key, occurrences in occurrences_by_product.items()]
    rows.sort(key=lambda row: (row["issue_time"], row["station"], row["product_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="taf_validity_", suffix=".csv.tmp", dir=output_path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TAF_VALIDITY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    statuses = Counter(row["status"] for row in rows)
    return {"manifest": _relative(manifest_path), "local_archive_count": len(archives),
            "archive_diagnostics": archive_diagnostics, "validity_rows": len(rows),
            "unique_products": len({row["product_id"] for row in rows}),
            "status_counts": {name: statuses.get(name, 0)
                              for name in ("normal", "cancelled", "nil", "ambiguous")},
            "ambiguous_duplicate_products": [
                {"product_id": row["product_id"], "station": row["station"],
                 "occurrence_count": row["occurrence_count"],
                 "source_archives": json.loads(row["source_archives"]),
                 "raw_names": json.loads(row["raw_name"]),
                 "raw_sha256s": json.loads(row["raw_sha256s"]),
                 "reason": row.get("ambiguity_reason")}
                for row in rows if row["status"] == "ambiguous"],
            "duplicate_products": [
                {"product_id": key[0], "station": key[1], "occurrence_count": len(occurrences),
                 "differing_parsed_fields": [field for field in
                     ("issue_time", "valid_from", "valid_to", "status")
                     if len({item["row"].get(field, "") for item in occurrences}) > 1],
                 "occurrences": [{"source_archive": item["source_archive"],
                     "raw_name": item["raw_name"], "raw_sha256": item["raw_sha256"],
                     "body_sha256": item["body_sha256"],
                     "issue_time": item["row"].get("issue_time", ""),
                     "valid_from": item["row"].get("valid_from", ""),
                     "valid_to": item["row"].get("valid_to", ""),
                     "status": item["row"].get("status", "")}
                     for item in occurrences]}
                for key, occurrences in sorted(occurrences_by_product.items()) if len(occurrences) > 1],
            "normal_missing_validity_window": sum(
                row["status"] == "normal" and (not row["valid_from"] or not row["valid_to"])
                for row in rows)}


def _source_paths(config, taf_manifest_path, config_path=None):
    supplied = config.get("source_files", [])
    paths = []
    if supplied:
        if isinstance(supplied, (str, os.PathLike)):
            raise TypeError("source_files must be a list of project-relative file paths")
        paths.extend(_root_relative(item, label="source file") for item in supplied)
    # Hash every file the fixed preparer actually reads, even when the caller
    # also supplies a source_files receipt list.
    for subdir, pattern in (("raw/weather/metar", "batch_*.csv.gz"),
                            ("raw/weather/taf", "batch_*.csv.gz"),
                            ("raw/weather/noaa", "GHCNh_*.psv.gz"),
                            ("raw/weather/taf_text", "*.zip")):
        paths.extend(sorted((ROOT / subdir).glob(pattern)))
    # The manifests, station map, and coverage table define interpretation and
    # source completeness, so include them even when the caller lists raw files.
    paths.extend([taf_manifest_path, ROOT / "tables/weather_coverage.csv",
                  ROOT / "tables/weather_download_manifest.json", ROOT / "references/airports.csv"])
    try:
        taf_manifest = json.loads(taf_manifest_path.read_text(encoding="utf-8"))
        paths.extend(_root_relative(item["file"], label="TAF archive")
                     for item in taf_manifest.get("requests", []) if item.get("file"))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid TAF text manifest: {_relative(taf_manifest_path)}") from exc
    if config_path is not None:
        paths.append(config_path)
    unique = {path.resolve() for path in paths if path.exists() and path.is_file()}
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required source file(s) are missing: {missing[:5]}")
    return sorted(unique, key=lambda path: _relative(path))


def _coverage_summary(path):
    data = pd.read_csv(path)
    required = {"airport_id", "month", "kind", "rows", "days"}
    if required - set(data.columns):
        raise ValueError(f"weather coverage missing columns: {sorted(required - set(data.columns))}")
    grouped = data.groupby(["kind", "month"], dropna=False).agg(
        airport_count=("airport_id", "nunique"), source_rows=("rows", "sum"),
        airport_days=("days", "sum")).reset_index()
    return {"coverage_rows": len(data), "by_kind_month": grouped.to_dict(orient="records")}


def _noaa_qc_counts(noaa_path, chunksize=200_000):
    totals = Counter()
    eligible = Counter()
    try:
        chunks = pd.read_csv(noaa_path, compression="gzip", chunksize=chunksize, low_memory=False)
        for chunk in chunks:
            qc_cols = [column for column in chunk.columns if column.endswith("_qc_eligible")]
            for column in qc_cols:
                values = chunk[column].astype("string").str.lower().isin(["true", "1", "yes"])
                totals[column.removesuffix("_qc_eligible")] += len(chunk)
                eligible[column.removesuffix("_qc_eligible")] += int(values.sum())
    except pd.errors.EmptyDataError:
        return {}
    return {element: {"rows": totals[element], "eligible_rows": eligible[element]}
            for element in sorted(totals)}


def _tree_records(folder, exclude=()):
    excluded = {Path(item).resolve() for item in exclude}
    return [_file_record(path) for path in sorted(Path(folder).rglob("*"))
            if path.is_file() and path.resolve() not in excluded]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run(config, run_dir):
    """Prepare a W-series weather dataset from already-downloaded local inputs."""
    settings, config_path = _read_config(config)
    output_value = settings.get("weather_output_dir", f"processed/{settings.get('data_id', 'W0001')}")
    output_dir = _root_relative(output_value, label="weather_output_dir")
    run_path = _root_relative(run_dir, label="run_dir")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty weather output directory: {_relative(output_dir)}")
    summary_path = run_path / "weather_preparation_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing weather summary: {_relative(summary_path)}")
    run_manifest_path = run_path / MANIFEST_NAME
    if run_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing data manifest: {_relative(run_manifest_path)}")
    run_config_path = run_path / "weather_config.json"
    if run_config_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing weather config: {_relative(run_config_path)}")

    input_path = settings.get("input_path", "raw/weather")
    input_dir = _root_relative(input_path, label="input_path")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Weather input path is not a directory: {_relative(input_dir)}")
    if input_dir != (ROOT / "raw/weather").resolve():
        raise ValueError("prepare_weather currently reads the canonical raw/weather directory; input_path must be raw/weather")
    taf_manifest_path = _root_relative(settings.get("taf_text_manifest", "tables/taf_text_manifest.json"),
                                       label="taf_text_manifest")
    source_files = _source_paths(settings, taf_manifest_path, config_path)
    source_records = [_file_record(path) for path in source_files]

    taf_validity_path = ROOT / "references/taf_validity.csv"
    taf_info = _prepare_validity(taf_manifest_path, taf_validity_path)
    config_record = {"config": settings, "input_path": _relative(input_dir),
                     "source_files": [_relative(path) for path in source_files]}
    preparation = prepare_weather.run(output_dir=output_dir, summary_path=summary_path)
    shutil.copy2(taf_validity_path, output_dir / "taf_validity.csv")
    config_archive = output_dir / "weather_config.json"
    _write_json(config_archive, config_record)
    _write_json(output_dir / "weather_preparation_summary.json", preparation)
    # The existing prepare_weather writer emits its summary to run_dir; retain
    # that exact copy and archive one in the dataset as well.
    qc_counts = _noaa_qc_counts(output_dir / "noaa_observations.csv.gz")
    coverage = _coverage_summary(ROOT / "tables/weather_coverage.csv")
    source_manifest = {"sources": source_records, "generated_taf_validity": _file_record(taf_validity_path),
                       "taf_validity": taf_info}
    source_manifest_path = output_dir / "source_manifest.json"
    _write_json(source_manifest_path, source_manifest)

    run_path.mkdir(parents=True, exist_ok=True)
    _write_json(run_config_path, config_record)
    manifest = {
        "data_id": settings.get("data_id", "W0001"), "input_path": _relative(input_dir),
        "inputs": source_records,
        "outputs": [],
        "taf_validity": taf_info,
        "weather_preparation": preparation,
        "noaa_qc_counts": qc_counts,
        "coverage": coverage,
        "assumptions": {
            "metar_availability": "observation_time + 5 minutes; sensitivity proxy, receipt time absent",
            "taf_availability": "max(body issue time, archive issue time); proxy, receipt time absent",
            "noaa_availability": "observation_time + 5 minutes is retrospective sensitivity only; historical receipt not verified",
        },
    }
    if settings.get("source_data_id") is not None:
        manifest["source_data_id"] = settings["source_data_id"]
    processed_manifest = output_dir / MANIFEST_NAME
    output_files = _tree_records(output_dir, exclude=(processed_manifest,))
    manifest["outputs"] = output_files
    manifest["run_artifacts"] = [_file_record(path) for path in (run_config_path, summary_path)]
    _write_json(processed_manifest, manifest)
    _write_json(run_manifest_path, manifest)
    return {"weather_preparation": preparation, "data_manifest": manifest,
            "processed_manifest": _relative(processed_manifest), "run_manifest": _relative(run_manifest_path),
            "weather_output_dir": _relative(output_dir), "summary_path": _relative(summary_path)}
