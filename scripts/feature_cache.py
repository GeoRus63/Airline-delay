"""Verified development-only feature reuse for fixed-feature algorithm comparisons."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import train_models as training
except ImportError:
    import train_models as training

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_KEYS = ("feature_group", "noaa_backfill", "connection_path", "data_id", "weather_manifest_path")


def _path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _setting(config, key):
    if key == "feature_group":
        return str(config.get(key, "G0")).upper()
    if key == "noaa_backfill":
        return config.get(key, False)
    value = config.get(key)
    if key == "connection_path":
        value = value or "processed/flight_connections.csv.gz"
    if key.endswith("_path") and value:
        return str(_path(value))
    return value


def load_development_archive(config):
    """Read only a receipt-verified completed run's development feature pickle.

    No source flight/weather reads or fitting occur here. An invalid archive is
    rejected, never filtered to make it eligible. Pickles are trusted local run
    artifacts, not an interchange format for untrusted external files.
    """
    if config.get("mode") != "development":
        raise ValueError("Feature archives are restricted to mode=development")
    if not config.get("feature_archive_path") or not config.get("feature_source_run_id"):
        raise ValueError("feature_archive_path and feature_source_run_id are required")
    archive = _path(config["feature_archive_path"])
    source_dir = archive.parent.parent
    if archive.name != "features.pkl" or archive.parent.name != "data" or source_dir.name != config["feature_source_run_id"]:
        raise ValueError("Archive must be source_run_id/data/features.pkl")
    manifest = _read_json(source_dir / "manifest.json")
    if manifest.get("status") != "completed" or manifest.get("run_id") != config["feature_source_run_id"]:
        raise ValueError("Feature source must be the named completed run")
    if manifest.get("invalidated") or manifest.get("valid") is False:
        raise ValueError("Feature source has been invalidated")
    receipts = {}
    for item in manifest.get("outputs", []):
        path = _path(item["path"])
        if path in receipts:
            raise ValueError("Duplicate source output receipt")
        receipts[path] = item
    checked = []
    for path in [source_dir / "config.json", source_dir / "model_manifest.json", archive]:
        receipt = receipts.get(path)
        if receipt is None or path.stat().st_size != receipt.get("bytes") or _sha(path) != receipt.get("sha256"):
            raise ValueError(f"Source output receipt mismatch: {path.name}")
        checked.append({"path": str(path), "bytes": receipt["bytes"], "sha256": receipt["sha256"]})
    source_config = _read_json(source_dir / "config.json")
    model = _read_json(source_dir / "model_manifest.json")
    if source_config.get("mode") != "development" or model.get("development_only") is not True:
        raise ValueError("Source must explicitly be development-only")
    for key in SCIENTIFIC_KEYS:
        if _setting(config, key) != _setting(source_config, key):
            raise ValueError(f"Scientific feature setting differs: {key}")
    for key in ("features", "categorical_features", "numeric_features"):
        if key not in config or config[key] != model.get(key):
            raise ValueError(f"Explicit requested features differ from source model: {key}")
    for candidate in [config, source_config]:
        resolved = training.resolve_features(candidate)
        if any(resolved[key] != model.get(key) for key in ("features", "categorical_features", "numeric_features")):
            raise ValueError("Resolved scientific features differ from source model")
    frame = pd.read_pickle(archive)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Feature archive must be a nonempty DataFrame")
    required = set(config["features"] + training.PREDICTION_METADATA + ["upstream_arr_delay_minutes"])
    if required - set(frame):
        raise ValueError(f"Feature archive lacks columns: {sorted(required - set(frame))}")
    # Check time scope before interpreting targets, including any August sentinel.
    prediction = pd.to_datetime(frame.prediction_time, utc=True, errors="coerce")
    if not (prediction.ge(training.DEVELOPMENT_START) & prediction.lt(training.DEVELOPMENT_END)).all():
        raise ValueError("Archive contains timestamps outside April-July development scope")
    if frame.connection_id.isna().any() or not frame.connection_id.is_unique:
        raise ValueError("Feature archive IDs must be nonmissing and unique")
    label = pd.to_datetime(frame.label_time, utc=True, errors="coerce")
    target = pd.to_numeric(frame.target_delay, errors="coerce")
    if not (prediction.lt(label) & np.isfinite(target) & target.ge(0)).all():
        raise ValueError("Feature archive violates label/target eligibility")
    data = frame.sort_values(["prediction_time", "connection_id"], kind="stable").reset_index(drop=True)
    audits = model.get("folds", [])
    if [a.get("fold") for a in audits] != [f[0] for f in training.DEVELOPMENT_FOLDS]:
        raise ValueError("Source evaluation folds differ")
    payload_hash = hashlib.sha256(json.dumps(audits, sort_keys=True).encode("utf-8")).hexdigest()
    if payload_hash != model.get("split_sha256"):
        raise ValueError("Source split hash differs")
    ids = []
    for (fold, cutoff, end, train_mask, validation_mask), audit in zip(training.development_splits(data), audits):
        train, validation = data.loc[train_mask], data.loc[validation_mask]
        actual = {"train_rows": len(train), "validation_rows": len(validation),
                  "train_id_sha256": training.hash_ids(train.connection_id),
                  "validation_id_sha256": training.hash_ids(validation.connection_id),
                  "train_target_sha256": training.hash_targets(train),
                  "validation_target_sha256": training.hash_targets(validation)}
        if train.empty or validation.empty or any(audit.get(k) != v for k, v in actual.items()):
            raise ValueError(f"Cached cohort differs from source evaluation: {fold}")
        ids.extend(fold + ":" + validation.connection_id.astype(str))
    evaluation_hash = training.hash_ids(ids)
    if evaluation_hash != model.get("evaluation_ids_sha256") or len(ids) != model.get("evaluation_rows"):
        raise ValueError("Source evaluation ID hash/count differs")
    frame.attrs["cache_audit"] = {"feature_source_run_id": config["feature_source_run_id"],
        "feature_archive_path": str(archive), "verified_outputs": checked,
        "evaluation_ids_sha256": evaluation_hash, "split_sha256": payload_hash,
        "development_rows": len(frame), "scientific_settings_checked": list(SCIENTIFIC_KEYS),
        "development_only": True, "refitted": False, "weather_rejoined": False}
    return frame
