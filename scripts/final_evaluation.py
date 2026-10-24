"""Frozen P6 assessment and shared computation for exact, separately archived reproduction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import train_models as training
    from . import evaluate_models as evaluation
    from . import compare_experiments as comparison
    from .build_features import load_configured_features
except ImportError:
    import train_models as training
    import evaluate_models as evaluation
    import compare_experiments as comparison
    from build_features import load_configured_features

ROOT = Path(__file__).resolve().parents[1]
FINAL_FOLD = "2026-08"
BASELINES = ["training_mean", "training_median", "carry_forward"]
DEFAULT_PROTOCOL = {
    "timezone": "UTC", "train_start": "2026-04-01T00:00:00+00:00",
    "train_end_exclusive": "2026-08-01T00:00:00+00:00",
    "train_label_end_exclusive": "2026-08-01T00:00:00+00:00",
    "test_start": "2026-08-01T00:00:00+00:00", "test_end_exclusive": "2026-09-01T00:00:00+00:00",
    "target": "downstream_nonnegative_departure_delay_minutes",
    "eligibility": "finite target_delay >= 0; prediction_time < label_time",
    "prediction_postprocess": "clip_zero", "severity_threshold_minutes": 60, "delayed_threshold_minutes": 15,
    "bootstrap": {"block_days": 7, "repeats": 1000, "seed": 42},
}
REQUIRED_CONFIG_KEYS = {"feature_group", "features", "categorical_features", "numeric_features", "model_name",
                        "model_type", "model_params", "preprocessing", "threads", "linear_tolerance",
                        "include_baselines", "connection_path", "data_id"}
RUNTIME_KEYS = {"mode", "freeze_manifest_path", "freeze_manifest_sha256", "stage", "change", "parent_run_id", "run_id"}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _required_sources(feature_group):
    directory = Path(__file__).resolve().parent
    names = ["final_evaluation.py", "reproduce_final.py", "run_pipeline.py", "render_research_notebooks.py",
             "train_models.py", "build_features.py", "evaluate_models.py", "compare_experiments.py"]
    if feature_group.upper() != "G0":
        names.append("operational_features.py")
    if feature_group.upper() in {"G4A", "G4B"}:
        names.extend(["weather_pipeline_features.py", "weather_features.py", "prepare_weather.py"])
    return {(directory / name).resolve() for name in names}


def _required_data(config):
    connection = _path(config["connection_path"])
    result = {connection}
    cache = connection.with_name("flight_connections.pkl")
    if cache.is_file():
        result.add(cache)
    group = config["feature_group"].upper()
    if group != "G0":
        result.update({_path("processed/flights_clean.csv.gz"), _path("references/airports.csv")})
    if group not in {"G0", "G1"}:
        result.update({_path("references/aircraft_reference_2024.csv"), _path("references/aircraft_type_aliases.csv")})
    if group in {"G4A", "G4B"}:
        if not config.get("weather_manifest_path"):
            raise ValueError("Frozen G4 config requires weather_manifest_path")
        manifest_path = _path(config["weather_manifest_path"])
        weather = _json(manifest_path)
        records = weather.get("inputs", []) + weather.get("outputs", [])
        if not weather.get("outputs") or any(not entry.get("path") or not entry.get("sha256") for entry in records):
            raise ValueError("Weather manifest must declare pinned outputs and valid input/output receipts")
        result.add(manifest_path)
        result.update(_path(entry["path"]) for entry in records)
    return result


def validate_freeze(config):
    """Validate all protocol/config/code/data pins before any semantic August read.

    Freeze schema v1 requires status/frozen_at_utc/selection/primary_model_name,
    config/protocol/source_files/data_files/payload_sha256. Receipt lists contain
    {path,sha256}; paths are repository-relative or absolute. payload_sha256 is the
    canonical hash of every field except payload_sha256 itself. Runtime config pins
    the exact frozen JSON file bytes using freeze_manifest_sha256.
    """
    if config.get("mode") not in {"final", "reproduce_final"}:
        raise ValueError("August assessment requires mode='final' or mode='reproduce_final'")
    if {"feature_archive_path", "feature_source_run_id"} & set(config):
        raise ValueError("Final evaluation forbids development feature_archive_path/feature_source_run_id")
    if not config.get("freeze_manifest_path") or not config.get("freeze_manifest_sha256"):
        raise ValueError("freeze_manifest_path and freeze_manifest_sha256 are required")
    path = _path(config["freeze_manifest_path"])
    file_hash = sha256_file(path)
    if file_hash != config["freeze_manifest_sha256"]:
        raise ValueError("Frozen manifest file SHA-256 mismatch")
    frozen = _json(path)
    if frozen.get("schema_version") != 1 or frozen.get("status") != "frozen":
        raise ValueError("A schema_version=1 frozen manifest is required")
    payload = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    if frozen.get("payload_sha256") != canonical_sha256(payload):
        raise ValueError("Frozen manifest payload SHA-256 mismatch")
    timestamp = pd.Timestamp(frozen["frozen_at_utc"])
    if timestamp.tzinfo is None or timestamp > pd.Timestamp.now(tz="UTC"):
        raise ValueError("frozen_at_utc must be timezone-aware and no later than final access")
    if not frozen.get("selection", {}).get("selected_run_id") or not frozen["selection"].get("reason"):
        raise ValueError("Freeze must identify the selected development run and selection reason")
    locked = frozen["config"]
    if {"feature_archive_path", "feature_source_run_id"} & set(locked):
        raise ValueError("Final evaluation forbids development feature_archive_path/feature_source_run_id")
    if REQUIRED_CONFIG_KEYS - set(locked):
        raise ValueError(f"Frozen scientific config is incomplete: {sorted(REQUIRED_CONFIG_KEYS - set(locked))}")
    if frozen.get("protocol") != DEFAULT_PROTOCOL or locked["include_baselines"] is not True:
        raise ValueError("Frozen protocol must use the fixed final UTC/maturity/metric/bootstrap rules and all baselines")
    if any(key in locked for key in RUNTIME_KEYS):
        raise ValueError("Frozen scientific config must omit runtime metadata")
    preprocessing = locked["preprocessing"]
    if set(preprocessing) != {"drop", "min_frequency"}:
        raise ValueError("Freeze preprocessing must explicitly pin drop and min_frequency")
    minimum = preprocessing["min_frequency"]
    valid_minimum = (minimum is None or type(minimum) is int and minimum >= 1
                     or type(minimum) is float and 0 < minimum < 1)
    if not valid_minimum or type(locked["threads"]) is not int or locked["threads"] < 1:
        raise ValueError("Frozen min_frequency/threads are invalid")
    training.prediction_threads(locked)  # Optional RF-only execution setting; old freezes retain their default.
    if "prediction_threads" in config:
        training.prediction_threads({**locked, "prediction_threads": config["prediction_threads"]})
    for key in ["max_depth", "min_samples_leaf", "n_estimators", "random_state"]:
        if key in locked["model_params"] and type(locked["model_params"][key]) is not int:
            raise ValueError(f"Frozen model parameter {key} must be an integer")
    for key, value in config.items():
        if key not in RUNTIME_KEYS and (key not in locked or value != locked[key]):
            raise ValueError(f"Runtime config differs from frozen scientific setting: {key}")
    resolved = training.resolve_features(locked)
    if any(resolved[key] != locked[key] for key in ["features", "categorical_features", "numeric_features"]):
        raise ValueError("Frozen typed features differ from the executed feature resolver")
    names = training.model_names(locked)
    if frozen.get("primary_model_name") not in names:
        raise ValueError("primary_model_name must be the frozen learned model or a declared baseline")
    estimator = training.make_pipeline(locked).named_steps["regression"]
    if locked["model_type"] == "random_forest" and estimator.n_estimators != 200:
        raise ValueError("The formal final forest must use the frozen 200-tree specification")
    if int(locked["threads"]) < 1 or not str(locked["data_id"]).strip():
        raise ValueError("Frozen threads/data_id are invalid")
    receipts = {}
    for kind, required in [("source_files", _required_sources(locked["feature_group"])), ("data_files", _required_data(locked))]:
        entries = frozen.get(kind, [])
        indexed = {_path(entry["path"]): entry for entry in entries}
        if len(indexed) != len(entries) or not entries or required - set(indexed):
            raise ValueError(f"Frozen {kind} omit required executed files or contain duplicate paths")
        checked = []
        for actual, entry in indexed.items():
            observed = sha256_file(actual)
            if observed != entry["sha256"]:
                raise ValueError(f"Frozen {kind} SHA-256 mismatch: {entry['path']}")
            checked.append({"path": entry["path"], "sha256": observed, "bytes": actual.stat().st_size})
        receipts[kind] = checked
    if locked["feature_group"].upper() in {"G4A", "G4B"}:
        weather = _json(_path(locked["weather_manifest_path"]))
        pins = {_path(entry["path"]): entry["sha256"] for entry in receipts["data_files"]}
        for entry in weather.get("inputs", []) + weather["outputs"]:
            if pins[_path(entry["path"])] != entry["sha256"]:
                raise ValueError(f"Frozen weather manifest input/output SHA differs from pinned file: {entry['path']}")
    return frozen, path, file_hash, receipts


def _feature_hash(frame, features):
    digest = hashlib.sha256(json.dumps(features, separators=(",", ":")).encode())
    digest.update(np.asarray(pd.util.hash_pandas_object(frame[features], index=False), dtype="<u8").tobytes())
    return digest.hexdigest()


def _frame_fingerprint(frame, features):
    """Cover every executed predictor, baseline input, outcome and prediction identity.

    Schema is hashed too: equal string renderings must not hide dtype changes.
    The split has already sorted rows, so the fingerprint also seals row order.
    """
    columns = list(dict.fromkeys(["connection_id", "prediction_time", "label_time", "target_delay",
                                 *features, "upstream_arr_delay_minutes",
                                 *[name for name in training.PREDICTION_METADATA if name in frame],
                                 *(["aircraft_family"] if "aircraft_family" in frame else [])]))
    schema = [{"name": name, "dtype": str(frame[name].dtype)} for name in columns]
    return {"rows": len(frame), "columns": columns, "schema": schema,
            "schema_sha256": canonical_sha256(schema), "values_sha256": _feature_hash(frame, columns),
            "ids_sha256": training.hash_ids(frame.connection_id), "targets_sha256": training.hash_targets(frame)}


def split_final_frame(frame):
    """The test window is half-open; training labels mature strictly before August."""
    prediction = pd.to_datetime(frame.prediction_time, utc=True, errors="raise")
    label = pd.to_datetime(frame.label_time, utc=True, errors="raise")
    start, cutoff, stop = [pd.Timestamp(DEFAULT_PROTOCOL[key]) for key in ["train_start", "test_start", "test_end_exclusive"]]
    prior = prediction.ge(start) & prediction.lt(cutoff)
    train = frame.loc[prior & label.lt(cutoff)].copy()
    test = frame.loc[prediction.ge(cutoff) & prediction.lt(stop)].copy()
    for name, part in [("train", train), ("test", test)]:
        if part.empty or part.connection_id.isna().any() or not part.connection_id.is_unique:
            raise ValueError(f"Final {name} must be nonempty with unique IDs")
        part["prediction_time"] = prediction.loc[part.index]
        part["label_time"] = label.loc[part.index]
        part["target_delay"] = pd.to_numeric(part.target_delay, errors="coerce")
        if not (np.isfinite(part.target_delay) & part.target_delay.ge(0) & part.prediction_time.lt(part.label_time)).all():
            raise ValueError(f"Final {name} violates outcome availability/target eligibility")
        part.sort_values(["prediction_time", "connection_id"], kind="stable", inplace=True)
    if set(train.connection_id) & set(test.connection_id):
        raise ValueError("Training and August IDs overlap")
    audit = {"fold": FINAL_FOLD, "train_start": start.isoformat(), "train_end_exclusive": cutoff.isoformat(),
             "validation_start": cutoff.isoformat(), "validation_end_exclusive": stop.isoformat(),
             "train_rows": len(train), "validation_rows": len(test),
             "excluded_immature_training_labels": int((prior & ~label.lt(cutoff)).sum()),
             "max_train_prediction_time": train.prediction_time.max().isoformat(), "max_train_label_time": train.label_time.max().isoformat(),
             "min_validation_prediction_time": test.prediction_time.min().isoformat(), "max_validation_prediction_time": test.prediction_time.max().isoformat(),
             "train_id_sha256": training.hash_ids(train.connection_id), "validation_id_sha256": training.hash_ids(test.connection_id),
             "train_target_sha256": training.hash_targets(train), "validation_target_sha256": training.hash_targets(test)}
    return train, test, audit


def _boundary_counts(connection_path):
    # Only timestamps/local dates are used outside the main UTC test window.
    metadata = pd.read_csv(connection_path, usecols=["upstream_actual_arr_utc", "downstream_flight_date"])
    prediction = pd.to_datetime(metadata.upstream_actual_arr_utc, utc=True, errors="coerce")
    local = pd.to_datetime(metadata.downstream_flight_date, errors="coerce")
    return {"local_date_definition": "downstream BTS FlightDate", "local_august_prediction_at_or_after_sep1": int(
        (local.dt.year.eq(2026) & local.dt.month.eq(8) & prediction.ge(pd.Timestamp(DEFAULT_PROTOCOL["test_end_exclusive"]))).sum())}


def _saved_predictions(run_dir, names):
    results = {}
    for name in names:
        frame = pd.read_csv(run_dir / "predictions" / f"{FINAL_FOLD}_{name}.csv.gz", float_precision="round_trip")
        frame["prediction_time"] = pd.to_datetime(frame.prediction_time, utc=True)
        frame["label_time"] = pd.to_datetime(frame.label_time, utc=True)
        if not frame.fold.eq(FINAL_FOLD).all() or not frame.model.eq(name).all() or frame.connection_id.duplicated().any():
            raise ValueError("Saved final predictions have invalid identities")
        results[name] = frame
    reference = results[names[0]]
    for frame in results.values():
        comparison.align_predictions(reference, frame, align_rows=False)
    return results


def _postprocess_predictions(output, config, primary, *, reproduction=False):
    names = training.model_names(config)
    predictions = _saved_predictions(output, names)
    metrics = evaluation.metrics_from_summaries(pd.concat([evaluation.summarize_predictions(frame)
                                                          for frame in predictions.values()], ignore_index=True))
    metrics.to_csv(output / "metrics.csv", index=False)
    residual_dir = output / "residuals"
    residual_dir.mkdir(exist_ok=True)
    residual_summary = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    edges = [-np.inf, -300, -120, -60, -30, -15, 0, 15, 30, 60, 120, 300, np.inf]
    labels = ["<-300", "-300:-120", "-120:-60", "-60:-30", "-30:-15", "-15:0", "0:15", "15:30", "30:60", "60:120", "120:300", "300+"]
    for ax, (name, frame) in zip(axes.flat, predictions.items()):
        residual = frame[["fold", "connection_id", "prediction_time", "target_delay", "predicted_delay"]].copy()
        residual["signed_error"] = residual.predicted_delay - residual.target_delay
        residual["absolute_error"] = residual.signed_error.abs()
        residual["squared_error"] = residual.signed_error ** 2
        residual.to_csv(residual_dir / f"{FINAL_FOLD}_{name}.csv.gz", index=False, compression=training.GZIP)
        error = residual.signed_error
        residual_summary.append({"model": name, "n": len(error), "mean_signed_error": float(error.mean()),
                                 "median_signed_error": float(error.median()), "p10_signed_error": float(error.quantile(.1)),
                                 "p90_signed_error": float(error.quantile(.9)), "p95_absolute_error": float(error.abs().quantile(.95))})
        n, _ = np.histogram(error, bins=edges)
        ax.bar(np.arange(len(n)), n / len(error) * 100, color="#356A91")
        ax.set(xticks=np.arange(len(n)), xticklabels=labels, ylabel="Test rows (%)", xlabel="Prediction - target (min)", title=f"{name} | n={len(error):,}")
        ax.tick_params(axis="x", labelrotation=50, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    caption = "Reproduction of frozen August assessment" if reproduction else "Frozen August assessment"
    fig.suptitle(caption + " | complete residual distribution, including tails")
    fig.tight_layout()
    (output / "figures").mkdir(exist_ok=True)
    fig.savefig(output / "figures/final_residual_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(residual_summary).to_csv(output / "residual_summary.csv", index=False)
    calendar = pd.date_range(DEFAULT_PROTOCOL["test_start"], DEFAULT_PROTOCOL["test_end_exclusive"], inclusive="left", freq="D")
    comparisons = []
    for baseline in BASELINES:
        daily = comparison.daily_statistics(predictions[baseline], predictions[primary], calendar)
        daily.to_csv(output / f"final_comparison_{baseline}_daily.csv")
        interval = comparison.comparison_intervals(daily, bootstrap_repeats=1000, block_days=7, seed=42)
        interval = interval.loc[interval.scope.eq("pooled")].copy()
        interval["reference_model"], interval["candidate_model"], interval["sample_scope"] = baseline, primary, "final_test"
        interval.to_csv(output / f"final_comparison_{baseline}.csv", index=False)
        comparison._comparison_figure(interval, output / "figures" / f"final_comparison_{baseline}.png", baseline, primary)
        comparisons.append(interval)
    pd.concat(comparisons, ignore_index=True).to_csv(output / "final_comparisons.csv", index=False)
    return metrics


def _fit_frozen(frozen, freeze_path, freeze_hash, verified, output, ledger_path, *, mode):
    """Shared final/reproduction computation; caller owns authorization and ledger policy."""
    locked = frozen["config"]
    shutil.copy2(freeze_path, output / "model_selection.json")
    _write_json(output / "final_verified_inputs.json", verified)
    _write_json(output / "final_effective_config.json", locked)
    frame = load_configured_features(locked, development_only=False)
    train, test, audit = split_final_frame(frame)
    features = locked["features"]
    required = set(features + ["connection_id", "prediction_time", "label_time", "target_delay", "upstream_arr_delay_minutes"])
    if required - set(frame):
        raise ValueError(f"Frozen final feature fields missing: {sorted(required - set(frame))}")
    fingerprints = {"schema_version": 1, "typed_features": training.resolve_features(locked),
                    "train": _frame_fingerprint(train, features), "test": _frame_fingerprint(test, features)}
    _write_json(output / "final_feature_fingerprints.json", fingerprints)
    audit["train_features_sha256"], audit["validation_features_sha256"] = _feature_hash(train, features), _feature_hash(test, features)
    pd.DataFrame([audit]).to_csv(output / "split_audit.csv", index=False)
    train[["connection_id", "prediction_time", "label_time", "target_delay"]].to_csv(
        output / "training_ids_targets.csv.gz", index=False, compression=training.GZIP)
    ids = test[["connection_id", "prediction_time", "label_time", "target_delay"]].copy()
    ids.insert(0, "fold", FINAL_FOLD)
    ids.to_csv(output / "evaluation_ids.csv.gz", index=False, compression=training.GZIP)
    (output / "models").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    validation_columns = list(dict.fromkeys(features + [name for name in training.PREDICTION_METADATA if name in test]
                                           + ["upstream_arr_delay_minutes"]
                                           + (["aircraft_family"] if "aircraft_family" in test else [])))
    _, timing = training._run_fold(FINAL_FOLD, train[features + ["target_delay"]].copy(), test[validation_columns].copy(),
                                   locked, str(output), audit)
    metrics = _postprocess_predictions(output, locked, frozen["primary_model_name"], reproduction=mode == "reproduce_final")
    result = {"mode": mode, "development_only": False, "freeze_manifest_path": str(freeze_path),
              "access_ledger_path": str(ledger_path), "claim_type": "formal" if mode == "final" else "reproduction",
              "freeze_manifest_sha256": freeze_hash, "freeze_payload_sha256": frozen["payload_sha256"],
              "primary_model_name": frozen["primary_model_name"], "model_name": locked["model_name"],
              "models": training.model_names(locked), "features": features, "config": locked,
              "actual_model_params": training._regressor(locked).get_params(deep=False),
              "protocol": frozen["protocol"], "evaluation_rows": len(test), "folds": [audit],
              "evaluation_ids_sha256": training.hash_ids(ids.fold.astype(str) + ":" + ids.connection_id.astype(str)),
              "split_sha256": canonical_sha256(audit), "timing": timing, "verified_inputs": verified,
              "feature_fingerprints": fingerprints,
              "utc_boundary_audit": _boundary_counts(_path(locked["connection_path"])),
              "primary_metric": metrics.loc[metrics.model.eq(frozen["primary_model_name"]) & metrics.fold.eq("overall") & metrics["group"].eq("all")].to_dict("records")}
    _write_json(output / "model_manifest.json", {
        **result, **training.resolve_features(locked), "model_type": locked["model_type"],
        "model_params": result["actual_model_params"], "include_baselines": True,
        "threads": locked["threads"], "fit_threads": locked["threads"],
        "prediction_threads": training.prediction_threads(locked), "fold_workers": 1, "fold_timings": [timing],
        "linear_tolerance": locked["linear_tolerance"],
        "effective_preprocessing": {"min_frequency": locked["preprocessing"]["min_frequency"],
                                    "drop": "first", "sparse": True,
                                    "numeric_imputation": "median; training-only fitted; missing indicators",
                                    "numeric_scaling": "StandardScaler(with_mean=False)",
                                    "categorical_imputation": "constant __MISSING__"}})
    result["artifact_hashes"] = {str(path.relative_to(output)): sha256_file(path) for path in sorted(output.rglob("*"))
                                 if path.is_file() and path.name not in {"manifest.json", "run.log", "resources.csv", "final_result.json"}}
    _write_json(output / "final_result.json", result)
    return result

def _validate_output(output, freeze_path):
    output = Path(output).resolve()
    # Existing run_pipeline metadata is permitted, fitted/result files never are.
    protected = freeze_path.parent
    if output == protected or output.is_relative_to(protected):
        raise ValueError("Final output must not modify the source freeze archive")
    if any((output / name).exists() for name in ["final_result.json", "model_manifest.json", "models", "predictions"]):
        raise FileExistsError("Final model artifacts already exist; use a new run_id")
    output.mkdir(parents=True, exist_ok=True)
    return output


def run(config, run_dir):
    """Validate freeze, claim the single formal test, fit once and archive everything."""
    if config.get("mode") != "final":
        raise ValueError("Formal August assessment requires mode='final'")
    frozen, freeze_path, freeze_hash, verified = validate_freeze(config)
    output = _validate_output(run_dir, freeze_path)
    ledger_path = ROOT / "experiments/final_access/2026-08.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    access = {"test_period": FINAL_FOLD, "status": "claimed", "run_dir": str(output), "run_id": output.name,
              "freeze_manifest_sha256": freeze_hash, "data_id": frozen["config"]["data_id"],
              "frozen_at_utc": frozen["frozen_at_utc"], "access_started_at_utc": _now()}
    # x-mode is intentional: another freeze or run cannot silently reopen August.
    with ledger_path.open("x", encoding="utf-8") as stream:
        json.dump(access, stream, indent=2)
    try:
        result = _fit_frozen(frozen, freeze_path, freeze_hash, verified, output, ledger_path, mode="final")
        access.update(status="completed", ended_at_utc=_now(), final_result_sha256=sha256_file(output / "final_result.json"))
        _write_json(ledger_path, access)
        return result
    except BaseException as error:
        access.update(status="failed", ended_at_utc=_now(), error=repr(error))
        _write_json(ledger_path, access)
        (output / "final_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def verify_saved(run_dir, *, atol=1e-10):
    """Read saved predictions and metrics only; no feature read, fit or test reopen."""
    output = Path(run_dir).resolve()
    result = _json(output / "final_result.json")
    ledger = Path(result["access_ledger_path"])
    ledger_checked = ledger.is_file()
    if ledger_checked:
        access = _json(ledger)
        if access.get("status") != "completed" or access.get("final_result_sha256") != sha256_file(output / "final_result.json"):
            raise ValueError("Final result differs from the completed single-access ledger")
    for name, expected in result["artifact_hashes"].items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"Saved final artifact SHA-256 mismatch: {name}")
    predictions = _saved_predictions(output, result["models"])
    summaries = pd.concat([evaluation.summarize_predictions(frame) for frame in predictions.values()], ignore_index=True)
    actual = evaluation.metrics_from_summaries(summaries)
    saved = pd.read_csv(output / "metrics.csv", float_precision="round_trip")
    keys = ["model", "fold", "group", "value"]
    actual["value"], saved["value"] = actual.value.astype(str), saved.value.astype(str)
    actual, saved = actual.sort_values(keys).reset_index(drop=True), saved.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, saved, check_dtype=False, check_exact=False, atol=atol, rtol=0)
    reference = predictions[result["models"][0]]
    if len(reference) != result["evaluation_rows"] or training.hash_targets(reference) != result["folds"][0]["validation_target_sha256"]:
        raise ValueError("Saved final target/ID cohort differs from the final audit")
    return {"verified": True, "evaluation_rows": len(reference), "models": result["models"], "metrics_recomputed": True,
            "artifact_hashes_checked": len(result["artifact_hashes"]), "access_ledger_checked": ledger_checked,
            "refitted": False, "raw_test_data_read": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-run", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_saved(args.verify_run), indent=2))
