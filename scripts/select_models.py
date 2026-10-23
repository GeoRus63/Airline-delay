"""Registered development selection and an explicit, separately invoked freeze.

This module reads saved May--July predictions only. It never loads raw test data,
fits an estimator, or creates the final-access ledger.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

try:
    from . import compare_experiments as comparison
    from . import final_evaluation as final
    from . import train_models as training
except ImportError:
    import compare_experiments as comparison
    import final_evaluation as final
    import train_models as training

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/model_selection_protocol.json"
REGISTRATION_COMMIT = "6357cd4d11882d01a70b64200356b9626004103d"
STAGES = ["G0", "G1", "G2", "G3", "G4A", "G4B"]
FOLDS = [item[0] for item in training.DEVELOPMENT_FOLDS]
BASELINES = final.BASELINES


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _git(*arguments):
    return subprocess.run(["git", *arguments], cwd=ROOT, check=True, capture_output=True).stdout


def _protocol():
    protocol = _json(PROTOCOL_PATH)
    registered = json.loads(_git("show", f"{REGISTRATION_COMMIT}:configs/model_selection_protocol.json"))
    if final.canonical_sha256(protocol) != final.canonical_sha256(registered):
        raise ValueError("Selection protocol differs from its registered Git version")
    return protocol


def _receipt(path):
    path = Path(path).resolve()
    try:
        name = path.relative_to(ROOT).as_posix()
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": final.sha256_file(path), "bytes": path.stat().st_size}


def _output(path):
    path = Path(path).resolve()
    for parent in [path, *path.parents]:
        manifest = parent / "manifest.json"
        if manifest.is_file() and _json(manifest).get("status") == "completed":
            raise ValueError("Selection must not modify an already completed run")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_run(path):
    path, manifest = comparison._completed_run(path)
    config = _json(path / "config.json")
    if manifest.get("development_only") is not True or config.get("mode") != "development":
        raise ValueError("Selection accepts completed development runs only")
    if [fold["fold"] for fold in manifest["folds"]] != FOLDS:
        raise ValueError("Selection requires all three fixed validation folds")
    comparison.validation_calendar(manifest)
    digest = hashlib.sha256(json.dumps(manifest["folds"], sort_keys=True).encode()).hexdigest()
    if digest != manifest.get("split_sha256"):
        raise ValueError("Split audit does not match split_sha256")
    spec = training.resolve_features(config)
    if any(spec[key] != manifest.get(key) for key in spec):
        raise ValueError("Archived configuration and manifest typed features differ")
    if manifest.get("model_type") != config.get("model_type", "linear"):
        raise ValueError("Archived model type differs from configuration")
    if training._regressor(config).get_params(deep=False) != manifest.get("model_params"):
        raise ValueError("Archived fitted estimator parameters differ from configuration")
    expected_preprocessing = {"drop": "first", "min_frequency": config.get("preprocessing", {}).get("min_frequency", 20)}
    if any(manifest.get("effective_preprocessing", {}).get(key) != value for key, value in expected_preprocessing.items()):
        raise ValueError("Archived preprocessing differs from configuration")
    model = manifest["model_name"]
    if model != config.get("model_name", "g0_linear"):
        raise ValueError("Archived model name differs from configuration")
    predictions = comparison._read_predictions(path, manifest, model)
    _check_prediction_hashes(predictions, manifest)
    metrics = pd.read_csv(path / "metrics.csv", float_precision="round_trip")
    run = {"path": path, "manifest": manifest, "config": config, "predictions": predictions,
           "model": model, "metrics": metrics, "spec": spec}
    run["scores"] = _scores(run, predictions, model)
    run["receipts"] = [_receipt(path / name) for name in ["manifest.json", "config.json", "model_manifest.json", "metrics.csv"]]
    run["receipts"] += [_receipt(path / "predictions" / f"{fold}_{model}.csv.gz") for fold in FOLDS]
    return run


def _check_prediction_hashes(predictions, manifest):
    actual = training.hash_ids(predictions.fold.astype(str) + ":" + predictions.connection_id.astype(str))
    if actual != manifest.get("evaluation_ids_sha256"):
        raise ValueError("Full saved predictions differ from evaluation_ids_sha256")
    for fold in manifest["folds"]:
        rows = predictions.loc[predictions.fold.eq(fold["fold"])]
        if training.hash_ids(rows.connection_id) != fold.get("validation_id_sha256"):
            raise ValueError("Fold evaluation ID hash differs")
        if training.hash_targets(rows) != fold.get("validation_target_sha256"):
            raise ValueError("Fold validation label hash differs")
        start, stop = pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end_exclusive"])
        if not (rows.prediction_time.ge(start) & rows.prediction_time.lt(stop)).all():
            raise ValueError("Prediction timestamp falls outside its fixed validation fold")
    if predictions.predicted_delay.lt(0).any():
        raise ValueError("Saved predictions must already use the fixed nonnegative clipping rule")


def _scores(run, predictions, model):
    result = []
    for fold in [*FOLDS, "overall"]:
        part = predictions if fold == "overall" else predictions.loc[predictions.fold.eq(fold)]
        errors = part.predicted_delay.to_numpy(float) - part.target_delay.to_numpy(float)
        row = {"run_id": run["path"].name, "model": model, "fold": fold, "n": len(part),
               "mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.square(errors).mean()))}
        saved = run["metrics"].loc[run["metrics"].model.eq(model) & run["metrics"].fold.eq(fold)
                                     & run["metrics"].group.eq("all")]
        if len(saved) != 1 or int(saved.iloc[0].n) != len(part):
            raise ValueError("Archived metrics do not contain exactly one complete all-sample score")
        if not np.allclose(saved.iloc[0][["mae", "rmse"]].to_numpy(float), [row["mae"], row["rmse"]], rtol=0, atol=1e-9):
            raise ValueError("Archived metric does not reproduce from complete saved predictions")
        result.append(row)
    return result


def _same_cohort(reference, candidate):
    for key in ["evaluation_ids_sha256", "split_sha256"]:
        if reference["manifest"][key] != candidate["manifest"][key]:
            raise ValueError(f"Candidate {key} differs")
    comparison.align_predictions(reference["predictions"], candidate["predictions"], align_rows=False)


def _pooled(run):
    return run["scores"][-1]["mae"]


def _scientific_config(run):
    # Cache/lineage/reporting switches are execution metadata, never final science.
    excluded = final.RUNTIME_KEYS | {"fold_workers", "preparation_evidence_dir", "compare_parent",
                                    "feature_source_run", "feature_source_run_id", "selected_feature_run_id",
                                    "feature_archive_path", "test_target_sealed"}
    prefixes = ("runtime", "cache", "feature_cache", "compare_", "common_weather", "preparation_evidence", "parent_")
    config = {key: value for key, value in run["config"].items()
              if key not in excluded and not key.startswith(prefixes)}
    config.update(run["spec"])
    manifest = run["manifest"]
    config.update(model_name=run["model"], model_type=manifest["model_type"], include_baselines=True,
                  threads=int(manifest.get("threads", run["config"].get("threads", 8))),
                  linear_tolerance=float(manifest.get("linear_tolerance", 1e-6)),
                  preprocessing={key: manifest["effective_preprocessing"][key] for key in ["drop", "min_frequency"]})
    allowed = {"tol", "fit_intercept"} if manifest["model_type"] == "linear" else {"max_depth", "min_samples_leaf", "random_state"}
    if manifest["model_type"] == "random_forest":
        allowed |= {"n_estimators", "max_features"}
    config["model_params"] = {key: value for key, value in manifest["model_params"].items() if key in allowed}
    return config


def _candidate_row(run):
    return {"run_id": run["path"].name, "run_dir": str(run["path"]), "model": run["model"],
            "feature_group": run["spec"]["feature_group"], "predictors": len(run["spec"]["features"]),
            "model_type": run["manifest"]["model_type"], "model_params": json.dumps(run["manifest"]["model_params"], sort_keys=True),
            "pooled_mae": _pooled(run), "pooled_rmse": run["scores"][-1]["rmse"]}


def _save_selection(output, filename, result, runs, extra_tables=None):
    if (output / filename).exists():
        raise FileExistsError(f"Refusing to overwrite selection evidence: {output / filename}")
    pd.DataFrame([_candidate_row(run) for run in runs]).to_csv(output / "candidates.csv", index=False)
    pd.DataFrame([row for run in runs for row in run["scores"]]).to_csv(output / "candidate_monthly_metrics.csv", index=False)
    for name, table in (extra_tables or {}).items():
        pd.DataFrame(table).to_csv(output / name, index=False)
    executed_sources = {Path(__file__).resolve()}
    for run in runs:
        executed_sources.update(final._required_sources(run["spec"]["feature_group"]))
    result.update(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol=_protocol(), registration_commit=REGISTRATION_COMMIT,
                  protocol_file=_receipt(PROTOCOL_PATH), code_files=[_receipt(path) for path in sorted(executed_sources)],
                  candidate_files=[receipt for run in runs for receipt in run["receipts"]],
                  output_files=[_receipt(path) for path in sorted(output.glob("*.csv"))])
    result["selection_sha256"] = final.canonical_sha256(result)
    _write(output / filename, result)
    return result


def select_features(mainline_run_dirs, output_dir):
    """Select one mainline linear feature set after all six stages have completed.

    Passed variant/NOAA branches and failed runs are recorded as exclusions; they
    never substitute for a missing valid mainline stage.
    """
    _protocol()
    output = _output(output_dir)
    runs, excluded, stages = [], [], {}
    for directory in mainline_run_dirs:
        path = Path(directory).resolve()
        config = _json(path / "config.json")
        group = str(config.get("feature_group", "")).upper()
        status = _json(path / "manifest.json").get("status")
        reason = ("variant is supplementary" if "VARIANT" in group else
                  "NOAA retrospective backfill is supplementary" if config.get("noaa_backfill", False) else
                  f"run status is {status}" if status != "completed" else None)
        if reason:
            excluded.append({"run_id": path.name, "reason": reason})
            continue
        group = "G2" if group == "G2_FAMILY" else group
        if group not in STAGES or group in stages:
            raise ValueError("Require exactly one valid run for each G0/G1/G2 family/G3/G4A/G4B mainline stage")
        run = _load_run(path)
        if run["manifest"]["model_type"] != "linear" or "aircraft_variant" in run["spec"]["features"]:
            raise ValueError("Feature selection must hold OLS fixed and use the family mainline")
        expected = training.resolve_features({"feature_group": group})
        if set(run["spec"]["features"]) != set(expected["features"]):
            raise ValueError("Feature mainline must use each complete registered cumulative information group")
        stages[group] = run
        runs.append(run)
    if set(stages) != set(STAGES):
        raise ValueError(f"Incomplete mainline; missing stages: {sorted(set(STAGES) - set(stages))}")
    runs = [stages[stage] for stage in STAGES]
    reference = runs[0]
    for run in runs[1:]:
        _same_cohort(reference, run)
        for key in ["model_params", "effective_preprocessing"]:
            if run["manifest"][key] != reference["manifest"][key]:
                raise ValueError("Feature comparison must hold linear/preprocessing settings fixed")
        for key in ["connection_path", "data_id", "data_manifest_path"]:
            if run["config"].get(key) != reference["config"].get(key):
                raise ValueError("Feature comparison data source differs")
    minimum = min(_pooled(run) for run in runs)
    tied = [run for run in runs if _pooled(run) - minimum <= 1e-9]
    chosen = min(tied, key=lambda run: (len(run["spec"]["features"]), STAGES.index("G2" if run["spec"]["feature_group"] == "G2_FAMILY" else run["spec"]["feature_group"])))
    result = {"selection_type": "features", "selected_run_id": chosen["path"].name,
              "selected_run_dir": str(chosen["path"]), "config": _scientific_config(chosen),
              "reason": "Minimum sample-weighted pooled May-July MAE; ties within 1e-9 prefer fewer predictors then earlier mainline stage",
              "selected_pooled_mae": _pooled(chosen), "excluded_runs": excluded,
              "excluded_primary_branches": ["G2_VARIANT", "NOAA retrospective backfill"],
              "evaluation_ids_sha256": chosen["manifest"]["evaluation_ids_sha256"], "split_sha256": chosen["manifest"]["split_sha256"]}
    return _save_selection(output, "feature_selection.json", result, runs, {"excluded_runs.csv": excluded})


def _feature_fingerprint(run):
    path = run["path"] / "data/features.pkl"
    if not path.is_file():
        raise ValueError("Fixed-feature algorithm selection requires archived data/features.pkl")
    frame = pd.read_pickle(path)
    columns = list(dict.fromkeys(["connection_id", "prediction_time", "label_time", "target_delay", *run["spec"]["features"]]))
    if set(columns) - set(frame) or not frame.connection_id.is_unique:
        raise ValueError("Archived feature cohort is incomplete or has duplicate IDs")
    prediction = pd.to_datetime(frame.prediction_time, utc=True, errors="raise")
    if not (prediction.ge(pd.Timestamp("2026-04-01", tz="UTC")) & prediction.lt(pd.Timestamp("2026-08-01", tz="UTC"))).all():
        raise ValueError("Feature archive must contain development rows only")
    # Include training rows and their labels, not only the common validation IDs.
    selected = frame[columns].sort_values("connection_id", kind="stable")
    digest = hashlib.sha256(json.dumps(columns).encode())
    digest.update(np.asarray(pd.util.hash_pandas_object(selected, index=False), dtype="<u8").tobytes())
    run["receipts"].append(_receipt(path))
    return digest.hexdigest()


def _family_best(runs):
    return min(runs, key=lambda run: (_pooled(run), run["manifest"]["model_params"].get("max_depth", 0),
                                      -run["manifest"]["model_params"].get("min_samples_leaf", 0)))


def select_model(feature_run, algorithm_runs, output_dir):
    """Compare the complete registered DT/RF grid on one fixed feature source."""
    _protocol()
    output = _output(output_dir)
    linear = _load_run(feature_run)
    if linear["manifest"]["model_type"] != "linear":
        raise ValueError("The feature source must be the selected mainline linear run")
    runs = [linear] + [_load_run(path) for path in algorithm_runs]
    if len({run["path"] for run in runs}) != len(runs):
        raise ValueError("Duplicate candidate run")
    fingerprint = _feature_fingerprint(linear)
    families = {"decision_tree": [], "random_forest": []}
    source_config = _scientific_config(linear)
    model_settings = {"model_name", "model_type", "model_params", "threads", "linear_tolerance", "include_baselines"}
    for run in runs[1:]:
        _same_cohort(linear, run)
        if run["spec"] != linear["spec"] or _feature_fingerprint(run) != fingerprint:
            raise ValueError("Algorithm comparison must use the identical typed features and complete development feature values")
        other_config = _scientific_config(run)
        if {k: v for k, v in source_config.items() if k not in model_settings} != {k: v for k, v in other_config.items() if k not in model_settings}:
            raise ValueError("Algorithm comparison scientific feature/data/preprocessing configuration differs")
        kind = run["manifest"]["model_type"]
        if kind not in families:
            raise ValueError("Algorithm grid must contain only DT and RF candidates after the linear reference")
        families[kind].append(run)
    expected = {"decision_tree": {(8, 50), (8, 200), (16, 50), (16, 200)}, "random_forest": {(16, 50), (16, 200)}}
    for kind, candidates in families.items():
        combinations = [(run["manifest"]["model_params"]["max_depth"], run["manifest"]["model_params"]["min_samples_leaf"]) for run in candidates]
        if len(combinations) != len(expected[kind]) or set(combinations) != expected[kind]:
            raise ValueError(f"Incomplete or duplicate registered {kind} grid")
        for run in candidates:
            params = run["manifest"]["model_params"]
            if params["random_state"] != 42 or kind == "random_forest" and (params["n_estimators"] != 200 or params["max_features"] != .5):
                raise ValueError("Candidate does not use the registered seed/forest specification")
    retained, decisions, interval_tables = linear, [], []
    calendar = comparison.validation_calendar(linear["manifest"])
    for family in ["decision_tree", "random_forest"]:
        challenger = _family_best(families[family])
        monthly_delta = [challenger["scores"][i]["mae"] - retained["scores"][i]["mae"] for i in range(3)]
        gain, required = _pooled(retained) - _pooled(challenger), max(.10, .005 * _pooled(retained))
        improved, worst = sum(value < 0 for value in monthly_delta), max(monthly_delta)
        reasons = []
        # Guard only binary subtraction roundoff at decimal protocol boundaries.
        if gain < required and not np.isclose(gain, required, rtol=0, atol=1e-12):
            reasons.append("pooled improvement below practical threshold")
        if improved < 2:
            reasons.append("fewer than two validation months improve")
        if worst > .10 and not np.isclose(worst, .10, rtol=0, atol=1e-12):
            reasons.append("worst monthly deterioration exceeds 0.10 minute")
        decisions.append({"family": family, "reference_run_id": retained["path"].name,
                          "candidate_run_id": challenger["path"].name, "pooled_improvement": gain,
                          "required_improvement": required, "improved_months": improved,
                          "worst_monthly_deterioration": worst, "promoted": not reasons,
                          "rejection_reasons": "; ".join(reasons), "monthly_candidate_minus_reference": monthly_delta})
        daily = comparison.daily_statistics(retained["predictions"], challenger["predictions"], calendar)
        intervals = comparison.comparison_intervals(daily, bootstrap_repeats=1000, block_days=7, seed=42)
        intervals.insert(0, "candidate_run_id", challenger["path"].name)
        intervals.insert(0, "reference_run_id", retained["path"].name)
        interval_tables.extend(intervals.to_dict("records"))
        if not reasons:
            retained = challenger
    baseline_scores, baseline_receipts = [], []
    for name in BASELINES:
        if name not in linear["manifest"]["models"]:
            raise ValueError("The feature source must archive all three baseline predictions")
        predictions = comparison._read_predictions(linear["path"], linear["manifest"], name)
        comparison.align_predictions(linear["predictions"], predictions, align_rows=False)
        baseline_scores.extend(_scores(linear, predictions, name))
        baseline_receipts.extend(_receipt(linear["path"] / "predictions" / f"{fold}_{name}.csv.gz") for fold in FOLDS)
    primary_candidates = [{"model": retained["model"], "mae": _pooled(retained), "run_id": retained["path"].name}]
    primary_candidates += [{"model": name, "mae": next(row["mae"] for row in baseline_scores if row["model"] == name and row["fold"] == "overall"), "run_id": linear["path"].name} for name in BASELINES]
    primary = min(primary_candidates, key=lambda row: row["mae"])
    linear["receipts"].extend(baseline_receipts)
    result = {"selection_type": "model", "selected_run_id": retained["path"].name, "selected_run_dir": str(retained["path"]),
              "feature_source_run_id": linear["path"].name, "feature_source_run_dir": str(linear["path"]),
              "development_feature_sha256": fingerprint, "config": _scientific_config(retained),
              "primary_model_name": primary["model"], "primary_development_mae": primary["mae"],
              "primary_candidates": primary_candidates, "learned_pooled_mae": _pooled(retained),
              "reason": "Registered family minima and sequential practical/monthly-stability promotion, followed by minimum pooled MAE against all three baselines",
              "primary_exact_tie_rule": "stable order: retained learned, training_mean, training_median, carry_forward; no RMSE override",
              "threshold_roundoff_tolerance": 1e-12,
              "promotion_decisions": decisions, "evaluation_ids_sha256": retained["manifest"]["evaluation_ids_sha256"],
              "split_sha256": retained["manifest"]["split_sha256"]}
    return _save_selection(output, "model_selection_decision.json", result, runs,
                           {"promotion_decisions.csv": decisions, "baseline_monthly_metrics.csv": baseline_scores,
                            "paired_intervals.csv": interval_tables})


def _assert_committed(paths):
    commit = _git("rev-parse", "HEAD").decode().strip()
    for path in paths:
        path = Path(path).resolve()
        try:
            relative = path.relative_to(ROOT).as_posix()
            committed = _git("show", f"{commit}:{relative}")
        except (ValueError, subprocess.CalledProcessError) as error:
            raise ValueError(f"Freeze source is not committed at HEAD: {path}") from error
        if hashlib.sha256(committed).hexdigest() != final.sha256_file(path):
            raise ValueError(f"Freeze source differs from its committed bytes: {relative}")
    return commit


def create_freeze(selection, path):
    """Explicit opt-in only: seal a completed model selection, never assess August.

    Returns file-byte and canonical-payload hashes. This is deliberately not called
    by either selection function. The caller commits all source changes first.
    """
    selection_path = Path(selection).resolve() if isinstance(selection, (str, Path)) else None
    selection = _json(selection_path) if selection_path else dict(selection)
    payload = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if selection.get("selection_type") != "model" or selection.get("selection_sha256") != final.canonical_sha256(payload):
        raise ValueError("Freeze requires an intact model-selection decision")
    _protocol()
    for receipt in [*selection["candidate_files"], *selection["output_files"], *selection["code_files"], selection["protocol_file"]]:
        actual = Path(receipt["path"])
        actual = actual if actual.is_absolute() else ROOT / actual
        if final.sha256_file(actual) != receipt["sha256"]:
            raise ValueError(f"Selection evidence changed before freeze: {actual}")
    source = _load_run(selection["selected_run_dir"])
    if source["path"].name != selection["selected_run_id"] or _scientific_config(source) != selection["config"]:
        raise ValueError("Selection source/config no longer matches its completed archive")
    config = selection["config"]
    sources = final._required_sources(config["feature_group"]) | {Path(__file__).resolve(), PROTOCOL_PATH.resolve()}
    for name in ["scripts/run_pipeline.py", "requirements.txt"]:
        if (ROOT / name).is_file():
            sources.add((ROOT / name).resolve())
    # All project computation scripts must be committed; do not freeze a dirty
    # auxiliary preparation/feature dependency omitted from an import list.
    commit = _assert_committed(sources | set((ROOT / "scripts").glob("*.py")))
    target = Path(path).resolve()
    _output(target.parent)
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing frozen selection")
    data = final._required_data(config)
    if config.get("data_manifest_path"):
        manifest_path = Path(config["data_manifest_path"])
        data.add(manifest_path.resolve() if manifest_path.is_absolute() else (ROOT / manifest_path).resolve())
    choice_files = selection["output_files"] + ([ _receipt(selection_path)] if selection_path else [])
    frozen = {"schema_version": 1, "status": "frozen", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection": {"selected_run_id": selection["selected_run_id"], "reason": selection["reason"],
                            "selection_sha256": selection["selection_sha256"], "actual_git_commit": commit,
                            "choice_files": choice_files, "feature_source_run_id": selection["feature_source_run_id"]},
              "primary_model_name": selection["primary_model_name"], "config": config,
              "protocol": final.DEFAULT_PROTOCOL, "source_files": [_receipt(item) for item in sorted(sources)],
              "data_files": [_receipt(item) for item in sorted(data)]}
    frozen["payload_sha256"] = final.canonical_sha256(frozen)
    # Validate the exact schema/pins before publishing the requested freeze path.
    # A temporary file in the same output directory is never a final-access claim.
    draft = target.with_name(target.name + ".draft")
    if draft.exists():
        raise FileExistsError(f"Freeze draft already exists: {draft}")
    _write(draft, frozen)
    final.validate_freeze({"mode": "final", "freeze_manifest_path": str(draft),
                           "freeze_manifest_sha256": final.sha256_file(draft)})
    draft.replace(target)
    return {"path": str(target), "payload_sha256": frozen["payload_sha256"],
            "sha256": final.sha256_file(target), "actual_git_commit": commit}
