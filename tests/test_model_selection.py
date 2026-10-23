"""Tiny saved development archives; no fitting and no real test-data access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts import select_models as selection
from scripts import train_models as training


def write_run(root, name, group="G0", kind="linear", errors=(10., 10., 10.), params=None,
              baselines=(11., 8., 12.)):
    path = root / name
    (path / "predictions").mkdir(parents=True)
    (path / "data").mkdir()
    config = {"mode": "development", "feature_group": group, "model_name": name.lower(), "model_type": kind,
              "model_params": params or {}, "include_baselines": True, "threads": 1,
              "preprocessing": {"drop": "first", "min_frequency": 20}, "linear_tolerance": 1e-6,
              "connection_path": str(root / "connections.csv"), "data_id": "SYNTHETIC",
              "stage": "synthetic", "parent_run_id": "ignored", "change": "unit test"}
    spec = training.resolve_features(config)
    config.update(spec)
    (root / "connections.csv").write_text("synthetic file hashed only; not a real August dataset")
    folds, parts, metric_rows = [], [], []
    names = [config["model_name"], *selection.BASELINES]
    for index, (fold, start, stop) in enumerate(training.DEVELOPMENT_FOLDS):
        n = index + 2
        rows = pd.DataFrame({"fold": fold, "connection_id": [f"{fold}-{i}" for i in range(n)],
                             "prediction_time": pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=12),
                             "target_delay": 0.})
        rows["label_time"] = rows.prediction_time + pd.Timedelta(minutes=10)
        audit = {"fold": fold, "validation_start": pd.Timestamp(start, tz="UTC").isoformat(),
                 "validation_end_exclusive": pd.Timestamp(stop, tz="UTC").isoformat(), "validation_rows": n,
                 "validation_id_sha256": training.hash_ids(rows.connection_id), "validation_target_sha256": training.hash_targets(rows),
                 "train_target_sha256": "same-mature-training-labels"}
        folds.append(audit)
        parts.append(rows)
        for model_index, model in enumerate(names):
            out = rows.copy()
            out["model"] = model
            out["predicted_delay"] = errors[index] if model_index == 0 else baselines[model_index - 1]
            out.to_csv(path / "predictions" / f"{fold}_{model}.csv.gz", index=False)
            metric_rows.append({"model": model, "fold": fold, "group": "all", "n": n,
                                "mae": float(out.predicted_delay.mean()), "rmse": float(np.sqrt(np.square(out.predicted_delay).mean()))})
    for model_index, model in enumerate(names):
        month_rows = [row for row in metric_rows if row["model"] == model]
        total = sum(row["n"] for row in month_rows)
        metric_rows.append({"model": model, "fold": "overall", "group": "all", "n": total,
                            "mae": sum(row["n"] * row["mae"] for row in month_rows) / total,
                            "rmse": np.sqrt(sum(row["n"] * row["rmse"] ** 2 for row in month_rows) / total)})
    frame = pd.concat(parts, ignore_index=True)
    for column in spec["categorical_features"]:
        frame[column] = "category"
    for column in spec["numeric_features"]:
        frame[column] = 1.
    frame.to_pickle(path / "data/features.pkl")
    manifest = {"development_only": True, **spec, "model_name": config["model_name"], "model_type": kind,
                "model_params": training._regressor(config).get_params(deep=False), "models": names,
                "folds": folds, "evaluation_rows": len(frame), "threads": 1, "linear_tolerance": 1e-6,
                "effective_preprocessing": config["preprocessing"],
                "evaluation_ids_sha256": training.hash_ids(frame.fold + ":" + frame.connection_id),
                "split_sha256": hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest()}
    for filename, value in [("manifest.json", {"status": "completed"}), ("model_manifest.json", manifest), ("config.json", config)]:
        (path / filename).write_text(json.dumps(value), encoding="utf-8")
    pd.DataFrame(metric_rows).to_csv(path / "metrics.csv", index=False)
    return path


def algorithm_grid(root, linear):
    result = []
    for depth in [8, 16]:
        for leaf in [50, 200]:
            result.append(write_run(root, f"DT{depth}_{leaf}", kind="decision_tree", errors=(9.6, 9.6, 10.3),
                                    params={"max_depth": depth, "min_samples_leaf": leaf, "random_state": 42}))
    for leaf in [50, 200]:
        result.append(write_run(root, f"RF{leaf}", kind="random_forest", errors=(9.7, 9.7, 9.7),
                                params={"n_estimators": 200, "max_depth": 16, "max_features": .5,
                                        "min_samples_leaf": leaf, "random_state": 42}))
    return result


class ModelSelection(unittest.TestCase):
    def test_complete_feature_chain_tie_and_excluded_branches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [write_run(root, group, group=group, errors=(score,) * 3)
                     for group, score in zip(selection.STAGES, [10., 9., 9. - 5e-10, 9.5, 9.4, 9.3])]
            variant = write_run(root, "VARIANT", group="G2_VARIANT", errors=(1.,) * 3)
            noaa = write_run(root, "NOAA", group="G4B", errors=(1.,) * 3)
            config = json.loads((noaa / "config.json").read_text())
            config["noaa_backfill"] = True
            (noaa / "config.json").write_text(json.dumps(config))
            failed = write_run(root, "R0011", group="G3")
            (failed / "manifest.json").write_text(json.dumps({"status": "failed"}))
            with patch.object(Pipeline, "fit", side_effect=AssertionError("must never train")):
                result = selection.select_features(paths + [variant, noaa, failed], root / "feature-choice")
            self.assertEqual(result["selected_run_id"], "G1")
            self.assertEqual(len(result["excluded_runs"]), 3)
            self.assertNotIn("parent_run_id", result["config"])
            self.assertEqual(len(pd.read_csv(root / "feature-choice/candidate_monthly_metrics.csv")), 24)
            self.assertFalse((root / "experiments/final_access").exists())
            with self.assertRaisesRegex(ValueError, "Incomplete mainline"):
                selection.select_features(paths[:-1], root / "missing")
            with self.assertRaisesRegex(ValueError, "already completed"):
                selection.select_features(paths, paths[0] / "choice")

    def test_model_grid_stability_sequential_promotion_and_baseline_primary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            linear = write_run(root, "LINEAR")
            algorithms = algorithm_grid(root, linear)
            with patch.object(Pipeline, "fit", side_effect=AssertionError("must never train")):
                result = selection.select_model(linear, algorithms, root / "model-choice")
            self.assertEqual(result["selected_run_id"], "RF200")
            self.assertEqual(result["primary_model_name"], "training_median")
            dt, rf = result["promotion_decisions"]
            self.assertEqual(dt["candidate_run_id"], "DT8_200")
            self.assertFalse(dt["promoted"])
            self.assertIn("deterioration", dt["rejection_reasons"])
            self.assertTrue(rf["promoted"])
            self.assertEqual(rf["reference_run_id"], "LINEAR")
            ci = pd.read_csv(root / "model-choice/paired_intervals.csv")
            self.assertEqual(len(ci), 16)
            self.assertTrue((root / "model-choice/baseline_monthly_metrics.csv").is_file())
            self.assertNotIn("mode", result["config"])
            with self.assertRaisesRegex(ValueError, "grid"):
                selection.select_model(linear, algorithms[:-1], root / "incomplete")

    def test_changed_labels_metrics_or_feature_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            linear = write_run(root, "LINEAR")
            algorithms = algorithm_grid(root, linear)
            prediction = algorithms[0] / "predictions/2026-05_dt8_50.csv.gz"
            original = prediction.read_bytes()
            rows = pd.read_csv(prediction)
            rows.loc[0, "target_delay"] = 1
            rows.to_csv(prediction, index=False)
            with self.assertRaisesRegex(ValueError, "label hash"):
                selection.select_model(linear, algorithms, root / "bad-label")
            prediction.write_bytes(original)
            features = algorithms[0] / "data/features.pkl"
            frame = pd.read_pickle(features)
            frame.loc[0, "scheduled_turnaround_minutes"] = 99
            frame.to_pickle(features)
            with self.assertRaisesRegex(ValueError, "feature values"):
                selection.select_model(linear, algorithms, root / "bad-feature")
            metrics = pd.read_csv(linear / "metrics.csv")
            metrics.loc[0, "mae"] = 100
            metrics.to_csv(linear / "metrics.csv", index=False)
            with self.assertRaisesRegex(ValueError, "reproduce"):
                selection._load_run(linear)

    def test_synthetic_explicit_freeze_schema_pins_and_dirty_source_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            linear = write_run(root, "LINEAR")
            result = selection.select_model(linear, algorithm_grid(root, linear), root / "choice")
            path = root / "frozen.json"
            # Only a synthetic fixture, mocked commit check and one tiny fake data
            # receipt. No actual project freeze or final ledger is created.
            with patch.object(selection, "_assert_committed", return_value="synthetic-commit"), \
                 patch.object(selection.final, "_required_data", return_value={root / "connections.csv"}), \
                 patch.object(selection.final, "load_configured_features", side_effect=AssertionError("no test loader")):
                frozen_receipt = selection.create_freeze(root / "choice/model_selection_decision.json", path)
                frozen = json.loads(path.read_text())
                self.assertEqual(frozen["primary_model_name"], "training_median")
                self.assertEqual(frozen_receipt["sha256"], selection.final.sha256_file(path))
                self.assertEqual(frozen["selection"]["actual_git_commit"], "synthetic-commit")
                self.assertEqual(set(frozen["config"]["model_params"]), {"n_estimators", "max_depth", "max_features", "min_samples_leaf", "random_state"})
                self.assertFalse((root / "experiments/final_access").exists())
                with self.assertRaises(FileExistsError):
                    selection.create_freeze(result, path)
            with patch.object(selection, "_git", return_value=b"not the source bytes"):
                with self.assertRaisesRegex(ValueError, "committed bytes"):
                    selection._assert_committed({Path(selection.__file__)})
            changed = dict(result, primary_model_name="tampered")
            with self.assertRaisesRegex(ValueError, "intact"):
                selection.create_freeze(changed, root / "tampered.json")


if __name__ == "__main__":
    unittest.main()
