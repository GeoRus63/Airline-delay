"""Synthetic P6 checks: freeze validation, strict maturity and prediction-only replay."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts import final_evaluation as final
from scripts.build_features import G0_CATEGORICAL, G0_FEATURES, G0_NUMERIC
from tests.test_parallel_training import development_frame


def fixture(root):
    frame = development_frame()
    extra = pd.concat([frame.iloc[[0]].copy()] * 5, ignore_index=True)
    extra["connection_id"] = ["immature-at-cutoff", "august-left", "august-mid", "august-right", "september-excluded"]
    extra["prediction_time"] = pd.to_datetime(["2026-07-31T23:59Z", "2026-08-01T00:00Z", "2026-08-15T12:00Z",
                                               "2026-08-31T23:59Z", "2026-09-01T00:00Z"])
    extra["label_time"] = extra.prediction_time + pd.Timedelta(minutes=10)
    extra.loc[0, "label_time"] = pd.Timestamp("2026-08-01T00:00Z")
    extra["target_delay"] = [999999, 5, 80, 0, "DO_NOT_PARSE_OUTSIDE_TEST"]
    frame = pd.concat([frame, extra], ignore_index=True)
    metadata = pd.DataFrame({"upstream_actual_arr_utc": frame.prediction_time,
                             "downstream_flight_date": frame.prediction_time.dt.strftime("%Y-%m-%d")})
    metadata.loc[metadata.index[-1], "downstream_flight_date"] = "2026-08-31"
    connection = root / "connections.csv"
    metadata.to_csv(connection, index=False)
    config = {"feature_group": "G0", "features": G0_FEATURES, "categorical_features": G0_CATEGORICAL,
              "numeric_features": G0_NUMERIC, "model_name": "frozen_linear", "model_type": "linear",
              "model_params": {"tol": 1e-6, "fit_intercept": True}, "preprocessing": {"drop": "first", "min_frequency": 2},
              "threads": 1, "linear_tolerance": 1e-6, "include_baselines": True,
              "connection_path": str(connection), "data_id": "SYNTHETIC"}
    frozen = {"schema_version": 1, "status": "frozen", "frozen_at_utc": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
              "selection": {"selected_run_id": "SYNTHETIC_DEV", "reason": "Synthetic unit test only; no real August data"},
              "primary_model_name": "training_median", "config": config, "protocol": copy.deepcopy(final.DEFAULT_PROTOCOL),
              "source_files": [{"path": str(path), "sha256": final.sha256_file(path)} for path in sorted(final._required_sources("G0"))],
              "data_files": [{"path": str(connection), "sha256": final.sha256_file(connection)}]}
    frozen["payload_sha256"] = final.canonical_sha256(frozen)
    directory = root / "freeze"
    directory.mkdir()
    path = directory / "model_selection.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    runtime = {"mode": "final", "freeze_manifest_path": str(path), "freeze_manifest_sha256": final.sha256_file(path)}
    return frame, frozen, runtime, path


class FinalEvaluation(unittest.TestCase):
    def test_freeze_pins_are_checked_before_reading_final_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, frozen, config, path = fixture(root)
            with patch.object(final, "ROOT", root), patch.object(final, "load_configured_features", side_effect=AssertionError("must not read")):
                with self.assertRaisesRegex(ValueError, "mode='final'"):
                    final.run({**config, "mode": "development"}, root / "bad-mode")
                with self.assertRaisesRegex(ValueError, "scientific setting"):
                    final.run({**config, "threads": 8}, root / "bad-config")
                with self.assertRaisesRegex(ValueError, "file SHA"):
                    final.run({**config, "freeze_manifest_sha256": "changed"}, root / "bad-file")
                frozen["protocol"]["severity_threshold_minutes"] = 30
                path.write_text(json.dumps(frozen))
                with self.assertRaisesRegex(ValueError, "payload SHA"):
                    final.run({**config, "freeze_manifest_sha256": final.sha256_file(path)}, root / "bad-payload")
                self.assertFalse((root / "experiments/final_access/2026-08.json").exists())

    def test_single_frozen_fit_strict_cutoffs_baseline_primary_and_verify_without_refit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame, frozen, config, _ = fixture(root)
            frame["aircraft_family"] = "SYNTHETIC_FAMILY"
            output = root / "RFINAL"
            with patch.object(final, "ROOT", root), patch.object(final, "load_configured_features", return_value=frame) as loader:
                result = final.run(config, output)
                loader.assert_called_once_with(frozen["config"], development_only=False)
                self.assertEqual(result["evaluation_rows"], 3)
                self.assertEqual(result["primary_model_name"], "training_median")
                self.assertEqual(result["folds"][0]["excluded_immature_training_labels"], 1)
                self.assertLess(pd.Timestamp(result["folds"][0]["max_train_label_time"]), pd.Timestamp("2026-08-01", tz="UTC"))
                ids = pd.read_csv(output / "evaluation_ids.csv.gz")
                self.assertEqual(set(ids.connection_id), {"august-left", "august-mid", "august-right"})
                self.assertEqual(result["utc_boundary_audit"]["local_august_prediction_at_or_after_sep1"], 1)
                self.assertEqual(len(list((output / "models").glob("*.joblib"))), 4)
                self.assertEqual(len(list((output / "residuals").glob("*.csv.gz"))), 4)
                predictions = pd.read_csv(output / "predictions/2026-08_frozen_linear.csv.gz")
                self.assertTrue(predictions.aircraft_family.eq("SYNTHETIC_FAMILY").all())
                comparisons = pd.read_csv(output / "final_comparisons.csv")
                self.assertEqual(set(comparisons.reference_model), set(final.BASELINES))
                self.assertTrue((comparisons.loc[comparisons.reference_model.eq("training_median"), "difference_candidate_minus_reference"] == 0).all())
                ledger = json.loads((root / "experiments/final_access/2026-08.json").read_text())
                self.assertEqual(ledger["status"], "completed")
                self.assertGreater(pd.Timestamp(ledger["access_started_at_utc"]), pd.Timestamp(ledger["frozen_at_utc"]))
                with self.assertRaises(FileExistsError):
                    final.run(config, root / "SECOND_FORMAL_ACCESS")
            with patch.object(final, "load_configured_features", side_effect=AssertionError("must not reopen test")), \
                 patch.object(Pipeline, "fit", side_effect=AssertionError("must not refit")):
                verified = final.verify_saved(output)
                self.assertTrue(verified["verified"])
                self.assertFalse(verified["refitted"])
            prediction_path = output / "predictions/2026-08_training_mean.csv.gz"
            with prediction_path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                final.verify_saved(output)

    def test_changed_data_hash_is_rejected_before_august_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, frozen, config, _ = fixture(root)
            connection = Path(frozen["config"]["connection_path"])
            connection.write_text(connection.read_text() + "\n")
            with patch.object(final, "ROOT", root), patch.object(final, "load_configured_features", side_effect=AssertionError("must not read")):
                with self.assertRaisesRegex(ValueError, "data_files SHA"):
                    final.run(config, root / "changed-data")
                self.assertFalse((root / "experiments/final_access/2026-08.json").exists())

    def test_weather_manifest_and_all_dependencies_are_pinned_before_august(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, frozen, config, freeze_path = fixture(root)
            frozen["config"]["feature_group"] = "G4A"
            frozen["config"]["weather_manifest_path"] = "processed/W_SYNTHETIC/data_manifest.json"
            weather_dir = root / "processed/W_SYNTHETIC"
            weather_dir.mkdir(parents=True)
            source, prepared = weather_dir / "raw_weather.csv", weather_dir / "metar_observations.pkl"
            source.write_text("SYNTHETIC input only")
            prepared.write_text("SYNTHETIC output only")
            weather = {"data_id": "W_SYNTHETIC", "inputs": [{"path": str(source), "sha256": final.sha256_file(source)}],
                       "outputs": [{"path": str(prepared), "sha256": final.sha256_file(prepared)}]}
            weather_path = weather_dir / "data_manifest.json"
            weather_path.write_text(json.dumps(weather))
            for relative in ["processed/flights_clean.csv.gz", "references/airports.csv", "references/aircraft_reference_2024.csv",
                             "references/aircraft_type_aliases.csv"]:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("SYNTHETIC dependency metadata only")
            with patch.object(final, "ROOT", root):
                frozen["source_files"] = [{"path": str(path), "sha256": final.sha256_file(path)}
                                          for path in sorted(final._required_sources("G4A"))]
                frozen["data_files"] = [{"path": str(path), "sha256": final.sha256_file(path)}
                                        for path in sorted(final._required_data(frozen["config"]))]
                self.assertTrue({"weather_pipeline_features.py", "weather_features.py", "prepare_weather.py"}.issubset(
                    {Path(item["path"]).name for item in frozen["source_files"]}))
                self.assertTrue({weather_path, source, prepared}.issubset({Path(item["path"]) for item in frozen["data_files"]}))
                frozen["payload_sha256"] = final.canonical_sha256({key: value for key, value in frozen.items() if key != "payload_sha256"})
                freeze_path.write_text(json.dumps(frozen))
                config["freeze_manifest_sha256"] = final.sha256_file(freeze_path)
                final.validate_freeze(config)
                # Change the weather manifest itself without touching its children.
                weather["data_id"] = "CHANGED_AFTER_FREEZE"
                weather_path.write_text(json.dumps(weather))
                with patch.object(final, "load_configured_features", side_effect=AssertionError("must not open August")):
                    with self.assertRaisesRegex(ValueError, "data_files SHA-256 mismatch"):
                        final.run(config, root / "changed-weather-manifest")
                self.assertFalse((root / "experiments/final_access/2026-08.json").exists())


if __name__ == "__main__":
    unittest.main()
