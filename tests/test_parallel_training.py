"""Small Windows-spawn regression checks for serial/parallel fold equivalence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.build_features import SOURCE_COLUMNS, build_base_features
from scripts.train_models import DEVELOPMENT_FOLDS, MODEL_NAMES, run_development


def development_frame():
    rng = np.random.default_rng(42)
    points = pd.DatetimeIndex([pd.Timestamp(f"2026-{month:02d}-01", tz="UTC") + pd.Timedelta(hours=i * 8)
                               for month in range(4, 8) for i in range(32)])
    source = pd.DataFrame({name: [1] * len(points) for name in SOURCE_COLUMNS})
    source["connection_id"] = [f"parallel-qa-{i:04d}" for i in range(len(points))]
    source["upstream_actual_arr_utc"] = points
    source["downstream_actual_dep_utc"] = points + pd.Timedelta(minutes=30)
    source["downstream_dep_delay_minutes"] = rng.integers(0, 120, len(points))
    source["upstream_arr_delay_minutes"] = rng.integers(0, 100, len(points))
    source["overnight_scheduled"] = rng.choice([False, True], len(points))
    source["airport_id"] = rng.integers(1, 8, len(points))
    source["downstream_origin_id"] = source.airport_id
    source["downstream_dest_id"] = rng.integers(1, 12, len(points))
    source["downstream_airline"] = rng.choice(["QA", "QB", "QC"], len(points))
    source["downstream_distance"] = rng.uniform(50, 2000, len(points))
    source["downstream_crs_elapsed_time"] = source.downstream_distance / 8 + 30
    source["downstream_day_of_week"] = points.dayofweek + 1
    source["downstream_sched_hour"] = points.hour
    source["scheduled_turnaround_minutes"] = rng.uniform(20, 300, len(points))
    source.loc[5, "downstream_distance"] = np.nan
    frame = build_base_features(source)
    frame["unused_metadata"] = "never send this column to a worker"
    return frame


class ParallelTraining(unittest.TestCase):
    def test_serial_parallel_equivalence_and_complete_artifacts(self):
        frame = development_frame()
        config = {"threads": 1, "preprocessing": {"min_frequency": 2}}
        with tempfile.TemporaryDirectory() as temp:
            serial, parallel = Path(temp) / "serial", Path(temp) / "parallel"
            serial_manifest = run_development(frame, {**config, "fold_workers": 1}, serial)
            parallel_manifest = run_development(frame, {**config, "fold_workers": 3}, parallel)
            self.assertEqual(serial_manifest["evaluation_ids_sha256"], parallel_manifest["evaluation_ids_sha256"])
            self.assertEqual(serial_manifest["split_sha256"], parallel_manifest["split_sha256"])
            self.assertEqual(serial_manifest["folds"], parallel_manifest["folds"])
            pd.testing.assert_frame_equal(pd.read_csv(serial / "evaluation_ids.csv.gz"),
                                          pd.read_csv(parallel / "evaluation_ids.csv.gz"))
            pd.testing.assert_frame_equal(pd.read_csv(serial / "split_audit.csv"), pd.read_csv(parallel / "split_audit.csv"))
            pd.testing.assert_frame_equal(pd.read_csv(serial / "metrics.csv"), pd.read_csv(parallel / "metrics.csv"),
                                          check_exact=False, atol=1e-10, rtol=0)
            for fold, _, _ in DEVELOPMENT_FOLDS:
                for name in MODEL_NAMES:
                    filename = f"{fold}_{name}.csv.gz"
                    pd.testing.assert_frame_equal(pd.read_csv(serial / "predictions" / filename),
                                                  pd.read_csv(parallel / "predictions" / filename),
                                                  check_exact=False, atol=1e-10, rtol=0)
                    self.assertTrue((parallel / "models" / f"{fold}_{name}.joblib").is_file())
                timing = json.loads((parallel / f"fold_{fold}_timing.json").read_text())
                self.assertEqual(timing["status"], "completed")
                self.assertGreater(timing["elapsed_seconds"], 0)
                self.assertEqual(set(timing["models_seconds"]), set(MODEL_NAMES))
                self.assertIn("status=completed", (parallel / f"fold_{fold}.log").read_text())
            self.assertEqual(len(list((parallel / "models").glob("*.joblib"))), 12)
            self.assertEqual(len(list((parallel / "predictions").glob("*.csv.gz"))), 12)

    def test_parallel_failure_preserves_completed_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            config = {"fold_workers": 3, "threads": 1, "preprocessing": {"drop": "invalid"}}
            with self.assertRaisesRegex(ValueError, "drop='first'"):
                run_development(development_frame(), config, output)
            self.assertFalse((output / "model_manifest.json").exists())
            self.assertTrue((output / "split_audit.csv").is_file())
            for fold, _, _ in DEVELOPMENT_FOLDS:
                timing = json.loads((output / f"fold_{fold}_timing.json").read_text())
                self.assertEqual(timing["status"], "failed")
                self.assertIn("Traceback", (output / f"fold_{fold}.log").read_text())
                for name in MODEL_NAMES[:3]:
                    self.assertTrue((output / "predictions" / f"{fold}_{name}.csv.gz").is_file())
                    self.assertTrue((output / "models" / f"{fold}_{name}.joblib").is_file())

    def test_invalid_worker_count_is_rejected_before_training(self):
        for value in [0, 2, 4, True, "3"]:
            with self.assertRaisesRegex(ValueError, "fold_workers"):
                run_development(development_frame(), {"fold_workers": value}, "unused")


if __name__ == "__main__":
    unittest.main()
