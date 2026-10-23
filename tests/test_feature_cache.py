"""Tiny archived-run fixtures; no real flight/weather input or model fitting."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import feature_cache as cache
from scripts import train_models as training
from tests.test_parallel_training import development_frame


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "runs/SYNTH"
        (self.directory / "data").mkdir(parents=True)
        self.frame = development_frame()
        self.frame.attrs["original_audit"] = {"retained": True}
        self.config = {"mode": "development", "feature_group": "G0", "data_id": "SYNTHETIC",
                       "connection_path": "unused.csv", **training.resolve_features({"feature_group": "G0"})}
        audits, ids = [], []
        data = self.frame.sort_values(["prediction_time", "connection_id"], kind="stable").reset_index(drop=True)
        for fold, _, _, train_mask, val_mask in training.development_splits(data):
            train, val = data.loc[train_mask], data.loc[val_mask]
            audits.append({"fold": fold, "train_rows": len(train), "validation_rows": len(val),
                "train_id_sha256": training.hash_ids(train.connection_id), "validation_id_sha256": training.hash_ids(val.connection_id),
                "train_target_sha256": training.hash_targets(train), "validation_target_sha256": training.hash_targets(val)})
            ids.extend(fold + ":" + val.connection_id.astype(str))
        self.model = {**training.resolve_features(self.config), "development_only": True, "folds": audits,
            "split_sha256": hashlib.sha256(json.dumps(audits, sort_keys=True).encode()).hexdigest(),
            "evaluation_rows": len(ids), "evaluation_ids_sha256": training.hash_ids(ids)}
        self.archive = self.directory / "data/features.pkl"
        self.runtime = {**self.config, "feature_archive_path": str(self.archive), "feature_source_run_id": "SYNTH",
                        "model_type": "random_forest", "model_params": {"n_estimators": 200}}
        self.persist()

    def persist(self):
        self.frame.to_pickle(self.archive)
        for name, value in [("config.json", self.config), ("model_manifest.json", self.model)]:
            (self.directory / name).write_text(json.dumps(value), encoding="utf-8")
        outputs = [{"path": str(path), "bytes": path.stat().st_size, "sha256": cache._sha(path)}
                   for path in [self.directory / "config.json", self.directory / "model_manifest.json", self.archive]]
        (self.directory / "manifest.json").write_text(json.dumps({"status": "completed", "run_id": "SYNTH", "outputs": outputs}), encoding="utf-8")

    def load(self, config=None):
        with patch.object(cache, "ROOT", self.root):
            return cache.load_development_archive(config or self.runtime)

    def test_algorithm_change_reuses_identical_frame_and_preserves_attrs(self):
        before = cache._sha(self.archive)
        actual = self.load()
        pd.testing.assert_frame_equal(actual, self.frame)
        self.assertEqual(actual.attrs["original_audit"], {"retained": True})
        self.assertEqual(actual.attrs["cache_audit"]["evaluation_ids_sha256"], self.model["evaluation_ids_sha256"])
        self.assertEqual(before, cache._sha(self.archive))

    def test_modified_archive_rejected_before_unpickle(self):
        with self.archive.open("ab") as stream:
            stream.write(b"changed")
        with patch.object(cache.pd, "read_pickle", side_effect=AssertionError("must not deserialize")):
            with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                self.load()

    def test_changed_features_and_scientific_inputs_rejected(self):
        for key, value in [("features", []), ("numeric_features", []), ("categorical_features", []),
                           ("noaa_backfill", True), ("weather_manifest_path", "other.json"),
                           ("connection_path", "other.csv"), ("data_id", "other"), ("feature_group", "G1")]:
            with self.subTest(key=key):
                altered = copy.deepcopy(self.runtime)
                altered[key] = value
                with self.assertRaises(ValueError):
                    self.load(altered)

    def test_august_sentinel_rejected_before_target_interpretation(self):
        self.frame.loc[0, "prediction_time"] = pd.Timestamp("2026-08-01", tz="UTC")
        self.frame["target_delay"] = self.frame.target_delay.astype(object)
        self.frame.loc[0, "target_delay"] = "DO_NOT_PARSE_AUGUST_TARGET"
        self.persist()
        with self.assertRaisesRegex(ValueError, "outside April-July"):
            self.load()

    def test_reissued_receipt_cannot_hide_changed_evaluation_target(self):
        self.frame.loc[40, "target_delay"] += 1
        self.persist()
        with self.assertRaisesRegex(ValueError, "Cached cohort differs"):
            self.load()

    def test_final_and_unfinished_or_invalid_sources_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode=development"):
            self.load({**self.runtime, "mode": "final"})
        path = self.directory / "manifest.json"
        original = json.loads(path.read_text())
        for changes in [{"status": "failed"}, {"invalidated": True}, {"valid": False}]:
            path.write_text(json.dumps({**original, **changes}), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.load()
        path.write_text(json.dumps(original), encoding="utf-8")
        self.model["development_only"] = False
        self.persist()
        with self.assertRaisesRegex(ValueError, "development-only"):
            self.load()

    def test_model_hash_and_source_run_identity_rejected(self):
        path = self.directory / "model_manifest.json"
        path.write_text(path.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            self.load()
        with self.assertRaisesRegex(ValueError, "source_run_id"):
            self.load({**self.runtime, "feature_source_run_id": "OTHER"})


if __name__ == "__main__":
    unittest.main()
