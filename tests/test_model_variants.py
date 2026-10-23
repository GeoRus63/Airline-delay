"""Whitelisted feature/model extensions preserve the development evaluation cohort."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.build_features import G0_CATEGORICAL, G0_FEATURES, G0_NUMERIC, G1_NUMERIC
from scripts.train_models import make_g0_pipeline, make_pipeline, resolve_features, run_development
from tests.test_parallel_training import development_frame


class ModelVariants(unittest.TestCase):
    def test_g0_defaults_match_explicit_configuration(self):
        frame = development_frame()
        base = {"threads": 1, "preprocessing": {"min_frequency": 2}}
        explicit = {**base, "feature_group": "G0", "features": G0_FEATURES,
                    "categorical_features": G0_CATEGORICAL, "numeric_features": G0_NUMERIC,
                    "model_name": "g0_linear", "model_type": "linear", "include_baselines": True}
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "default", Path(tmp) / "explicit"
            a, b = run_development(frame, base, first), run_development(frame, explicit, second)
            self.assertEqual(a["folds"], b["folds"])
            self.assertEqual(a["features"], G0_FEATURES)
            self.assertEqual(a["model_params"], b["model_params"])
            self.assertEqual(a["model_params"]["tol"], 1e-6)
            self.assertTrue(a["model_params"]["fit_intercept"])
            pd.testing.assert_frame_equal(pd.read_csv(first / "metrics.csv"), pd.read_csv(second / "metrics.csv"))
        self.assertEqual(make_g0_pipeline(base).get_params()["preprocess__categorical__encoder__drop"], "first")

    def test_added_features_keep_ids_targets_and_split_hash(self):
        frame = development_frame()
        for name in set(G1_NUMERIC) - set(frame):
            frame[name] = np.arange(len(frame)) % 5
        with tempfile.TemporaryDirectory() as tmp:
            base = run_development(frame, {"threads": 1}, Path(tmp) / "g0")
            added = run_development(frame, {"threads": 1, "feature_group": "G1", "model_name": "g1_linear",
                                             "include_baselines": False}, Path(tmp) / "g1")
            self.assertEqual(base["folds"], added["folds"])
            self.assertEqual(base["evaluation_ids_sha256"], added["evaluation_ids_sha256"])
            self.assertEqual(base["split_sha256"], added["split_sha256"])
            self.assertEqual(added["models"], ["g1_linear"])
            self.assertEqual(len(list((Path(tmp) / "g1/models").glob("*.joblib"))), 3)

    def test_tree_and_forest_run_with_locked_parameters(self):
        frame = development_frame()
        for kind in ["decision_tree", "random_forest"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                params = {"max_depth": 8 if kind == "decision_tree" else 16, "min_samples_leaf": 50}
                if kind == "random_forest":
                    params["n_estimators"] = 3  # Lightweight only; formal config locks 200.
                config = {"threads": 1, "model_name": kind, "model_type": kind,
                          "model_params": params, "include_baselines": False}
                manifest = run_development(frame, config, tmp)
                self.assertEqual(manifest["model_params"]["random_state"], 42)
                self.assertEqual(manifest["model_params"]["min_samples_leaf"], 50)
                self.assertIsNone(manifest["sparse_solver"])
                self.assertEqual(len(list((Path(tmp) / "models").glob("*.joblib"))), 3)
                if kind == "random_forest":
                    self.assertEqual(manifest["model_params"]["max_features"], .5)
                    self.assertEqual(manifest["model_params"]["n_jobs"], 1)
        self.assertEqual(make_pipeline({"model_type": "random_forest"}).named_steps["regression"].n_estimators, 200)

    def test_leakage_arbitrary_prefix_and_nested_types_are_rejected(self):
        for field in ["target_delay", "label_time", "actual_ground_minutes", "month", "metar_target_delay", "taf_actual_ground_minutes"]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_features({"feature_group": "G4B", "features": G0_FEATURES + [field]})
        with self.assertRaisesRegex(ValueError, "never both"):
            resolve_features({"feature_group": "G2", "features": G0_FEATURES + ["aircraft_family", "aircraft_variant"]})
        with self.assertRaises(ValueError):
            resolve_features({"features": G0_FEATURES, "categorical_features": G0_CATEGORICAL + ["downstream_distance"],
                              "numeric_features": G0_NUMERIC})
        weather = resolve_features({"feature_group": "G4B", "features": G0_FEATURES + ["metar_wind_knots", "taf_prevailing_visibility_miles"]})
        self.assertIn("metar_wind_knots", weather["numeric_features"])

    def test_unplanned_model_parameters_are_rejected(self):
        for config in [{"model_type": "decision_tree", "model_params": {"max_depth": 4}},
                       {"model_type": "random_forest", "model_params": {"min_samples_leaf": 1}},
                       {"model_type": "random_forest", "model_params": {"n_jobs": 32}},
                       {"model_type": "linear", "model_params": {"fit_intercept": False}}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                make_pipeline(config)


if __name__ == "__main__":
    unittest.main()
