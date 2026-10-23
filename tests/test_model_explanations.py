"""Small read-only explanation checks with already-fitted synthetic pipelines."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts.build_features import G0_FEATURES, G1_FEATURES
from scripts.explain_models import explain_run, permutation_groups, permute_input_group
from scripts.train_models import make_pipeline
from tests.test_parallel_training import development_frame


def fitted_run(root, model_type="linear", feature_group="G0"):
    source = root / "R0001"
    (source / "models").mkdir(parents=True)
    (source / "data").mkdir()
    frame = development_frame()
    features = G0_FEATURES if feature_group == "G0" else G1_FEATURES
    if feature_group == "G1":
        frame["tail_completed_24h_n"] = np.arange(len(frame)) % 6
        frame["tail_completed_24h_arr_delay_mean"] = np.arange(len(frame)) % 30
        frame["history_window_incomplete"] = 0
    train = frame.loc[frame.prediction_time < pd.Timestamp("2026-07-01", tz="UTC")]
    config = {"threads": 1, "feature_group": feature_group, "model_type": model_type, "preprocessing": {"min_frequency": 2}}
    model = make_pipeline(config).fit(train[features], train.target_delay)
    joblib.dump(model, source / "models/2026-07_demo.joblib")
    manifest = {"models": ["demo"], "model_name": "demo", "features": features,
                "folds": [{"fold": "2026-07", "validation_start": "2026-07-01T00:00Z",
                           "validation_end_exclusive": "2026-08-01T00:00Z", "validation_rows": 32}]}
    (source / "model_manifest.json").write_text(json.dumps(manifest))
    (source / "manifest.json").write_text(json.dumps({"status": "completed"}))
    # Outside the selected validation window: this target must never be parsed.
    future = frame.iloc[[0]].copy()
    future["connection_id"] = "never-explain-august"
    future["prediction_time"] = pd.Timestamp("2026-08-01T00:00Z")
    future["target_delay"] = "DO_NOT_PARSE_AUGUST"
    pd.concat([frame, future], ignore_index=True).to_pickle(source / "data/features.pkl")
    return source


class ModelExplanations(unittest.TestCase):
    def test_validation_sampling_no_fit_coefficients_skips_and_pdp_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = fitted_run(root)
            with patch.object(Pipeline, "fit", side_effect=AssertionError("explanations must never fit")):
                result = explain_run(source, root / "explanation", sample_n=12, repeats=5)
            ids = pd.read_csv(root / "explanation/explain_sample_ids.csv.gz")
            self.assertEqual(len(ids), 12)
            times = pd.to_datetime(ids.prediction_time, utc=True)
            self.assertTrue(times.ge(pd.Timestamp("2026-07-01", tz="UTC")).all())
            self.assertTrue(times.lt(pd.Timestamp("2026-08-01", tz="UTC")).all())
            self.assertFalse(result["fit_called"])
            self.assertTrue(any(item.get("group") == "G1_upstream_delay" for item in result["skips"]))
            self.assertTrue(any(item.get("feature") == "upstream_arr_delay_minutes" for item in result["skips"]))
            permutations = pd.read_csv(root / "explanation/permutation_repeats.csv")
            self.assertEqual(len(permutations), 5)
            self.assertTrue(permutations.features.str.split("|").map(set).map(lambda s: s == set(G0_FEATURES)).all())
            coefficients = pd.read_csv(root / "explanation/explain_coefficients.csv")
            model = joblib.load(source / "models/2026-07_demo.joblib")
            self.assertEqual(len(coefficients), len(model.named_steps["preprocess"].get_feature_names_out()))
            self.assertEqual(len(pd.read_csv(root / "explanation/explain_intercepts.csv")), 1)
            points = pd.read_csv(root / "explanation/pdp_values.csv")
            self.assertEqual(set(points.feature), {"scheduled_turnaround_minutes"})
            self.assertTrue((points.mean_prediction >= 0).all())
            support = pd.read_csv(root / "explanation/pdp_support.csv")
            self.assertEqual(int(support.n.sum()), 12)
            raw = pd.read_csv(root / "explanation/explain_sample_predictions.csv.gz")
            self.assertEqual(len(raw), 12)
            self.assertTrue((raw.original_predicted_delay >= 0).all())
            self.assertTrue((root / "explanation/pdp_2026-07_scheduled_turnaround_minutes.png").is_file())
            again = explain_run(source, root / "again", sample_n=12, repeats=5)
            self.assertEqual(result["folds"], again["folds"])
            pd.testing.assert_frame_equal(ids, pd.read_csv(root / "again/explain_sample_ids.csv.gz"))

    def test_group_permutation_keeps_plan_columns_together(self):
        x = pd.DataFrame({"airport_id": ["A", "B", "C"], "route": ["A->X", "B->Y", "C->Z"],
                          "upstream_arr_delay_minutes": [1, 2, 3]})
        groups = permutation_groups(x.columns.tolist())
        self.assertEqual(groups["G0_plan"], ["airport_id", "route"])
        changed = permute_input_group(x, groups["G0_plan"], np.array([2, 0, 1]))
        self.assertEqual(list(zip(changed.airport_id, changed.route)), [("C", "C->Z"), ("A", "A->X"), ("B", "B->Y")])
        self.assertEqual(changed.upstream_arr_delay_minutes.tolist(), [1, 2, 3])

    def test_completed_source_is_protected_and_tree_paths_are_archived(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = fitted_run(root, model_type="decision_tree")
            with self.assertRaisesRegex(ValueError, "completed source"):
                explain_run(source, source / "explanations")
            with patch.object(Pipeline, "fit", side_effect=AssertionError("must not refit")):
                summary = explain_run(source, root / "tree", sample_n=8, repeats=2)
            self.assertEqual(len(summary["tree_path_files"]), 1)
            text = (root / "tree" / summary["tree_path_files"][0]).read_text()
            self.assertIn("truncated at depth 3", text)
            self.assertTrue(pd.read_csv(root / "tree/explain_coefficients.csv").empty)

    def test_g1_both_numeric_pdps_and_complete_history_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = fitted_run(root, feature_group="G1")
            with patch.object(Pipeline, "fit", side_effect=AssertionError("must not refit")):
                result = explain_run(source, root / "g1", sample_n=16, repeats=2)
            pdp = pd.read_csv(root / "g1/pdp_values.csv")
            self.assertEqual(set(pdp.feature), {"upstream_arr_delay_minutes", "scheduled_turnaround_minutes"})
            groups = result["permutation_groups"]
            self.assertEqual(groups["G1_upstream_delay"], ["upstream_arr_delay_minutes"])
            self.assertEqual(set(groups["G1_tail_history"]), {"tail_completed_24h_n", "tail_completed_24h_arr_delay_mean", "history_window_incomplete"})
            self.assertEqual(set(pd.read_csv(root / "g1/permutation_summary.csv").group), {"G0_plan", "G1_upstream_delay", "G1_tail_history"})


if __name__ == "__main__":
    unittest.main()
