"""Small synthetic checks of temporal boundaries and the predictor whitelist."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.build_features import G0_FEATURES, SOURCE_COLUMNS, build_base_features
from scripts.train_models import development_splits, make_g0_pipeline, run_development
from scripts.evaluate_models import evaluate_predictions


def connections():
    prediction = pd.to_datetime(["2026-04-30T23:59Z", "2026-04-30T23:59Z", "2026-05-01T00:00Z",
                                 "2026-06-01T00:00Z", "2026-07-01T00:00Z", "2026-08-01T00:00Z"])
    frame = pd.DataFrame({column: [1] * len(prediction) for column in SOURCE_COLUMNS})
    frame["connection_id"] = [f"c{i}" for i in range(len(frame))]
    frame["upstream_actual_arr_utc"] = prediction
    frame["downstream_actual_dep_utc"] = prediction + pd.Timedelta(minutes=2)
    frame.loc[0, "downstream_actual_dep_utc"] = pd.Timestamp("2026-04-30T23:59:30Z")
    frame.loc[1, "downstream_actual_dep_utc"] = pd.Timestamp("2026-05-01T00:00Z")
    frame["downstream_dep_delay_minutes"] = [1, 2, 3, 4, 5, 9999999]
    frame["upstream_arr_delay_minutes"] = [0, 2, 3, 4, 5, 9999999]
    frame["overnight_scheduled"] = False
    frame["downstream_airline"] = "XX"
    return frame


class TemporalContract(unittest.TestCase):
    def test_half_open_and_mature_labels(self):
        frame = build_base_features(connections())
        self.assertNotIn("c5", set(frame.connection_id))
        splits = list(development_splits(frame))
        self.assertEqual(frame.loc[splits[0][3], "connection_id"].tolist(), ["c0"])
        self.assertEqual(frame.loc[splits[0][4], "connection_id"].tolist(), ["c2"])
        self.assertEqual(frame.loc[splits[1][4], "connection_id"].tolist(), ["c3"])
        self.assertEqual(frame.loc[splits[2][4], "connection_id"].tolist(), ["c4"])

    def test_same_time_missing_and_negative_outcomes_excluded(self):
        source = connections()
        source.loc[0, "downstream_actual_dep_utc"] = source.loc[0, "upstream_actual_arr_utc"]
        source.loc[1, "downstream_dep_delay_minutes"] = np.nan
        source.loc[2, "downstream_dep_delay_minutes"] = -1
        frame = build_base_features(source)
        self.assertEqual(set(frame.connection_id), {"c3", "c4"})
        counts = frame.attrs["feature_audit"]["counts"]
        self.assertEqual(counts["prediction_not_before_departure"], 1)
        self.assertEqual(counts["missing_or_nonfinite_target"], 1)
        self.assertEqual(counts["negative_target"], 1)

    def test_outcomes_never_in_design_matrix(self):
        frame = build_base_features(connections())
        pipeline = make_g0_pipeline({"preprocessing": {"min_frequency": 1}})
        pipeline.fit(frame, frame.target_delay)
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        for prohibited in ["target_delay", "label_time", "actual_ground", "upstream_arr_delay", "prediction_month"]:
            self.assertFalse(any(prohibited in column for column in names))
        self.assertEqual(pipeline.named_steps["preprocess"].transform(frame).shape[0], len(frame))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_development(frame, {"features": G0_FEATURES + ["target_delay"]}, tmp)

    def test_small_development_run_and_overall_weighting(self):
        frame = build_base_features(connections())
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_development(frame, {"threads": 1, "preprocessing": {"min_frequency": 1}}, tmp)
            self.assertEqual(summary["evaluation_rows"], 3)
            output = Path(tmp)
            metrics = pd.read_csv(output / "metrics.csv")
            all_metrics = metrics.loc[(metrics.group == "all") & (metrics.model == "carry_forward")]
            self.assertTrue((all_metrics.mae == 0).all())
            self.assertEqual(int(all_metrics.loc[all_metrics.fold == "overall", "n"].iloc[0]), 3)
            self.assertEqual(len(list((output / "models").glob("*.joblib"))), 12)
            self.assertEqual(len(list((output / "predictions").glob("*.csv.gz"))), 12)
            ids = pd.read_csv(output / "evaluation_ids.csv.gz")
            self.assertEqual(set(ids.connection_id), {"c2", "c3", "c4"})

    def test_high_cardinality_encoding_stays_sparse(self):
        frame = pd.concat([build_base_features(connections()).iloc[[0]]] * 100, ignore_index=True)
        frame["airport_id"] = [str(i) for i in range(len(frame))]
        frame["route"] = [f"{i}->2" for i in range(len(frame))]
        model = make_g0_pipeline({"preprocessing": {"min_frequency": 1}})
        model.fit(frame, frame.target_delay)
        self.assertTrue(sparse.issparse(model.named_steps["preprocess"].transform(frame)))

    def test_clipping_and_fixed_severity_boundary(self):
        frame = build_base_features(connections()).iloc[:3].copy()
        frame["model"], frame["fold"] = "test", "2026-05"
        frame["target_delay"] = [0, 59, 60]
        frame["predicted_delay"] = [-10, 59, 60]
        metrics = evaluate_predictions(frame)
        overall = metrics.loc[(metrics.fold == "overall") & (metrics.group == "all")].iloc[0]
        self.assertEqual(overall.mae, 0)
        severity = metrics.loc[(metrics.fold == "overall") & (metrics.group == "severity")].set_index("value")
        self.assertEqual(severity.loc["ordinary_lt60", "n"], 2)
        self.assertEqual(severity.loc["severe_ge60", "n"], 1)


if __name__ == "__main__":
    unittest.main()
