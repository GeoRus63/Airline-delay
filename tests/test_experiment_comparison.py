"""Synthetic archived-run comparisons, paired calendar blocks and subset contracts."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.compare_experiments import (align_predictions, compare_runs, comparison_intervals,
                                        daily_statistics, moving_block_draws)


def write_run(path, model, predictions, parent=""):
    path.mkdir()
    (path / "predictions").mkdir()
    manifest = {"status": "completed"}
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "metrics.csv").write_text("model,mae\n" + model + ",1\n")
    (path / "config.json").write_text(json.dumps({"parent_run_id": parent}))
    folds = []
    for month in [5, 6, 7]:
        name = f"2026-{month:02d}"
        rows = predictions.loc[predictions.fold.eq(name)].copy()
        rows["model"] = model
        rows.to_csv(path / "predictions" / f"{name}_{model}.csv.gz", index=False)
        folds.append({"fold": name, "validation_start": f"2026-{month:02d}-01T00:00:00+00:00",
                      "validation_end_exclusive": f"2026-{month+1:02d}-01T00:00:00+00:00", "validation_rows": len(rows)})
    model_manifest = {"models": [model], "model_name": model, "folds": folds,
                      "evaluation_rows": len(predictions), "evaluation_ids_sha256": "same-ids", "split_sha256": "same-splits"}
    (path / "model_manifest.json").write_text(json.dumps(model_manifest))


def paired_predictions():
    # Unequal day support: an unweighted average of day MAEs would be wrong.
    times = pd.to_datetime(["2026-05-01T12:00Z"] * 9 + ["2026-05-02T12:00Z", "2026-06-01T12:00Z", "2026-07-01T12:00Z"])
    frame = pd.DataFrame({"fold": times.strftime("%Y-%m"), "connection_id": [f"i{i}" for i in range(len(times))],
                          "prediction_time": times, "target_delay": 0., "predicted_delay": [1.] * 9 + [10., 2., 3.]})
    candidate = frame.copy()
    candidate["predicted_delay"] = frame.predicted_delay * 2
    return frame, candidate


class ExperimentComparison(unittest.TestCase):
    def test_exact_ids_can_align_but_missing_or_changed_label_is_rejected(self):
        reference, candidate = paired_predictions()
        shuffled = candidate.sample(frac=1, random_state=42)
        with self.assertRaisesRegex(ValueError, "order differs"):
            align_predictions(reference, shuffled)
        left, right = align_predictions(reference, shuffled, align_rows=True)
        self.assertEqual(left.connection_id.tolist(), right.connection_id.tolist())
        changed = candidate.copy()
        changed.loc[0, "target_delay"] = 1
        with self.assertRaisesRegex(ValueError, "Target labels"):
            align_predictions(reference, changed)
        with self.assertRaisesRegex(ValueError, "cohort differs"):
            align_predictions(reference, candidate.iloc[:-1])
        changed = candidate.copy()
        changed.loc[0, "prediction_time"] += pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "prediction_time"):
            align_predictions(reference, changed)

    def test_ratio_of_sums_and_shared_date_draws_give_correct_direction(self):
        reference, candidate = paired_predictions()
        calendar = pd.date_range("2026-05-01", "2026-08-01", inclusive="left", tz="UTC")
        daily = daily_statistics(reference, candidate, calendar)
        self.assertEqual(len(daily), 92)
        self.assertEqual(int(daily.n.sum()), 12)
        metrics = comparison_intervals(daily, bootstrap_repeats=1000, block_days=7, seed=42)
        may = metrics.loc[(metrics.scope == "2026-05") & (metrics.metric == "mae")].iloc[0]
        self.assertAlmostEqual(may.reference_error, 1.9)
        self.assertAlmostEqual(may.difference_candidate_minus_reference, 1.9)
        self.assertEqual(may.calendar_dates, 31)
        self.assertTrue((metrics.difference_candidate_minus_reference > 0).all())
        self.assertTrue((metrics.ci95_low.dropna() > 0).all())
        # On identical predictions shared date draws must produce exactly zero CI.
        same = comparison_intervals(daily_statistics(reference, reference, calendar), bootstrap_repeats=1000)
        self.assertTrue((same.ci95_low.dropna() == 0).all())
        self.assertTrue((same.ci95_high.dropna() == 0).all())

    def test_blocks_are_seven_adjacent_days_and_do_not_cross_missing_month(self):
        calendar = pd.date_range("2026-05-01", periods=31, tz="UTC").append(pd.date_range("2026-07-01", periods=31, tz="UTC"))
        draws = moving_block_draws(calendar, 20, 7, np.random.default_rng(42))
        self.assertEqual(draws.shape, (20, 62))
        # Full blocks in each independent contiguous segment are adjacent.
        for start in [0, 7, 14, 21, 31, 38, 45, 52]:
            np.testing.assert_array_equal(np.diff(draws[:, start:start+7], axis=1), np.ones((20, 6), dtype=int))
        self.assertTrue((draws[:, :31] < 31).all())
        self.assertTrue((draws[:, 31:] >= 31).all())

    def test_archived_comparison_weather_subset_and_parent_relation(self):
        reference, candidate = paired_predictions()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref, cand = root / "R0001", root / "R0002"
            write_run(ref, "base", reference)
            write_run(cand, "new", candidate, parent="R0001")
            (cand / "data").mkdir()
            features = pd.DataFrame({"connection_id": candidate.connection_id,
                                     "metar_wind_knots": [np.nan] + [10.] * 11})
            features.to_pickle(cand / "data/features.pkl")
            summary = compare_runs(ref, cand, root / "R0003/comparison", common_sample_columns=["metar_wind_knots"])
            self.assertEqual(summary["sample_scope"], "explicit_subset")
            self.assertEqual(summary["compared_n"], 11)
            self.assertEqual(summary["full_evaluation_n"], 12)
            self.assertEqual(summary["parent_run_id"], "R0001")
            for name in summary["outputs"]:
                self.assertTrue((root / "R0003/comparison" / name).is_file())
            with self.assertRaisesRegex(ValueError, "declared parent"):
                compare_runs(ref, cand, root / "alternate", parent_run_id="R0000")
            with self.assertRaisesRegex(ValueError, "must not modify"):
                compare_runs(ref, cand, ref / "comparison")
            with self.assertRaisesRegex(ValueError, "outside"):
                compare_runs(ref, cand, root / "bad-subset", subset_ids=["not-a-real-id"])
            manifest = json.loads((cand / "model_manifest.json").read_text())
            manifest["split_sha256"] = "changed"
            (cand / "model_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "split_sha256 mismatch"):
                compare_runs(ref, cand, root / "bad-hash")

    def test_explicit_running_candidate_is_ready_and_can_archive_before_sealing(self):
        reference, candidate = paired_predictions()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref, cand = root / "R0001", root / "R0002"
            write_run(ref, "base", reference)
            write_run(cand, "new", candidate, parent="R0001")
            (cand / "manifest.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "completed run"):
                compare_runs(ref, cand, cand / "comparison")
            summary = compare_runs(ref, cand, cand / "comparison", allow_running_candidate=True)
            self.assertEqual(summary["candidate_status_at_comparison"], "running")
            self.assertTrue((cand / "comparison/comparison.json").exists())
            (cand / "metrics.csv").unlink()
            with self.assertRaisesRegex(ValueError, "not ready"):
                compare_runs(ref, cand, cand / "not-ready", allow_running_candidate=True)
            (ref / "manifest.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "completed run"):
                compare_runs(ref, cand, root / "bad-reference", allow_running_candidate=True)


if __name__ == "__main__":
    unittest.main()
