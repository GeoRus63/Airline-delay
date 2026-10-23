"""Strict event-time history checks independent of training/validation partitions."""
import unittest

import numpy as np
import pandas as pd

from scripts.build_features import G0_FEATURES, G1_ADDED_NUMERIC, G1_FEATURES, augment_history


def queries():
    return pd.DataFrame({"connection_id": ["q0", "q1", "q2", "q3"],
                         "tail_number": [" a ", "A", None, "NO_HISTORY"],
                         "upstream_flight_id": ["up", "later_up", "missing", "none"],
                         "upstream_arr_delay_minutes": [15, 20, 0, 0],
                         "prediction_time": pd.to_datetime(["2026-05-01T00:00Z", "2026-05-01T01:00Z",
                                                           "2026-05-01T00:00Z", "2026-05-01T00:00Z"]),
                         "target_delay": [12345, 12345, 12345, 12345]})


def events():
    return pd.DataFrame({"flight_id": ["old", "boundary", "prior", "prior_tie", "up", "same_time_other", "future", "later_up"],
                         "Tail_Number": ["A"] * 8,
                         "actual_arr_utc": ["2026-04-29T23:59Z", "2026-04-30T00:00Z", "2026-04-30T12:00Z", "2026-04-30T12:00Z",
                                            "2026-05-01T00:00Z", "2026-05-01T00:00Z", "2026-05-01T00:30Z", "2026-05-01T01:00Z"],
                         "ArrDelayMinutes": [999, 10, 20, 30, 100, 200, 300, 400],
                         "normal_operation": True, "time_valid": True,
                         # Deliberately future-dependent QA: must have no effect.
                         "actual_order_conflict": True, "tail_schedule_incomplete": True})


class HistoryFeatures(unittest.TestCase):
    def test_strict_same_time_and_24_hour_boundary(self):
        result = augment_history(queries(), events().sample(frac=1, random_state=42))
        self.assertEqual(result.loc[0, "tail_completed_24h_n"], 3)
        self.assertEqual(result.loc[0, "tail_completed_24h_arr_delay_mean"], 20)
        # At 01:00, boundary is older than 24h; midnight legs and 00:30 are known.
        self.assertEqual(result.loc[1, "tail_completed_24h_n"], 5)
        self.assertEqual(result.loc[1, "tail_completed_24h_arr_delay_mean"], 130)
        has_history = result.history_latest_completion_time.notna()
        self.assertTrue((result.loc[has_history, "history_latest_completion_time"] < result.loc[has_history, "prediction_time"]).all())
        self.assertTrue(result.target_delay.eq(12345).all())
        self.assertTrue(set(G0_FEATURES).issubset(G1_FEATURES))
        self.assertIn("history_window_incomplete", G1_ADDED_NUMERIC)

    def test_optional_current_upstream_adds_only_matching_id(self):
        result = augment_history(queries(), events(), include_current_upstream=True)
        self.assertEqual(result.loc[0, "tail_completed_24h_n"], 4)
        self.assertEqual(result.loc[0, "tail_completed_24h_arr_delay_mean"], 40)
        self.assertEqual(result.attrs["history_audit"]["queries"]["current_upstream_added"], 2)
        self.assertEqual(result.loc[0, "history_latest_completion_time"], result.loc[0, "prediction_time"])

    def test_missing_tail_zero_history_and_source_left_edge(self):
        frame = queries()
        frame.loc[3, "prediction_time"] = pd.Timestamp("2026-04-01T12:00Z")
        result = augment_history(frame, events())
        self.assertTrue(np.isnan(result.loc[2, "tail_completed_24h_n"]))
        self.assertTrue(result.loc[2, "history_missing_tail"])
        self.assertEqual(result.loc[2, "history_window_incomplete"], 0)
        self.assertEqual(result.loc[3, "tail_completed_24h_n"], 0)
        self.assertTrue(np.isnan(result.loc[3, "tail_completed_24h_arr_delay_mean"]))
        self.assertFalse(result.loc[3, "history_missing_tail"])
        self.assertEqual(result.loc[3, "history_window_incomplete"], 1)

    def test_validation_uses_leg_history_outside_training_cohort(self):
        frame = queries().iloc[[0]].copy()
        frame.index = [999]
        frame["fold"] = "May validation"
        result = augment_history(frame, events())
        self.assertEqual(result.index.tolist(), [999])
        self.assertEqual(result.loc[999, "tail_completed_24h_n"], 3)
        self.assertEqual(result.loc[999, "history_latest_completion_time"], pd.Timestamp("2026-04-30T12:00Z"))

    def test_quality_ids_missing_and_no_future_value_parsing(self):
        source = events()
        bad = source.iloc[[2]].copy()
        bad["flight_id"] = "bad"
        bad["ArrDelayMinutes"] = np.nan
        duplicate = source.iloc[[2]].copy()
        duplicate["flight_id"] = "prior"
        missing_tail = source.iloc[[2]].copy()
        missing_tail["flight_id"], missing_tail["Tail_Number"] = "no-tail", None
        future = source.iloc[[2]].copy()
        future["flight_id"], future["actual_arr_utc"], future["ArrDelayMinutes"] = "august", "2026-08-01T00:00Z", "DO_NOT_PARSE"
        result = augment_history(queries().iloc[[0]], pd.concat([source, bad, missing_tail, future], ignore_index=True))
        self.assertEqual(result.iloc[0].tail_completed_24h_n, 3)
        audit = result.attrs["history_audit"]["counts"]
        self.assertEqual(audit["duplicate_flight_id_rows"], 0)
        self.assertEqual(audit["missing_tail_rows"], 1)
        self.assertEqual(audit["invalid_arrival_delay_rows"], 1)
        self.assertEqual(audit["valid_history_event_rows"], audit["unique_history_flight_ids"])
        with self.assertRaisesRegex(ValueError, "2 duplicate flight_id"):
            augment_history(queries(), pd.concat([source, duplicate], ignore_index=True))

    def test_no_history_events_is_valid(self):
        result = augment_history(queries(), events().iloc[:0])
        self.assertEqual(result.loc[0, "tail_completed_24h_n"], 0)
        self.assertTrue(np.isnan(result.loc[0, "tail_completed_24h_arr_delay_mean"]))
        self.assertEqual(result.attrs["history_audit"]["counts"]["valid_history_event_rows"], 0)


if __name__ == "__main__":
    unittest.main()
