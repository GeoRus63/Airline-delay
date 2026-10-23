"""Strict event-time history checks independent of training/validation partitions."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.build_features import G0_FEATURES, G1_ADDED_NUMERIC, G1_FEATURES, augment_history, load_configured_features
from scripts.operational_features import source_calendar_starts


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
    def test_local_month_conservative_boundary_and_carry_in_caveat(self):
        airports = pd.DataFrame({'airport_id': [1, 2], 'timezone': ['America/New_York', 'America/Los_Angeles']})
        starts = source_calendar_starts(airports)
        frame = queries().iloc[[0]].copy()
        frame['prediction_time'] = pd.Timestamp('2026-04-02T01:00Z')
        result = augment_history(frame, events().iloc[:0], source_start_utc=starts.max())
        self.assertEqual(starts.max(), pd.Timestamp('2026-04-01T07:00Z'))
        self.assertEqual(result.iloc[0].history_window_incomplete, 1)
        frame['prediction_time'] = pd.Timestamp('2026-04-02T07:00Z')
        result = augment_history(frame, events().iloc[:0], source_start_utc=starts.max())
        self.assertEqual(result.iloc[0].history_window_incomplete, 0)
        self.assertIn('not a completeness guarantee', result.attrs['history_audit']['left_truncation_rule'])
        self.assertIn('prior-local-month', result.attrs['history_audit']['carry_in_limitation'])

    def test_variant_loader_and_explicit_feature_audit(self):
        frame = queries().iloc[[0]].copy()
        frame['downstream_airline'] = 'AA'
        airports = pd.DataFrame({'airport_id': [1], 'timezone': ['America/Los_Angeles']})
        ref = pd.DataFrame({'airline': ['AA'], 'tail_number': ['a'], 'aircraft_model': ['A320']})
        aliases = pd.DataFrame({'aircraft_model_raw': ['A320'], 'aircraft_family': ['A320'], 'aircraft_variant': ['A320-200']})
        def read(path, **kwargs):
            if str(path).endswith('airports.csv'):
                return airports
            if 'aircraft_reference' in str(path):
                return ref
            if 'aircraft_type_aliases' in str(path):
                return aliases
            return events()
        config = {'feature_group': 'G2_VARIANT', 'features': ['aircraft_variant', 'upstream_arr_delay_minutes']}
        with patch('scripts.build_features.load_base_features', return_value=frame), patch('scripts.build_features.pd.read_csv', side_effect=read):
            result = load_configured_features(config)
        audit = result.attrs['feature_audit']
        self.assertEqual(audit['feature_group'], 'G2_VARIANT')
        self.assertEqual(audit['features'], config['features'])
        self.assertEqual(audit['categorical_features'], ['aircraft_variant'])
        self.assertEqual(audit['numeric_features'], ['upstream_arr_delay_minutes'])
        self.assertEqual(result.attrs['history_audit']['source_start_utc'], '2026-04-01T07:00:00+00:00')

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
