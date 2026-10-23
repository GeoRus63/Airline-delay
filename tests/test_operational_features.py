import unittest
import numpy as np
import pandas as pd

from scripts.operational_features import augment_aircraft, augment_airport_operations, source_calendar_starts, _event_windows


class OperationalFeatures(unittest.TestCase):
    def test_empty_model_does_not_mask_unique_known_model(self):
        frame = pd.DataFrame({'downstream_airline': ['AA'], 'tail_number': ['N1']}, index=[99])
        ref = pd.DataFrame({'airline': ['AA', 'AA'], 'tail_number': ['N1', 'N1'], 'aircraft_model': [None, 'A320']})
        aliases = pd.DataFrame({'aircraft_model_raw': ['A320'], 'aircraft_family': ['A320'], 'aircraft_variant': ['A320-200']})
        result = augment_aircraft(frame, ref, aliases)
        self.assertEqual(result.loc[99, 'aircraft_family'], 'A320')

    def test_float_query_id_joins_integer_event_id(self):
        queries = pd.DataFrame({'airport_id': [1.0], 'prediction_time': pd.to_datetime(['2026-05-01T10:00Z'])}, index=[999])
        source = pd.DataFrame({'OriginAirportID': [1], 'actual_dep_utc': pd.to_datetime(['2026-05-01T09:30Z'])})
        counts, _ = _event_windows(queries, source, 'OriginAirportID', 'actual_dep_utc', 60, 0)
        self.assertEqual(counts.tolist(), [1])

    def test_airport_local_month_boundary(self):
        t = pd.Timestamp('2026-04-01T07:30Z')
        frame = pd.DataFrame({'airport_id': ['1', '2'], 'prediction_time': [t, t]}, index=[92, 5])
        source = pd.DataFrame({'flight_id': ['a'], 'OriginAirportID': [1], 'DestAirportID': [1],
                               'scheduled_dep_utc': [t], 'scheduled_arr_utc': [t], 'actual_dep_utc': [t],
                               'actual_dep_clock_error': [0.], 'DepDelay': [0.], 'DepDelayMinutes': [0.], 'duplicate_key': [False]})
        starts = source_calendar_starts(pd.DataFrame({'airport_id': [1, 2], 'timezone': ['America/Los_Angeles', 'America/New_York']}))
        result = augment_airport_operations(frame, source, source_start_utc=starts)
        self.assertEqual(result.airport_history_window_incomplete.tolist(), [1, 0])
        self.assertIn('zero does not guarantee', result.attrs['airport_operations_audit']['left_truncation_rule'])

    def test_inventory_join_has_unknowns_and_no_nested_predictor_decision(self):
        frame = pd.DataFrame({'downstream_airline': ['AA', 'AA', 'XX'], 'tail_number': ['N1', 'N2', 'N1']})
        ref = pd.DataFrame({'airline': ['AA'], 'tail_number': ['N1'], 'aircraft_model': ['A320']})
        aliases = pd.DataFrame({'aircraft_model_raw': ['A320'], 'aircraft_family': ['A320 family'],
                               'aircraft_variant': ['A320-214']})
        result = augment_aircraft(frame, ref, aliases)
        self.assertEqual(result.aircraft_family.tolist()[0], 'A320 family')
        self.assertEqual(result.aircraft_family.notna().sum(), 1)
        self.assertEqual(result.attrs['aircraft_audit']['rows'], 3)
        bad = pd.concat([ref, ref.assign(aircraft_model='B737')], ignore_index=True)
        with self.assertRaises(ValueError):
            augment_aircraft(frame, bad, aliases)

    def test_airport_windows_exclude_future_outcomes_and_include_plan_cancellations(self):
        base = pd.Timestamp('2026-04-02T10:00Z')
        query = pd.DataFrame({'airport_id': ['1', '2'], 'prediction_time': [base, base]})
        dep = [base-pd.Timedelta(minutes=m) for m in [60, 1, 0, -1, 61]]
        schedule = [base + pd.Timedelta(minutes=m) for m in [0, 59, 60, -1, 30]]
        source = pd.DataFrame({'flight_id': list('abcde'), 'OriginAirportID': [1]*5, 'DestAirportID': [2]*5,
                               'scheduled_dep_utc': schedule, 'scheduled_arr_utc': schedule,
                               'actual_dep_utc': dep, 'actual_dep_clock_error': [0.]*5,
                               'DepDelay': [10., 20., 100., 200., 300.],
                               'DepDelayMinutes': [10., 20., 100., 200., 300.],
                               'duplicate_key': [False]*5, 'Cancelled': [False]*4+[True],
                               'Diverted': [True]*5, 'time_valid': [False]*5})
        result = augment_airport_operations(query, source)
        self.assertEqual(result.airport_departures_60m_n.tolist(), [2, 0])
        self.assertEqual(result.airport_departures_60m_delay_mean.iloc[0], 15)
        self.assertTrue(np.isnan(result.airport_departures_60m_delay_mean.iloc[1]))
        self.assertEqual(result.airport_planned_departures_next60m_n.tolist(), [3, 0])
        self.assertEqual(result.airport_planned_arrivals_next60m_n.tolist(), [0, 3])
        changed = source.copy()
        changed.loc[2:, 'DepDelayMinutes'] = 10000
        changed['Diverted'], changed['time_valid'] = False, True
        again = augment_airport_operations(query, changed)
        pd.testing.assert_frame_equal(result, again)

    def test_left_window_is_flagged_and_ambiguous_source_rejected(self):
        t = pd.Timestamp('2026-04-01T00:30Z')
        frame = pd.DataFrame({'airport_id': ['1'], 'prediction_time': [t]})
        source = pd.DataFrame({'flight_id': ['a'], 'OriginAirportID': [1], 'DestAirportID': [1],
                               'scheduled_dep_utc': [t], 'scheduled_arr_utc': [t], 'actual_dep_utc': [t],
                               'actual_dep_clock_error': [0.], 'DepDelay': [0.], 'DepDelayMinutes': [0.],
                               'duplicate_key': [False]})
        result = augment_airport_operations(frame, source)
        self.assertEqual(result.airport_history_window_incomplete.iloc[0], 1)
        with self.assertRaises(ValueError):
            augment_airport_operations(frame, pd.concat([source, source]))


if __name__ == '__main__':
    unittest.main()
