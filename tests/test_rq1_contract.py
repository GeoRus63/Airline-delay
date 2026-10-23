import unittest

import pandas as pd

from scripts.analyze_rq1 import select_rq1, summarize_airports


class RQ1Contract(unittest.TestCase):
    def test_airport_aliases_are_one_stable_entity_and_ground_zero_is_retained(self):
        source = pd.DataFrame({
            'connection_id': ['a', 'b', 'sealed'], 'airport_id': [14027] * 3,
            'airport_code': ['PBI', 'DJT', 'DJT'],
            'upstream_actual_arr_utc': ['2026-04-01T10:00Z', '2026-05-01T10:00Z', '2026-08-01T10:00Z'],
            'downstream_dep_delay_minutes': [10, 30, 'sealed'],
            'upstream_arr_delay_minutes': [20, 20, 'sealed'],
            'scheduled_turnaround_minutes': [60] * 3,
            'actual_ground_minutes': [0, 30, 30], 'overnight_scheduled': [False] * 3,
        })
        selected, audit = select_rq1(source)
        result = summarize_airports(selected)
        self.assertEqual(audit['main_ground_zero_n'], 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['n'], 2)
        self.assertEqual(result.iloc[0].s_net, 0)
        self.assertEqual(result.iloc[0].airport_code, 'DJT|PBI')


if __name__ == '__main__':
    unittest.main()
