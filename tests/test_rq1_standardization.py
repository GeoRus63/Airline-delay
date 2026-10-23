"""Small synthetic checks of shared support and paired date-block inference."""
import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.rq1_standardization import (evaluate, freeze_reference, load_protocol,
                                         moving_block_counts, moving_block_indices, network_tables, run)


def protocol():
    value = load_protocol()
    value.update(airport_min_main_rows=5, airport_min_prediction_days=5,
                 cell_min_rows=5, cell_min_prediction_days=5, fallback_airport_limits=[2])
    value["bootstrap"] = dict(value["bootstrap"], replicates=60, min_valid_replicates=50)
    return value


def sample(include_sparse=False):
    rows = []
    for date in pd.date_range("2026-04-01", periods=14, tz="UTC"):
        for airport, counts, changes in [(1, (1, 3), (-10, 20)), (2, (3, 1), (-5, 10))]:
            for delay, count, change in zip((20, 40), counts, changes):
                for _ in range(count):
                    rows.append(dict(connection_id=str(len(rows)), airport_id=airport, airport_code=f"A{airport}",
                                     upstream_actual_arr_utc=date + pd.Timedelta(hours=12),
                                     upstream_arr_delay_minutes=delay, downstream_dep_delay_minutes=delay + change,
                                     scheduled_turnaround_minutes=60., actual_ground_minutes=40., overnight_scheduled=False))
        if include_sparse:
            rows.append(dict(rows[-1], connection_id=str(len(rows)), airport_id=3,
                             upstream_arr_delay_minutes=20, downstream_dep_delay_minutes=10))
    return pd.DataFrame(rows)


class StandardizationTests(unittest.TestCase):
    def test_fixed_weights_and_cumulative_identity(self):
        source = sample()
        frozen = freeze_reference(source, protocol())
        self.assertTrue(frozen["supported"])
        np.testing.assert_allclose(frozen["references"]["ge15"]["weights"], [.5, .5])
        result = evaluate(source, frozen)
        main = result["airports"].query("threshold == 'ge15'").set_index("airport_id")
        self.assertAlmostEqual(main.loc[1, "standardized_mean"], 5.)
        self.assertAlmostEqual(main.loc[2, "standardized_mean"], 2.5)
        self.assertNotEqual(main.loc[1, "mean_delta"], main.loc[1, "standardized_mean"])
        np.testing.assert_allclose(main.s_net, main.s_plus - main.s_minus)
        paired = result["pairwise_differences"].query("threshold == 'ge15'").iloc[0]
        self.assertAlmostEqual(paired.mean_difference, 2.5)
        self.assertAlmostEqual(paired.ci_low, 2.5)
        self.assertAlmostEqual(paired.ci_high, 2.5)

    def test_support_fallback_uses_counts(self):
        frozen = freeze_reference(sample(True), protocol())
        self.assertFalse(frozen["selection_attempts"][0]["passed"])
        self.assertEqual(frozen["airport_ids"], [1, 2])
        altered = sample(True)
        altered["downstream_dep_delay_minutes"] = 99999
        self.assertEqual(frozen, freeze_reference(altered, protocol()))

    def test_unknown_and_missing_frozen_cells_are_na(self):
        frozen = freeze_reference(sample(), protocol())
        before = copy.deepcopy(frozen)
        source = sample(True)
        source = source.loc[~(source.airport_id.eq(1) & source.upstream_arr_delay_minutes.eq(40))]
        result = evaluate(source, frozen)["airports"].query("threshold == 'ge15'").set_index("airport_id")
        self.assertTrue(pd.isna(result.loc[1, "standardized_mean"]))
        self.assertTrue(pd.isna(result.loc[3, "rank"]))
        self.assertTrue(pd.isna(result.loc[3, "standardized_mean"]))
        self.assertTrue(result.loc[2, "supported"])
        self.assertEqual(before, frozen)

    def test_calendar_blocks_and_common_draws(self):
        indexes = moving_block_indices(20, replicates=10, block_days=7, seed=42)
        for start, end in [(0, 7), (7, 14), (14, 20)]:
            np.testing.assert_array_equal(np.diff(indexes[:, start:end], axis=1), np.ones((10, end-start-1)))
        draws = moving_block_counts(20, dict(replicates=10, block_days=7, seed=42))
        np.testing.assert_array_equal(draws.sum(axis=1), np.full(10, 20))
        np.testing.assert_array_equal(draws, moving_block_counts(20, dict(replicates=10, block_days=7, seed=42)))
        source = sample()
        result = evaluate(source, freeze_reference(source, protocol()))
        np.testing.assert_allclose(result["replicates"]["gt0"], result["replicates"]["ge15"], equal_nan=True)

    def test_threshold_conditioning_on_fixed_master_reference(self):
        source = sample()
        low = source.copy()
        low["connection_id"] = "low_" + low.connection_id
        low["upstream_arr_delay_minutes"] = 2
        low["downstream_dep_delay_minutes"] = 0
        combined = pd.concat([source, low], ignore_index=True)
        frozen = freeze_reference(combined, protocol())
        self.assertEqual(frozen["airport_ids"], [1, 2])
        self.assertEqual(len(frozen["references"]["gt0"]["cells"]), 3)
        self.assertEqual(len(frozen["references"]["ge15"]["cells"]), 2)
        np.testing.assert_allclose(frozen["references"]["ge15"]["weights"], [.5, .5])
        result = evaluate(combined, frozen)["airports"].query("threshold == 'gt0'").set_index("airport_id")
        self.assertAlmostEqual(result.loc[1, "standardized_mean"], 1.5)
        self.assertAlmostEqual(result.loc[2, "standardized_mean"], .25)

    def test_negative_sign_and_network_units(self):
        source = sample()
        source.loc[source.airport_id.eq(2), "downstream_dep_delay_minutes"] = source.loc[source.airport_id.eq(2), "upstream_arr_delay_minutes"] - 2
        result = evaluate(source, freeze_reference(source, protocol()))
        main = result["airports"].query("threshold == 'ge15'").set_index("airport_id")
        self.assertAlmostEqual(main.loc[2, "standardized_mean"], -2)
        self.assertAlmostEqual(main.loc[2, "ci_high"], -2)
        with TemporaryDirectory() as temporary:
            reference = Path(temporary) / "airports.csv"
            pd.DataFrame({"airport_id": [1, 2, 3], "airport_code": ["A", "B", "C"], "latitude": [1, 2, 3], "longitude": [1, 2, 3]}).to_csv(reference, index=False)
            flights = pd.DataFrame({"OriginAirportID": [1, 1, 2], "DestAirportID": [2, 2, 1],
                                    "scheduled_dep_utc": pd.to_datetime(["2026-04-01"] * 3, utc=True)})
            nodes, edges = network_tables(flights, source, result["airports"], reference_path=reference)
            self.assertEqual(edges.flight_count.sum(), 3)
            self.assertEqual(nodes.valid_connection_count.sum(), len(source))
            self.assertTrue(pd.isna(nodes.set_index("airport_id").loc[3, "standardized_mean"]))

    def test_archive_and_final_requires_freeze(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                run(sample(), root / "final", development=False)
            summary = run(sample(), root / "development", protocol=protocol())
            self.assertEqual(summary["selected_airports"], 2)
            self.assertTrue((root / "development/frozen_reference.json").is_file())
            with np.load(root / "development/paired_bootstrap.npz") as saved:
                self.assertEqual(saved["ge15"].shape, (60, 2))


if __name__ == "__main__":
    unittest.main()
