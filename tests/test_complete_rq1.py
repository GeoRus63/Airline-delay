"""Monthly calendars, immutable reference reuse, rates and tiny orchestration."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from scripts import rq1_standardization as standard
from scripts.complete_rq1 import monthly_archives, run
from tests.test_rq1_standardization import sample, protocol


class CompleteRQ1Tests(unittest.TestCase):
    def test_period_calendar_and_development_guard(self):
        source = sample()
        frozen = standard.freeze_reference(source, protocol())
        april = standard.evaluate(source, frozen, period=("2026-04-01T00:00Z", "2026-05-01T00:00Z"))
        self.assertEqual(len(april["calendar"]), 30)
        np.testing.assert_array_equal(april["bootstrap_day_counts"].sum(axis=1), np.full(60, 30))
        may = standard.evaluate(source, frozen, period=("2026-05-01T00:00Z", "2026-06-01T00:00Z"))
        self.assertEqual(len(may["calendar"]), 31)
        self.assertTrue(may["airports"].standardized_mean.isna().all())
        with self.assertRaises(ValueError):
            standard.evaluate(source, frozen, period=("2026-08-01T00:00Z", "2026-09-01T00:00Z"))

    def test_rates_cell_support_and_totals_identity(self):
        source = sample()
        frozen = standard.freeze_reference(source, protocol())
        result = standard.evaluate(source, frozen, period=("2026-04-01T00:00Z", "2026-05-01T00:00Z"))
        main = result["airports"].query("threshold == 'ge15'")
        np.testing.assert_allclose(main.standardized_reduction_share, .5)
        np.testing.assert_allclose(main.standardized_amplification_share, .5)
        np.testing.assert_allclose(main.standardized_unchanged_share, 0)
        np.testing.assert_allclose(main.standardized_reduction_share + main.standardized_amplification_share + main.standardized_unchanged_share, 1)
        np.testing.assert_allclose(main.s_net, main.s_plus - main.s_minus)
        cells = result["cells"].query("threshold == 'ge15'")
        self.assertEqual(cells.n.sum(), len(source))
        self.assertTrue(cells.loc[cells.n.eq(0), "mean_delta"].isna().all())
        self.assertTrue(cells.loc[cells.n.gt(0), "prediction_days"].eq(14).all())

    def test_monthly_reference_bytes_unchanged(self):
        source = sample()
        frozen = standard.freeze_reference(source, protocol())
        original = copy.deepcopy(frozen)
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            reference = path / "frozen.json"
            exact = json.dumps(frozen, ensure_ascii=False, indent=4).encode("utf-8")
            reference.write_bytes(exact)
            tables, summaries = monthly_archives(source, frozen, path / "monthly", frozen_path=reference)
            self.assertEqual([s["calendar_days"] for s in summaries], [30, 31, 30, 31])
            self.assertEqual(set(tables.month_utc), {"2026-04", "2026-05", "2026-06", "2026-07"})
            for month in tables.month_utc.unique():
                self.assertEqual((path / "monthly" / month / "frozen_reference.json").read_bytes(), exact)
            self.assertEqual(frozen, original)

    def test_tiny_complete_run_and_final_freeze(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = sample()
            source.to_pickle(root / "flight_connections.pkl")
            flights = pd.DataFrame({"OriginAirportID": [1, 1, 2], "DestAirportID": [2, 2, 1],
                                    "scheduled_dep_utc": pd.to_datetime(["2026-04-01"] * 3, utc=True)})
            flights.to_csv(root / "flights.csv", index=False)
            pd.DataFrame({"airport_id": [1, 2], "airport_code": ["A", "B"], "latitude": [35, 40], "longitude": [-100, -80]}).to_csv(root / "airports.csv", index=False)
            (root / "protocol.json").write_text(json.dumps(protocol()), encoding="utf-8")
            config = {"mode": "development", "connection_path": str(root / "flight_connections.csv.gz"),
                      "flights_path": str(root / "flights.csv"), "airport_reference_path": str(root / "airports.csv"),
                      "rq1_protocol_path": str(root / "protocol.json")}
            summary = run(config, root / "development")
            self.assertEqual(summary["network"]["bts_flights"], 3)
            self.assertEqual(summary["network"]["valid_connections"], len(source))
            frozen_path = root / "development/rq1_standardized/frozen_reference.json"
            original = frozen_path.read_bytes()
            final_config = dict(config, mode="final", rq1_frozen_reference_path=str(frozen_path))
            final = run(final_config, root / "final")
            self.assertEqual((root / "final/rq1_standardized/frozen_reference.json").read_bytes(), original)
            self.assertEqual(frozen_path.read_bytes(), original)
            self.assertEqual(final["standardized"]["calendar_days"], 153)
            self.assertEqual(final["monthly"][-1]["calendar_days"], 31)
            self.assertTrue((root / "final/figures/rq1_network_geographic.png").exists())
            self.assertTrue((root / "final/rq1_standardized/rq1_cells.csv").exists())
            with self.assertRaises(ValueError):
                run(dict(config, mode="final"), root / "invalid_final")


if __name__ == "__main__":
    unittest.main()
