"""Synthetic checks for date-scoped single-flight EDA."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import analyze_single_flights as eda


class SingleFlightEdaTests(unittest.TestCase):
    def _project(self, root):
        (root / "processed/D0003").mkdir(parents=True)
        (root / "references").mkdir()
        flights = pd.DataFrame([
            ["2026-04-01", "AA", "N1", 1, 2, 3, 800, 8, -10, -5, 0, 0, 0, 0],
            ["2026-05-02", "AA", "N1", 1, 2, 6, 900, 9, 20, 12, 20, 12, 0, 0],
            ["2026-06-03", "BB", "", 2, 1, 7, 1000, 10, "bad-cancel-target", 0, 0, 0, 1, 0],
            ["2026-07-04", "BB", "N2", 2, 1, 1, 1100, 11, 10, 15, 10, 15, 0, 1],
            # An August target that cannot be parsed in the default development run.
            ["2026-08-05", "AA", "N1", 1, 2, 3, 1200, 12, "BAD_AUGUST", "BAD_AUGUST", "BAD", "BAD", 0, 0],
        ], columns=["FlightDate", "Reporting_Airline", "Tail_Number", "OriginAirportID", "DestAirportID",
                    "DayOfWeek", "CRSDepTime", "scheduled_dep_hour", "DepDelay", "ArrDelay",
                    "DepDelayMinutes", "ArrDelayMinutes", "Cancelled", "Diverted"])
        flights.to_csv(root / "processed/flights_clean.csv.gz", index=False, compression="gzip")
        connections = pd.DataFrame({"airport_id": [1, 1, 2], "qa_cohort_month": ["2026-04", "2026-08", "2026-04"],
                                    "main_analysis_eligible": [True, True, False],
                                    "rq2_eligible": [True, True, False]})
        connections.to_csv(root / "processed/D0003/flight_connections.csv.gz", index=False, compression="gzip")
        pd.DataFrame({"airport_id": [1, 2], "airport_code": ["AAA", "BBB"],
                      "latitude": [40.0, 41.0], "longitude": [-70.0, -71.0]}).to_csv(
            root / "references/airports.csv", index=False)
        for year in (2024, 2025):
            pd.DataFrame({"airline": ["AA"], "tail_number": ["N1"], "aircraft_model": ["320-214"],
                          "manufacturer": ["Airbus"], "inventory_year": [str(year)]}).to_csv(
                root / f"references/aircraft_reference_{year}.csv", index=False)
        pd.DataFrame({"aircraft_model": ["320-214"], "aircraft_family": ["A320"],
                      "aircraft_variant": ["A320-214"]}).to_csv(
            root / "references/aircraft_model_mapping.csv", index=False)

    def test_development_filters_august_before_target_parsing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._project(root)
            run_dir = root / "experiments/runs/EDA001"
            with patch.object(eda, "ROOT", root):
                result = eda.run({}, run_dir)
            stats = pd.read_csv(run_dir / "tables/single_flight_delay_stats.csv")
            monthly = stats.loc[(stats.dimension == "month") & (stats.outcome == "DepDelay")]
            self.assertEqual(set(monthly.group_value), {"2026-04", "2026-05"})
            april = monthly.loc[monthly.group_value.eq("2026-04")].iloc[0]
            self.assertEqual(april.n, 1)
            self.assertEqual(april.negative, 1)
            self.assertEqual(april.zero, 0)
            self.assertEqual(result["single_flight_summary"]["scoped_flight_rows"], 4)
            self.assertEqual(result["single_flight_summary"]["normal_completed_rows"], 2)
            operations = pd.read_csv(run_dir / "tables/single_flight_operations.csv")
            status = operations.loc[operations.dimension.eq("month")].set_index("group_value")
            self.assertEqual(status.loc["2026-06", "cancelled"], 1)
            self.assertEqual(status.loc["2026-07", "diverted"], 1)
            airports = pd.read_csv(run_dir / "tables/airport_activity.csv").set_index("airport_id")
            self.assertEqual(airports.loc[1, "origin_flights"], 2)
            self.assertEqual(airports.loc[1, "accepted_connection_count"], 1)
            self.assertEqual(airports.loc[1, "rq2_eligible_connection_count"], 1)
            aircraft = pd.read_csv(run_dir / "tables/airline_aircraft_coverage.csv")
            self.assertEqual(set(aircraft.inventory_year), {2024, 2025})
            unknown = pd.read_csv(run_dir / "tables/airline_aircraft_models.csv")
            self.assertIn("unknown", set(unknown.aircraft_model))
            self.assertTrue((run_dir / "figures/monthly_delay_quantiles.png").is_file())
            summary = json.loads((run_dir / "tables/eda_complete_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "development")

    def test_final_must_be_explicit_and_named_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._project(root)
            run_dir = root / "experiments/runs/EDA002"
            with patch.object(eda, "ROOT", root):
                with self.assertRaises(ValueError):
                    eda.run({"mode": "unexpected"}, run_dir)
                eda.run({"mode": "development"}, run_dir)
                with self.assertRaises(FileExistsError):
                    eda.run({"mode": "development"}, run_dir)


if __name__ == "__main__":
    unittest.main()
