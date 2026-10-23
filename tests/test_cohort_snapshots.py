"""Structural isolation checks on tiny synthetic flight histories."""
import unittest

import pandas as pd

from scripts.rebuild_cohorts import rebuild


def flights(times):
    dep = pd.to_datetime(times, utc=True)
    n = len(dep)
    return pd.DataFrame({
        "flight_id": [f"f{i}" for i in range(n)], "Tail_Number": ["TEST"] * n,
        "Reporting_Airline": ["XX"] * n, "FlightDate": dep.tz_localize(None).normalize(),
        "Origin": [f"A{i}" for i in range(n)], "Dest": [f"A{i+1}" for i in range(n)],
        "OriginAirportID": list(range(n)), "DestAirportID": list(range(1, n+1)),
        "origin_tz": ["UTC"] * n, "dest_tz": ["UTC"] * n,
        "scheduled_dep_utc": dep, "scheduled_arr_utc": dep + pd.Timedelta(minutes=30),
        "actual_dep_utc": dep, "actual_arr_utc": dep + pd.Timedelta(minutes=30),
        "scheduled_dep_hour": dep.hour, "ArrDelay": [0.] * n, "ArrDelayMinutes": [0.] * n,
        "DepDelay": [0.] * n, "DepDelayMinutes": [0.] * n, "DayOfWeek": dep.dayofweek + 1,
        "Distance": [100.] * n, "CRSElapsedTime": [30.] * n,
        "normal_operation": [True] * n, "time_valid": [True] * n, "duplicate_key": [False] * n,
    })


class MonthlyCohortTests(unittest.TestCase):
    def test_future_corruption_does_not_change_mature_pairs(self):
        base = flights(["2026-04-10 09:00Z", "2026-04-10 10:00Z", "2026-08-10 09:00Z"])
        before, _ = rebuild(base)
        broken = base.copy()
        broken.loc[2, "scheduled_dep_utc"] = pd.NaT
        broken.loc[2, "actual_dep_utc"] = pd.Timestamp("2026-04-10 09:10Z")
        after, audit = rebuild(broken)
        cols = [c for c in before if c.startswith("check_")] + ["rq2_eligible", "first_exclusion_reason"]
        pd.testing.assert_series_equal(before.iloc[0][cols], after.iloc[0][cols])
        self.assertTrue(after.iloc[0].rq2_eligible)
        self.assertTrue(after.iloc[0].label_time < pd.Timestamp("2026-05-01", tz="UTC"))
        self.assertTrue(audit.actual_events_at_or_after_cutoff.eq(0).all())

    def test_appended_future_missing_plan_does_not_change_pair(self):
        base = flights(["2026-04-10 09:00Z", "2026-04-10 10:00Z"])
        extended = flights(["2026-04-10 09:00Z", "2026-04-10 10:00Z", "2026-08-10 09:00Z"])
        extended.loc[2, "scheduled_dep_utc"] = pd.NaT
        original, _ = rebuild(base)
        result, _ = rebuild(extended)
        self.assertEqual(original.iloc[0].first_exclusion_reason, result.iloc[0].first_exclusion_reason)
        self.assertTrue(result.iloc[0].rq2_eligible)

    def test_cancelled_middle_is_not_skipped(self):
        frame = flights(["2026-04-10 09:00Z", "2026-04-10 10:00Z", "2026-04-10 11:00Z"])
        frame.loc[1, "normal_operation"] = False
        result, _ = rebuild(frame)
        self.assertEqual(result.connection_id.tolist(), ["f0->f1", "f1->f2"])
        self.assertFalse(result.main_analysis_eligible.any())

    def test_schedule_tie_rejected(self):
        frame = flights(["2026-04-10 09:00Z", "2026-04-10 09:00Z"])
        result, _ = rebuild(frame)
        self.assertFalse(result.main_analysis_eligible.any())
        self.assertFalse(result.check_unique_schedule.any())

    def test_boundary_endpoint_retained_without_flag_propagation(self):
        frame = flights(["2026-07-31 22:00Z", "2026-08-01 00:10Z", "2026-08-03 09:00Z"])
        frame.loc[2, "scheduled_dep_utc"] = pd.NaT
        result, _ = rebuild(frame)
        pair = result.iloc[0]
        self.assertTrue(pair.downstream_outside_qa_window)
        self.assertTrue(pair.rq2_eligible)
        self.assertEqual(pair.qa_cohort_month, "2026-07")
        self.assertGreater(pair.label_time, pair.qa_cutoff_utc)
        self.assertFalse(pair.full_period_audit_main_analysis_eligible)

    def test_equal_arrival_departure_not_rq2(self):
        frame = flights(["2026-04-10 09:00Z", "2026-04-10 09:30Z"])
        result, _ = rebuild(frame)
        self.assertTrue(result.iloc[0].main_analysis_eligible)
        self.assertFalse(result.iloc[0].rq2_eligible)


if __name__ == "__main__":
    unittest.main()
