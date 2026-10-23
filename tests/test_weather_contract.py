"""Small synthetic checks for strict weather as-of and NOAA QC contracts."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

from scripts.prepare_weather import (
    _nullable_weather_flag,
    match_metar_asof,
    noaa_qc_is_eligible,
    noaa_precip_status,
    prepare_metar,
    prepare_taf,
    prepare_noaa_frame,
    run as run_weather_preparation,
    select_taf_asof,
    taf_availability_proxy,
    taf_query_time,
)


class WeatherContractTests(unittest.TestCase):
    def _prepare_missing_text_fixture(self):
        ids = ["202604010800-KAAA-FTUS00-TAFAAA", "202604010900-KAAA-FTUS00-TAFAAA-AAA",
               "202604011000-KAAA-FTUS00-TAFAAA"]
        times = ["2026-04-01 08:00", "2026-04-01 09:00", "2026-04-01 10:00"]
        raw = pd.DataFrame({"airport_id": [1] * 3, "station": ["KAAA"] * 3,
            "product_id": ids, "valid": times, "fx_valid": times,
            "fx_valid_end": [None] * 3, "ftype": ["Observation"] * 3,
            "raw": ["20010KT"] * 3, "skyc": ["[]"] * 3, "skyl": ["[]"] * 3,
            "presentwx": ["[]"] * 3, "sknt": [10, 99, 20], "drct": [200] * 3,
            "gust": [None] * 3, "visibility": [6] * 3, "is_amendment": [False, True, False]})
        validity = pd.DataFrame({"product_id": [ids[0], ids[2]], "station": ["KAAA"] * 2,
            "issue_time": [times[0], times[2]], "valid_from": [times[0], times[2]],
            "valid_to": ["2026-04-01 18:00"] * 2, "status": ["normal"] * 2,
            "raw_header": ["TAF KAAA 010800Z 0108/0118", "TAF KAAA 011000Z 0110/0118"]})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "references").mkdir()
            validity.to_csv(root / "references/taf_validity.csv", index=False)
            with patch("scripts.prepare_weather.ROOT", root), \
                    patch("scripts.prepare_weather.read_batches", return_value=raw), \
                    patch("scripts.prepare_weather.station_lookup", return_value={"KAAA": 1}):
                return prepare_taf()

    def test_missing_full_text_product_blocks_old_forecast(self):
        segments, bulletins = self._prepare_missing_text_fixture()
        unknown = bulletins.loc[bulletins.status.eq("unverified")].iloc[0]
        self.assertEqual(unknown.archive_issue_time, pd.Timestamp("2026-04-01T09:00Z"))
        self.assertTrue(pd.isna(unknown.body_issue_time))
        self.assertTrue(pd.isna(unknown.availability_time))
        self.assertFalse(segments.loc[segments.product_id.eq(unknown.product_id), "interval_usable"].any())
        before = select_taf_asof(segments, bulletins, 1, "2026-04-01T08:30Z", "2026-04-01T11:00Z")
        self.assertEqual(before["status"], "matched")
        for allow in [False, True]:
            blocked = select_taf_asof(segments, bulletins, 1, "2026-04-01T09:30Z", "2026-04-01T11:00Z",
                                      allow_unverified_revisions=allow)
            self.assertEqual(blocked["status"], "unverified_issue_time")
            self.assertTrue(blocked["segments"].empty)

    def test_normal_product_after_missing_full_text_restores_forecast(self):
        segments, bulletins = self._prepare_missing_text_fixture()
        restored = select_taf_asof(segments, bulletins, 1, "2026-04-01T10:30Z", "2026-04-01T11:00Z")
        self.assertEqual(restored["status"], "matched")
        self.assertEqual(restored["segments"].wind_knots.tolist(), [20])

    def test_taf_query_time_uses_later_schedule_or_prediction(self):
        flights = pd.DataFrame({
            "scheduled_dep_utc": pd.to_datetime(["2026-04-01T10:00Z", "2026-04-01T08:00Z"], utc=True),
            "prediction_time": pd.to_datetime(["2026-04-01T09:00Z", "2026-04-01T09:00Z"], utc=True),
        })
        self.assertEqual(taf_query_time(flights).tolist(), list(pd.to_datetime(
            ["2026-04-01T10:00Z", "2026-04-01T09:00Z"], utc=True)))

    def test_taf_availability_proxy_uses_later_timestamp_and_requires_both(self):
        self.assertEqual(taf_availability_proxy("2026-04-01T08:05Z", "2026-04-01T08:10Z"),
                         pd.Timestamp("2026-04-01T08:10Z"))
        self.assertTrue(pd.isna(taf_availability_proxy(pd.NaT, "2026-04-01T08:10Z")))

    def test_metar_missing_and_trace_are_distinct(self):
        raw = pd.DataFrame({
            "airport_id": [1, 1, 1], "station": ["KAAA"] * 3,
            "valid": ["2026-04-01 09:00", "2026-04-01 09:10", "2026-04-01 09:20"],
            "tmpf": ["50", "50", "50"], "dwpf": ["40", "40", "40"],
            "relh": ["", "", ""], "drct": ["180", "180", "180"],
            "sknt": ["5", "5", "5"], "gust": ["M"] * 3,
            "vsby": ["10", "10", "10"], "p01i": ["M", "T", "0.00"],
            "skyc1": ["M", "M", "CLR"], "skyc2": ["M"] * 3,
            "skyc3": ["M"] * 3, "skyc4": ["M"] * 3,
            "skyl1": ["M"] * 3, "skyl2": ["M"] * 3,
            "skyl3": ["M"] * 3, "skyl4": ["M"] * 3,
            "wxcodes": ["M", "-SN", "M"], "metar": ["METAR KAAA"] * 3,
        })
        with patch("scripts.prepare_weather.read_batches", return_value=raw):
            prepared = prepare_metar().sort_values("precip_measurement_status").reset_index(drop=True)
        statuses = dict(zip(prepared.precip_measurement_status, prepared.precip_flag.astype(object)))
        self.assertEqual(statuses["trace"], 1)
        self.assertEqual(statuses["measured"], 0)
        self.assertTrue(pd.isna(statuses["missing"]))
        snow = prepared.loc[prepared.snow_flag.eq(1)]
        self.assertEqual(len(snow), 1)

    def test_missing_taf_weather_codes_are_not_encoded_as_clear_weather(self):
        self.assertTrue(pd.isna(_nullable_weather_flag([], False, ("TS",))))
        self.assertEqual(_nullable_weather_flag([], True, ("TS",)), 0)
        self.assertEqual(_nullable_weather_flag(["-TSRA"], True, ("TS",)), 1)

    def test_metar_correction_is_excluded_and_untimed_duplicates_are_ambiguous(self):
        raw = pd.DataFrame({
            "airport_id": [1, 1, 1, 1], "station": ["KAAA"] * 4,
            "valid": ["2026-04-01 09:00", "2026-04-01 09:00", "2026-04-01 10:00", "2026-04-01 11:00"],
            "tmpf": ["50", "51", "50", "50"], "dwpf": ["40"] * 4,
            "relh": ["M"] * 4, "drct": ["180"] * 4, "sknt": ["5"] * 4,
            "gust": ["M"] * 4, "vsby": ["10"] * 4, "p01i": ["0.00"] * 4,
            "skyc1": ["CLR"] * 4, "skyc2": ["M"] * 4, "skyc3": ["M"] * 4,
            "skyc4": ["M"] * 4, "skyl1": ["M"] * 4, "skyl2": ["M"] * 4,
            "skyl3": ["M"] * 4, "skyl4": ["M"] * 4,
            "wxcodes": ["M"] * 4,
            "metar": ["KAAA 010900Z 18005KT", "KAAA 010900Z 19005KT",
                      "KAAA 011000Z COR 18005KT", "KAAA 011100Z 18005KT"],
        })
        with patch("scripts.prepare_weather.read_batches", return_value=raw):
            prepared = prepare_metar()
        self.assertEqual(len(prepared), 2)
        ambiguous = prepared.loc[prepared.weather_time.eq(pd.Timestamp("2026-04-01T09:00Z"))].iloc[0]
        self.assertTrue(ambiguous.metar_ambiguous)
        self.assertTrue(pd.isna(ambiguous.wind_knots))
        self.assertFalse(prepared.weather_time.eq(pd.Timestamp("2026-04-01T10:00Z")).any())
        self.assertTrue(pd.isna(prepared.loc[prepared.weather_time.eq(pd.Timestamp("2026-04-01T11:00Z")),
                                            "thunder_flag"]).all())

    def test_metar_match_preserves_input_order_and_normalizes_airport_ids(self):
        observations = pd.DataFrame({
            "airport_id": [1], "weather_time": pd.to_datetime(["2026-04-01T09:00Z"], utc=True),
            "available_at": pd.to_datetime(["2026-04-01T09:05Z"], utc=True),
            "wind_knots": [8.0], "metar_ambiguous": [False],
        })
        flights = pd.DataFrame({
            "flight_id": ["first", "second"], "airport_id": ["1", 1.0],
            "prediction_time": pd.to_datetime(["2026-04-01T09:05Z", "2026-04-01T09:06Z"], utc=True),
        }, index=[7, 2])
        result = match_metar_asof(flights, observations)
        self.assertEqual(result.flight_id.tolist(), ["first", "second"])
        self.assertEqual(result.index.tolist(), [7, 2])
        self.assertEqual(result.wind_knots.tolist(), [8.0, 8.0])

    def test_metar_asof_lag_and_age_boundary(self):
        observations = pd.DataFrame({
            "airport_id": [1, 1],
            "weather_time": pd.to_datetime(["2026-04-01T09:00Z", "2026-04-01T08:00Z"], utc=True),
            "available_at": pd.to_datetime(["2026-04-01T09:05Z", "2026-04-01T08:05Z"], utc=True),
            "wind_knots": [12.0, 7.0],
            "precip_measurement_status": ["missing", "trace"],
            "precip_flag": pd.array([pd.NA, 1], dtype="Int8"),
        })
        flights = pd.DataFrame({
            "flight_id": ["before_lag", "latest", "age_limit", "too_old"],
            "airport_id": [1, 1, 1, 1],
            "prediction_time": pd.to_datetime([
                "2026-04-01T09:04Z", "2026-04-01T09:05Z", "2026-04-01T11:00Z", "2026-04-01T11:06Z",
            ], utc=True),
        })
        matched = match_metar_asof(flights, observations).set_index("flight_id")
        self.assertEqual(matched.loc["before_lag", "weather_time"], pd.Timestamp("2026-04-01T08:00Z"))
        self.assertNotEqual(matched.loc["before_lag", "weather_time"], pd.Timestamp("2026-04-01T09:00Z"))
        self.assertEqual(matched.loc["latest", "wind_knots"], 12.0)
        self.assertEqual(matched.loc["age_limit", "metar_age_minutes"], 120.0)
        self.assertEqual(matched.loc["too_old", "metar_missing"], 1)
        self.assertTrue(pd.isna(matched.loc["too_old", "wind_knots"]))
        self.assertTrue(pd.isna(matched.loc["latest", "precip_flag"]))

    def test_metar_missing_reason_separates_conflict_stale_and_no_observation(self):
        observations = pd.DataFrame({
            "airport_id": [1, 2],
            "weather_time": pd.to_datetime(["2026-04-01T09:00Z", "2026-04-01T06:00Z"], utc=True),
            "available_at": pd.to_datetime(["2026-04-01T09:05Z", "2026-04-01T06:05Z"], utc=True),
            "wind_knots": [np.nan, 7.0], "metar_ambiguous": [True, False],
        })
        flights = pd.DataFrame({
            "flight_id": ["conflict", "stale", "no_obs"],
            "airport_id": [1, 2, 3],
            "prediction_time": pd.to_datetime([
                "2026-04-01T09:06Z", "2026-04-01T08:06Z", "2026-04-01T09:00Z",
            ], utc=True),
        })
        result = match_metar_asof(flights, observations).set_index("flight_id")
        self.assertEqual(result.loc["conflict", "metar_missing_reason"], "conflicting_reports")
        self.assertEqual(result.loc["stale", "metar_missing_reason"], "stale_observation")
        self.assertEqual(result.loc["no_obs", "metar_missing_reason"], "no_observation")

    def test_taf_latest_cancel_or_nil_never_falls_back(self):
        segments = pd.DataFrame({
            "airport_id": [1, 1, 1],
            "product_id": ["routine", "cancel", "routine"],
            "station": ["KAAA"] * 3,
            "status": ["normal", "cancelled", "normal"],
            "message_type": ["routine", "routine", "routine"],
            "availability_time": pd.to_datetime([
                "2026-04-01T08:00Z", "2026-04-01T09:00Z", "2026-04-01T08:00Z",
            ], utc=True),
            "archive_issue_time": pd.to_datetime([
                "2026-04-01T08:00Z", "2026-04-01T09:00Z", "2026-04-01T08:00Z",
            ], utc=True),
            "body_issue_time": pd.to_datetime([
                "2026-04-01T08:00Z", "2026-04-01T09:00Z", "2026-04-01T08:00Z",
            ], utc=True),
            "availability_verified": [False] * 3,
            "interval_usable": [True, False, True],
            "segment_start": pd.to_datetime(["2026-04-01T08:00Z"] * 3, utc=True),
            "segment_end": pd.to_datetime(["2026-04-01T12:00Z"] * 3, utc=True),
        })
        bulletins = segments.drop_duplicates(["product_id", "station"])
        # Cancellation/NIL may have no parsed forecast segments in IEM's CSV.
        forecast_segments = segments.loc[segments.product_id.eq("routine")]
        result = select_taf_asof(forecast_segments, bulletins, 1,
                                 "2026-04-01T10:00Z", "2026-04-01T10:30Z")
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["segments"].empty)

        segments.loc[1, "status"] = "nil"
        bulletins = segments.drop_duplicates(["product_id", "station"])
        result = select_taf_asof(forecast_segments, bulletins, 1,
                                 "2026-04-01T10:00Z", "2026-04-01T10:30Z")
        self.assertEqual(result["status"], "nil")
        self.assertTrue(result["segments"].empty)

    def test_taf_same_time_distinct_bulletins_are_ambiguous(self):
        rows = pd.DataFrame({
            "airport_id": [1, 1], "product_id": ["routine", "cancel"],
            "station": ["KAAA", "KAAA"], "status": ["normal", "cancelled"],
            "message_type": ["routine", "routine"],
            "availability_time": pd.to_datetime(["2026-04-01T09:00Z"] * 2, utc=True),
            "archive_issue_time": pd.to_datetime(["2026-04-01T09:00Z"] * 2, utc=True),
            "body_issue_time": pd.to_datetime(["2026-04-01T09:00Z"] * 2, utc=True),
            "availability_verified": [False, False], "interval_usable": [True, False],
            "segment_start": pd.to_datetime(["2026-04-01T08:00Z"] * 2, utc=True),
            "segment_end": pd.to_datetime(["2026-04-01T12:00Z"] * 2, utc=True),
        })
        result = select_taf_asof(rows.iloc[:1], rows, 1,
                                 "2026-04-01T10:00Z", "2026-04-01T10:30Z")
        self.assertEqual(result["status"], "ambiguous_bulletin")
        self.assertTrue(result["segments"].empty)

    def test_conflicting_duplicate_product_status_blocks_asof_forecast(self):
        rows = pd.DataFrame({
            "airport_id": [1, 1], "product_id": ["routine", "conflict"],
            "station": ["KAAA", "KAAA"], "status": ["normal", "ambiguous"],
            "message_type": ["routine", "routine"],
            "availability_time": pd.to_datetime(["2026-04-01T08:00Z", "2026-04-01T09:00Z"], utc=True),
            "archive_issue_time": pd.to_datetime(["2026-04-01T08:00Z", "2026-04-01T09:00Z"], utc=True),
            "body_issue_time": pd.to_datetime(["2026-04-01T08:00Z", "2026-04-01T09:00Z"], utc=True),
            "availability_verified": [False, False], "interval_usable": [True, False],
            "segment_start": pd.to_datetime(["2026-04-01T08:00Z"] * 2, utc=True),
            "segment_end": pd.to_datetime(["2026-04-01T12:00Z"] * 2, utc=True),
        })
        result = select_taf_asof(rows.iloc[:1], rows, 1,
                                 "2026-04-01T10:00Z", "2026-04-01T10:30Z")
        self.assertEqual(result["status"], "ambiguous_bulletin")
        self.assertEqual(result["bulletin"]["product_id"], "conflict")
        self.assertTrue(result["segments"].empty)

    def test_unverified_amendment_blocks_fallback_by_default(self):
        segments = pd.DataFrame({
            "airport_id": [1, 1], "product_id": ["routine", "amendment"],
            "station": ["KAAA", "KAAA"], "status": ["normal", "normal"],
            "message_type": ["routine", "amended"],
            "availability_time": pd.to_datetime(["2026-04-01T08:00Z", "2026-04-01T09:00Z"], utc=True),
            "archive_issue_time": pd.to_datetime(["2026-04-01T08:00Z", "2026-04-01T09:00Z"], utc=True),
            "body_issue_time": pd.to_datetime(["2026-04-01T08:00Z", "2026-04-01T09:00Z"], utc=True),
            "availability_verified": [False, False],
            "interval_usable": [True, True],
            "segment_start": pd.to_datetime(["2026-04-01T08:00Z"] * 2, utc=True),
            "segment_end": pd.to_datetime(["2026-04-01T12:00Z"] * 2, utc=True),
        })
        bulletins = segments.drop_duplicates(["product_id", "station"])
        blocked = select_taf_asof(segments, bulletins, 1,
                                  "2026-04-01T10:00Z", "2026-04-01T10:30Z")
        self.assertEqual(blocked["status"], "disabled_revision")
        self.assertTrue(blocked["segments"].empty)
        assumed = select_taf_asof(segments, bulletins, 1,
                                  "2026-04-01T10:00Z", "2026-04-01T10:30Z",
                                  allow_unverified_revisions=True)
        self.assertEqual(assumed["status"], "matched")
        self.assertEqual(assumed["segments"].product_id.tolist(), ["amendment"])

    def test_taf_untimed_bulletin_blocks_fallback(self):
        segments = pd.DataFrame({
            "airport_id": [1], "product_id": ["routine"], "station": ["KAAA"],
            "status": ["normal"], "message_type": ["routine"],
            "availability_time": pd.to_datetime(["2026-04-01T08:00Z"], utc=True),
            "archive_issue_time": pd.to_datetime(["2026-04-01T08:00Z"], utc=True),
            "body_issue_time": pd.to_datetime(["2026-04-01T08:00Z"], utc=True),
            "availability_verified": [False], "interval_usable": [True],
            "segment_start": pd.to_datetime(["2026-04-01T08:00Z"], utc=True),
            "segment_end": pd.to_datetime(["2026-04-01T12:00Z"], utc=True),
        })
        bulletins = pd.DataFrame({
            "airport_id": [1, 1], "product_id": ["routine", "untimed"], "station": ["KAAA", "KAAA"],
            "status": ["normal", "normal"], "message_type": ["routine", "delayed"],
            "availability_time": pd.to_datetime(["2026-04-01T08:00Z", None], utc=True),
            "archive_issue_time": pd.to_datetime(["2026-04-01T08:00Z", "2026-04-01T09:00Z"], utc=True),
            "body_issue_time": pd.to_datetime(["2026-04-01T08:00Z", None], utc=True),
            "availability_verified": [False, False],
        })
        result = select_taf_asof(segments, bulletins, 1,
                                 "2026-04-01T10:00Z", "2026-04-01T10:30Z")
        self.assertEqual(result["status"], "unverified_issue_time")
        self.assertTrue(result["segments"].empty)
        bulletins.loc[1, "archive_issue_time"] = pd.Timestamp("2026-04-01T08:00Z")
        result = select_taf_asof(segments, bulletins, 1,
                                 "2026-04-01T10:00Z", "2026-04-01T10:30Z")
        self.assertEqual(result["status"], "unverified_issue_time")

    def test_noaa_qc_whitelist_is_source_specific(self):
        self.assertTrue(noaa_qc_is_eligible("313", "1", "temperature"))
        self.assertTrue(noaa_qc_is_eligible("313", "5", "temperature"))
        self.assertFalse(noaa_qc_is_eligible("313", "0", "temperature"))
        self.assertFalse(noaa_qc_is_eligible("313", "2", "temperature"))
        self.assertTrue(noaa_qc_is_eligible("345", "0", "wind_speed"))
        self.assertFalse(noaa_qc_is_eligible("345", "0", "precipitation"))
        self.assertFalse(noaa_qc_is_eligible("345", "", "wind_speed"))
        self.assertTrue(noaa_qc_is_eligible("382", "", "precipitation"))
        self.assertFalse(noaa_qc_is_eligible("382", "", "temperature"))
        self.assertFalse(noaa_qc_is_eligible("382", "X", "precipitation"))
        self.assertFalse(noaa_qc_is_eligible("413", "", "temperature"))
        self.assertFalse(noaa_qc_is_eligible("413", "x", "temperature"))

    def test_noaa_trace_missing_and_provenance_are_preserved(self):
        self.assertEqual(noaa_precip_status(0.2, "", "313"), "measured")
        self.assertEqual(noaa_precip_status(0.0, "2", "313"), "trace")
        self.assertEqual(noaa_precip_status(0.0, "T", "382"), "trace")
        self.assertEqual(noaa_precip_status(0.0, "2", "382"), "source_specific_measurement_code")
        self.assertEqual(noaa_precip_status(0.2, "1", "313"), "invalid_measurement_code")
        self.assertEqual(noaa_precip_status(0.0, "2", "413"), "source_specific_measurement_code")
        self.assertEqual(noaa_precip_status(np.nan, "", "313"), "missing")
        raw = pd.DataFrame({
            "STATION": ["USW00000001", "USW00000001"],
            "DATE": ["2026-04-01T09:00:00", "2026-04-01T10:00:00"],
            "wind_speed": [2.0, 3.0], "wind_speed_Quality_Code": ["1", "2"],
            "wind_speed_Source_Code": ["313", "313"],
            "precipitation": [0.0, -9999], "precipitation_Measurement_Code": ["2", ""],
            "precipitation_Quality_Code": ["", ""], "precipitation_Source_Code": ["313", "313"],
        })
        result = prepare_noaa_frame(raw, "sample.psv.gz")
        self.assertTrue(result.wind_speed_qc_eligible.tolist()[0])
        self.assertFalse(result.wind_speed_qc_eligible.tolist()[1])
        self.assertEqual(result.precipitation_measurement_status.tolist(), ["trace", "missing"])
        self.assertTrue(pd.isna(result.precipitation.iloc[1]))
        self.assertTrue(result.source_file.eq("sample.psv.gz").all())
        self.assertTrue(result.precipitation_period_note.str.contains("nominal_hour").all())

    def test_weather_run_writes_pickles_returns_summary_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "processed" / "W0001"
            summary_file = root / "run" / "weather_preparation_summary.json"
            metar = pd.DataFrame({"airport_id": [1], "wind_knots": [5.0]})
            taf = pd.DataFrame({"airport_id": [1], "product_id": ["p"],
                                "validity_matched": [True], "interval_usable": [True]})
            bulletins = pd.DataFrame({"airport_id": [1], "product_id": ["p"]})
            noaa_chunk = pd.DataFrame({"STATION": ["USW00000001"]})
            with patch("scripts.prepare_weather.prepare_metar", return_value=metar), \
                    patch("scripts.prepare_weather.prepare_taf", return_value=(taf, bulletins)), \
                    patch("scripts.prepare_weather.iter_noaa_batches", return_value=iter([noaa_chunk])):
                summary = run_weather_preparation(output_dir=output, summary_path=summary_file)
            self.assertEqual(summary["metar_rows"], 1)
            self.assertTrue(summary_file.is_file())
            for name in ("metar_observations", "taf_segments", "taf_bulletins"):
                self.assertTrue((output / f"{name}.pkl").is_file())
                self.assertTrue((output / f"{name}.csv.gz").is_file())
            self.assertTrue((output / "noaa_observations.csv.gz").is_file())
            with self.assertRaises(FileExistsError):
                run_weather_preparation(output_dir=output, summary_path=root / "other.json")

    def test_metar_asof_normalizes_mixed_datetime_units(self):
        flights = pd.DataFrame({"airport_id": [1], "prediction_time": pd.to_datetime(
            ["2026-04-01T09:05:00Z"], utc=True).as_unit("ns")})
        observations = pd.DataFrame({"airport_id": [1], "station": ["KAAA"],
            "weather_time": pd.to_datetime(["2026-04-01T09:00:00Z"], utc=True).as_unit("us"),
            "available_at": pd.to_datetime(["2026-04-01T09:05:00Z"], utc=True).as_unit("us"),
            "wind_knots": [8.0], "metar_ambiguous": [False]})
        result = match_metar_asof(flights, observations)
        self.assertEqual(result.wind_knots.iloc[0], 8.0)


if __name__ == "__main__":
    unittest.main()
