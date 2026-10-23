"""Batch weather feature join contracts on small synthetic tables."""
import unittest

import numpy as np
import pandas as pd

from scripts.prepare_weather import prepare_noaa_frame
from scripts.weather_features import augment_metar, augment_noaa, augment_taf, noaa_overlap_diagnostics


def ts(values):
    return pd.to_datetime(values, utc=True)


class WeatherFeatureTests(unittest.TestCase):
    def test_taf_and_noaa_asof_normalize_mixed_datetime_units(self):
        flights = pd.DataFrame({
            "airport_id": [1], "prediction_time": ts(["2026-04-01T09:05:00Z"]).as_unit("ns"),
            "scheduled_dep_utc": ts(["2026-04-01T09:10:00Z"]).as_unit("ns"),
        })
        bulletins = pd.DataFrame({
            "airport_id": [1], "product_id": ["routine"], "station": ["KAAA"],
            "status": ["normal"], "message_type": ["routine"],
            "availability_time": ts(["2026-04-01T08:00:00Z"]).as_unit("us"),
            "archive_issue_time": ts(["2026-04-01T08:00:00Z"]).as_unit("us"),
            "body_issue_time": ts(["2026-04-01T08:00:00Z"]).as_unit("us"),
        })
        segments = pd.DataFrame({
            "airport_id": [1], "product_id": ["routine"], "ftype": ["Forecast"],
            "segment_start": ts(["2026-04-01T08:00:00Z"]).as_unit("us"),
            "segment_end": ts(["2026-04-01T12:00:00Z"]).as_unit("us"),
            "interval_usable": [True], "wind_knots": [12.0], "is_tempo": [False],
            "is_probability": [False],
        })
        taf_result = augment_taf(flights, segments, bulletins)
        self.assertEqual(taf_result.taf_prevailing_wind_knots.iloc[0], 12.0)

        airports = pd.DataFrame({"airport_id": [1], "ghcnh_id": ["USW00000001"]})
        noaa = prepare_noaa_frame(pd.DataFrame({
            "STATION": ["USW00000001"], "DATE": ["2026-04-01T09:00:00Z"],
            "temperature": [20.0], "temperature_Source_Code": ["313"],
            "temperature_Quality_Code": ["1"],
            "wind_speed": [5.0], "wind_speed_Source_Code": ["313"],
            "wind_speed_Quality_Code": ["1"],
        }))
        noaa["observation_time"] = noaa.observation_time.dt.as_unit("us")
        fill_frame = pd.DataFrame({"airport_id": [1], "prediction_time": ts(
            ["2026-04-01T09:10:00Z"]).as_unit("ns"), "metar_missing": [1]})
        for column in ("metar_temperature_f", "metar_dewpoint_f", "metar_wind_direction_deg",
                       "metar_relative_humidity", "metar_wind_knots", "metar_gust_knots",
                       "metar_visibility_miles", "metar_ceiling_ft", "metar_precip_amount_in",
                       "metar_precip_flag", "metar_rain_flag", "metar_snow_flag", "metar_thunder_flag"):
            fill_frame[column] = np.nan
        result = augment_noaa(fill_frame, noaa, airports)
        self.assertAlmostEqual(result.metar_wind_knots.iloc[0], 5.0 * 1.943844)
        self.assertTrue(pd.isna(result.metar_temperature_f.iloc[0]))

    def test_taf_untimed_archive_blocks_fallback_with_nondefault_flight_index(self):
        flights = pd.DataFrame({"airport_id": [1],
            "prediction_time": ts(["2026-04-01T09:00:00Z"]),
            "scheduled_dep_utc": ts(["2026-04-01T10:00:00Z"])}, index=[99])
        bulletins = pd.DataFrame({
            "airport_id": [1, 1], "product_id": ["routine", "untimed"],
            "status": ["normal", "normal"], "message_type": ["routine", "routine"],
            "availability_time": ts(["2026-04-01T08:00:00Z", None]),
            "archive_issue_time": ts(["2026-04-01T08:00:00Z", "2026-04-01T08:30:00Z"]),
            "body_issue_time": ts(["2026-04-01T08:00:00Z", None]),
        })
        segments = pd.DataFrame({
            "airport_id": [1], "product_id": ["routine"], "ftype": ["Observation"],
            "segment_start": ts(["2026-04-01T08:00:00Z"]),
            "segment_end": ts(["2026-04-01T12:00:00Z"]), "interval_usable": [True],
            "wind_knots": [10.0], "is_tempo": [False], "is_probability": [False],
        })
        result = augment_taf(flights, segments, bulletins)
        self.assertEqual(result.index.tolist(), [99])
        self.assertEqual(result.taf_status.iloc[0], "unverified_issue_time")
        self.assertTrue(pd.isna(result.taf_prevailing_wind_knots.iloc[0]))

    def test_augment_metar_prefixes_features_and_keeps_rows(self):
        flights = pd.DataFrame({
            "flight_id": ["later", "earlier"], "airport_id": [1, "1"],
            "prediction_time": ts(["2026-04-01T10:00Z", "2026-04-01T09:05Z"]),
        }, index=[8, 3])
        observations = pd.DataFrame({
            "airport_id": [1], "station": ["KAAA"],
            "weather_time": ts(["2026-04-01T09:00Z"]), "available_at": ts(["2026-04-01T09:05Z"]),
            "availability_assumption": ["weather_time_plus_5_minutes"], "wind_knots": [12.0],
            "wind_direction_deg": [180.0], "gust_knots": [20.0], "visibility_miles": [5.0],
            "ceiling_ft": [2000.0], "precip_flag": pd.array([pd.NA], dtype="Int8"),
            "rain_flag": pd.array([pd.NA], dtype="Int8"), "snow_flag": pd.array([pd.NA], dtype="Int8"),
            "thunder_flag": pd.array([pd.NA], dtype="Int8"), "metar_ambiguous": [False],
        })
        result = augment_metar(flights, observations)
        self.assertEqual(result.flight_id.tolist(), ["later", "earlier"])
        self.assertEqual(result.index.tolist(), [8, 3])
        self.assertEqual(result.metar_wind_knots.tolist(), [12.0, 12.0])
        self.assertEqual(result.metar_missing.tolist(), [0, 0])
        self.assertTrue(result.metar_weather_time.notna().all())

    def test_augment_taf_selects_then_aggregates_each_segment_type(self):
        flights = pd.DataFrame({
            "flight_id": ["tempo", "amend", "cancel", "before", "other_airport"],
            "airport_id": [1, 1, 1, 1, 2],
            "scheduled_dep_utc": ts(["2026-04-01T09:30Z", "2026-04-01T11:00Z",
                                     "2026-04-01T11:00Z", "2026-04-01T09:00Z", "2026-04-01T09:30Z"]),
            "prediction_time": ts(["2026-04-01T08:50Z", "2026-04-01T09:20Z",
                                   "2026-04-01T10:00Z", "2026-04-01T07:00Z", "2026-04-01T09:00Z"]),
        }, index=[14, 11, 8, 5, 2])
        bulletins = pd.DataFrame({
            "airport_id": [1, 1, 1], "product_id": ["routine", "amendment", "cancel"],
            "station": ["KAAA"] * 3, "status": ["normal", "normal", "cancelled"],
            "message_type": ["routine", "amended", "routine"],
            "availability_time": ts(["2026-04-01T08:00Z", "2026-04-01T09:10Z", "2026-04-01T09:55Z"]),
            "archive_issue_time": ts(["2026-04-01T08:00Z", "2026-04-01T09:10Z", "2026-04-01T09:55Z"]),
            "body_issue_time": ts(["2026-04-01T08:00Z", "2026-04-01T09:10Z", "2026-04-01T09:55Z"]),
            "issue_time_mismatch_minutes": [0.0, 0.0, 0.0],
            "availability_assumption": ["proxy"] * 3,
        })
        rows = [
            # Routine prevailing forecast, split at 10Z.
            ("routine", "Observation", "2026-04-01T08:00Z", "2026-04-01T10:00Z", 10, 10, 6, 25000, 0, 0, 0, 0, False, False),
            ("routine", "Forecast", "2026-04-01T10:00Z", "2026-04-01T12:00Z", 15, 18, 5, 18000, 0, 0, 0, 0, False, False),
            ("routine", "Temporary", "2026-04-01T09:00Z", "2026-04-01T10:00Z", 20, 30, 3, 2000, 1, 1, 0, 0, True, False),
            ("routine", "Temporary", "2026-04-01T09:15Z", "2026-04-01T09:45Z", 25, 35, 1, 1000, 1, 0, 1, 1, True, False),
            ("routine", "Probability", "2026-04-01T09:15Z", "2026-04-01T09:45Z", 18, 28, 2, 1500, 1, 1, 0, 1, False, True),
            # Disabled amendment contains values that must not leak or fall back.
            ("amendment", "Observation", "2026-04-01T09:10Z", "2026-04-01T12:00Z", 99, 99, 0.1, 100, 1, 1, 1, 1, False, False),
        ]
        segments = pd.DataFrame(rows, columns=["product_id", "ftype", "segment_start", "segment_end",
            "wind_knots", "gust_knots", "visibility_miles", "ceiling_ft", "precip_flag", "rain_flag",
            "snow_flag", "thunder_flag", "is_tempo", "is_probability"])
        segments["airport_id"] = 1
        segments["segment_start"] = ts(segments.segment_start)
        segments["segment_end"] = ts(segments.segment_end)
        segments["interval_usable"] = True

        result = augment_taf(flights, segments, bulletins)
        self.assertEqual(result.flight_id.tolist(), flights.flight_id.tolist())
        self.assertEqual(result.index.tolist(), flights.index.tolist())
        got = result.set_index("flight_id")
        self.assertEqual(got.loc["tempo", "taf_status"], "matched")
        self.assertEqual(got.loc["tempo", "taf_query_time"], pd.Timestamp("2026-04-01T09:30Z"))
        self.assertEqual(got.loc["tempo", "taf_prevailing_wind_knots"], 10)
        self.assertEqual(got.loc["tempo", "taf_tempo_segment_count"], 2)
        self.assertEqual(got.loc["tempo", "taf_tempo_wind_knots"], 25)
        self.assertEqual(got.loc["tempo", "taf_tempo_visibility_miles"], 1)
        self.assertEqual(got.loc["tempo", "taf_probability_segment_count"], 1)
        self.assertEqual(got.loc["tempo", "taf_probability_rain_flag"], 1)
        self.assertEqual(got.loc["amend", "taf_status"], "disabled_revision")
        self.assertTrue(pd.isna(got.loc["amend", "taf_prevailing_wind_knots"]))
        self.assertEqual(got.loc["cancel", "taf_status"], "cancelled")
        self.assertEqual(got.loc["before", "taf_status"], "no_available_bulletin")
        self.assertEqual(got.loc["other_airport", "taf_status"], "no_available_bulletin")
        self.assertEqual(got.loc["cancel", "taf_missing"], 1)

    def test_augment_noaa_converts_units_respects_asof_qc_and_only_fills_missing(self):
        airports = pd.DataFrame({"airport_id": [1], "ghcnh_id": ["USW00000001"]})
        noaa = prepare_noaa_frame(pd.DataFrame({
            "STATION": ["USW00000001", "USW00000001"],
            "DATE": ["2026-04-01T08:00:00Z", "2026-04-01T09:00:00Z"],
            "temperature": [20.0, 21.0], "temperature_Quality_Code": ["1", "1"],
            "temperature_Source_Code": ["313", "313"],
            "wind_speed": [5.0, 12.0], "wind_speed_Quality_Code": ["1", "2"],
            "wind_speed_Source_Code": ["313", "313"],
            "wind_gust": [8.0, 9.0], "wind_gust_Quality_Code": ["1", "1"],
            "wind_gust_Source_Code": ["313", "313"],
            "relative_humidity": [50.0, 55.0], "relative_humidity_Quality_Code": ["1", "1"],
            "relative_humidity_Source_Code": ["313", "313"],
            "visibility": [10.0, 12.0], "visibility_Quality_Code": ["1", "1"],
            "visibility_Source_Code": ["313", "313"],
            "ceiling_height": [300.0, 400.0], "ceiling_height_Quality_Code": ["1", "1"],
            "ceiling_height_Source_Code": ["313", "313"],
            "precipitation": [1.0, 2.0], "precipitation_Quality_Code": ["1", "1"],
            "precipitation_Source_Code": ["313", "313"],
        }))
        frame = pd.DataFrame({
            "airport_id": [1, 1],
            "prediction_time": ts(["2026-04-01T09:10Z", "2026-04-01T08:04Z"]),
            "metar_wind_knots": [np.nan, np.nan], "metar_gust_knots": [np.nan, np.nan],
            "metar_visibility_miles": [np.nan, np.nan], "metar_ceiling_ft": [np.nan, np.nan],
            "metar_temperature_f": [np.nan, np.nan], "metar_dewpoint_f": [np.nan, np.nan],
            "metar_wind_direction_deg": [np.nan, np.nan], "metar_precip_amount_in": [np.nan, np.nan],
            "metar_precip_flag": [pd.NA, pd.NA], "metar_rain_flag": [pd.NA, pd.NA],
            "metar_snow_flag": [pd.NA, pd.NA], "metar_thunder_flag": [pd.NA, pd.NA],
            "metar_relative_humidity": [np.nan, np.nan], "metar_missing": [1, 1],
        }, index=[7, 3])
        frame.loc[7, "metar_wind_knots"] = 77.0  # Existing IEM measurement must win.
        result = augment_noaa(frame, noaa, airports)
        self.assertEqual(result.index.tolist(), [7, 3])
        self.assertEqual(result.loc[7, "metar_wind_knots"], 77.0)
        self.assertEqual(result.loc[7, "metar_wind_knots_source"], "IEM")
        # NOAA fields outside the fixed G4a set remain audit/overlap-only.
        self.assertTrue(pd.isna(result.loc[7, "metar_temperature_f"]))
        self.assertTrue(pd.isna(result.loc[7, "metar_relative_humidity"]))
        self.assertAlmostEqual(result.loc[7, "metar_wind_knots"], 77.0)
        # The 09:00Z wind fails QC; older valid 08:00Z data is selected as-of.
        frame.loc[7, "metar_wind_knots"] = np.nan
        result = augment_noaa(frame, noaa, airports)
        self.assertAlmostEqual(result.loc[7, "metar_wind_knots"], 5.0 * 1.943844)
        self.assertAlmostEqual(result.loc[7, "metar_wind_knots_noaa_age_minutes"], 70.0)
        self.assertEqual(result.loc[7, "metar_wind_knots_noaa_source_code"], "313")
        self.assertTrue(pd.isna(result.loc[7, "metar_precip_amount_in"]))
        self.assertTrue(pd.isna(result.loc[3, "metar_temperature_f"]))  # No future availability.
        self.assertEqual(result.attrs["noaa_fill_counts"]["metar_temperature_f"], 0)

    def test_noaa_backfill_respects_structural_gust_and_ceiling_missingness(self):
        airports = pd.DataFrame({"airport_id": [1], "ghcnh_id": ["USW00000001"]})
        noaa = prepare_noaa_frame(pd.DataFrame({
            "STATION": ["USW00000001"], "DATE": ["2026-04-01T08:00:00Z"],
            "wind_speed": [5.0], "wind_speed_Quality_Code": ["1"], "wind_speed_Source_Code": ["313"],
            "wind_gust": [8.0], "wind_gust_Quality_Code": ["1"], "wind_gust_Source_Code": ["313"],
            "visibility": [10.0], "visibility_Quality_Code": ["1"], "visibility_Source_Code": ["313"],
            "ceiling_height": [300.0], "ceiling_height_Quality_Code": ["1"], "ceiling_height_Source_Code": ["313"],
        }))
        frame = pd.DataFrame({
            "airport_id": [1, 1, 1, 1],
            "prediction_time": ts(["2026-04-01T09:10Z"] * 4),
            "metar_missing": [0, 0, 1, 1],
            "metar_sky_reported": [1, 0, pd.NA, pd.NA],
            # This is the name produced by augment_metar's current prefixing.
            "metar_metar_ambiguous": [False, False, False, True],
            "metar_wind_knots": [np.nan] * 4,
            "metar_gust_knots": [np.nan] * 4,
            "metar_visibility_miles": [np.nan] * 4,
            "metar_ceiling_ft": [np.nan] * 4,
            "metar_temperature_f": [np.nan] * 4, "metar_dewpoint_f": [np.nan] * 4,
            "metar_wind_direction_deg": [np.nan] * 4,
            "metar_relative_humidity": [np.nan] * 4,
            "metar_precip_amount_in": [np.nan] * 4,
            "metar_precip_flag": [pd.NA] * 4, "metar_rain_flag": [pd.NA] * 4,
            "metar_snow_flag": [pd.NA] * 4, "metar_thunder_flag": [pd.NA] * 4,
        }, index=[40, 10, 99, 55])
        result = augment_noaa(frame, noaa, airports)
        # Existing sky observation can mean clear/FEW/SCT; do not invent a ceiling.
        self.assertTrue(pd.isna(result.loc[40, "metar_ceiling_ft"]))
        # Gust absence on an otherwise usable METAR is not equivalent to missing weather.
        self.assertTrue(pd.isna(result.loc[40, "metar_gust_knots"]))
        self.assertTrue(pd.isna(result.loc[10, "metar_gust_knots"]))
        # Explicitly unreported sky and a wholly unavailable report can be backfilled.
        self.assertAlmostEqual(result.loc[10, "metar_ceiling_ft"], 300.0 / 0.3048)
        self.assertAlmostEqual(result.loc[99, "metar_gust_knots"], 8.0 * 1.943844)
        self.assertAlmostEqual(result.loc[99, "metar_ceiling_ft"], 300.0 / 0.3048)
        # A conflicting duplicate is not a safe missing report to backfill.
        for column in ("metar_wind_knots", "metar_gust_knots", "metar_visibility_miles", "metar_ceiling_ft"):
            self.assertTrue(pd.isna(result.loc[55, column]), column)

    def test_noaa_overlap_diagnostics_reports_converted_exact_time_differences(self):
        airports = pd.DataFrame({"airport_id": [1], "ghcnh_id": ["USW00000001"]})
        noaa = prepare_noaa_frame(pd.DataFrame({
            "STATION": ["USW00000001"], "DATE": ["2026-04-01T08:00:00Z"],
            "wind_speed": [5.0], "wind_speed_Quality_Code": ["1"], "wind_speed_Source_Code": ["313"],
            "temperature": [20.0], "temperature_Quality_Code": ["1"], "temperature_Source_Code": ["313"],
        }))
        metar = pd.DataFrame({
            "airport_id": [1], "station": ["KAAA"], "weather_time": ts(["2026-04-01T08:00Z"]),
            "wind_knots": [10.0], "temperature_f": [68.0], "metar_ambiguous": [False],
        })
        report = noaa_overlap_diagnostics(metar, noaa, airports)
        wind = report["summary"].set_index("feature").loc["metar_wind_knots"]
        self.assertEqual(wind.overlap_count, 1)
        self.assertAlmostEqual(wind.mean_signed_difference, 5.0 * 1.943844 - 10.0)
        self.assertEqual(wind.unit_consistency, "converted_for_comparison")
        self.assertEqual(len(report["pairs"]), 2)


if __name__ == "__main__":
    unittest.main()
