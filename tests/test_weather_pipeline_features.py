"""Synthetic end-to-end checks for the pipeline weather composition layer."""
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import prepare_weather
from scripts import weather_pipeline_features as pipeline_weather


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class WeatherPipelineFeatureTests(unittest.TestCase):
    def _weather_project(self, root):
        output = root / "processed/W0001"
        output.mkdir(parents=True)
        refs = root / "references"
        refs.mkdir()
        pd.DataFrame({"airport_id": [1], "ghcnh_id": ["USW00000001"]}).to_csv(
            refs / "airports.csv", index=False)

        metar = pd.DataFrame({
            "airport_id": [1], "station": ["KAAA"],
            "weather_time": pd.to_datetime(["2026-04-01T09:00:00Z"], utc=True).as_unit("us"),
            "available_at": pd.to_datetime(["2026-04-01T09:05:00Z"], utc=True).as_unit("us"),
            "availability_assumption": ["weather_time_plus_5_minutes"],
            "wind_knots": [np.nan], "wind_direction_deg": [180.0], "gust_knots": [15.0],
            "visibility_miles": [10.0], "ceiling_ft": [6000.0], "ceiling_reported": [1],
            "sky_reported": [1], "precip_amount_in": [np.nan],
            "precip_measurement_status": ["missing"], "precip_flag": pd.array([pd.NA], dtype="Int8"),
            "rain_flag": pd.array([pd.NA], dtype="Int8"), "snow_flag": pd.array([pd.NA], dtype="Int8"),
            "weather_codes_reported": [0], "thunder_flag": pd.array([pd.NA], dtype="Int8"),
            "temperature_f": [np.nan], "dewpoint_f": [np.nan], "relative_humidity": [np.nan],
            "metar_ambiguous": [False], "metar": ["METAR KAAA 010900Z 18010KT 10SM SCT060"],
        })
        taf_segments = pd.DataFrame({
            "airport_id": [1, 1], "station": ["KAAA", "KAAA"], "product_id": ["routine", "amended"],
            "segment_start": pd.to_datetime(["2026-04-01T08:00:00Z", "2026-04-01T09:20:00Z"],
                                             utc=True).as_unit("us"),
            "segment_end": pd.to_datetime(["2026-04-01T12:00:00Z", "2026-04-01T12:00:00Z"],
                                           utc=True).as_unit("us"),
            "ftype": ["Observation", "Observation"], "interval_usable": [True, True],
            "is_tempo": [False, False], "is_probability": [False, False],
            "wind_knots": [11.0, 99.0], "wind_direction_deg": [180.0, 270.0],
            "gust_knots": [np.nan, np.nan], "visibility_miles": [6.0, 0.5],
            "ceiling_ft": [4000.0, 200.0], "precip_flag": [0, 1], "rain_flag": [0, 1],
            "snow_flag": [0, 0], "thunder_flag": [0, 1],
        })
        bulletins = pd.DataFrame({
            "airport_id": [1, 1], "station": ["KAAA", "KAAA"],
            "product_id": ["routine", "amended"], "status": ["normal", "normal"],
            "message_type": ["routine", "amended"],
            "availability_time": pd.to_datetime(["2026-04-01T08:00:00Z", "2026-04-01T09:20:00Z"],
                                                  utc=True).as_unit("us"),
            "archive_issue_time": pd.to_datetime(["2026-04-01T08:00:00Z", "2026-04-01T09:20:00Z"],
                                                   utc=True).as_unit("us"),
            "body_issue_time": pd.to_datetime(["2026-04-01T08:00:00Z", "2026-04-01T09:20:00Z"],
                                                utc=True).as_unit("us"),
            "issue_time_mismatch_minutes": [0.0, 0.0],
            "availability_assumption": ["proxy", "proxy"],
        })
        paths = {}
        for name, frame in (("metar_observations", metar), ("taf_segments", taf_segments),
                            ("taf_bulletins", bulletins)):
            path = output / f"{name}.pkl"
            frame.to_pickle(path)
            paths[name] = path

        noaa = prepare_weather.prepare_noaa_frame(pd.DataFrame({
            "STATION": ["USW00000001"], "DATE": ["2026-04-01T09:00:00Z"],
            "wind_speed": [5.0], "wind_speed_Source_Code": ["313"],
            "wind_speed_Quality_Code": ["1"],
        }), source_file="tiny.psv.gz")
        noaa.to_csv(output / "noaa_observations.csv.gz", index=False, compression="gzip")
        paths["noaa_observations"] = output / "noaa_observations.csv.gz"

        records = [{"path": path.relative_to(root).as_posix(), "sha256": _hash(path)}
                   for path in paths.values()]
        input_record = {"path": "references/airports.csv", "sha256": _hash(refs / "airports.csv")}
        manifest = {"data_id": "W0001", "outputs": records, "inputs": [input_record],
                    "assumptions": {"METAR": "obs+5m proxy", "TAF": "issue proxy", "NOAA": "sensitivity"}}
        manifest_path = output / "data_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return output, manifest_path

    def _flights(self):
        return pd.DataFrame({
            "connection_id": ["c_later", "c_earlier"], "airport_id": [1, 1],
            "prediction_time": pd.to_datetime(["2026-04-01T09:30:00Z", "2026-04-01T09:10:00Z"],
                                               utc=True).as_unit("ns"),
            "downstream_scheduled_dep_utc": pd.to_datetime(["2026-04-01T10:00:00Z",
                                                             "2026-04-01T09:30:00Z"],
                                                            utc=True).as_unit("ns"),
            "target_delay": [42.0, 7.0],
        }, index=[97, 12])

    def test_augment_weather_preserves_contract_and_blocks_unverified_taf(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, manifest_path = self._weather_project(root)
            flights = self._flights()
            flights.attrs["caller_audit"] = {"token": "preserve"}
            original = flights.copy(deep=True)
            original.attrs = dict(flights.attrs)
            config = {"feature_group": "G4B", "weather_manifest_path": "processed/W0001/data_manifest.json",
                      "noaa_backfill": True}
            with patch.object(pipeline_weather, "ROOT", root):
                result = pipeline_weather.augment_weather(flights, config)
            self.assertEqual(result.index.tolist(), [97, 12])
            self.assertEqual(result.connection_id.tolist(), ["c_later", "c_earlier"])
            self.assertTrue(result.target_delay.equals(flights.target_delay))
            self.assertEqual(result.attrs["caller_audit"], {"token": "preserve"})
            self.assertIn("noaa_fill_counts", result.attrs)
            self.assertEqual(flights.equals(original), True)
            self.assertEqual(flights.attrs, original.attrs)
            self.assertEqual(result.loc[97, "taf_status"], "disabled_revision")
            self.assertTrue(pd.isna(result.loc[97, "taf_prevailing_wind_knots"]))
            self.assertEqual(result.loc[12, "taf_status"], "matched")
            self.assertEqual(result.loc[12, "taf_prevailing_wind_knots"], 11.0)
            self.assertAlmostEqual(result.loc[12, "metar_wind_knots"], 5.0 * 1.943844)
            self.assertEqual(result.loc[12, "metar_wind_knots_source"], "GHCNh")
            self.assertEqual(result.loc[12, "metar_wind_knots_noaa_source_code"], "313")
            for column in pipeline_weather.METAR_MODEL_NUMERIC + pipeline_weather.TAF_MODEL_NUMERIC:
                self.assertIn(column, result)
                self.assertTrue(pd.api.types.is_float_dtype(result[column].dtype), column)
            self.assertTrue(pd.isna(result.loc[97, "taf_prevailing_wind_knots"]))
            noaa_loaded = pipeline_weather.load_noaa(manifest_path.parent)
            self.assertEqual(str(noaa_loaded.wind_speed_Source_Code.dtype), "string")
            self.assertEqual(noaa_loaded.wind_speed_Source_Code.iloc[0], "313")

            # Manifest validation checks all listed input/output hashes.
            with patch.object(pipeline_weather, "ROOT", root):
                pipeline_weather.load_manifest(config)
                with (manifest_path.parent / "metar_observations.pkl").open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "Weather input changed"):
                    pipeline_weather.load_manifest(config)

    def test_write_weather_audit_never_requests_target_and_writes_coverage_only(self):
        class TargetAccessGuard(pd.DataFrame):
            @property
            def _constructor(self):
                return TargetAccessGuard

            def __getitem__(self, key):
                if key == "target_delay" or (isinstance(key, (list, tuple)) and "target_delay" in key):
                    raise AssertionError("write_weather_audit must not access target_delay")
                return super().__getitem__(key)

        with tempfile.TemporaryDirectory() as temp:
            frame = TargetAccessGuard({
                "airport_id": [1, 1],
                "prediction_time": pd.to_datetime(["2026-04-01T09:10Z", "2026-04-01T10:10Z"], utc=True),
                "metar_missing": [0, 1], "taf_missing": [1, 0], "target_delay": [99.0, 0.0],
            })
            summary = pipeline_weather.write_weather_audit(frame, temp, {})
            self.assertEqual(summary["rows"], 2)
            tables = Path(temp) / "tables"
            coverage = pd.read_csv(tables / "weather_coverage_month.csv")
            self.assertNotIn("target_delay", coverage.columns)
            self.assertEqual(coverage.metar_available_sum.iloc[0], 1)
            self.assertEqual(coverage.taf_available_sum.iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
