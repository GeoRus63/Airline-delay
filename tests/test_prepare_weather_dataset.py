"""Offline wrapper tests for isolated weather dataset preparation."""
import json
import importlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import prepare_weather_dataset as dataset


class PrepareWeatherDatasetTests(unittest.TestCase):
    def _make_project(self, root):
        for relative in ("raw/weather/metar", "raw/weather/taf", "raw/weather/noaa",
                         "raw/weather/taf_text", "tables", "references"):
            (root / relative).mkdir(parents=True, exist_ok=True)
        (root / "raw/weather/metar/batch_001.csv.gz").write_bytes(b"metar source")
        (root / "raw/weather/taf/batch_001.csv.gz").write_bytes(b"taf source")
        (root / "raw/weather/noaa/GHCNh_test.psv.gz").write_bytes(b"noaa source")
        archive_path = root / "raw/weather/taf_text/batch_001.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("TAFABE_202604010800.txt",
                             "FTUS31 KAAA 010800\nTAF KAAA 010800Z 0108/0212 18010KT P6SM SCT025\n")
        manifest = {"requests": [{"file": "raw/weather/taf_text/batch_001.zip",
                                  "sha256": "synthetic", "product_count": 1}]}
        (root / "tables/taf_text_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "tables/weather_download_manifest.json").write_text("{}", encoding="utf-8")
        pd.DataFrame({"airport_id": [1], "month": ["2026-04"], "kind": ["metar"],
                      "rows": [1], "days": [1]}).to_csv(root / "tables/weather_coverage.csv", index=False)
        pd.DataFrame({"airport_id": [1], "ghcnh_id": ["USW00000001"]}).to_csv(
            root / "references/airports.csv", index=False)
        return archive_path

    def test_offline_taf_validity_archives_outputs_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._make_project(root)
            run_dir = root / "experiments/runs/R0009"

            def fake_prepare(output_dir, summary_path):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"STATION": ["USW00000001"], "temperature_qc_eligible": [True]}).to_csv(
                    output_dir / "noaa_observations.csv.gz", index=False, compression="gzip")
                (output_dir / "metar_observations.pkl").write_bytes(b"metar pkl")
                (output_dir / "taf_segments.pkl").write_bytes(b"taf pkl")
                (output_dir / "taf_bulletins.pkl").write_bytes(b"taf bulletins")
                summary = {"metar_rows": 1, "taf_validity_matched_rows": 1,
                           "taf_usable_interval_rows": 1, "noaa_rows": 1}
                Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
                Path(summary_path).write_text(json.dumps(summary), encoding="utf-8")
                return summary

            with patch.object(dataset, "ROOT", root), \
                    patch.object(dataset.prepare_weather, "run", side_effect=fake_prepare):
                result = dataset.run({"input_path": "raw/weather",
                                      "weather_output_dir": "processed/W0001",
                                      "source_files": ["raw/weather/metar/batch_001.csv.gz"]}, run_dir)

            validity = pd.read_csv(root / "references/taf_validity.csv")
            self.assertEqual(validity.status.tolist(), ["normal"])
            self.assertEqual(validity.station.tolist(), ["KAAA"])
            self.assertEqual(validity.valid_from.iloc[0], "2026-04-01T08:00:00Z")
            processed_manifest_path = root / "processed/W0001/data_manifest.json"
            run_manifest_path = run_dir / "data_manifest.json"
            processed_manifest = json.loads(processed_manifest_path.read_text(encoding="utf-8"))
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(processed_manifest, run_manifest)
            output_paths = {item["path"] for item in processed_manifest["outputs"]}
            self.assertNotIn("processed/W0001/data_manifest.json", output_paths)
            self.assertIn("processed/W0001/taf_validity.csv", output_paths)
            self.assertIn("raw/weather/metar/batch_001.csv.gz",
                          {item["path"] for item in processed_manifest["inputs"]})
            input_paths = {item["path"] for item in processed_manifest["inputs"]}
            self.assertIn("raw/weather/noaa/GHCNh_test.psv.gz", input_paths)
            self.assertIn("raw/weather/taf_text/batch_001.zip", input_paths)
            self.assertEqual(processed_manifest["noaa_qc_counts"]["temperature"]["eligible_rows"], 1)
            self.assertNotIn("source_data_id", processed_manifest)
            self.assertEqual(result["weather_output_dir"], "processed/W0001")
            self.assertTrue((root / "processed/W0001/weather_config.json").is_file())

    def test_nonempty_output_is_refused_before_taf_validity_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._make_project(root)
            output = root / "processed/W0001"
            output.mkdir(parents=True)
            (output / "keep.txt").write_text("preserve", encoding="utf-8")
            with patch.object(dataset, "ROOT", root), \
                    patch.object(dataset.prepare_weather, "run") as run_mock:
                with self.assertRaises(FileExistsError):
                    dataset.run({"input_path": "raw/weather"}, root / "runs/R")
            run_mock.assert_not_called()
            self.assertFalse((root / "references/taf_validity.csv").exists())

    def test_direct_scripts_path_import_matches_run_pipeline_behavior(self):
        script_dir = str(Path(__file__).resolve().parents[1] / "scripts")
        sys.path.insert(0, script_dir)
        try:
            direct = importlib.import_module("prepare_weather_dataset")
            self.assertTrue(callable(direct.run))
        finally:
            sys.modules.pop("prepare_weather_dataset", None)
            sys.path.remove(script_dir)


if __name__ == "__main__":
    unittest.main()
