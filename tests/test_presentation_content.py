"""Synthetic, sealed summary archives only. No real August access or training."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import build_presentation_content as presentation
from scripts import build_report as report

fixture_spec = importlib.util.spec_from_file_location("presentation_report_fixture", Path(__file__).with_name("test_report_evidence.py"))
fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture)


def presentation_fixture(root, fills=None):
    config = fixture.make_fixture(root)
    run_id = config["noaa_run_id"]
    directory = root / "experiments/runs" / run_id
    fixture.write_json(directory / "feature_audit.json", {"noaa_fill_counts": fills or {"synthetic": 0},
                       "noaa_fill_assumption": "Synthetic retrospective availability", "weather_audit": {"noaa_backfill": True}})
    fixture.seal(root, run_id, json.loads((directory / "config.json").read_text(encoding="utf-8")))
    return config


class PresentationContent(unittest.TestCase):
    def test_complete_deck_is_dynamic_summary_only_and_nonmutating(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = presentation_fixture(root)
            original = {str(p): report.digest(p) for p in root.rglob("*") if p.is_file()}
            with patch.object(pd, "read_pickle", side_effect=AssertionError("No pickles")):
                result = presentation.build_content(config, "reports/presentation/content.json", root=root)
            deck = json.loads(Path(result["content"]).read_text(encoding="utf-8"))
            self.assertEqual(len(deck["slides"]), 12)
            self.assertEqual(sum(s["duration"] for s in deck["slides"]), 600)
            self.assertGreaterEqual(result["speaker_words"], 1050)
            self.assertLessEqual(result["speaker_words"], 1600)
            self.assertIn("Training median", deck["slides"][0]["title"])
            self.assertIn("8.000", deck["slides"][0]["title"])
            self.assertIn("cannot establish that NOAA information has no predictive value", deck["slides"][6]["notes"])
            self.assertIn("NOAA filled 0 values, 0 in selected model columns", deck["slides"][6]["claims"])
            for slide in deck["slides"]:
                self.assertLessEqual(len(slide["claims"]), 4)
                self.assertTrue(slide["notes"])
                self.assertTrue(slide["citations"])
                self.assertLessEqual(sum(key in slide for key in ["table", "chart", "image"]), 1)
                self.assertTrue(all("SHA256=" in ref for ref in slide["citations"]))
                if "table" in slide:
                    self.assertLessEqual(len(slide["table"]), 8)
                if "chart" in slide:
                    self.assertTrue(all(len(s["values"]) == len(slide["chart"]["categories"]) for s in slide["chart"]["series"]))
            text = json.dumps(deck)
            for required in ["not selection-adjusted", "NA rather than", "retrospective", "revision block", "R0017", "R0019"]:
                self.assertIn(required, text)
            sources = json.loads(Path(result["sources"]).read_text(encoding="utf-8"))
            self.assertFalse(sources["raw_or_test_targets_read"])
            for source in sources["source_files"]:
                self.assertFalse(any(part in source["path"].split("/") for part in ["raw", "processed", "predictions", "models"]))
            for filename, digest in original.items():
                self.assertEqual(report.digest(filename), digest)
            with self.assertRaises(FileExistsError):
                presentation.build_content(config, result["content"], root=root)

    def test_tampered_summary_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = presentation_fixture(root)
            source = root / "experiments/runs/R0017/metrics.csv"
            source.write_text(source.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "receipt"):
                presentation.build_content(config, "reports/presentation/content.json", root=root)
            self.assertFalse((root / "reports").exists())

    def test_invalidated_run_is_never_presentation_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = presentation_fixture(root)
            index_path = root / "experiments/index.csv"
            index = pd.read_csv(index_path)
            index.loc[index.run_id.eq(config["noaa_run_id"]), "status"] = "invalidated"
            index.to_csv(index_path, index=False)
            with self.assertRaisesRegex(ValueError, "completed and valid"):
                presentation.build_content(config, "reports/presentation/content.json", root=root)
            self.assertFalse((root / "reports").exists())

    def test_positive_noaa_fills_use_the_actual_selected_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = presentation_fixture(root, {"metar_wind_knots": 2, "metar_temperature_f": 4})
            # This case owns its selected NOAA fields rather than relying on the
            # report fixture's default G0 feature names. Keep one inserted field
            # inside the model and one outside it, so the intersection matters.
            directory = root / "experiments/runs" / config["noaa_run_id"]
            model_path = directory / "model_manifest.json"
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["features"].append("metar_wind_knots")
            self.assertNotIn("metar_temperature_f", model["features"])
            fixture.write_json(model_path, model)
            fixture.seal(root, config["noaa_run_id"], json.loads((directory / "config.json").read_text(encoding="utf-8")))
            result = presentation.build_content(config, "reports/presentation/content.json", root=root)
            slide = json.loads(Path(result["content"]).read_text(encoding="utf-8"))["slides"][6]
            self.assertIn("NOAA filled 6 values, 2 in selected model columns", slide["claims"])
            self.assertNotIn("No NOAA values entered", slide["notes"])
            self.assertIn("field values, not distinct connections", slide["notes"])


if __name__ == "__main__":
    unittest.main()
