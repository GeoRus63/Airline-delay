"""Synthetic, sealed summary archives only. No real August access or training."""
import importlib.util
from decimal import Decimal
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
            self.assertEqual(deck["slides"][7]["title"], "Training median wins the registered comparison")
            self.assertEqual(deck["slides"][8]["table"][1][0], "OLS")
            self.assertIn("cannot establish that NOAA information has no predictive value", deck["slides"][6]["notes"])
            self.assertIn("NOAA filled 0 values, 0 in selected model columns", deck["slides"][6]["claims"])
            severity = deck["slides"][9]
            self.assertEqual(severity["duration"], 45)
            self.assertEqual(len(severity["claims"]), 4)
            self.assertIn("Ordinary MAE: OLS 10.0000 vs median 8.0000 min", severity["claims"])
            self.assertIn("Ordinary RMSE: OLS 10.0000 vs median 8.0000 min", severity["claims"])
            self.assertIn("50.0000% of AE", severity["claims"][3])
            self.assertIn("post-hoc diagnostics", severity["notes"])
            self.assertLessEqual(len(severity["notes"].split()), 115)
            self.assertIn("not a joint 95% statement", deck["slides"][8]["notes"])
            monthly = deck["slides"][4]
            self.assertEqual(monthly["chart"]["minimum"], 0)
            self.assertEqual(monthly["chart"]["maximum"], 2)
            self.assertEqual(monthly["chart"]["series"][0]["values"], [1])
            self.assertIn("not a fixed-cohort trend", monthly["caption"])
            self.assertIn("not a ranking", deck["slides"][3]["caption"])
            self.assertIn("table", deck["slides"][3])
            self.assertNotIn("reproduction passed", json.dumps(deck).lower())
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
            self.assertEqual(sources["chart_display_precision"]["decimal_places"], 4)
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

    def test_severity_contributions_reconcile_counts_and_additive_errors(self):
        ordinary = pd.Series(dict(n=80, mae=2., rmse=3., absolute_error_sum=160., squared_error_sum=720.))
        severe = pd.Series(dict(n=20, mae=8., rmse=12., absolute_error_sum=160., squared_error_sum=2880.))
        overall = pd.Series(dict(n=100, mae=3.2, rmse=6., absolute_error_sum=320., squared_error_sum=3600.))
        expected = dict(sample_share=.2, absolute_error_share=.5, squared_error_share=.8)
        self.assertEqual(presentation.severity_contributions(overall, ordinary, severe), expected)
        without_sums = lambda row: row.drop(["absolute_error_sum", "squared_error_sum"])
        self.assertEqual(presentation.severity_contributions(*map(without_sums, [overall, ordinary, severe])), expected)
        mismatched = severe.copy()
        mismatched["n"] += 1
        with self.assertRaisesRegex(ValueError, "partition"):
            presentation.severity_contributions(overall, ordinary, mismatched)
        inconsistent = overall.copy()
        inconsistent["absolute_error_sum"] = 321
        with self.assertRaisesRegex(ValueError, "reconcile"):
            presentation.severity_contributions(inconsistent, ordinary, severe)

    def test_excel_chart_precision_changes_display_values_only(self):
        values = [13.655428499127, 22.4984761293, 18.546450012, 0.000049999]
        rows = [pd.Series({"mae": value}) for value in values]
        displayed = presentation.chart_mae_values(rows)
        self.assertEqual([float(row.mae) for row in rows], values)
        for raw, shown in zip(values, displayed):
            self.assertEqual(Decimal(str(shown)).quantize(Decimal("0.0001")), Decimal(str(shown)))
            self.assertLessEqual(abs(Decimal(str(raw)) - Decimal(str(shown))), Decimal("0.00005"))

    def test_display_names_follow_model_type_without_claiming_a_fixed_winner(self):
        self.assertEqual(presentation.display_model_name("rf_d16_l50", "random_forest"), "Random forest")
        self.assertEqual(presentation.display_model_name("dt_d8_l200", "decision_tree"), "Decision tree")
        self.assertEqual(presentation.display_model_name("linear_g4b", "linear"), "OLS")
        self.assertEqual(presentation.display_model_name("training_median", "linear"), "Training median")

    def test_unique_positive_airport_is_named_with_its_archived_interval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = presentation_fixture(root)
            run_id = config["final_rq1_run_id"]
            directory = root / "experiments/runs" / run_id
            airport_file = directory / "rq1_standardized/rq1_airports.csv"
            airports = pd.read_csv(airport_file)
            selected = airports.airport_id.eq(2) & airports.threshold.eq("ge15")
            airports.loc[selected, ["standardized_mean", "ci_low", "ci_high"]] = [.436, -2.403, 3.012]
            airports.to_csv(airport_file, index=False)
            node_file = directory / "rq1_standardized/network_nodes.csv"
            nodes = pd.read_csv(node_file)
            nodes.loc[nodes.airport_id.eq(2), "airport_code"] = "FLL"
            nodes.to_csv(node_file, index=False)
            fixture.seal(root, run_id, json.loads((directory / "config.json").read_text(encoding="utf-8")))
            result = presentation.build_content(config, "reports/presentation/content.json", root=root)
            slide = json.loads(Path(result["content"]).read_text(encoding="utf-8"))["slides"][3]
            self.assertIn("1 negative estimates; FLL interval includes zero", slide["claims"])
            self.assertEqual(len(slide["claims"]), 2)
            self.assertEqual(slide["table"][0], ["Airport", "Retained n", "Δ (min)", "95% CI"])
            self.assertIn(["FLL", "2", "0.436", "[-2.403, 3.012]"], slide["table"])
            self.assertEqual(len(slide["table"]), 3)
            self.assertIn("includes zero", slide["notes"])
            self.assertIn("not an independent August validation", slide["notes"])
            self.assertIn('"airport_id": 2', " ".join(slide["citations"]))


if __name__ == "__main__":
    unittest.main()
