"""Execution disclosure tests use only synthetic completed summary archives."""
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

import pandas as pd

from scripts import build_report as report
from tests.test_report_evidence import make_fixture, seal, write_json, write_table


class ReportContentDisclosure(unittest.TestCase):
    def test_chinese_metric_citations_match_the_displayed_model_and_population(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            archive = report.Archive(root)
            validated = report.validate_inputs(archive, config)
            _, chinese = report.compose(archive, config, validated)
            evidence = {item["evidence_id"]: item for item in archive.evidence}
            matches = re.findall(r"^(R\d+) / ([^：]+)：MAE [^\n]+。\[(E\d+)\]$", chinese, re.M)
            self.assertEqual(len(matches), len(config["mainline_run_ids"]) + len(config["algorithm_run_ids"]) + 3)
            for run_id, model, evidence_id in matches:
                item = evidence[evidence_id]
                self.assertEqual(item["run_id"], run_id)
                self.assertEqual(item["source_file"], f"experiments/runs/{run_id}/metrics.csv")
                self.assertEqual(json.loads(item["selector"]), dict(model=model, fold="overall", group="all", value="all"))
            g0 = config["mainline_run_ids"]["G0"]
            first_g0_metric = next(item for item in archive.evidence if item["run_id"] == g0 and item["source_file"].endswith("/metrics.csv"))
            self.assertEqual(json.loads(first_g0_metric["selector"])["model"], "training_mean")
            g0_citation = next(evidence_id for run_id, _, evidence_id in matches if run_id == g0)
            self.assertNotEqual(g0_citation, first_g0_metric["evidence_id"])

    def test_code_overview_and_eda_rationale_are_explicit_and_source_backed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            archive = report.Archive(root)
            english, chinese = report.compose(archive, config, report.validate_inputs(archive, config))
            overview = english.split("## Overview of code structure\n\n", 1)[1].split("## Limitations", 1)[0]
            for module in ["run_pipeline.py", "prepare_flights.py", "build_features.py", "train_models.py", "evaluate_models.py",
                           "select_models.py", "final_evaluation.py", "reproduce_final.py", "render_research_notebooks.py",
                           "build_report.py", "build_presentation_content.py", "rq1_standardization.py"]:
                self.assertIn(module, overview)
            self.assertIn("do not fit models or reopen prediction rows", overview)
            rationale = next(paragraph for paragraph in english.split("\n\n") if paragraph.startswith("These exploratory observations"))
            self.assertLess(english.index(rationale), english.index("## Prediction timing"))
            for phrase in ["right tail", "month-to-month", "MAE", "RMSE", "training-median", "carry-forward", "severe-delay",
                           "forward validation", "explicit unknown", "family versus variant", "missingness", "common-availability",
                           "not a retrospective change", "preregistered before P0"]:
                self.assertIn(phrase, rationale)
            ids = re.findall(r"\[(E\d+)\]", rationale)
            referenced = [item for item in archive.evidence if item["evidence_id"] in ids]
            self.assertEqual(len(referenced), 3)
            self.assertEqual({Path(item["source_file"]).name for item in referenced},
                             {"eda_monthly_target.csv", "airline_aircraft_coverage.csv", "weather_coverage_summary.json"})
            self.assertTrue(all(item["research_log_heading_lines"] for item in referenced))
            self.assertIn("这是对既定方法的描述性解释", chinese)
            self.assertIn("未据此事后更改评价规则", chinese)


class ReproductionDisclosure(unittest.TestCase):
    def fixture(self, root):
        config = make_fixture(root)
        config["reproduction_run_id"] = "R0021"
        formal = root / "experiments/runs/R0017"
        directory = root / "experiments/runs/R0021"
        directory.mkdir()
        frozen = json.loads((formal / "model_selection.json").read_text())
        frozen.update(config={"threads": 8}, protocol={"timezone": "UTC"})
        frozen["payload_sha256"] = report.canonical({k: v for k, v in frozen.items() if k != "payload_sha256"})
        write_json(formal / "model_selection.json", frozen)
        model = json.loads((formal / "model_manifest.json").read_text())
        model.update(threads=8, fold_workers=1)
        write_json(formal / "model_manifest.json", model)
        original = json.loads((formal / "final_result.json").read_text())
        models = [model["model_name"], *report.BASELINES]
        original.update(mode="final", claim_type="formal", config=frozen["config"], protocol=frozen["protocol"],
                        models=models, evaluation_rows=2, folds=model["folds"], split_sha256=model["split_sha256"],
                        evaluation_ids_sha256=model["evaluation_ids_sha256"], feature_fingerprints={"synthetic_train": "a", "synthetic_test": "b"},
                        verified_inputs={"synthetic_sources": "same"}, freeze_payload_sha256=frozen["payload_sha256"],
                        freeze_manifest_sha256=report.digest(formal / "model_selection.json"))
        write_json(formal / "final_result.json", original)
        formal_config = json.loads((formal / "config.json").read_text())
        formal_config["threads"] = 8
        seal(root, "R0017", formal_config)
        ledger = root / "experiments/final_access/2026-08.json"
        access = {"status": "completed", "run_id": "R0017", "final_result_sha256": report.digest(formal / "final_result.json"),
                  "freeze_manifest_sha256": original["freeze_manifest_sha256"]}
        write_json(ledger, access)
        reference = {"mode": "reproduce_final", "claim_type": "reproduction", "formal_run_id": "R0017", "formal_run_dir": str(formal),
                     "formal_final_result_sha256": access["final_result_sha256"], "freeze_manifest_sha256": access["freeze_manifest_sha256"],
                     "access_ledger_path": str(ledger), "ledger_sha256_before": report.digest(ledger),
                     "creates_new_formal_claim": False, "refitted": True, "raw_test_data_read": True}
        comparison = {"prediction_rows_compared": 8, "metric_rows_compared": len(pd.read_csv(formal / "metrics.csv")),
                      "all_feature_values_and_types_match": True, "ids_and_targets_match": True,
                      "prediction_atol": 1e-10, "metric_atol": 1e-10, "rtol": 0}
        repeated = json.loads(json.dumps(original))
        repeated.update(mode="reproduce_final", claim_type="reproduction", reproduction_reference=reference,
                        reproduction_comparison=comparison)
        verified = {**reference, **comparison, "verified": True, "ledger_sha256_after": report.digest(ledger), "formal_ledger_unchanged": True}
        differences = [{"model": model_name, "column": column, "rows": 2, "max_absolute_difference": 2e-14 if column.endswith("raw") else 1e-14,
                        "atol": 1e-10, "rtol": 0.} for model_name in models for column in ["predicted_delay_raw", "predicted_delay"]]
        shutil.copyfile(formal / "model_selection.json", directory / "model_selection.json")
        shutil.copyfile(formal / "metrics.csv", directory / "metrics.csv")
        self.persist(root, reference, verified, repeated, differences)
        index = pd.read_csv(root / "experiments/index.csv").to_dict("records")
        index.append({"run_id": "R0021", "parent_run_id": "R0017", "status": "completed", "artifact_dir": "experiments/runs/R0021",
                      "change": "Synthetic full reproduction", "decision": "same frozen computation"})
        write_table(root / "experiments/index.csv", index)
        with (root / "research/研究记录.md").open("a", encoding="utf-8") as stream:
            stream.write("\n### Synthetic — R0021\n")
        return config, reference, verified, repeated, differences

    def persist(self, root, reference, verified, repeated, differences):
        directory = root / "experiments/runs/R0021"
        for name, value in [("reproduction_reference.json", reference), ("reproduction_verification.json", verified), ("final_result.json", repeated)]:
            write_json(directory / name, value)
        write_table(directory / "reproduction_prediction_differences.csv", differences)
        seal(root, "R0021", {"mode": "reproduce_final", "parent_run_id": "R0017"})

    def test_completed_reproduction_prose_uses_only_archived_summary_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, *_ = self.fixture(root)
            before = {path: report.digest(path) for path in (root / "experiments").rglob("*") if path.is_file()}
            result = report.build_report(config, root=root)
            text = (Path(result["output_dir"]) / "report_en.md").read_text(encoding="utf-8")
            chinese = (Path(result["output_dir"]) / "结果解读_中文.md").read_text(encoding="utf-8")
            self.assertIn("Completed reproduction R0021", text)
            self.assertIn("2e-14/1e-14 min", text)
            self.assertIn("eight raw/clipped comparison rows", text)
            self.assertIn("already-seen test set, not a new independent test", text)
            self.assertIn("仅读归档JSON/CSV执行证据", chinese)
            self.assertEqual(before, {path: report.digest(path) for path in before})

    def test_missing_optional_reproduction_does_not_open_an_extra_run(self):
        self.assertEqual(report.reproduction_prose(None, {}, None), ([], []))

    def test_invalid_reproduction_evidence_is_rejected(self):
        faults = ["wrong_mode", "wrong_formal", "freeze", "ledger", "refitted", "new_claim", "types", "ids", "tolerance",
                  "missing_pair", "duplicate_pair", "nonfinite", "difference", "rows", "metric_rows", "missing_receipt"]
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config, reference, verified, repeated, differences = self.fixture(root)
                directory = root / "experiments/runs/R0021"
                if fault == "wrong_formal":
                    reference["formal_run_id"] = verified["formal_run_id"] = "R0001"
                elif fault == "freeze":
                    with (directory / "model_selection.json").open("a") as stream:
                        stream.write("\n")
                elif fault == "ledger":
                    verified["ledger_sha256_after"] = "changed"
                elif fault == "refitted":
                    reference["refitted"] = verified["refitted"] = False
                elif fault == "new_claim":
                    reference["creates_new_formal_claim"] = verified["creates_new_formal_claim"] = True
                elif fault == "types":
                    verified["all_feature_values_and_types_match"] = False
                elif fault == "ids":
                    repeated["evaluation_ids_sha256"] = "different"
                elif fault == "tolerance":
                    verified["prediction_atol"] = 1e-8
                elif fault == "missing_pair":
                    differences.pop()
                elif fault == "duplicate_pair":
                    differences[-1] = differences[0].copy()
                elif fault == "nonfinite":
                    differences[0]["max_absolute_difference"] = float("nan")
                elif fault == "difference":
                    differences[0]["max_absolute_difference"] = 1.1e-10
                elif fault == "rows":
                    differences[0]["rows"] = 3
                elif fault == "metric_rows":
                    verified["metric_rows_compared"] += 1
                self.persist(root, reference, verified, repeated, differences)
                if fault == "wrong_mode":
                    seal(root, "R0021", {"mode": "final", "parent_run_id": "R0017"})
                if fault == "missing_receipt":
                    manifest_path = directory / "manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["outputs"] = [item for item in manifest["outputs"] if not item["path"].endswith("reproduction_prediction_differences.csv")]
                    write_json(manifest_path, manifest)
                archive = report.Archive(root)
                with self.assertRaises(ValueError):
                    report.reproduction_prose(archive, config, report.model_archive(archive, "R0017", development=False))
                self.assertFalse((root / "reports").exists())


class OutcomeTradeoffDisclosure(unittest.TestCase):
    def test_network_redraw_keeps_global_route_selection_then_filters_within_panels(self):
        with tempfile.TemporaryDirectory() as temp:
            nodes = pd.DataFrame([{"airport_id": i, "longitude": -110 + i / 10, "latitude": 35., "valid_connection_count": i,
                                   "standardized_mean": -1. if i == 1 else float("nan")} for i in range(1, 102)])
            nodes = pd.concat([nodes, pd.DataFrame([{"airport_id": 102, "longitude": 170., "latitude": 60., "valid_connection_count": 10,
                                                     "standardized_mean": float("nan")}])], ignore_index=True)
            # The largest route crosses panels. Exactly 99 of the global top100
            # should be shown; taking top100 separately per panel would show 100.
            edges = pd.DataFrame([{"origin_airport_id": 1, "destination_airport_id": i, "flight_count": 102 - i} for i in range(2, 102)] +
                                 [{"origin_airport_id": 1, "destination_airport_id": 102, "flight_count": 1000}])
            destination = Path(temp) / "network.png"
            result = report.render_network_figure(nodes, edges, destination)
            self.assertEqual(result["coordinate_supported_nodes"], 102)
            self.assertEqual(len(result["global_top_routes"]), 100)
            self.assertEqual(result["displayed_routes"], 99)
            self.assertEqual([len(panel["node_ids"]) for panel in result["panels"]], [101, 1, 0, 0])
            self.assertEqual(result["gray_nodes"], 101)
            self.assertFalse(result["airport_code_labels"])
            self.assertGreaterEqual(result["minimum_font_points"], 10)
            self.assertEqual(report.digest(destination), result["image_sha256"])
            with self.assertRaises(FileExistsError):
                report.render_network_figure(nodes, edges, destination)

    def test_overall_gain_does_not_hide_ordinary_mae_loss_or_severe_error_concentration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            directory = root / "experiments/runs/R0017"
            rows = []
            for model, ordinary_mae, ordinary_rmse, severe_mae, severe_rmse in [("r0001", 4., 5., 20., 25.), ("training_median", 2., 6., 30., 40.)]:
                for value, n, mae, rmse in [("ordinary_lt60", 8, ordinary_mae, ordinary_rmse), ("severe_ge60", 2, severe_mae, severe_rmse),
                                           ("all", 10, (8 * ordinary_mae + 2 * severe_mae) / 10, ((8 * ordinary_rmse**2 + 2 * severe_rmse**2) / 10)**.5)]:
                    rows.append({"model": model, "fold": "overall", "group": "all" if value == "all" else "severity", "value": value,
                                 "n": n, "mae": mae, "rmse": rmse, "unit": "minutes"})
            write_table(directory / "metrics.csv", rows)
            seal(root, "R0017", json.loads((directory / "config.json").read_text()))
            archive = report.Archive(root)
            run = {"id": "R0017", "metrics": archive.table("R0017", "metrics.csv", report.METRIC_COLUMNS)}
            en, zh = report.final_tradeoff_prose(archive, run, "r0001")
            text = " ".join(en)
            self.assertIn("overall MAE by 5.26%", text)
            self.assertIn("4.0000 versus 2.0000 min (worse; signed change +2.0000)", text)
            self.assertIn("5.0000 versus 6.0000 min (better)", text)
            self.assertIn("55.5556% of primary absolute error", text)
            self.assertIn("86.2069% of primary squared error", text)
            self.assertIn("no deployment switching rule or subgroup significance test", text)
            self.assertIn("这不是航班实际延误节省", " ".join(zh))
            run["metrics"].loc[run["metrics"].value.eq("ordinary_lt60"), "n"] = 9
            with self.assertRaisesRegex(ValueError, "do not reconcile"):
                report.final_tradeoff_prose(archive, run, "r0001")

    def test_rq1_three_retention_denominators_and_cross_zero_example_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            directory = root / "experiments/runs/R0019"
            write_table(directory / "tables/rq1_airport_raw.csv", [
                {"airport_id": airport, "airport_code": code, "n": n} for airport, code, n in [(1, "AAA", 100), (2, "FLL", 200), (3, "CCC", 700)]])
            write_table(directory / "rq1_standardized/rq1_airports.csv", [
                {"airport_id": airport, "threshold": "ge15", "supported": True, "retained_n": retained, "retained_share": .9,
                 "standardized_mean": mean, "ci_low": low, "ci_high": high} for airport, retained, mean, low, high in [(1, 90, -1., -2., -.1), (2, 180, 1., -1., 2.)]])
            path = directory / "rq1_standardized/rq1_threshold_stability.csv"
            stability = pd.read_csv(path)
            stability.loc[stability.threshold.eq("gt0"), "direction_agreement"] = .5
            stability.to_csv(path, index=False)
            seal(root, "R0019", json.loads((directory / "config.json").read_text()))
            en, zh = report.rq1_support_prose(report.Archive(root), "R0019")
            text = " ".join(en)
            self.assertIn("270/1,000 (27.000%)", text)
            self.assertIn("270/300 (90.000%)", text)
            self.assertIn("1 negative and 1 positive", text)
            self.assertIn("FLL: 1.000 [-1.000, 2.000]", text)
            self.assertIn("gt0: 1/2, ge5: 2/2", text)
            self.assertIn("并非完全方向稳健", " ".join(zh))


class CoefficientDisclosure(unittest.TestCase):
    def test_included_upstream_requires_its_numeric_row_and_is_then_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            source_dir = root / "experiments/runs/R0001"
            model = json.loads((source_dir / "model_manifest.json").read_text())
            model["features"].append("upstream_arr_delay_minutes")
            write_json(source_dir / "model_manifest.json", model)
            seal(root, "R0001", json.loads((source_dir / "config.json").read_text()))
            directory = root / "experiments/runs/R0016"
            base = directory / "explanations/R0001"
            summary = json.loads((base / "explain_summary.json").read_text())
            summary["features"] = model["features"]
            write_json(base / "explain_summary.json", summary)
            seal(root, "R0016", json.loads((directory / "config.json").read_text()))
            archive = report.Archive(root)
            with self.assertRaisesRegex(ValueError, "Expected exactly one result row"):
                report.ols_coefficient_prose(archive, "R0016", report.model_archive(archive, "R0001"))
            path = base / "explain_coefficients.csv"
            rows = pd.read_csv(path).to_dict("records")
            rows.append({"fold": "2026-05", "encoded_feature": "numeric__upstream_arr_delay_minutes", "coefficient": 12.5})
            write_table(path, rows)
            seal(root, "R0016", json.loads((directory / "config.json").read_text()))
            archive = report.Archive(root)
            en, _ = report.ols_coefficient_prose(archive, "R0016", report.model_archive(archive, "R0001"))
            self.assertIn("upstream_arr_delay_minutes=12.500000", " ".join(en))

    def test_prespecified_coefficients_use_feature_source_even_when_tree_is_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            archive = report.Archive(root)
            source = report.model_archive(archive, "R0001")
            en, zh = report.ols_coefficient_prose(archive, "R0016", source)
            self.assertIn("feature-source run R0001", " ".join(en))
            self.assertIn("upstream_arr_delay_minutes was not included", " ".join(en))
            self.assertIn("downstream_distance=1.250000", " ".join(en))
            self.assertIn("1个训练期标准差", " ".join(zh))
            self.assertTrue(any('numeric__scheduled_turnaround_minutes' in row["selector"] for row in archive.evidence))
            self.assertTrue(all("R0009" not in row["source_file"] for row in archive.evidence))

    def test_wrong_source_missing_coefficient_and_missing_intercept_are_rejected(self):
        for fault in ["source", "coefficient", "intercept", "refit", "nonlinear"]:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                make_fixture(root)
                directory = root / "experiments/runs/R0016"
                base = directory / "explanations/R0001"
                summary = json.loads((base / "explain_summary.json").read_text())
                if fault == "source":
                    summary["source_run"] = "R0009"
                elif fault == "refit":
                    summary["fit_called"] = True
                elif fault == "coefficient":
                    path = base / "explain_coefficients.csv"
                    pd.read_csv(path).iloc[1:].to_csv(path, index=False)
                elif fault == "intercept":
                    write_table(base / "explain_intercepts.csv", [{"fold": "2026-07", "intercept": 1.}])
                write_json(base / "explain_summary.json", summary)
                seal(root, "R0016", json.loads((directory / "config.json").read_text()))
                archive = report.Archive(root)
                source = report.model_archive(archive, "R0009" if fault == "nonlinear" else "R0001")
                with self.assertRaises(ValueError):
                    report.ols_coefficient_prose(archive, "R0016", source)


class ExecutionDisclosure(unittest.TestCase):
    def test_algorithm_archive_can_declare_no_repeated_baselines(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            for run_id in ["R0009", "R0013"]:
                directory = root / "experiments/runs" / run_id
                config = json.loads((directory / "config.json").read_text())
                model = json.loads((directory / "model_manifest.json").read_text())
                config["include_baselines"] = model["include_baselines"] = False
                write_json(directory / "model_manifest.json", model)
                metrics = pd.read_csv(directory / "metrics.csv")
                metrics.loc[metrics.model.eq(model["model_name"])].to_csv(directory / "metrics.csv", index=False)
                seal(root, run_id, config)
                actual = report.model_archive(report.Archive(root), run_id)
                self.assertEqual(set(actual["metrics"].model), {model["model_name"]})

    def run_with_budget(self, root, run_id, *, threads=None, workers=None, manifest=True):
        directory = root / "experiments/runs" / run_id
        config = json.loads((directory / "config.json").read_text())
        model_path = directory / "model_manifest.json"
        model = json.loads(model_path.read_text())
        for name, value in [("threads", threads), ("fold_workers", workers)]:
            if value is not None:
                config[name] = value
                if manifest:
                    model[name] = value
        write_json(model_path, model)
        seal(root, run_id, config)

    def freeze(self, root, threads, delta=None):
        directory = root / "experiments/runs/R0017"
        path = directory / "model_selection.json"
        value = json.loads(path.read_text())
        value["config"] = {"threads": threads}
        if delta is not None:
            value["execution_delta"] = delta
        value["payload_sha256"] = report.canonical({k: v for k, v in value.items() if k != "payload_sha256"})
        write_json(path, value)
        seal(root, "R0017", json.loads((directory / "config.json").read_text()))

    def prose(self, root, ids, selected):
        archive = report.Archive(root)
        runs = [report.model_archive(archive, run_id) for run_id in ids]
        final = report.model_archive(archive, "R0017", development=False)
        en, zh = report.execution_prose(archive, runs, next(run for run in runs if run["id"] == selected), final)
        return " ".join(en), " ".join(zh), archive.evidence

    def test_mixed_budgets_and_explicit_override_are_source_backed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            for run_id in ["R0001", "R0013"]:
                self.run_with_budget(root, run_id, threads=8, workers=3, manifest=run_id != "R0001")
            self.run_with_budget(root, "R0014", threads=24, workers=1)
            self.run_with_budget(root, "R0017", threads=24, workers=1)
            self.freeze(root, 24, {"explicitly_requested": True, "learning_parameters_unchanged": True,
                                   "threads": {"development": 8, "final": 24}})
            en, zh, evidence = self.prose(root, ["R0001", "R0013", "R0014"], "R0013")
            self.assertIn("R0001, R0013 used 8 threads per fold and up to 3", en)
            self.assertIn("R0014 used 24 threads per fold and up to 1", en)
            self.assertIn("from 8 development threads per fold to 24 final threads", en)
            self.assertIn("same freeze, including its 24 training threads and 24 prediction threads", en)
            self.assertIn("atol=1e-10 and rtol=0", en)
            self.assertIn("不能据运行时长判断算法优劣", zh)
            self.assertTrue(any(row["source_file"].endswith("R0001/config.json") for row in evidence))
            self.assertTrue(any("execution_delta" in row["selector"] for row in evidence))

    def test_same_budget_needs_no_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            for run_id in ["R0014", "R0017"]:
                self.run_with_budget(root, run_id, threads=24, workers=1)
            self.freeze(root, 24)
            en, _, _ = self.prose(root, ["R0014"], "R0014")
            self.assertIn("no execution override is recorded", en)
            self.assertNotIn("from 8 development", en)

    def rf_prediction_fixture(self, root, development_threads=24, final_fit=24, final_prediction=1):
        make_fixture(root)
        self.run_with_budget(root, "R0013", threads=development_threads, workers=1)
        self.run_with_budget(root, "R0017", threads=24, workers=1)
        directory = root / "experiments/runs/R0017"
        model = json.loads((directory / "model_manifest.json").read_text())
        model.update(model_type="random_forest", fit_threads=final_fit, prediction_threads=final_prediction)
        write_json(directory / "model_manifest.json", model)
        config = json.loads((directory / "config.json").read_text())
        config["model_type"] = "random_forest"
        seal(root, "R0017", config)
        delta = {"explicitly_requested": True, "learning_parameters_unchanged": True,
                 "prediction_threads": {"development": development_threads, "final": 1}}
        if development_threads != 24:
            delta["threads"] = {"development": development_threads, "final": 24}
        self.freeze(root, 24, delta)
        path = directory / "model_selection.json"
        frozen = json.loads(path.read_text())
        frozen["config"].update(model_type="random_forest", prediction_threads=1)
        frozen["payload_sha256"] = report.canonical({k: v for k, v in frozen.items() if k != "payload_sha256"})
        write_json(path, frozen)
        seal(root, "R0017", config)

    def test_frozen_rf_prediction_budget_is_separate_with_prediction_only_or_combined_delta(self):
        for development_threads in [24, 8]:
            with self.subTest(development_threads=development_threads), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.rf_prediction_fixture(root, development_threads)
                en, zh, evidence = self.prose(root, ["R0013"], "R0013")
                self.assertIn("24 threads for training and 1 for prediction", en)
                self.assertIn(f"prediction threads changing from {development_threads} to 1", en)
                self.assertIn("restores the training n_jobs", en)
                self.assertIn("bare predict does not preserve this prediction policy", en)
                self.assertIn("including unnormalised error sums", en)
                self.assertIn("训练24线程、预测1线程", zh)
                self.assertTrue(any('config.prediction_threads' in row["selector"] for row in evidence))
                self.assertTrue(any('fit_threads' in row["selector"] for row in evidence))
                if development_threads == 24:
                    self.assertIn("no training-thread override is recorded", en)
                    self.assertNotIn("from 24 development threads per fold to", en)

    def test_rf_prediction_and_fit_manifest_conflicts_are_rejected(self):
        for fault in ["prediction", "fit", "missing_prediction", "delta", "non_rf"]:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.rf_prediction_fixture(root, final_fit=8 if fault == "fit" else 24,
                                           final_prediction=24 if fault == "prediction" else 1)
                directory = root / "experiments/runs/R0017"
                if fault == "missing_prediction":
                    path = directory / "model_manifest.json"
                    model = json.loads(path.read_text())
                    model.pop("prediction_threads")
                    write_json(path, model)
                if fault in {"delta", "non_rf"}:
                    path = directory / "model_selection.json"
                    frozen = json.loads(path.read_text())
                    if fault == "delta":
                        frozen["execution_delta"]["prediction_threads"]["development"] = 8
                    else:
                        frozen["config"]["model_type"] = "linear"
                    write_json(path, frozen)
                seal(root, "R0017", json.loads((directory / "config.json").read_text()))
                with self.assertRaisesRegex(ValueError, "prediction|fit_threads"):
                    self.prose(root, ["R0013"], "R0013")

    def test_legacy_fixture_missing_fields_is_not_assumed_to_use_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            en, zh, _ = self.prose(root, ["R0001"], "R0001")
            self.assertIn("threads per fold=not recorded", en)
            self.assertIn("does not record frozen execution threads", en)
            self.assertIn("未记录冻结线程数", zh)
            self.assertNotIn("8 threads", en)

    def test_unrecorded_override_or_conflicting_manifest_is_rejected(self):
        for conflict in ["override", "manifest"]:
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                make_fixture(root)
                self.run_with_budget(root, "R0013", threads=8, workers=3)
                self.run_with_budget(root, "R0017", threads=24, workers=1)
                self.freeze(root, 24)
                if conflict == "manifest":
                    directory = root / "experiments/runs/R0013"
                    path = directory / "model_manifest.json"
                    model = json.loads(path.read_text())
                    model["threads"] = 4
                    write_json(path, model)
                    seal(root, "R0013", json.loads((directory / "config.json").read_text()))
                with self.assertRaisesRegex(ValueError, "execution_delta|manifest/config disagree"):
                    self.prose(root, ["R0013"], "R0013")


if __name__ == "__main__":
    unittest.main()
