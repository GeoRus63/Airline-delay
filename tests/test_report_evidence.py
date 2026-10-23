"""Small synthetic report archives: no real August read and no estimator fit."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import build_report as report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def seal(root, run_id, config):
    directory = root / "experiments/runs" / run_id
    write_json(directory / "config.json", config)
    outputs = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": report.digest(path)}
               for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.json"]
    write_json(directory / "manifest.json", {"run_id": run_id, "status": "completed", "config": config, "outputs": outputs})


def make_fixture(root):
    configs = {}

    def run(run_id, mode="development", parent="", group="G0", kind="linear", params=None, features=None, score=10., **extra):
        config = dict(mode=mode, parent_run_id=parent, feature_group=group, model_type=kind, model_name=run_id.lower(), **extra)
        configs[run_id] = config
        directory = root / "experiments/runs" / run_id
        directory.mkdir(parents=True)
        if mode != "development" and mode != "final":
            return directory
        final = mode == "final"
        months = [8] if final else [5, 6, 7]
        folds = []
        for month in months:
            folds.append({"fold": f"2026-{month:02d}", "validation_start": f"2026-{month:02d}-01T00:00:00+00:00",
                          "validation_end_exclusive": f"2026-{month + 1:02d}-01T00:00:00+00:00",
                          "max_train_prediction_time": "2026-04-30T23:00:00+00:00", "max_train_label_time": "2026-04-30T23:30:00+00:00",
                          "validation_target_sha256": "synthetic-targets"})
        model = {"development_only": not final, "model_name": run_id.lower(), "model_type": kind, "feature_group": group,
                 "features": features or ["synthetic"], "model_params": params or {"tol": 1e-6},
                 "effective_preprocessing": {"drop": "first"}, "evaluation_ids_sha256": "synthetic-ids-final" if final else "synthetic-ids",
                 "split_sha256": "synthetic-splits-final" if final else "synthetic-splits", "evaluation_rows": len(months) * 2, "folds": folds}
        write_json(directory / "model_manifest.json", model)
        metrics = []
        for model_name in [run_id.lower(), *report.BASELINES]:
            error = score if model_name == run_id.lower() else 8.
            for fold in [*[f"2026-{month:02d}" for month in months], "overall"]:
                metrics.append({"model": model_name, "fold": fold, "group": "all", "value": "all", "n": len(months) * 2 if fold == "overall" else 2,
                                "mae": error, "rmse": error, "unit": "minutes"})
            if final:
                for grouping, value in [("severity", "ordinary_lt60"), ("severity", "severe_ge60"),
                                         *[(group, "synthetic") for group in ["airport", "airline", "aircraft_family", "day_of_week", "scheduled_hour", "overnight"]]]:
                    metrics.append({"model": model_name, "fold": "overall", "group": grouping, "value": value,
                                    "n": 1, "mae": error, "rmse": error, "unit": "minutes"})
        write_table(directory / "metrics.csv", metrics)
        return directory

    ids = {group: f"R{index:04d}" for index, group in enumerate(report.STAGES, 1)}
    main = list(ids.values())
    for index, (group, run_id) in enumerate(ids.items()):
        run(run_id, parent=main[index - 1] if index else "", group=group, score=10. + index)
    run("R0007", parent="R0003", group="G2_VARIANT", score=12.)
    run("R0008", parent="R0006", group="G4B", noaa_backfill=True, score=15.)
    algorithm_ids = []
    for index, (kind, depth, leaf) in enumerate([("decision_tree", depth, leaf) for depth in [8, 16] for leaf in [50, 200]] + [("random_forest", 16, leaf) for leaf in [50, 200]], 9):
        run_id = f"R{index:04d}"
        params = dict(max_depth=depth, min_samples_leaf=leaf, random_state=42)
        if kind == "random_forest":
            params.update(n_estimators=200, max_features=.5)
        run(run_id, parent="R0001", kind=kind, params=params, feature_source_run_id="R0001")
        algorithm_ids.append(run_id)
    feature_dir = run("R0015", mode="select_features", parent="R0006")
    model_dir = run("R0016", mode="select_model", parent="R0015")
    def candidate_receipts(candidates):
        result = []
        for run_id in candidates:
            seal(root, run_id, configs[run_id])
            for filename in ["manifest.json", "config.json", "model_manifest.json", "metrics.csv"]:
                path = root / "experiments/runs" / run_id / filename
                result.append({"path": path.relative_to(root).as_posix(), "sha256": report.digest(path), "bytes": path.stat().st_size})
        return result
    feature = {"selection_type": "features", "selected_run_id": "R0001", "selected_pooled_mae": 10., "reason": "synthetic minimum", "candidate_files": []}
    feature["selection_sha256"] = report.canonical(feature)
    write_json(feature_dir / "selection/feature_selection.json", feature)
    choice = {"selection_type": "model", "selected_run_id": "R0001", "feature_source_run_id": "R0001", "primary_model_name": "training_median", "reason": "synthetic baseline wins", "promotion_decisions": [], "candidate_files": []}
    choice["selection_sha256"] = report.canonical(choice)
    write_json(model_dir / "selection/model_selection_decision.json", choice)
    write_json(model_dir / "explanations/R0001/explain_summary.json", {"source_run": "R0001", "fit_called": False})
    write_table(model_dir / "explanations/R0001/permutation_summary.csv", [{"fold": "2026-05", "group": "G0", "mean_delta_mae": .2, "sd_delta_mae": .1, "repeats": 3}])
    for directory, candidates in [(feature_dir, main), (model_dir, ["R0001", *algorithm_ids])]:
        rows = []
        for value in candidates:
            metrics = pd.read_csv(root / "experiments/runs" / value / "metrics.csv")
            pooled = metrics.loc[metrics.model.eq(value.lower()) & metrics.fold.eq("overall")].iloc[0]
            rows.append({"run_id": value, "model": value.lower(), "pooled_mae": pooled.mae, "pooled_rmse": pooled.rmse})
        write_table(directory / "selection/candidates.csv", rows)

    def comparison(owner, reference, candidate, folder="comparison", difference=0.):
        directory = root / "experiments/runs" / owner / folder
        write_json(directory / "comparison.json", {"reference_run": reference, "candidate_run": candidate, "sample_scope": "full_evaluation"})
        write_table(directory / "comparison_metrics.csv", [{"scope": "pooled", "metric": "mae", "n": 6,
                    "difference_candidate_minus_reference": difference, "ci95_low": difference - 1., "ci95_high": difference + 1.}])
    for left, right in zip(main, main[1:]):
        comparison(right, left, right, difference=1.)
    comparison("R0007", "R0003", "R0007")
    comparison("R0008", "R0006", "R0008")
    for reference, candidate, label in [("R0004", "R0005", "G3_to_G4a"), ("R0005", "R0006", "G4a_to_G4b"), ("R0004", "R0006", "G3_to_G4b")]:
        comparison("R0006", reference, candidate, folder="common_weather/" + label)
    write_json(root / "experiments/runs/R0006/common_weather/definition.json", {"refitted": False, "uses_target_for_selection": False,
               "n": 6, "full_validation_n": 6, "definition": "synthetic availability only"})
    for run_id in ["R0005", "R0006", "R0008"]:
        write_json(root / "experiments/runs" / run_id / "weather_coverage_summary.json", {"rows": 8, "availability": {"metar_available": 7}})
    first = root / "experiments/runs/R0001"
    write_json(first / "eda_summary.json", {"n": 8, "airport_n": 1, "route_n": 1, "overnight_n": 1})
    write_table(first / "tables/eda_monthly_target.csv", [{"month_utc": "2026-04", "n": 8, "mean": 10., "median": 0., "p99": 60., "target_ge60_share": .1}])
    write_table(first / "tables/eda_target_bins.csv", [{"bin": "0", "n": 4}])
    (first / "figures").mkdir()
    (first / "figures/eda_target_distribution.png").write_bytes(b"synthetic png stand-in")

    final_dir = run("R0017", mode="final", parent="R0016")
    configs["R0017"]["model_name"] = "r0001"
    model = json.loads((final_dir / "model_manifest.json").read_text())
    model["model_name"] = "r0001"
    write_json(final_dir / "model_manifest.json", model)
    metrics = pd.read_csv(final_dir / "metrics.csv")
    metrics.loc[metrics.model.eq("r0017"), "model"] = "r0001"
    metrics.to_csv(final_dir / "metrics.csv", index=False)
    frozen = {"status": "frozen", "selection": {"selection_sha256": choice["selection_sha256"], "selected_run_id": "R0001"}, "primary_model_name": "training_median"}
    frozen["payload_sha256"] = report.canonical(frozen)
    write_json(final_dir / "model_selection.json", frozen)
    final = {"freeze_payload_sha256": frozen["payload_sha256"], "freeze_manifest_sha256": report.digest(final_dir / "model_selection.json"), "primary_model_name": "training_median"}
    write_json(final_dir / "final_result.json", final)
    write_json(final_dir / "final_verification.json", {"verified": True, "metrics_recomputed": True, "refitted": False, "raw_test_data_read": False})
    write_json(root / "experiments/final_access/2026-08.json", {"status": "completed", "run_id": "R0017", "final_result_sha256": report.digest(final_dir / "final_result.json")})
    write_table(final_dir / "final_comparisons.csv", [{"scope": "pooled", "metric": "mae", "n": 2, "difference_candidate_minus_reference": 0., "ci95_low": 0., "ci95_high": 0., "reference_model": name, "candidate_model": "training_median"} for name in report.BASELINES])
    write_table(final_dir / "residual_summary.csv", [{"model": "training_median", "n": 2, "mean_signed_error": -2., "median_signed_error": 0., "p95_absolute_error": 20.}])
    (final_dir / "figures").mkdir()
    (final_dir / "figures/final_residual_distribution.png").write_bytes(b"synthetic png stand-in")

    for run_id, mode in [("R0018", "analysis"), ("R0019", "rq1_final")]:
        directory = run(run_id, mode=mode, parent="R0017" if mode == "rq1_final" else "R0006", prediction_final_run_id="R0017")
        write_json(directory / "rq1_standardized/frozen_reference.json", {"synthetic": True})
        write_json(directory / "rq1_complete_summary.json", {"development": mode != "rq1_final", "frozen_reference_sha256": report.digest(directory / "rq1_standardized/frozen_reference.json")})
        write_table(directory / "tables/rq1_airport_raw.csv", [{"airport_id": 1, "airport_code": "AAA", "n": 2, "mean_delta": -5., "s_plus": 1., "s_minus": 11., "s_net": -10.}])
        write_table(directory / "rq1_standardized/rq1_airports.csv", [{"airport_id": 1, "threshold": "ge15", "supported": True, "retained_n": 2, "retained_share": 1., "standardized_mean": -5., "ci_low": -6., "ci_high": -4., "valid_replicates": 1000}])
        write_table(directory / "rq1_standardized/rq1_threshold_stability.csv", [{"threshold": threshold, "common_airports": 1, "frozen_airports": 1, "direction_agreement": 1., "spearman": None} for threshold in ["gt0", "ge5", "ge15"]])
        write_table(directory / "rq1_standardized/rq1_airports_monthly.csv", [{"month_utc": "2026-04", "threshold": "ge15", "supported": True, "standardized_mean": -5., "retained_share": 1.}])
        write_table(directory / "rq1_standardized/network_nodes.csv", [{"airport_id": 1, "valid_connection_count": 2, "standardized_mean": -5.}])
        write_table(directory / "rq1_standardized/network_edges.csv", [{"origin_airport_id": 1, "destination_airport_id": 2, "flight_count": 8}])
        (directory / "figures").mkdir()
        for name in ["rq1_standardized_airports", "rq1_threshold_stability", "rq1_network_geographic"]:
            (directory / "figures" / (name + ".png")).write_bytes(b"synthetic png stand-in")

    for run_id, config in configs.items():
        seal(root, run_id, config)
    # Seal candidate receipts only after all candidate comparison/EDA outputs exist.
    feature["candidate_files"] = candidate_receipts(main)
    feature["selection_sha256"] = report.canonical({k: v for k, v in feature.items() if k != "selection_sha256"})
    choice["candidate_files"] = candidate_receipts(["R0001", *algorithm_ids])
    choice["selection_sha256"] = report.canonical({k: v for k, v in choice.items() if k != "selection_sha256"})
    write_json(feature_dir / "selection/feature_selection.json", feature)
    write_json(model_dir / "selection/model_selection_decision.json", choice)
    frozen["selection"]["selection_sha256"] = choice["selection_sha256"]
    frozen["payload_sha256"] = report.canonical({k: v for k, v in frozen.items() if k != "payload_sha256"})
    write_json(final_dir / "model_selection.json", frozen)
    final.update(freeze_payload_sha256=frozen["payload_sha256"], freeze_manifest_sha256=report.digest(final_dir / "model_selection.json"))
    write_json(final_dir / "final_result.json", final)
    write_json(root / "experiments/final_access/2026-08.json", {"status": "completed", "run_id": "R0017", "final_result_sha256": report.digest(final_dir / "final_result.json")})
    for run_id in ["R0015", "R0016", "R0017"]:
        seal(root, run_id, configs[run_id])
    rows = [{"run_id": run_id, "parent_run_id": config["parent_run_id"], "status": "completed", "artifact_dir": f"experiments/runs/{run_id}",
             "change": "synthetic change", "decision": "synthetic retained"} for run_id, config in configs.items()]
    rows.append({"run_id": "R0020", "parent_run_id": "", "status": "invalidated", "artifact_dir": "experiments/runs/R0020", "change": "synthetic boundary correction", "decision": "not evidence"})
    write_table(root / "experiments/index.csv", rows)
    (root / "research").mkdir()
    (root / "research/研究记录.md").write_text("\n".join("### Synthetic — " + row["run_id"] for row in rows), encoding="utf-8")
    (root / "research/literature_review.md").write_text("Synthetic literature fixture", encoding="utf-8")
    for filename in ["references/ind5003-2610_projects.pdf", "references/papers/1-s2.0-S1366554524001169-main.pdf", "references/papers/2211.08140v2.pdf"]:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic reference receipt only")
    return {"mainline_run_ids": ids, "variant_run_id": "R0007", "noaa_run_id": "R0008", "algorithm_run_ids": algorithm_ids,
            "feature_selection_run_id": "R0015", "model_selection_run_id": "R0016", "final_prediction_run_id": "R0017",
            "development_rq1_run_id": "R0018", "final_rq1_run_id": "R0019"}


class ReportEvidence(unittest.TestCase):
    def test_complete_report_is_summary_only_traceable_and_nonmutating(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            original = {str(path): report.digest(path) for path in root.rglob("*") if path.is_file()}
            with patch.object(pd, "read_pickle", side_effect=AssertionError("pickle forbidden")):
                result = report.build_report(config, root=root)
            self.assertGreaterEqual(result["evidence_items"], 80)
            self.assertFalse(result["pdf_verified"])
            for name, expected in original.items():
                self.assertEqual(report.digest(name), expected)
            english = (root / "reports/report_en.md").read_text(encoding="utf-8")
            self.assertIn("training_median", english)
            self.assertIn("not estimable", english)
            self.assertIn("invalidated", english)
            self.assertFalse(any(line.startswith(("- ", "* ")) for line in english.splitlines()))
            with (root / "reports/evidence_index.csv").open(encoding="utf-8", newline="") as stream:
                evidence = list(csv.DictReader(stream))
            self.assertTrue(all(row["research_log_heading_lines"] for row in evidence))
            self.assertEqual(len({row["evidence_id"] for row in evidence}), len(evidence))
            receipts = json.loads((root / "reports/source_receipts.json").read_text(encoding="utf-8"))
            self.assertFalse(receipts["data_access"]["prediction_rows"])
            self.assertTrue(all(not any(token in item["path"] for token in [".pkl", "/raw/", "/predictions/"]) for item in receipts["sources"]))
            with self.assertRaisesRegex(ValueError, "new directory"):
                report.build_report(config, root=root)

    def test_bad_or_missing_receipt_rejected_before_output(self):
        for failure in ["tampered", "missing"]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = make_fixture(root)
                path = root / "experiments/runs/R0001/metrics.csv"
                if failure == "tampered":
                    path.write_text(path.read_text() + "\n")
                else:
                    manifest_path = path.parent / "manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["outputs"] = [item for item in manifest["outputs"] if not item["path"].endswith("metrics.csv")]
                    write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "receipt"):
                    report.build_report(config, root=root)
                self.assertFalse((root / "reports").exists())

    def test_invalidated_source_and_incomplete_chain_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            index = pd.read_csv(root / "experiments/index.csv")
            index.loc[index.run_id.eq("R0001"), "status"] = "invalidated"
            index.to_csv(root / "experiments/index.csv", index=False)
            with self.assertRaisesRegex(ValueError, "completed and valid"):
                report.build_report(config, root=root)
            config["mainline_run_ids"].pop("G4B")
            with self.assertRaisesRegex(ValueError, "six registered"):
                report.build_report(config, root=root)

    def test_path_escapes_and_large_data_are_forbidden(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            archive = report.Archive(root)
            for name in ["raw/flights.csv", "processed/features.json", "experiments/runs/R0001/data/features.pkl", "../outside.csv"]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    archive.path(name)

    def test_final_freeze_and_rq1_reference_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            path = root / "experiments/runs/R0019/rq1_standardized/frozen_reference.json"
            write_json(path, {"synthetic": "changed"})
            archive_config = json.loads((path.parents[1] / "config.json").read_text())
            seal(root, "R0019", archive_config)
            with self.assertRaisesRegex(ValueError, "reuse the development reference"):
                report.build_report(config, root=root)


if __name__ == "__main__":
    unittest.main()
