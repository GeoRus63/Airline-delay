"""Execution disclosure tests use only synthetic completed summary archives."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts import build_report as report
from tests.test_report_evidence import make_fixture, seal, write_json


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
            self.assertIn("same freeze and its 24-thread setting", en)
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
