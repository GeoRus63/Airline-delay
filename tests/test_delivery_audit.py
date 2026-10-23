"""Synthetic delivery audits; never open real final predictions or raw inputs."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from scripts import audit_delivery, run_pipeline, verify_research_archive


class DeliveryAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "experiments/runs/R0099"
        self.output.mkdir(parents=True)
        ledger = self.root / "experiments/final_access/2026-08.json"
        ledger.parent.mkdir(parents=True)
        ledger.write_text(json.dumps({"status": "completed", "run_id": "R0090"}))
        (self.root / "notebooks").mkdir()
        nbformat.write(nbformat.v4.new_notebook(), self.root / "notebooks/00_数据清单.ipynb")
        self.delivery = self.root / "report.txt"
        self.delivery.write_text("Synthetic published report")
        self.config = {"mode": "audit_delivery", "delivery_paths": ["report.txt"]}
        self.audit = {"passed": True, "errors": [], "final_verified": True,
                      "unsealed_runs_skipped": ["R0099"],
                      "runs": [{"run_id": "R0091", "eligible_completed": True,
                                "reproduction_verification": {"verified": True}}]}

    def invoke(self, render=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(audit_delivery, "ROOT", self.root))
            stack.enter_context(patch.object(run_pipeline, "ROOT", self.root))
            raw = stack.enter_context(patch.object(audit_delivery.verify_downloads, "verify", return_value={"passed": True}))
            archive = stack.enter_context(patch.object(audit_delivery.verify_research_archive, "verify_archive", return_value=self.audit))
            renderer = stack.enter_context(patch.object(audit_delivery, "_render", side_effect=render,
                                                         return_value={"executed": True}))
            result = audit_delivery.run(self.config, self.output)
            raw.assert_called_once_with(raw_only=True)
            archive.assert_called_once_with(all_runs=True)
            renderer.assert_called_once()
            return result

    def test_only_self_unsealed_is_permitted_and_source_files_unchanged(self):
        before = self.delivery.read_bytes()
        ledger = self.root / "experiments/final_access/2026-08.json"
        ledger_before = ledger.read_bytes()
        result = self.invoke()
        self.assertTrue(result["archive_integrity_passed"])
        self.assertEqual(result["verified_reproduction_runs"], ["R0091"])
        self.assertEqual(before, self.delivery.read_bytes())
        self.assertEqual(ledger_before, ledger.read_bytes())

    def test_other_running_run_prevents_delivery(self):
        self.audit["unsealed_runs_skipped"].append("R0019")
        with self.assertRaisesRegex(ValueError, "pending=.*R0019"):
            self.invoke()

    def test_formal_and_reproduction_verification_are_both_required(self):
        self.audit["final_verified"] = False
        with self.assertRaisesRegex(ValueError, "Archive audit incomplete"):
            self.invoke()
        self.audit["final_verified"] = True
        self.audit["runs"] = []
        with self.assertRaisesRegex(ValueError, "reproduction"):
            self.invoke()

    def test_archive_running_state_is_skipped_before_receipt_checks(self):
        (self.output / "manifest.json").write_text(json.dumps({"run_id": "R0099", "status": "running"}))
        row = {"status": "running", "artifact_dir": "experiments/runs/R0099"}
        actual = verify_research_archive.verify_run("R0099", root=self.root, index_row=row)
        self.assertTrue(actual["skipped_unsealed"])
        self.assertEqual(actual["errors"], [])
        self.assertFalse(actual["eligible_completed"])

    def test_receipt_describes_delivered_notebook_after_rendering(self):
        # Renderer normally updates the main 00 notebook. Hashes must describe
        # the published bytes after this permitted update, not the input template.
        self.config["delivery_paths"] = ["notebooks/00_数据清单.ipynb"]
        notebook = self.root / self.config["delivery_paths"][0]

        def render(*args, **kwargs):
            notebook.write_text("synthetic executed notebook")
            return {"executed": True}

        self.invoke(render)
        receipt = json.loads((self.output / "delivery_receipts.json").read_text())[0]
        self.assertEqual(receipt["sha256"], hashlib.sha256(notebook.read_bytes()).hexdigest())

    def test_pipeline_mutable_and_current_run_paths_are_rejected(self):
        for relative in ["experiments/index.csv", "research/研究记录.md",
                         "experiments/runs/R0099/config.json"]:
            with self.subTest(path=relative):
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic mutable artifact")
                self.config["delivery_paths"] = [relative]
                with self.assertRaisesRegex(ValueError, "cannot be sealed"):
                    self.invoke()

    def test_aliases_of_same_delivery_file_are_rejected(self):
        self.config["delivery_paths"] = ["report.txt", "notebooks/../report.txt"]
        with self.assertRaisesRegex(ValueError, "Duplicate resolved"):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
