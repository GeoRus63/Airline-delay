"""Synthetic receipt audits: never train, decode raw inputs or deserialize models."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_research_archive as audit


SOURCE = b"print('historical source')\n"
REQUIREMENTS = b"pandas==3.0.3\n"


def fixture(root, status="completed", indexed_status=None, final=False):
    run = root / "experiments/runs/R0001"
    (run / "models").mkdir(parents=True)
    (root / "scripts").mkdir()
    source = root / "scripts/example.py"
    source.write_bytes(SOURCE)
    (root / "requirements.txt").write_bytes(REQUIREMENTS)
    # A deliberately invalid pickle must only be hashed, never deserialized.
    model = run / "models/example.joblib"
    model.write_bytes(b"not-a-pickle; receipt-only audit")
    config = {"mode": "final" if final else "development"}
    config_path = run / "config.json"
    config_path.write_text(json.dumps(config))
    receipts = lambda path: {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": audit._sha(path)}
    outputs = [receipts(model), receipts(config_path)]
    if final:
        (run / "final_result.json").write_text("{}")
        outputs.append(receipts(run / "final_result.json"))
    manifest = {"run_id": "R0001", "status": status, "git_commit": "synthetic-commit", "config": config,
                "source_files": [receipts(source)], "outputs": outputs, "environment": {"pandas": "3.0.3"}, "python": "synthetic-python"}
    (run / "manifest.json").write_text(json.dumps(manifest))
    with (root / "experiments/index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run_id", "status", "artifact_dir", "decision"])
        writer.writeheader()
        writer.writerow({"run_id": "R0001", "status": indexed_status or status, "artifact_dir": "experiments/runs/R0001", "decision": "retained or superseded"})
    return run, model


def fake_blob(root, commit, path):
    return REQUIREMENTS if path == "requirements.txt" else SOURCE


class ArchiveVerification(unittest.TestCase):
    def test_reproduction_routes_to_saved_reproduction_verifier_without_formal_claim(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "_git_blob", side_effect=fake_blob), \
             patch.object(audit, "_ignored", return_value=False), \
             patch.object(audit, "current_environment", return_value={"requirements_match": True}), \
             patch.object(audit, "_verify_final_saved", side_effect=AssertionError("reproduction is not formal")), \
             patch.object(audit, "_verify_reproduction_saved", return_value={"verified": True, "refitted": False,
                                                                           "raw_test_data_read": False}) as replay:
            root = Path(temp)
            run, _ = fixture(root, final=True)
            config = {"mode": "reproduce_final"}
            (run / "config.json").write_text(json.dumps(config))
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["config"] = config
            for item in manifest["outputs"]:
                path = root / item["path"]
                item.update(bytes=path.stat().st_size, sha256=audit._sha(path))
            (run / "manifest.json").write_text(json.dumps(manifest))
            result = audit.verify_archive(root=root, all_runs=True)
            self.assertTrue(result["passed"])
            self.assertTrue(result["reproduction_verified"])
            self.assertFalse(result["final_verified"])
            self.assertFalse(result["formal_final_verified"])
            self.assertNotIn("final_verification", result["runs"][0])
            replay.assert_called_once_with(run, root)

    def test_receipts_reject_tampering_and_missing_ignored_models(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "_git_blob", side_effect=fake_blob), \
             patch.object(audit, "_ignored", return_value=True):
            root = Path(temp)
            run, model = fixture(root)
            result = audit.verify_run("R0001", root=root)
            self.assertTrue(result["integrity_passed"])
            self.assertTrue(result["eligible_completed"])
            self.assertTrue(result["output_receipts"][0]["git_ignored"])
            model.write_bytes(b"tampered")
            result = audit.verify_run("R0001", root=root)
            self.assertFalse(result["integrity_passed"])
            self.assertFalse(result["eligible_completed"])
            self.assertTrue(any("SHA-256" in error for error in result["errors"]))
            model.unlink()
            result = audit.verify_run("R0001", root=root)
            self.assertTrue(any("Missing local artifact" in error for error in result["errors"]))

    def test_failed_and_invalidated_never_become_completed(self):
        for status, index_status in [("failed", "failed"), ("completed", "invalidated")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp, \
                 patch.object(audit, "_git_blob", side_effect=fake_blob), patch.object(audit, "_ignored", return_value=False):
                root = Path(temp)
                fixture(root, status, index_status)
                result = audit.verify_run("R0001", root=root)
                self.assertTrue(result["integrity_passed"])
                self.assertFalse(result["eligible_completed"])
                self.assertEqual(result["effective_status"], index_status)

    def test_git_blob_is_authoritative_historical_source_receipt(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "_git_blob", side_effect=fake_blob), \
             patch.object(audit, "_ignored", return_value=False):
            root = Path(temp)
            run, _ = fixture(root)
            (root / "scripts/example.py").write_text("legitimate later source")
            result = audit.verify_run("R0001", root=root)
            self.assertTrue(result["integrity_passed"])
            self.assertFalse(result["source_receipts"][0]["working_file_matches_receipt"])
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["source_files"][0]["sha256"] = hashlib.sha256(b"forged receipt").hexdigest()
            (run / "manifest.json").write_text(json.dumps(manifest))
            self.assertFalse(audit.verify_run("R0001", root=root)["integrity_passed"])

    def test_final_delegates_only_saved_replay_and_report_cannot_write_archive(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "_git_blob", side_effect=fake_blob), \
             patch.object(audit, "_ignored", return_value=False), \
             patch.object(audit, "_verify_final_saved", return_value={"verified": True, "refitted": False, "raw_test_data_read": False}) as replay:
            root = Path(temp)
            run, _ = fixture(root, final=True)
            result = audit.verify_run("R0001", root=root)
            self.assertTrue(result["integrity_passed"])
            replay.assert_called_once_with(run)
            before = (run / "manifest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "outside"):
                audit.write_report(result, run / "audit.json", root=root)
            output = root / "audit_reports/new.json"
            audit.write_report(result, output, root=root)
            with self.assertRaises(FileExistsError):
                audit.write_report(result, output, root=root)
            self.assertEqual((run / "manifest.json").read_bytes(), before)
            self.assertFalse((root / "experiments/final_access").exists())

    def test_claimed_final_ledger_and_running_run_are_never_verified(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "_git_blob", side_effect=fake_blob), \
             patch.object(audit, "_ignored", return_value=False), \
             patch.object(audit, "current_environment", return_value={"requirements_match": True}), \
             patch.object(audit, "_verify_final_saved", side_effect=AssertionError("not completed")):
            root = Path(temp)
            fixture(root, status="running", final=True)
            ledger = root / "experiments/final_access/2026-08.json"
            ledger.parent.mkdir()
            ledger.write_text(json.dumps({"run_id": "R0001", "status": "claimed"}))
            result = audit.verify_archive(root=root, final_only=True)
            self.assertFalse(result["passed"])
            self.assertFalse(result["final_verified"])
            self.assertEqual(result["unsealed_runs_skipped"], ["R0001"])


if __name__ == "__main__":
    unittest.main()
