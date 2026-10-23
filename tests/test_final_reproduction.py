"""Full final reproduction tests using synthetic rows only; never read repository data."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from scripts import final_evaluation as final
from scripts import reproduce_final as reproduction
from scripts import run_pipeline as pipeline
from tests.test_final_evaluation import fixture


class FinalReproduction(unittest.TestCase):
    def _saved_reproduction(self, root):
        frame, _, config, _, reference, original = self._formal(root)
        output = root / "RREPRODUCED"
        with patch.object(final, "ROOT", root), patch.object(final, "load_configured_features", return_value=frame):
            result = reproduction.run(config, output)
        (output / "config.json").write_text(json.dumps(config))
        self._seal(root, output, config)
        return output, reference, original, result

    def _seal(self, root, output, config):
        receipts = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                     "sha256": final.sha256_file(path)} for path in sorted(output.rglob("*"))
                    if path.is_file() and path.name != "manifest.json"]
        (output / "manifest.json").write_text(json.dumps({"run_id": output.name, "status": "completed",
                                                         "config": config, "outputs": receipts}))

    def test_saved_reproduction_verifier_is_read_only_and_never_opens_raw_or_refits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, _, _, _ = self._saved_reproduction(root)
            before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            real_hash = final.sha256_file

            def no_raw_hash(path):
                if Path(path).name == "connections.csv":
                    raise AssertionError("Must not hash raw input during saved audit")
                return real_hash(path)

            with patch.object(final, "load_configured_features", side_effect=AssertionError("no raw read")), \
                    patch.object(final.training, "_run_fold", side_effect=AssertionError("no refit")), \
                    patch.object(final, "validate_freeze", side_effect=AssertionError("no live raw/source pins")), \
                    patch.object(final, "sha256_file", side_effect=no_raw_hash), \
                    patch.object(Path, "write_text", side_effect=AssertionError("no archive write")), \
                    patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("no archive write")):
                result = reproduction.verify_saved_reproduction(output, root=root)
            self.assertTrue(result["verified"])
            self.assertFalse(result["refitted"])
            self.assertFalse(result["raw_test_data_read"])
            self.assertFalse(result["archive_written"])
            self.assertEqual(before, {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()})

    def test_saved_reproduction_rejects_missing_receipt_prediction_and_ledger_hash_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, _, _, _ = self._saved_reproduction(root)
            config = final._json(output / "config.json")
            receipt_path = output / "reproduction_verification.json"
            receipt_bytes = receipt_path.read_bytes()
            receipt_path.unlink()
            with self.assertRaisesRegex(ValueError, "SHA-256|evidence"):
                reproduction.verify_saved_reproduction(output, root=root)
            receipt_path.write_bytes(receipt_bytes)
            prediction = output / "predictions/2026-08_frozen_linear.csv.gz"
            prediction_bytes = prediction.read_bytes()
            prediction.write_bytes(prediction_bytes + b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                reproduction.verify_saved_reproduction(output, root=root)
            prediction.write_bytes(prediction_bytes)
            receipt = final._json(receipt_path)
            receipt["ledger_sha256_after"] = "wrong"
            receipt_path.write_text(json.dumps(receipt))
            self._seal(root, output, config)
            with self.assertRaisesRegex(ValueError, "ledger after hash"):
                reproduction.verify_saved_reproduction(output, root=root)
            receipt_path.write_bytes(receipt_bytes)
            self._seal(root, output, config)
            ledger = root / "experiments/final_access/2026-08.json"
            ledger.write_bytes(ledger.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "ledger hash"):
                reproduction.verify_saved_reproduction(output, root=root)

    def test_compare_read_only_mode_never_writes_on_prediction_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, reference, original, result = self._saved_reproduction(root)
            path = output / "predictions/2026-08_frozen_linear.csv.gz"
            frame = pd.read_csv(path, float_precision="round_trip")
            frame.loc[0, "predicted_delay_raw"] += 1
            frame.to_csv(path, index=False, compression=final.training.GZIP)
            with patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("no archive write")):
                with self.assertRaisesRegex(ValueError, "prediction difference"):
                    reproduction._compare(reference, output, original, result, write_evidence=False)

    def test_pipeline_dispatches_reproduction_and_archives_its_notebook_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = {"mode": "reproduce_final"}
            run = Mock(return_value={"mode": "reproduce_final"})
            render = Mock(return_value={"executed_notebook": "new-run/07.ipynb", "update_main": False})
            with patch.dict(sys.modules, {"reproduce_final": SimpleNamespace(run=run, render_notebook=render)}):
                result = pipeline.execute(config, output)
            run.assert_called_once_with(config, output)
            render.assert_called_once_with(output)
            self.assertEqual(result["mode"], "reproduce_final")
            self.assertFalse(final._json(output / "reproduction_notebook_receipt.json")["update_main"])

    def test_reproduction_notebook_is_labeled_and_renders_only_to_new_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            with patch("scripts.render_research_notebooks._render", return_value={"update_main": False}) as render:
                reproduction.render_notebook(output)
            arguments, keywords = render.call_args
            self.assertIn("完整复现", arguments[0].cells[0].source)
            self.assertIn("不是新的独立测试", arguments[0].cells[0].source)
            self.assertEqual(arguments[1], "07_最终评估.ipynb")
            self.assertEqual(keywords["output_dir"], output)
            self.assertFalse(keywords["update_main"])
            verification_guidance = [cell.source for cell in arguments[0].cells
                                     if cell.cell_type == "markdown" and "--verify-run" in cell.source]
            self.assertTrue(all("仅适用于" in text and "原正式 run" in text for text in verification_guidance))

    def _formal(self, root):
        frame, frozen, runtime, freeze_path = fixture(root)
        reference = root / "RFORMAL"
        with patch.object(final, "ROOT", root), patch.object(final, "load_configured_features", return_value=frame):
            result = final.run(runtime, reference)
        (reference / "manifest.json").write_text(json.dumps({"status": "completed"}))
        config = {**runtime, "mode": "reproduce_final", "stage": "P7", "change": "Exact frozen reproduction"}
        return frame, frozen, config, freeze_path, reference, result

    def test_missing_or_incomplete_ledger_rejected_without_feature_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, config, _ = fixture(root)
            config["mode"] = "reproduce_final"
            ledger = root / "experiments/final_access/2026-08.json"
            with patch.object(final, "ROOT", root), patch.object(final, "load_configured_features", side_effect=AssertionError("must not read")):
                with self.assertRaisesRegex(ValueError, "completed formal ledger"):
                    reproduction.run(config, root / "RREPLAY")
                ledger.parent.mkdir(parents=True)
                for state in ["claimed", "failed"]:
                    ledger.write_text(json.dumps({"status": state, "test_period": final.FINAL_FOLD}))
                    before = ledger.read_bytes()
                    with self.assertRaisesRegex(ValueError, "completed formal ledger"):
                        reproduction.run(config, root / "RREPLAY")
                    self.assertEqual(before, ledger.read_bytes())

    def test_success_rebuilds_and_refits_all_four_models_without_new_formal_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame, frozen, config, _, reference, original = self._formal(root)
            ledger = root / "experiments/final_access/2026-08.json"
            before = ledger.read_bytes()
            output = root / "RREPRODUCED"
            real_fold = final.training._run_fold
            with patch.object(final, "ROOT", root), patch.object(final, "load_configured_features", return_value=frame) as loader, \
                    patch.object(final.training, "_run_fold", wraps=real_fold) as fit:
                result = reproduction.run(config, output)
                loader.assert_called_once_with(frozen["config"], development_only=False)
                fit.assert_called_once()
            self.assertEqual(before, ledger.read_bytes())
            self.assertEqual(result["mode"], "reproduce_final")
            self.assertEqual(result["claim_type"], "reproduction")
            self.assertEqual(result["feature_fingerprints"], original["feature_fingerprints"])
            self.assertEqual(len(list((output / "models").glob("*.joblib"))), 4)
            verification = final._json(output / "reproduction_verification.json")
            self.assertTrue(verification["formal_ledger_unchanged"])
            self.assertEqual(verification["ledger_sha256_before"], verification["ledger_sha256_after"])
            self.assertFalse(verification["creates_new_formal_claim"])
            pd.testing.assert_frame_equal(pd.read_csv(reference / "metrics.csv"), pd.read_csv(output / "metrics.csv"))
            # Execute the real 07 renderer against synthetic archives, in the new output only.
            import nbformat
            rendered = reproduction.render_notebook(output)
            notebook = nbformat.read(rendered["executed_notebook"], as_version=4)
            self.assertFalse(rendered["update_main"])
            self.assertTrue(Path(rendered["html"]).is_file())
            self.assertTrue(all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code"))
            self.assertFalse(any(item.output_type == "error" for cell in notebook.cells if cell.cell_type == "code"
                                 for item in cell.outputs))
            self.assertEqual(before, ledger.read_bytes())
            with patch.object(final, "ROOT", root):
                self.assertTrue(final.verify_saved(reference)["verified"])

    def test_wrong_freeze_science_source_and_input_rejected_before_feature_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, config, _, reference, _ = self._formal(root)
            ledger = root / "experiments/final_access/2026-08.json"
            before = ledger.read_bytes()
            with patch.object(final, "ROOT", root), patch.object(final, "load_configured_features", side_effect=AssertionError("must not read")):
                with self.assertRaisesRegex(ValueError, "original ledger freeze"):
                    reproduction.run({**config, "freeze_manifest_sha256": "wrong"}, root / "RWRONGFREEZE")
                with self.assertRaisesRegex(ValueError, "scientific setting"):
                    reproduction.run({**config, "threads": 2}, root / "RWRONGSCIENCE")
                actual_hash = final.sha256_file
                source = next(iter(final._required_sources("G0")))
                with patch.object(final, "sha256_file", side_effect=lambda path: "changed-source" if Path(path).resolve() == source else actual_hash(path)):
                    with self.assertRaisesRegex(ValueError, "source_files SHA-256"):
                        reproduction.run(config, root / "RWRONGSOURCE")
                connection = root / "connections.csv"
                connection.write_text(connection.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "data_files SHA-256"):
                    reproduction.run(config, root / "RWRONGDATA")
            self.assertEqual(before, ledger.read_bytes())

    def test_feature_and_prediction_differences_fail_and_preserve_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame, _, config, _, _, _ = self._formal(root)
            ledger = root / "experiments/final_access/2026-08.json"
            before = ledger.read_bytes()
            changed = frame.copy()
            changed.loc[0, "scheduled_turnaround_minutes"] += 1
            with patch.object(final, "ROOT", root), patch.object(final, "load_configured_features", return_value=changed):
                with self.assertRaisesRegex(ValueError, "folds|feature_fingerprints"):
                    reproduction.run(config, root / "RCHANGEDFEATURES")
            self.assertTrue((root / "RCHANGEDFEATURES/reproduction_failure.txt").is_file())
            real_fold = final.training._run_fold

            def tampered_fold(*args, **kwargs):
                value = real_fold(*args, **kwargs)
                prediction = Path(args[4]) / "predictions/2026-08_frozen_linear.csv.gz"
                rows = pd.read_csv(prediction, float_precision="round_trip")
                rows.loc[0, "predicted_delay_raw"] += 0.1
                rows.to_csv(prediction, index=False, compression=final.training.GZIP)
                return value

            with patch.object(final, "ROOT", root), patch.object(final, "load_configured_features", return_value=frame), \
                    patch.object(final.training, "_run_fold", side_effect=tampered_fold):
                with self.assertRaisesRegex(ValueError, "prediction difference"):
                    reproduction.run(config, root / "RCHANGEDPREDICTIONS")
            self.assertEqual(before, ledger.read_bytes())
            self.assertTrue((root / "RCHANGEDPREDICTIONS/reproduction_prediction_differences.csv").is_file())

    def test_fingerprint_seals_schema_and_baseline_only_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame, frozen, _, _ = fixture(Path(temporary))
            train, _, _ = final.split_final_frame(frame)
            features = frozen["config"]["features"]
            original = final._frame_fingerprint(train, features)
            changed = train.copy()
            changed["upstream_arr_delay_minutes"] += 1
            self.assertNotEqual(original, final._frame_fingerprint(changed, features))
            changed = train.copy()
            changed["target_delay"] = changed.target_delay.astype("float32")
            self.assertNotEqual(original["schema_sha256"], final._frame_fingerprint(changed, features)["schema_sha256"])


if __name__ == "__main__":
    unittest.main()
