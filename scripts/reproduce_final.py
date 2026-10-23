"""Refit a completed sealed final experiment into a new run; never claim a new test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

try:
    from . import final_evaluation as final
except ImportError:
    import final_evaluation as final

# Numerical comparison is fixed before first freeze; configuration cannot relax it.
PREDICTION_ATOL = 1e-10
METRIC_ATOL = 1e-10


def _completed_reference(config, output, *, root=None):
    """Check the sole completed claim and its archived artifacts before opening inputs."""
    root = Path(final.ROOT if root is None else root).resolve()
    resolve = lambda value: Path(value).resolve() if Path(value).is_absolute() else (root / value).resolve()
    ledger_path = root / "experiments/final_access/2026-08.json"
    if not ledger_path.is_file():
        raise ValueError("Reproduction requires the existing completed formal ledger")
    ledger_hash = final.sha256_file(ledger_path)
    access = final._json(ledger_path)
    if access.get("status") != "completed" or access.get("test_period") != final.FINAL_FOLD:
        raise ValueError("Reproduction requires the completed formal ledger, never claimed/failed access")
    reference = resolve(access["run_dir"])
    if reference.name != access.get("run_id"):
        raise ValueError("Formal ledger run identity is inconsistent")
    if output == reference or output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError("Reproduction must use a new run outside the formal archive")
    manifest = reference / "manifest.json"
    if not manifest.is_file() or final._json(manifest).get("status") != "completed":
        raise ValueError("Formal run archive must have completed pipeline status")
    result_path = reference / "final_result.json"
    if not result_path.is_file() or final.sha256_file(result_path) != access.get("final_result_sha256"):
        raise ValueError("Formal result differs from the unique ledger")
    result = final._json(result_path)
    if result.get("mode") != "final" or result.get("claim_type") != "formal":
        raise ValueError("Reference must be the original formal final, not a reproduction")
    if (result.get("freeze_manifest_sha256") != access.get("freeze_manifest_sha256")
            or config.get("freeze_manifest_sha256") != access.get("freeze_manifest_sha256")):
        raise ValueError("Reproduction must use the original ledger freeze SHA-256")
    if resolve(result["access_ledger_path"]) != ledger_path.resolve():
        raise ValueError("Formal result points to a different access ledger")
    if not result.get("feature_fingerprints"):
        raise ValueError("Formal run lacks complete feature fingerprints; cannot certify full reproduction")
    final.verify_saved(reference)
    return ledger_path, ledger_hash, access, reference, result


def _compare(reference, output, original, repeated, *, write_evidence=True):
    """Exact IDs/outcomes/features plus rowwise raw/clipped prediction and metric checks."""
    for key in ["config", "protocol", "models", "primary_model_name", "folds", "split_sha256",
                "evaluation_rows", "evaluation_ids_sha256", "feature_fingerprints", "verified_inputs"]:
        if original[key] != repeated[key]:
            raise ValueError(f"Reproduction differs from formal run: {key}")
    for filename in ["training_ids_targets.csv.gz", "evaluation_ids.csv.gz"]:
        expected = pd.read_csv(reference / filename, float_precision="round_trip")
        actual = pd.read_csv(output / filename, float_precision="round_trip")
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    before = final._saved_predictions(reference, original["models"])
    after = final._saved_predictions(output, repeated["models"])
    differences = []
    prediction_columns = ["predicted_delay_raw", "predicted_delay"]
    for name in original["models"]:
        expected, actual = before[name], after[name]
        pd.testing.assert_frame_equal(expected.drop(columns=prediction_columns), actual.drop(columns=prediction_columns),
                                      check_exact=True)
        for column in prediction_columns:
            left, right = expected[column].to_numpy(), actual[column].to_numpy()
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError("Reproduction predictions must be finite")
            difference = np.abs(left - right)
            differences.append({"model": name, "column": column, "rows": len(left),
                                "max_absolute_difference": float(difference.max(initial=0)),
                                "atol": PREDICTION_ATOL, "rtol": 0.0})
            if not np.allclose(left, right, atol=PREDICTION_ATOL, rtol=0):
                if write_evidence:
                    pd.DataFrame(differences).to_csv(output / "reproduction_prediction_differences.csv", index=False)
                raise ValueError(f"Reproduction prediction difference exceeds fixed tolerance: {name}/{column}")
    if write_evidence:
        pd.DataFrame(differences).to_csv(output / "reproduction_prediction_differences.csv", index=False)
    expected = pd.read_csv(reference / "metrics.csv", float_precision="round_trip")
    actual = pd.read_csv(output / "metrics.csv", float_precision="round_trip")
    identity = ["model", "fold", "group", "value", "n"]
    expected["value"], actual["value"] = expected.value.astype(str), actual.value.astype(str)
    expected, actual = (frame.sort_values(identity).reset_index(drop=True) for frame in [expected, actual])
    pd.testing.assert_frame_equal(expected[identity], actual[identity], check_exact=True)
    metric_columns = [name for name in expected if name not in identity]
    if list(expected.columns) != list(actual.columns):
        raise ValueError("Reproduction metric schema differs")
    pd.testing.assert_frame_equal(expected[metric_columns], actual[metric_columns], check_dtype=False,
                                  check_exact=False, atol=METRIC_ATOL, rtol=0)
    return {"prediction_rows_compared": sum(len(frame) for frame in before.values()),
            "metric_rows_compared": len(expected), "all_feature_values_and_types_match": True,
            "ids_and_targets_match": True, "prediction_atol": PREDICTION_ATOL, "metric_atol": METRIC_ATOL, "rtol": 0}


def verify_saved_reproduction(run_dir, *, root=None):
    """Audit completed saved reproduction artifacts only; no fitting/raw reads/writes.

    Archived freeze copies and recorded source/input receipts identify the original
    computation. Current raw inputs and current source bytes are deliberately not
    opened: this is an archive audit, not another full reproduction.
    """
    root = Path(final.ROOT if root is None else root).resolve()
    output = Path(run_dir).resolve()
    manifest = final._json(output / "manifest.json")
    runtime = final._json(output / "config.json")
    if manifest.get("status") != "completed" or manifest.get("config") != runtime or runtime.get("mode") != "reproduce_final":
        raise ValueError("Saved reproduction requires its completed matching pipeline manifest/config")
    if manifest.get("run_id") != output.name:
        raise ValueError("Saved reproduction manifest run identity differs")
    result = final._json(output / "final_result.json")
    if result.get("mode") != "reproduce_final" or result.get("claim_type") != "reproduction":
        raise ValueError("Saved result is not a full final reproduction")
    ledger, ledger_hash, access, reference, original = _completed_reference(runtime, output, root=root)

    def check_hash(relative, expected, *, size=None):
        path = Path(relative)
        path = path.resolve() if path.is_absolute() else (output / path).resolve()
        if not path.is_relative_to(output) or not path.is_file() or final.sha256_file(path) != expected:
            raise ValueError(f"Saved reproduction artifact SHA-256 mismatch or invalid path: {relative}")
        if size is not None and path.stat().st_size != size:
            raise ValueError(f"Saved reproduction artifact size mismatch: {relative}")
        return path

    artifacts = result.get("artifact_hashes", {})
    if not artifacts:
        raise ValueError("Saved reproduction lacks artifact hashes")
    for relative, expected in artifacts.items():
        check_hash(relative, expected)
    output_receipts = manifest.get("outputs", [])
    if not output_receipts:
        raise ValueError("Saved reproduction lacks completed output receipts")
    pinned = set()
    for receipt in output_receipts:
        path = Path(receipt["path"])
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path in pinned:
            raise ValueError("Saved reproduction has duplicate output receipts")
        pinned.add(check_hash(path, receipt["sha256"], size=receipt.get("bytes")))
    required = {"config.json", "final_result.json", "model_selection.json", "final_feature_fingerprints.json",
                "reproduction_reference.json", "reproduction_verification.json", "reproduction_prediction_differences.csv",
                "training_ids_targets.csv.gz", "evaluation_ids.csv.gz", "metrics.csv"}
    required.update(f"predictions/{final.FINAL_FOLD}_{name}.csv.gz" for name in original["models"])
    if {output / name for name in required} - pinned:
        raise ValueError("Saved reproduction output receipts omit required reproduction evidence")
    provenance = final._json(output / "reproduction_reference.json")
    receipt = final._json(output / "reproduction_verification.json")
    expected_provenance = {"mode": "reproduce_final", "claim_type": "reproduction", "formal_run_id": access["run_id"],
                           "formal_run_dir": str(reference), "formal_final_result_sha256": access["final_result_sha256"],
                           "freeze_manifest_sha256": access["freeze_manifest_sha256"], "access_ledger_path": str(ledger),
                           "ledger_sha256_before": ledger_hash, "creates_new_formal_claim": False,
                           "refitted": True, "raw_test_data_read": True}
    if provenance != expected_provenance or result.get("reproduction_reference") != provenance:
        raise ValueError("Saved reproduction provenance/original ledger hash differs")
    if any(receipt.get(key) != value for key, value in provenance.items()) or receipt.get("verified") is not True:
        raise ValueError("Saved reproduction verification receipt is missing or differs")
    if receipt.get("ledger_sha256_after") != ledger_hash or receipt.get("formal_ledger_unchanged") is not True:
        raise ValueError("Saved reproduction ledger after hash differs")
    for directory in [reference, output]:
        if final.sha256_file(directory / "model_selection.json") != access["freeze_manifest_sha256"]:
            raise ValueError("Saved reproduction freeze identity differs from the completed formal ledger")
    frozen = final._json(output / "model_selection.json")
    payload = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    if (frozen.get("payload_sha256") != final.canonical_sha256(payload)
            or result.get("freeze_payload_sha256") != frozen["payload_sha256"]
            or result.get("freeze_manifest_sha256") != access["freeze_manifest_sha256"]
            or result.get("config") != frozen.get("config") or result.get("protocol") != frozen.get("protocol")):
        raise ValueError("Saved reproduction frozen payload/config differs")
    if final._json(output / "final_feature_fingerprints.json") != result.get("feature_fingerprints"):
        raise ValueError("Saved reproduction complete feature fingerprint evidence differs")
    comparison = _compare(reference, output, original, result, write_evidence=False)
    if result.get("reproduction_comparison") != comparison or any(receipt.get(key) != value for key, value in comparison.items()):
        raise ValueError("Saved reproduction comparison receipt differs from recalculated saved artifacts")
    if final.sha256_file(ledger) != ledger_hash:
        raise ValueError("Formal ledger changed during saved reproduction verification")
    return {"verified": True, "mode": "reproduce_final", "formal_run_id": access["run_id"],
            "formal_ledger_unchanged": True, "artifact_hashes_checked": len(artifacts),
            "output_receipts_checked": len(output_receipts), **comparison,
            "refitted": False, "raw_test_data_read": False, "archive_written": False,
            "verification_scope": "saved reproduction comparison; full refit remains the archived execution"}


def run(config, run_dir):
    """Rebuild features and refit exactly the original frozen final and three baselines."""
    if config.get("mode") != "reproduce_final":
        raise ValueError("Full final reproduction requires mode='reproduce_final'")
    output = Path(run_dir).resolve()
    ledger_path, ledger_hash, access, reference, original = _completed_reference(config, output)
    frozen, freeze_path, freeze_hash, verified = final.validate_freeze(config)
    if (original["config"] != frozen["config"] or original["freeze_payload_sha256"] != frozen["payload_sha256"]
            or original["verified_inputs"] != verified):
        raise ValueError("Original formal freeze/config/source/input receipts differ")
    # The exact original JSON bytes are also retained inside the formal run.
    if final.sha256_file(reference / "model_selection.json") != freeze_hash:
        raise ValueError("Archived formal freeze differs from the original ledger freeze")
    output = final._validate_output(output, freeze_path)
    provenance = {"mode": "reproduce_final", "claim_type": "reproduction", "formal_run_id": access["run_id"],
                  "formal_run_dir": str(reference), "formal_final_result_sha256": access["final_result_sha256"],
                  "freeze_manifest_sha256": freeze_hash, "access_ledger_path": str(ledger_path),
                  "ledger_sha256_before": ledger_hash, "creates_new_formal_claim": False,
                  "refitted": True, "raw_test_data_read": True}
    final._write_json(output / "reproduction_reference.json", provenance)
    try:
        result = final._fit_frozen(frozen, freeze_path, freeze_hash, verified, output, ledger_path, mode="reproduce_final")
        compared = _compare(reference, output, original, result)
        ledger_after = final.sha256_file(ledger_path)
        if ledger_after != ledger_hash:
            raise RuntimeError("Formal ledger changed during reproduction; reproduction cannot be certified")
        result.update(reproduction_reference=provenance, reproduction_comparison=compared)
        # Refresh the artifact receipt after adding comparison evidence.
        result["artifact_hashes"] = {str(path.relative_to(output)): final.sha256_file(path)
                                     for path in sorted(output.rglob("*")) if path.is_file()
                                     and path.name not in {"manifest.json", "run.log", "resources.csv", "final_result.json"}}
        final._write_json(output / "final_result.json", result)
        final._write_json(output / "reproduction_verification.json", {**provenance, **compared, "verified": True,
                              "ledger_sha256_after": ledger_after, "formal_ledger_unchanged": True})
        return result
    except BaseException:
        (output / "reproduction_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        # No code path in this module writes or restores the formal ledger.
        if final.sha256_file(ledger_path) != ledger_hash:
            raise RuntimeError("Formal ledger changed during reproduction; reproduction cannot be certified")


def render_notebook(run_dir):
    """Reuse 07's renderer with a reproduction title and explicit verification evidence."""
    import nbformat
    try:
        from .render_research_notebooks import _render, final_template
    except ImportError:
        from render_research_notebooks import _render, final_template
    output = Path(run_dir).resolve()
    notebook = final_template()
    notebook.cells[0].source = (
        "# P7：冻结最终实验的完整复现\n\n"
        "本次使用唯一已完成正式实验的原冻结文件、相同源码及输入，重新构建特征、按相同成熟标签切分，"
        "重新拟合冻结学习模型与三个基线并逐行核对预测及指标。这是已见测试集的复现，不是新的独立测试，"
        "不重新选模、不产生第二个正式测试声明。原正式账本只读，前后 SHA-256 必须一致。\n\n"
        "本 Notebook 仅展示本次新 run 的归档结果；实际重建与重拟合由统一入口完成。")
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and "--verify-run" in cell.source:
            cell.source = (
                "## 7. 复现核对与结论边界\n\n"
                "本次是否完成完整复现，以本 run 的 `reproduction_verification.json`、逐行预测差异表及"
                "完整特征指纹为依据。只读取保存预测重算指标的 `final_evaluation.py --verify-run` 仅适用于"
                "原正式 run，不能作为本次重新构建和重拟合的替代，也不能对本复现 run 使用正式账本核对。"
                "原正式 run 的位置记录在 `reproduction_reference.json`。已见的 8 月仍是原冻结方案的测试集；"
                "本次不增加独立验证证据，后续方法变化应记为测试后探索。")
    notebook.cells.extend([
        nbformat.v4.new_markdown_cell("## 完整复现核对凭据\n\n下列文件记录原正式实验、训练/测试完整特征指纹、"
                                     "预测和指标比较及账本前后哈希。比较容差在冻结前的源码中固定。"),
        nbformat.v4.new_code_cell('archived_json("reproduction_reference.json")\n'
                                 'archived_json("final_feature_fingerprints.json")\n'
                                 'archived_table("reproduction_prediction_differences.csv", limit=None)\n'
                                 'archived_json("reproduction_verification.json")'),
    ])
    return _render(notebook, "07_最终评估.ipynb", output, final.ROOT, output_dir=output, update_main=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(final._json(arguments.config), arguments.run_dir), indent=2))
