"""Read-only run receipts, historical Git source bytes and environment audit.

No training, raw-data decoding, pickle loading, artifact repair or final-access
claim is performed. Final prediction replay delegates only to verify_saved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "failed", "invalidated"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value, root):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _git_blob(root, commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root,
                          capture_output=True, check=True).stdout


def _normal_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(text):
    pins, unsupported = {}, []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match:
            pins[_normal_name(match[1])] = match[2]
        else:
            unsupported.append(line)
    return pins, unsupported


def _environment(values):
    if isinstance(values, dict):
        return {_normal_name(name): str(version) for name, version in values.items()}
    if isinstance(values, list):
        return _pins("\n".join(values))[0]
    return {}


def _environment_check(actual, required):
    return [{"package": name, "required": version, "observed": actual.get(name),
             "matches": actual.get(name) == version} for name, version in sorted(required.items())]


def current_environment(root=ROOT):
    root = Path(root).resolve()
    required, unsupported = _pins((root / "requirements.txt").read_text(encoding="utf-8-sig"))
    actual = {_normal_name(item.metadata["Name"]): item.version for item in importlib.metadata.distributions()}
    matches = _environment_check(actual, required)
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "requirements_sha256": _sha(root / "requirements.txt"), "packages": matches,
            "unsupported_requirement_lines": unsupported,
            "requirements_match": bool(required) and not unsupported and all(item["matches"] for item in matches),
            "scope": "Direct pinned requirements only; complete historical environments remain in each run manifest"}


def _ignored(path, root):
    try:
        relative = Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return None
    result = subprocess.run(["git", "check-ignore", "--no-index", "--", relative], cwd=root,
                            capture_output=True, check=False)
    return result.returncode == 0


def _check_receipt(receipt, root, *, source_commit=None, allowed_directory=None):
    result = {"path": receipt.get("path"), "errors": []}
    if not receipt.get("path") or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256", ""))):
        result["errors"].append("Malformed path/SHA-256 receipt")
        return result
    path = _path(receipt["path"], root)
    if allowed_directory is not None and not path.is_relative_to(allowed_directory):
        result["errors"].append("Output receipt escapes its own run directory")
        return result
    result["git_ignored"] = _ignored(path, root)
    result["exists_locally"] = path.is_file()
    if not path.is_file():
        result["working_file_matches_receipt"] = False
        if source_commit is None:
            result["errors"].append("Missing local artifact; Git clone does not restore ignored predictions/models/features/processed data")
    else:
        result["observed_bytes"] = path.stat().st_size
        result["observed_sha256"] = _sha(path)
        result["working_file_matches_receipt"] = (result["observed_sha256"] == receipt["sha256"]
                                                    and ("bytes" not in receipt or receipt["bytes"] == path.stat().st_size))
        # Historical sources may have legitimately evolved in the working tree.
        # Their recorded Git blob, rather than today's source, is authoritative.
        if not result["working_file_matches_receipt"] and source_commit is None:
            result["errors"].append("Artifact bytes/SHA-256 differ from recorded receipt")
    if source_commit:
        try:
            relative = path.relative_to(root).as_posix()
            blob = _git_blob(root, source_commit, relative)
            result["git_commit"] = source_commit
            result["git_blob_sha256"] = hashlib.sha256(blob).hexdigest()
            result["git_blob_matches_receipt"] = result["git_blob_sha256"] == receipt["sha256"]
            if "bytes" in receipt:
                result["git_blob_matches_receipt"] &= len(blob) == receipt["bytes"]
            if not result["git_blob_matches_receipt"]:
                result["errors"].append("Recorded source receipt does not match its historical Git commit bytes")
        except (ValueError, subprocess.CalledProcessError) as error:
            result["errors"].append(f"Historical source blob unavailable: {type(error).__name__}; fetch full Git history if shallow")
    result["passed"] = not result["errors"]
    return result


def _index(root):
    with (root / "experiments/index.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    identifiers = [row["run_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Experiment index contains duplicate run IDs")
    return {row["run_id"]: row for row in rows}


def _verify_final_saved(path):
    try:
        from .final_evaluation import verify_saved
    except ImportError:
        from final_evaluation import verify_saved
    return verify_saved(path)


def verify_run(run_id, *, root=ROOT, index_row=None):
    """Integrity is distinct from eligibility: a preserved failed run stays failed."""
    root = Path(root).resolve()
    if not re.fullmatch(r"R\d+", run_id):
        raise ValueError("run_id must be R followed by digits")
    index_row = _index(root).get(run_id) if index_row is None else index_row
    result = {"run_id": run_id, "errors": [], "warnings": [], "eligible_completed": False,
              "refitted": False, "raw_data_read": False}
    if index_row is None:
        result["errors"].append("Run directory is absent from experiments/index.csv")
    path = root / "experiments/runs" / run_id
    if not (path / "manifest.json").is_file():
        result["errors"].append("Run manifest is missing")
        result["integrity_passed"] = False
        return result
    manifest = _json(path / "manifest.json")
    status = manifest.get("status")
    effective = index_row["status"] if index_row else status
    result.update(manifest_status=status, effective_status=effective)
    if manifest.get("run_id") != run_id:
        result["errors"].append("Manifest run_id differs from its directory/index")
    if index_row:
        if _path(index_row["artifact_dir"], root) != path:
            result["errors"].append("Index artifact_dir does not identify this run directory")
        if effective != status:
            if effective == "invalidated" and status in TERMINAL and index_row.get("decision"):
                result["warnings"].append("Index invalidates a retained historical execution; it is not an eligible completed result")
            else:
                result["errors"].append("Index and manifest statuses disagree")
    if effective not in TERMINAL:
        result["warnings"].append("Run is not terminal; receipts may still be changing and are not sealed")
        result.update(skipped_unsealed=True, integrity_passed=False)
        return result
    result["eligible_completed"] = (effective == status == "completed" and not manifest.get("invalidated")
                                      and manifest.get("valid") is not False)
    commit = manifest.get("git_commit")
    sources = manifest.get("source_files", [])
    if not sources and manifest.get("source_sha256"):
        matched = re.search(r"(?:^|\s)(scripts/[^\s\"']+\.py)(?:\s|$)", manifest.get("command", ""))
        if matched:
            sources = [{"path": matched[1], "sha256": manifest["source_sha256"]}]
            result["warnings"].append("Legacy preflight manifest: one source hash; no modern output receipt list")
    if not commit or not sources:
        result["errors"].append("Missing historical Git commit/source receipts")
    source_paths = [item.get("path") for item in sources]
    if len(set(source_paths)) != len(source_paths):
        result["errors"].append("Duplicate source receipts")
    result["source_receipts"] = [_check_receipt(item, root, source_commit=commit) for item in sources]
    outputs = manifest.get("outputs", [])
    if not outputs:
        (result["errors"] if status == "completed" else result["warnings"]).append("No output receipt list was archived")
    output_paths = [_path(item["path"], root) for item in outputs]
    if len(set(output_paths)) != len(output_paths):
        result["errors"].append("Duplicate output receipts")
    result["output_receipts"] = [_check_receipt(item, root, allowed_directory=path) for item in outputs]
    result["input_manifest_receipts"] = [_check_receipt(manifest[key], root)
                                          for key in ["input_manifest", "derived_input_manifest"] if manifest.get(key)]
    # Input manifests contain raw/derived file receipts, but this audit never opens
    # those datasets. verify_downloads is the separate explicit raw-data checker.
    archived_config = path / "config.json"
    if archived_config.is_file() and "config" in manifest and _json(archived_config) != manifest["config"]:
        result["errors"].append("Archived config.json differs from manifest config")
    environment = _environment(manifest.get("environment"))
    if not environment:
        result["errors"].append("Run did not record an environment")
    try:
        requirement_bytes = _git_blob(root, commit, "requirements.txt")
        required, unsupported = _pins(requirement_bytes.decode("utf-8-sig"))
        matches = _environment_check(environment, required)
        result["historical_environment"] = {"python": manifest.get("python"),
            "requirements_git_sha256": hashlib.sha256(requirement_bytes).hexdigest(), "packages": matches,
            "unsupported_requirement_lines": unsupported, "requirements_match": bool(required) and not unsupported and all(item["matches"] for item in matches)}
        if not result["historical_environment"]["requirements_match"]:
            result["warnings"].append("Recorded environment differs from its committed direct requirements; see per-package evidence")
    except (subprocess.CalledProcessError, TypeError) as error:
        result["warnings"].append(f"Historical requirements unavailable: {type(error).__name__}")
    for category in ["source_receipts", "output_receipts", "input_manifest_receipts"]:
        result["errors"].extend(f"{item['path']}: {error}" for item in result[category] for error in item["errors"])
    final_path = path / "final_result.json"
    mode = manifest.get("config", {}).get("mode")
    if final_path.is_file() and result["eligible_completed"]:
        if result["errors"]:
            result["warnings"].append("Final prediction replay skipped because archive receipts already failed")
        else:
            try:
                result["final_verification"] = _verify_final_saved(path)
                if result["final_verification"].get("refitted") is not False or result["final_verification"].get("raw_test_data_read") is not False:
                    raise ValueError("Final replay did not affirm no fitting/no raw test reads")
            except Exception as error:
                result["errors"].append(f"Saved final prediction verification failed: {type(error).__name__}: {error}")
    elif mode == "final" and result["eligible_completed"]:
        result["errors"].append("Completed final run is missing final_result.json")
    result["integrity_passed"] = not result["errors"]
    # Eligibility includes receipt integrity, not only a status string.
    result["eligible_completed"] &= result["integrity_passed"]
    return result


def verify_archive(*, root=ROOT, run_ids=None, all_runs=False, final_only=False):
    root = Path(root).resolve()
    index = _index(root)
    errors, warnings = [], []
    if final_only:
        ledger = root / "experiments/final_access/2026-08.json"
        access = _json(ledger) if ledger.is_file() else None
        selected = [access["run_id"]] if access else []
        if not selected:
            warnings.append("No final-access ledger exists; final assessment has not been verified")
        elif access.get("status") != "completed":
            errors.append("Final-access ledger is not completed; final assessment has not been verified")
    elif all_runs:
        directories = {path.name for path in (root / "experiments/runs").iterdir() if path.is_dir() and re.fullmatch(r"R\d+", path.name)}
        selected = sorted(set(index) | directories)
    else:
        selected = list(run_ids or [])
    runs = []
    for run_id in selected:
        try:
            runs.append(verify_run(run_id, root=root, index_row=index.get(run_id)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            runs.append({"run_id": run_id, "integrity_passed": False, "eligible_completed": False,
                         "errors": [f"Malformed/unreadable archive: {type(error).__name__}: {error}"], "warnings": []})
    environment = current_environment(root)
    if not environment["requirements_match"]:
        errors.append("Current environment does not match current pinned requirements")
    failed = [run["run_id"] for run in runs if not run.get("integrity_passed") and not run.get("skipped_unsealed")]
    if failed:
        errors.append("Run integrity checks failed: " + ", ".join(failed))
    final_verified = any(item.get("final_verification", {}).get("verified") is True for item in runs)
    if final_only and selected and not final_verified:
        errors.append("The final ledger does not identify a completed integrity-verified saved prediction archive")
    return {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
            "passed": not errors, "errors": errors, "warnings": warnings, "environment": environment, "runs": runs,
            "terminal_runs_checked": sum(not item.get("skipped_unsealed", False) for item in runs),
            "unsealed_runs_skipped": [item["run_id"] for item in runs if item.get("skipped_unsealed")],
            "final_verified": final_verified,
            "raw_data_read": False, "refitted": False, "pickle_or_joblib_loaded": False,
            "raw_verification_entry": "python scripts/verify_downloads.py (separate explicit command)",
            "clone_limit": "Git ignores processed/* and run data/, predictions/, models/; local absence fails full receipt verification rather than being silently skipped"}


def write_report(result, output, *, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root / "experiments/runs"):
        raise ValueError("Audit output must be outside experiment run archives")
    for ancestor in [output.parent, *output.parents]:
        if (ancestor / "manifest.json").is_file():
            raise ValueError("Audit output must not modify an existing artifact archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run-id")
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--final", action="store_true")
    parser.add_argument("--output", help="Optional new JSON path outside all run archives")
    args = parser.parse_args()
    result = verify_archive(run_ids=[args.run_id] if args.run_id else None, all_runs=args.all, final_only=args.final)
    if args.output:
        write_report(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
