"""Package preserved model/prediction bytes; never fit or modify a run."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "research-artifacts-v1"
REPO = "GeoRus63/Airline-delay"
OUT = ROOT / "tmp" / TAG
MANIFEST = ROOT / "artifacts" / "model_prediction_manifest.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_run(record: dict) -> dict | None:
    run_id = record["run_id"]
    directory = ROOT / "experiments" / "runs" / run_id
    files = sorted(p for kind in ("models", "predictions")
                   for p in (directory / kind).rglob("*") if p.is_file())
    if not files:
        return None
    original = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    receipts = {entry["path"].replace("\\", "/"): entry
                for entry in original["outputs"]}
    entries = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        actual = {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)}
        expected = receipts[relative]
        if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError(f"Original run receipt mismatch: {relative}")
        entries.append(actual)
    name = f"{run_id}-models-predictions.zip"
    target = OUT / name
    # Source gzip/joblib files are already compressed. ZIP_STORED preserves fast,
    # transparent archives; fixed member metadata makes packaging deterministic.
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for path, entry in zip(files, entries):
            info = zipfile.ZipInfo(entry["path"], date_time=(2026, 10, 24, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as dest:
                while chunk := source.read(8 * 1024 * 1024):
                    dest.write(chunk)
    with zipfile.ZipFile(target) as archive:
        if set(archive.namelist()) != {e["path"] for e in entries}:
            raise ValueError(f"Archive membership mismatch: {name}")
        for entry in entries:
            with archive.open(entry["path"]) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                    raise ValueError(f"Archive content mismatch: {entry['path']}")
    result = {"run_id": run_id, "run_status": record["status"], "asset": name,
              "bytes": target.stat().st_size, "sha256": digest(target),
              "download_url": f"https://github.com/{REPO}/releases/download/{TAG}/{name}",
              "files": entries}
    print(f"Packaged {run_id}: {len(entries)} verified files, {result['bytes']} bytes", flush=True)
    return result


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError("Manifest already exists; use a new archive version for new bytes")
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "experiments/index.csv").open(encoding="utf-8-sig", newline="") as stream:
        runs = list(csv.DictReader(stream))
    with ThreadPoolExecutor(max_workers=4) as pool:
        archives = [item for item in pool.map(package_run, runs) if item]
    members = [item for archive in archives for item in archive["files"]]
    # Also account for prediction samples already stored directly in Git.
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    samples = [{"path": p, "bytes": (ROOT / p).stat().st_size, "sha256": digest(ROOT / p)}
               for p in tracked if p and "predictions" in Path(p).name]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "release_tag": TAG, "release_url": f"https://github.com/{REPO}/releases/tag/{TAG}",
        "scope": "All existing per-run models and full prediction files, including failed-run outputs; explanation prediction samples are already in Git. Processed inputs and per-run feature data are not included.",
        "aircraft_reference": "2024 primary; 2025 update remains pending. These are preserved original results, not retrained results.",
        "file_count": len(members), "uncompressed_file_bytes": sum(e["bytes"] for e in members),
        "model_file_count": sum("/models/" in e["path"] for e in members),
        "prediction_file_count": sum("/predictions/" in e["path"] for e in members),
        "archive_count": len(archives), "archives": archives,
        "prediction_samples_in_git": samples,
    }
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Manifest: {len(archives)} archives / {len(members)} files", flush=True)


if __name__ == "__main__":
    main()
