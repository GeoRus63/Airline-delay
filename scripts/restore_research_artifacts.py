"""Download and verify archived models/predictions without loading models or training."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]


def destination(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe member: {name}")
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Member outside destination: {name}")
    return target


def restore_archive(archive_path: Path, entry: dict, root: Path) -> None:
    if not valid(archive_path, entry):
        raise ValueError(f"Archive SHA-256/size mismatch: {archive_path.name}")
    expected = {item["path"]: item for item in entry["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != set(expected):
            raise ValueError(f"Archive member mismatch: {archive_path.name}")
        # Preflight the entire archive before writing anything.
        for name, item in expected.items():
            path = destination(root, name)
            if path.exists() and not valid(path, item):
                raise FileExistsError(f"Refusing to overwrite different existing file: {path}")
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                    raise ValueError(f"Member SHA-256 mismatch: {name}")
        for name, item in expected.items():
            path = destination(root, name)
            if valid(path, item):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".restore-", delete=False) as dest:
                temporary = Path(dest.name)
                with archive.open(name) as source:
                    shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
            if not valid(temporary, item):
                raise ValueError(f"Extracted file mismatch: {name}; temporary file retained")
            # A concurrent writer must never be overwritten.
            with temporary.open("rb") as source, path.open("xb") as dest:
                shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
            temporary.unlink()
            if not valid(path, item):
                raise ValueError(f"Restored file mismatch: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/model_prediction_manifest.json")
    parser.add_argument("--runs", nargs="+", help="Run IDs, e.g. R0028 R0030; default all available runs")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository or empty restore destination")
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "tmp/research-artifacts-downloads")
    parser.add_argument("--verify-only", action="store_true", help="Check local extracted files; never access network")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    archives = manifest["archives"]
    if args.runs:
        unknown = set(args.runs) - {item["run_id"] for item in archives}
        if unknown:
            parser.error(f"No archived model/prediction files for: {sorted(unknown)}")
        archives = [item for item in archives if item["run_id"] in args.runs]
    root = args.root.resolve()
    count = 0
    for entry in archives:
        files = entry["files"]
        if args.verify_only:
            for item in files:
                if not valid(destination(root, item["path"]), item):
                    raise ValueError(f"Missing or mismatched local file: {item['path']}")
        else:
            args.archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = args.archive_dir / entry["asset"]
            if not valid(archive_path, entry):
                partial = archive_path.with_suffix(".zip.partial")
                request = urllib.request.Request(entry["download_url"], headers={"User-Agent": "Airline-delay-artifact-restore"})
                with urllib.request.urlopen(request, timeout=60) as source, partial.open("wb") as dest:
                    shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
                if not valid(partial, entry):
                    raise ValueError(f"Downloaded archive mismatch: {entry['asset']}")
                partial.replace(archive_path)
            restore_archive(archive_path, entry, root)
        count += len(files)
        print(f"Verified {entry['run_id']}: {len(files)} files", flush=True)
    print(f"Complete: {len(archives)} runs / {count} files", flush=True)


if __name__ == "__main__":
    main()
