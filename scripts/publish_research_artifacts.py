"""Publish the explicit model/prediction manifest to this project's GitHub release."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = "GeoRus63/Airline-delay"
API = f"https://api.github.com/repos/{REPO}"
TAG = "research-artifacts-v1"
MANIFEST = ROOT / "artifacts/model_prediction_manifest.json"
RECEIPT = ROOT / "research/audits/model_prediction_release_upload.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("upload", "publish", "verify"))
    args = parser.parse_args()
    # Reuse the existing Git credential helper. Credentials stay in memory and
    # are never written to receipts, shell arguments, logs, or download requests.
    result = subprocess.run(["git", "credential", "fill"],
                            input="protocol=https\nhost=github.com\n\n",
                            text=True, capture_output=True, check=True, cwd=ROOT)
    credential = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = credential["password"]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}

    def api(method: str, path: str, **kwargs):
        response = requests.request(method, API + path, headers=headers, timeout=60, **kwargs)
        if not response.ok:
            raise RuntimeError(f"GitHub API {method} {path}: HTTP {response.status_code}")
        return response.json()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = [{"path": ROOT / "tmp" / TAG / e["asset"], "name": e["asset"],
                 "size": e["bytes"], "sha256": e["sha256"]} for e in manifest["archives"]]
    expected.append({"path": MANIFEST, "name": MANIFEST.name,
                     "size": MANIFEST.stat().st_size, "sha256": digest(MANIFEST)})
    releases = api("GET", "/releases?per_page=100")
    matching = [r for r in releases if r["tag_name"] == TAG]
    if len(matching) > 1:
        raise RuntimeError("Ambiguous release")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    if not matching:
        if args.stage != "upload":
            raise RuntimeError("Release has not been staged")
        release = api("POST", "/releases", json={
            "tag_name": TAG, "target_commitish": head, "draft": True, "prerelease": False,
            "name": "Research archive v1 — all preserved models and predictions",
            "body": "2024 aircraft-reference research snapshot; updating to the latest 2025 aircraft reference remains pending. No retraining was performed for this archive.\n\n17 per-run ZIP assets contain all 134 model files and 134 full prediction files (268 files, 2,243,334,942 bytes), including preserved failed run R0011. Each source file was checked against its original run receipt. Two explanation prediction samples are already tracked in Git.\n\nDownload model_prediction_manifest.json for file sizes/SHA-256. Use scripts/restore_research_artifacts.py from codex/apr-aug-research to restore original relative paths and verify every file. Processed inputs and per-run feature tables are excluded; this is not a complete computational backup. Existing reports, freezes and run records are unchanged."
        })
    else:
        release = matching[0]
    release_id = release["id"]
    if args.stage in ("upload", "publish") and not release["draft"]:
        raise RuntimeError("Published archive is immutable; use verify or a new release version")

    def assets():
        return {a["name"]: a for a in api("GET", f"/releases/{release_id}/assets?per_page=100")}

    def matches(remote: dict, entry: dict) -> bool:
        return (remote["size"] == entry["size"] and remote["state"] == "uploaded"
                and remote.get("digest") == "sha256:" + entry["sha256"])

    if args.stage == "upload":
        existing = assets()
        upload_url = release["upload_url"].split("{")[0]
        if not upload_url.startswith("https://uploads.github.com/repos/GeoRus63/Airline-delay/"):
            raise RuntimeError("Unexpected GitHub upload host/path")

        def upload(entry):
            if entry["name"] in existing:
                if not matches(existing[entry["name"]], entry):
                    raise RuntimeError(f"Different existing asset: {entry['name']}")
                print(f"Already verified: {entry['name']}", flush=True)
                return
            if entry["path"].stat().st_size != entry["size"] or digest(entry["path"]) != entry["sha256"]:
                raise ValueError(f"Local asset mismatch: {entry['name']}")
            upload_headers = dict(headers, **{"Content-Type": "application/octet-stream"})
            with entry["path"].open("rb") as stream:
                response = requests.post(upload_url, params={"name": entry["name"]},
                                         headers=upload_headers, data=stream, timeout=(30, 600))
            if response.status_code != 201:
                raise RuntimeError(f"Asset upload {entry['name']}: HTTP {response.status_code}")
            if not matches(response.json(), entry):
                raise ValueError(f"Remote upload digest mismatch: {entry['name']}")
            print(f"Uploaded and SHA-256 verified: {entry['name']}", flush=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(upload, expected))

    remote = assets()
    if set(remote) != {e["name"] for e in expected}:
        raise ValueError("Remote release membership mismatch")
    for entry in expected:
        if not matches(remote[entry["name"]], entry):
            raise ValueError(f"Remote digest/size/state mismatch: {entry['name']}")
    if args.stage == "publish":
        release = api("PATCH", f"/releases/{release_id}", json={
            "draft": False, "target_commitish": head, "make_latest": "true"})
    receipt = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "release_id": release_id,
               "release_url": release["html_url"], "tag": TAG, "draft": release["draft"],
               "manifest_sha256": digest(MANIFEST), "expected_assets": len(expected),
               "all_remote_digests_match": True,
               "assets": [{key: a.get(key) for key in ("name", "id", "size", "digest", "state", "browser_download_url")}
                          for a in sorted(remote.values(), key=lambda x: x["name"])]}
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Release verified: {len(remote)} assets, draft={release['draft']}", flush=True)


if __name__ == "__main__":
    main()
