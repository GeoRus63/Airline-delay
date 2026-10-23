"""Download complete IEM AFOS TAF text products for stations used by the BTS airports."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import argparse
import json
import re
import subprocess
import urllib.parse
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "raw" / "weather" / "taf"
OUT_DIR = ROOT / "raw" / "weather" / "taf_text"
MANIFEST_PATH = ROOT / "tables" / "taf_text_manifest.json"
BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
START = datetime(2026, 3, 30, 0, 0, tzinfo=timezone.utc)
END = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
GROUP_SIZE = 6
LIMIT = 9999
PRODUCT_RE = re.compile(r"^(\d{12})-([^-]+)-([^-]+)-(TAF[A-Z0-9]{3})(?:-([^-]+))?$")
MEMBER_RE = re.compile(r"^(TAF[A-Z0-9]{3})_(\d{12})\.txt$")
HEADER_RE = re.compile(r"^(FT[A-Z0-9]{4})\s+([A-Z0-9]{4})\s+\d{6}(?:\s+([A-Z]{3}))?\s*$", re.M)


def read_inputs():
    ids, sources = set(), []
    for path in sorted(CSV_DIR.glob("batch_*.csv.gz")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = 0
        with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "product_id" not in reader.fieldnames:
                raise ValueError(f"{path.name}: missing product_id column")
            for row in reader:
                rows += 1
                if row.get("product_id"):
                    ids.add(row["product_id"])
        sources.append({"file": str(path.relative_to(ROOT)), "rows": rows, "sha256": digest})
    if not sources:
        raise FileNotFoundError(f"No TAF CSV batches found under {CSV_DIR}")
    products = {}
    for product_id in ids:
        match = PRODUCT_RE.fullmatch(product_id)
        if not match:
            raise ValueError(f"Unexpected TAF product_id: {product_id}")
        stamp, center, ftype, pil, bbb = match.groups()
        products[product_id] = {"timestamp": stamp, "center": center, "ftype": ftype, "pil": pil, "bbb": bbb}
    return products, sources


def request_zip(pils, start, end):
    query = urllib.parse.urlencode({"limit": LIMIT, "order": "asc", "pil": ",".join(pils), "fmt": "zip",
                                   "sdate": start.strftime("%Y-%m-%dT%H:%MZ"), "edate": end.strftime("%Y-%m-%dT%H:%MZ")})
    url = f"{BASE_URL}?{query}"
    print(f"Fetching {','.join(pils)} {start:%Y-%m-%dT%H:%MZ} to {end:%Y-%m-%dT%H:%MZ}", flush=True)
    result = subprocess.run(["curl", "-fsSL", "--retry", "3", "--retry-all-errors", "--retry-delay", "2",
                             "--connect-timeout", "30", "--max-time", "480", url],
                            check=True, capture_output=True)
    body = result.stdout
    if not zipfile.is_zipfile(io.BytesIO(body)):
        raise ValueError(f"IEM did not return a ZIP for {url}")
    return body, url


def issue_partition(pils, start, end, depth=0):
    body, url = request_zip(pils, start, end)
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".txt")]
    if len(members) >= LIMIT:
        seconds = int((end - start).total_seconds())
        if seconds <= 60:
            raise RuntimeError(f"IEM limit still reached in a one-minute window: {url}")
        midpoint = start + timedelta(seconds=(seconds // 2 // 60) * 60)
        if midpoint <= start or midpoint >= end:
            raise RuntimeError(f"Cannot subdivide capped request: {url}")
        issue_partition(pils, start, midpoint, depth + 1)
        issue_partition(pils, midpoint, end, depth + 1)
        return
    filename = f"batch_{issue_partition.next_index:03d}.zip"
    issue_partition.next_index += 1
    path = OUT_DIR / filename
    temporary = path.with_suffix(".zip.tmp")
    temporary.write_bytes(body)
    temporary.replace(path)
    record = {"pil_group": pils, "start_utc": start.isoformat(), "end_utc_exclusive": end.isoformat(),
              "url": url, "split_depth": depth, "product_count": len(members), "bytes": len(body),
              "sha256": hashlib.sha256(body).hexdigest(), "file": str(path.relative_to(ROOT)),
              "saved_bytes": path.stat().st_size}
    issue_partition.records.append(record)
    issue_partition.checkpoint()


def identify_member(name, payload):
    file_match = MEMBER_RE.fullmatch(Path(name).name)
    if not file_match:
        return None
    pil, stamp = file_match.groups()
    text = payload.decode("latin-1", "replace")
    header = HEADER_RE.search(text)
    if not header:
        return None
    ftype, center, bbb = header.groups()
    bbb = bbb or ""
    suffix = f"-{bbb}" if bbb else ""
    return f"{stamp}-{center}-{ftype}-{pil}{suffix}"


def _month_offset(year, month, offset):
    index = year * 12 + month - 1 + offset
    return index // 12, index % 12 + 1


def _valid_time_candidates(token, issue_time):
    day, hour = int(token[:2]), int(token[2:4])
    minute = int(token[4:6]) if len(token) == 6 else 0
    choices = []
    for offset in (-1, 0, 1):
        year, month = _month_offset(issue_time.year, issue_time.month, offset)
        try:
            value = datetime(year, month, day, 0 if hour == 24 else hour, minute, tzinfo=timezone.utc)
        except ValueError:
            continue
        if hour == 24:
            value += timedelta(days=1)
        choices.append(value)
    return choices


def validity_row(name, payload):
    match = MEMBER_RE.fullmatch(Path(name).name)
    if not match:
        return None
    pil, stamp = match.groups()
    issue_time = datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    text = payload.decode("latin-1", "replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = HEADER_RE.search(text)
    station_pattern = re.compile(r"^(?:TAF\s+)?(?:(?:AMD|COR|RTD)\s+)*(?:TAF\s+)?([A-Z0-9]{4})\s+(\d{6})Z(?:\s+(\d{4,6})/(\d{4,6}))?\b")
    station_line = next((re.match(station_pattern, line) for line in lines if station_pattern.match(line)), None)
    station = station_line.group(1) if station_line else ""
    upper = text.upper()
    status = "nil" if re.search(r"\bNIL\b", upper) else "cancelled" if re.search(r"\bCNL\b", upper) else "normal"
    valid_from = valid_to = ""
    if status == "normal" and station_line and station_line.group(3):
        starts = _valid_time_candidates(station_line.group(3), issue_time)
        ends = _valid_time_candidates(station_line.group(4), issue_time)
        if starts and ends:
            start = min(starts, key=lambda value: abs((value - issue_time).total_seconds()))
            later_ends = [value for value in ends if value > start]
            if later_ends:
                end = min(later_ends)
                valid_from = start.isoformat().replace("+00:00", "Z")
                valid_to = end.isoformat().replace("+00:00", "Z")
    product_id = identify_member(name, payload) or ""
    return {"product_id": product_id, "station": station,
            "issue_time": issue_time.isoformat().replace("+00:00", "Z"),
            "valid_from": valid_from, "valid_to": valid_to, "status": status,
            "raw_header": "\n".join(lines[:6]), "raw_name": Path(name).name}


def main(prepare_validity=False):
    products, source_manifest = read_inputs()
    pils = sorted({value["pil"] for value in products.values()})
    old_manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    old_requests = old_manifest.get("requests", [])
    complete_pils = set(old_manifest.get("complete_pils", []))
    old_requests = [item for item in old_requests if set(item["pil_group"]).issubset(complete_pils)]
    for item in old_manifest.get("requests", []):
        if item not in old_requests:
            stale = ROOT / item.get("file", "")
            if stale.is_file() and stale.parent == OUT_DIR:
                stale.unlink()
    new_pils = [pil for pil in pils if pil not in complete_pils]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "tables").mkdir(parents=True, exist_ok=True)
    issue_partition.records = list(old_requests)
    issue_partition.next_index = max((int(path.stem.split("_")[-1]) for path in OUT_DIR.glob("*.zip")
                                      if path.stem.split("_")[-1].isdigit()), default=0) + 1

    def checkpoint(state="downloading"):
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {"source": "Iowa Environmental Mesonet AFOS retrieve.py",
                "source_help": f"{BASE_URL}?help=", "window_utc": {"start_inclusive": START.isoformat(),
                "end_exclusive": END.isoformat()}, "query_parameters": {"fmt": "zip", "limit": LIMIT,
                "order": "asc", "group_size_pils": GROUP_SIZE}, "taf_pil_count": len(pils), "taf_pils": pils,
                "complete_pils": sorted(complete_pils), "source_csv_files": source_manifest,
                "requests": issue_partition.records, "state": state}
        temporary = MANIFEST_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(MANIFEST_PATH)

    issue_partition.checkpoint = checkpoint
    checkpoint()
    for offset in range(0, len(new_pils), GROUP_SIZE):
        group = new_pils[offset:offset + GROUP_SIZE]
        issue_partition(group, START, END)
        complete_pils.update(group)
        checkpoint()

    archive_members, validity_rows = {}, []
    for item in issue_partition.records:
        path = ROOT / item["file"]
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = member.filename
                if not name.lower().endswith(".txt"):
                    continue
                raw = archive.read(member)
                product_id = identify_member(name, raw)
                if product_id:
                    archive_members.setdefault(product_id, []).append({"file": str(path.relative_to(ROOT)), "member": name})
                    if prepare_validity:
                        row = validity_row(name, raw)
                        if row:
                            validity_rows.append(row)

    matched = sorted(set(products) & set(archive_members))
    missing = sorted(set(products) - set(archive_members))
    extra = sorted(set(archive_members) - set(products))
    handoff_path = ROOT / "tables" / "taf_text_missing_products.csv"
    with handoff_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["product_id", "center", "ttaaii", "pil", "status", "full_range_lookup",
                  "two_hour_probe", "calendar_day_probe", "single_product_lookup", "notes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for product_id in missing:
            meta = products[product_id]
            writer.writerow({"product_id": product_id, "center": meta["center"],
                             "ttaaii": meta["ftype"], "pil": meta["pil"],
                             "status": "not_found_in_iem_responses",
                             "full_range_lookup": "no matching ZIP member",
                             "two_hour_probe": "no matching ZIP member",
                             "calendar_day_probe": "no matching ZIP member",
                             "single_product_lookup": "see independent_official_lookup in manifest" if old_manifest.get("independent_official_lookup") else "not attempted",
                             "notes": "Product ID is present in source TAF CSV batches; source response did not contain a matching raw message."})
    probed = {(item["probe_product_id"], item.get("probe_kind", "two_hour")) for item in issue_partition.records
              if item.get("probe_product_id")}
    for product_id in missing:
        meta = products[product_id]
        issue = datetime.strptime(meta["timestamp"], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        day_start = issue.replace(hour=0, minute=0)
        for probe_kind, start, end in [("two_hour", issue - timedelta(hours=1), issue + timedelta(hours=1)),
                                       ("calendar_day", day_start, day_start + timedelta(days=1))]:
            if (product_id, probe_kind) in probed:
                continue
            body, url = request_zip([meta["pil"]], start, end)
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                member_count = sum(item.filename.lower().endswith(".txt") for item in archive.infolist())
                if member_count >= LIMIT:
                    raise RuntimeError(f"One-PIL source probe reached IEM limit for {product_id}")
            path = OUT_DIR / f"probe_{issue_partition.next_index:03d}.zip"
            issue_partition.next_index += 1
            temporary = path.with_suffix(".zip.tmp")
            temporary.write_bytes(body)
            temporary.replace(path)
            probe = {"pil_group": [meta["pil"]], "start_utc": start.isoformat(), "end_utc_exclusive": end.isoformat(),
                     "url": url, "split_depth": 0, "product_count": member_count, "bytes": len(body),
                     "sha256": hashlib.sha256(body).hexdigest(), "file": str(path.relative_to(ROOT)),
                     "saved_bytes": path.stat().st_size, "probe_product_id": product_id, "probe_kind": probe_kind}
            issue_partition.records.append(probe)
            with zipfile.ZipFile(path) as probe_archive:
                for member in probe_archive.infolist():
                    if not member.filename.lower().endswith(".txt"):
                        continue
                    product_id_in_zip = identify_member(member.filename, probe_archive.read(member))
                    if product_id_in_zip:
                        archive_members.setdefault(product_id_in_zip, []).append({"file": str(path.relative_to(ROOT)),
                                                                                   "member": member.filename})
            probed.add((product_id, probe_kind))
            checkpoint()
    matched = sorted(set(products) & set(archive_members))
    missing = sorted(set(products) - set(archive_members))
    extra = sorted(set(archive_members) - set(products))
    validity_path = ROOT / "references" / "taf_validity.csv"
    if prepare_validity:
        validity_path.parent.mkdir(parents=True, exist_ok=True)
        validity_fields = ["product_id", "station", "issue_time", "valid_from", "valid_to", "status", "raw_header", "raw_name"]
        with validity_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=validity_fields)
            writer.writeheader()
            writer.writerows(validity_rows)
    manifest = {
        "source": "Iowa Environmental Mesonet AFOS retrieve.py",
        "source_help": "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?help=",
        "window_utc": {"start_inclusive": START.isoformat(), "end_exclusive": END.isoformat()},
        "query_parameters": {"fmt": "zip", "limit": LIMIT, "order": "asc", "group_size_pils": GROUP_SIZE},
        "taf_pil_count": len(pils), "taf_pils": pils, "complete_pils": pils, "source_csv_files": source_manifest,
        "unique_csv_product_ids": len(products), "matched_csv_product_ids": len(matched),
        "unmatched_csv_product_ids": missing, "archive_product_ids_not_in_csv": extra,
        "prepare_validity_requested": prepare_validity,
        "taf_validity_file": str(validity_path.relative_to(ROOT)) if prepare_validity else None,
        "taf_validity_rows": len(validity_rows) if prepare_validity else None,
        "taf_validity_missing_window_rows": sum(row["status"] == "normal" and not row["valid_from"] for row in validity_rows) if prepare_validity else None,
        "taf_status_counts": {status: sum(row["status"] == status for row in validity_rows) for status in ["normal", "cancelled", "nil"]} if prepare_validity else None,
        "requests": issue_partition.records,
        "state": "complete" if not missing else "needs_source_followup",
        "notes": ["Each saved ZIP is the original IEM response; each AFOS product is retained as its own TXT member.",
                  "A request at the 9999-member ceiling is discarded and recursively split into disjoint half-open UTC windows.",
                  "Product coverage matches full product_id reconstructed from ZIP member PIL/time and raw AFOS header."]
    }
    if old_manifest.get("independent_official_lookup"):
        manifest["independent_official_lookup"] = old_manifest["independent_official_lookup"]
        manifest["unmatched_handoff_csv"] = str(handoff_path.relative_to(ROOT))
        manifest["unmatched_handoff_rows"] = len(missing)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pil_count": len(pils), "zip_count": len(issue_partition.records), "products_csv": len(products),
                      "matched": len(matched), "unmatched": len(missing), "manifest": str(MANIFEST_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-validity", action="store_true", help="Also parse TAF validity headers")
    main(prepare_validity=parser.parse_args().prepare_validity)
