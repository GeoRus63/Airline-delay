"""D0002 monthly retrospective QA snapshots; not an online deployment cohort.

Full-plan adjacency is retained. Future endpoint outcomes remain necessary for
completed-connection labels, but never propagate QA flags to earlier pairs.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd

try:
    from .prepare_flights import build_connections, local_schedule_to_utc
except ImportError:
    from prepare_flights import build_connections, local_schedule_to_utc

ROOT = Path(__file__).resolve().parents[1]
DATA_ID = "D0002"
FLAGS = ["tail_schedule_incomplete", "schedule_tie", "actual_order_conflict"]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_flags(flights, cutoff):
    """Only prefix plans and prefix actual departures contribute global flags."""
    missing = flights.scheduled_dep_utc.isna()
    membership_time = flights.scheduled_dep_utc.copy()
    if missing.any():
        subset = flights.loc[missing]
        membership_time.loc[missing] = local_schedule_to_utc(
            subset.FlightDate, pd.Series("0000", index=subset.index), subset.origin_tz)
    # Unknown calendar dates cannot safely be assigned to a future window.
    inside = membership_time.lt(cutoff) | membership_time.isna()
    ordered = flights.loc[inside & flights.Tail_Number.notna(),
                          ["flight_id", "Tail_Number", "scheduled_dep_utc", "actual_dep_utc"]].copy()
    ordered = ordered.sort_values(["Tail_Number", "scheduled_dep_utc", "flight_id"], kind="stable")
    ordered["tail_schedule_incomplete"] = ordered.scheduled_dep_utc.isna().groupby(ordered.Tail_Number).transform("any")
    ordered["schedule_tie"] = ordered.duplicated(["Tail_Number", "scheduled_dep_utc"], keep=False)
    known = ordered.loc[ordered.actual_dep_utc.lt(cutoff)]
    previous = known.groupby("Tail_Number").actual_dep_utc.cummax().groupby(known.Tail_Number).shift()
    reverse = known.iloc[::-1]
    following = reverse.groupby("Tail_Number").actual_dep_utc.cummin().groupby(reverse.Tail_Number).shift().reindex(known.index)
    ordered["actual_order_conflict"] = False
    ordered.loc[known.index, "actual_order_conflict"] = known.actual_dep_utc.le(previous) | known.actual_dep_utc.ge(following)
    return ordered.set_index("flight_id")[FLAGS], {
        "cutoff_utc": cutoff.isoformat(), "qa_plan_rows": len(ordered),
        "qa_actual_departures": len(known),
        "missing_schedule_rows_in_window": int(ordered.scheduled_dep_utc.isna().sum()),
        "max_actual_event_utc": str(known.actual_dep_utc.max()),
        "actual_events_at_or_after_cutoff": int(known.actual_dep_utc.ge(cutoff).sum()),
    }


def rebuild(flights, candidates=None):
    if candidates is None:
        candidates, _ = build_connections(flights)
    pairs = candidates.copy()
    original_checks = [c for c in pairs if c.startswith("check_")]
    for column in original_checks + ["first_exclusion_reason", "main_analysis_eligible"]:
        pairs["full_period_audit_" + column] = pairs[column]
    pairs["prediction_time"] = pairs.upstream_actual_arr_utc
    pairs["label_time"] = pairs.downstream_actual_dep_utc
    pairs["qa_cohort_month"] = pairs.prediction_time.dt.strftime("%Y-%m")
    pairs["qa_cutoff_utc"] = pd.Series(pd.NaT, index=pairs.index, dtype="datetime64[ns, UTC]")
    pairs["downstream_outside_qa_window"] = False
    pairs["check_qa_cohort_in_scope"] = False
    lookup = flights.set_index("flight_id")
    for side in ["upstream", "downstream"]:
        for field in ["source_file", "source_row"]:
            if field in lookup:
                pairs[side + "_" + field] = pairs[side + "_flight_id"].map(lookup[field])
    audits = []
    for month in range(4, 9):
        cutoff = pd.Timestamp(year=2026, month=month + 1, day=1, tz="UTC")
        mask = pairs.qa_cohort_month.eq(f"2026-{month:02d}")
        flags, audit = snapshot_flags(flights, cutoff)
        sample = pairs.loc[mask]
        up = sample.upstream_flight_id
        down = sample.downstream_flight_id
        up_inside = up.isin(flags.index)
        down_inside = down.isin(flags.index)
        boundary = ~down_inside
        pairs.loc[mask, "qa_cutoff_utc"] = cutoff
        pairs.loc[mask, "downstream_outside_qa_window"] = boundary
        pairs.loc[mask, "check_qa_cohort_in_scope"] = up_inside
        pairs.loc[mask, "check_tail_schedule_complete"] = ~up.map(flags.tail_schedule_incomplete).fillna(True).astype(bool)
        up_bad = up.map(flags.schedule_tie).fillna(True).astype(bool) | up.map(lookup.duplicate_key).fillna(True).astype(bool)
        down_bad = down.map(flags.schedule_tie).fillna(False).astype(bool)
        # The boundary endpoint is checked locally; its whole-tail flags do not propagate.
        down_bad |= down.map(lookup.duplicate_key).fillna(True).astype(bool) & down_inside
        pairs.loc[mask, "check_unique_schedule"] = ~(up_bad | down_bad) & sample.upstream_scheduled_dep_utc.ne(sample.downstream_scheduled_dep_utc)
        order_bad = up.map(flags.actual_order_conflict).fillna(True).astype(bool) | down.map(flags.actual_order_conflict).fillna(False).astype(bool)
        pairs.loc[mask, "check_actual_order_consistent"] = ~order_bad & sample.upstream_actual_dep_utc.lt(sample.downstream_actual_dep_utc)
        audit.update(cohort_pairs=len(sample), boundary_downstream_pairs=int(boundary.sum()))
        audits.append(audit)
    # Shared RQ1 connection validity allows zero ground time. RQ2 is strictly later.
    pairs["first_exclusion_reason"] = "accepted"
    for check in ["check_qa_cohort_in_scope"] + original_checks:
        pairs[check] = pairs[check].fillna(False).astype(bool)
        pairs.loc[pairs.first_exclusion_reason.eq("accepted") & ~pairs[check], "first_exclusion_reason"] = check.removeprefix("check_")
    pairs["main_analysis_eligible"] = pairs.first_exclusion_reason.eq("accepted")
    pairs["rq2_eligible"] = pairs.main_analysis_eligible & pairs.prediction_time.lt(pairs.label_time) & pairs.downstream_dep_delay_minutes.notna()
    return pairs, pd.DataFrame(audits)


def run(output_dir, data_dir=ROOT / "processed/D0002"):
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite derived data: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    flights = pd.read_csv(ROOT / "processed/flights_clean.csv.gz", low_memory=False)
    candidates = pd.read_csv(ROOT / "processed/connection_candidates.csv.gz", low_memory=False)
    for frame in [flights, candidates]:
        for column in frame:
            if column.endswith("_utc"):
                frame[column] = pd.to_datetime(frame[column], utc=True)
            elif column == "FlightDate" or column.endswith("_flight_date"):
                frame[column] = pd.to_datetime(frame[column])
    pairs, audit = rebuild(flights, candidates)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("connection_candidates", pairs), ("flight_connections", pairs.loc[pairs.main_analysis_eligible])]:
        frame.to_csv(data_dir / f"{name}.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 3, "mtime": 0})
        if name == 'flight_connections':
            frame.to_pickle(data_dir / 'flight_connections.pkl')
    changed = pairs.main_analysis_eligible.ne(pairs.full_period_audit_main_analysis_eligible)
    differences = []
    for column in [c for c in candidates if c.startswith("check_")] + ["main_analysis_eligible"]:
        previous = pairs["full_period_audit_" + column].astype(bool)
        current = pairs[column].astype(bool)
        differences.append({"field": column, "changed_rows": int(previous.ne(current).sum()),
                            "false_to_true": int((~previous & current).sum()),
                            "true_to_false": int((previous & ~current).sum())})
    pd.DataFrame(differences).to_csv(output_dir / "cohort_check_differences.csv", index=False)
    pairs.loc[changed, ["connection_id", "qa_cohort_month", "first_exclusion_reason", "full_period_audit_first_exclusion_reason"]].to_csv(output_dir / "cohort_membership_changes.csv", index=False)
    audit.to_csv(output_dir / "cohort_structural_audit.csv", index=False)
    summary = {"data_id": DATA_ID, "source_data_id": "D0001", "candidate_rows": len(pairs),
               "accepted_rows": int(pairs.main_analysis_eligible.sum()), "rq2_rows": int(pairs.rq2_eligible.sum()),
               "changed_membership_rows": int(changed.sum()), "data_dir": data_dir.relative_to(ROOT).as_posix(),
               "boundary_accepted_rows": int((pairs.main_analysis_eligible & pairs.downstream_outside_qa_window).sum()),
               "qa_out_of_scope_rows": int((~pairs.check_qa_cohort_in_scope).sum()),
               "rules": {"qa_month": "upstream actual arrival UTC month", "missing_schedule_membership": "origin local FlightDate midnight; unknown dates conservatively included",
                         "scope": "retrospective normally completed connections; full-plan adjacency; future endpoint labels allowed but no propagation of endpoint QA",
                         "actual_event_cutoff": "strictly before monthly cutoff", "months": ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]}}
    (output_dir / "cohort_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    exclusions = pairs.first_exclusion_reason.value_counts().rename_axis("reason").reset_index(name="pairs")
    exclusions["share_of_candidates"] = exclusions.pairs / max(len(pairs), 1)
    exclusions.to_csv(data_dir / "connection_exclusions.csv", index=False)
    source_summary = ROOT / "tables/cleaning_summary.json"
    cleaning = json.loads(source_summary.read_text(encoding="utf-8"))
    cleaning["data_id"] = DATA_ID
    cleaning["source_data_id"] = "D0001"
    cleaning["audit_inheritance"] = "Raw-flight audit inherited from D0001. Full-period flight flags are audit only; D0002 monthly connection qualification is authoritative."
    cleaning["summary"].update(candidate_connections=len(pairs), accepted_connections=summary["accepted_rows"],
                               accepted_share_of_candidates=summary["accepted_rows"] / max(len(pairs), 1),
                               accepted_overnight_connections=int(pairs.loc[pairs.main_analysis_eligible, "overnight_scheduled"].sum()),
                               accepted_cross_month_connections=int((pairs.main_analysis_eligible & pairs.upstream_flight_date.dt.month.ne(pairs.downstream_flight_date.dt.month)).sum()),
                               rq2_connections=summary["rq2_rows"], cohort_changed_membership=summary["changed_membership_rows"])
    cleaning["cohort_rules"] = summary["rules"]
    (data_dir / "cleaning_summary.json").write_text(json.dumps(cleaning, ensure_ascii=False, indent=2), encoding="utf-8")
    sources = [ROOT / "processed/flights_clean.csv.gz", ROOT / "processed/connection_candidates.csv.gz", source_summary]
    products = sorted(data_dir.iterdir())
    manifest = {"data_id": DATA_ID, "source_data_id": "D0001",
                "inputs": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in sources],
                "outputs": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in products],
                "code_sha256": file_sha256(Path(__file__))}
    (output_dir / "cohort_data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_dir / 'data_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary
