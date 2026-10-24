"""Build P7 prose and its evidence index from completed lightweight archives only.

No training, raw-data, pickle or prediction-file access occurs here. Invoke only
after P6: python scripts/build_report.py --config configs/report.json
The JSON keys are mainline_run_ids (G0/G1/G2/G3/G4A/G4B dictionary),
variant_run_id, noaa_run_id, algorithm_run_ids (the four DT and two RF runs),
development_rq1_run_id, feature_selection_run_id, model_selection_run_id,
final_prediction_run_id, final_rq1_run_id, optional reproduction_run_id, and
optional output_dir (reports).
output_dir must not exist. Markdown typography metadata is for the downstream
PDF renderer; this script does not certify PDF pages, fonts, slides or video.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["G0", "G1", "G2", "G3", "G4A", "G4B"]
BASELINES = ["training_mean", "training_median", "carry_forward"]
MAX_BYTES = 20 * 1024 * 1024
METRIC_COLUMNS = ["model", "fold", "group", "value", "n", "mae", "rmse", "unit"]
COMPARISON_COLUMNS = ["scope", "metric", "n", "difference_candidate_minus_reference", "ci95_low", "ci95_high"]


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, digits=3):
    return "not estimable" if pd.isna(value) else f"{float(value):,.{digits}f}"


def flag(series):
    return series.astype(str).str.lower().isin(["true", "1"])


class Archive:
    """The deliberately small allowlist for report source access."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.receipts, self.runs, self.evidence, self.figures = {}, {}, [], []
        self.index = self.table_path("experiments/index.csv", ["run_id", "status", "parent_run_id", "artifact_dir", "change", "decision"])
        if self.index.run_id.duplicated().any():
            raise ValueError("Duplicate run IDs in experiment index")
        self.log_path = self.path("research/研究记录.md")
        self.log = self.text_path(self.log_path).splitlines()
        self.text_path("research/literature_review.md")

    def path(self, name):
        path = Path(name)
        path = (path if path.is_absolute() else self.root / path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Source outside project: {path}")
        rel = path.relative_to(self.root)
        if any(part.lower() in {"raw", "processed", "predictions", "models", "residuals"} for part in rel.parts):
            raise ValueError(f"Report forbids data/model source: {rel}")
        if path.suffix.lower() not in {".json", ".csv", ".md", ".png", ".pdf"}:
            raise ValueError(f"Source extension not permitted: {rel}")
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError(f"Missing or oversized report source: {rel}")
        return path

    def record(self, path, expected=None):
        path = self.path(path)
        item = {"path": path.relative_to(self.root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
        if expected is not None and (item["sha256"] != expected.get("sha256") or item["bytes"] != expected.get("bytes")):
            raise ValueError(f"Source receipt mismatch: {item['path']}")
        self.receipts[item["path"]] = item
        return path

    def text_path(self, path):
        return self.record(path).read_text(encoding="utf-8-sig")

    def json_path(self, path):
        return json.loads(self.text_path(path))

    def table_path(self, path, columns):
        table = pd.read_csv(self.record(path), float_precision="round_trip", keep_default_na=True)
        absent = set(columns) - set(table)
        if absent or table.empty:
            raise ValueError(f"Empty table or missing columns {sorted(absent)}: {path}")
        return table

    def run(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"R\d{4,}", run_id):
            raise ValueError(f"Invalid run ID: {run_id}")
        if run_id in self.runs:
            return self.runs[run_id]
        index = self.index.loc[self.index.run_id.eq(run_id)]
        if len(index) != 1 or index.iloc[0].status != "completed":
            raise ValueError(f"Run is not completed and valid in experiment index: {run_id}")
        directory = self.root / "experiments/runs" / run_id
        if (self.root / str(index.iloc[0].artifact_dir)).resolve() != directory.resolve():
            raise ValueError(f"Index artifact_dir differs for {run_id}")
        manifest = self.json_path(directory / "manifest.json")
        if manifest.get("status") != "completed" or manifest.get("run_id") != run_id:
            raise ValueError(f"Completed manifest required for {run_id}")
        receipts = {}
        for item in manifest.get("outputs", []):
            if not isinstance(item, dict) or not all(key in item for key in ["path", "bytes", "sha256"]):
                raise ValueError(f"Malformed output receipt in {run_id}")
            path = Path(item["path"])
            path = (path if path.is_absolute() else self.root / path).resolve()
            if not path.is_relative_to(directory.resolve()) or path in receipts:
                raise ValueError(f"Duplicate or escaping output receipt in {run_id}")
            receipts[path] = item
        self.runs[run_id] = {"path": directory, "manifest": manifest, "receipts": receipts}
        config = self.json(run_id, "config.json")
        if config != manifest.get("config"):
            raise ValueError(f"Archived config and manifest config differ: {run_id}")
        parent = str(index.iloc[0].parent_run_id) if pd.notna(index.iloc[0].parent_run_id) else ""
        if parent != config.get("parent_run_id", ""):
            raise ValueError(f"Index/config parent differs: {run_id}")
        self.runs[run_id]["config"] = config
        if not self.log_entries(run_id):
            raise ValueError(f"Research journal has no run heading: {run_id}")
        return self.runs[run_id]

    def source(self, run_id, filename):
        run = self.run(run_id)
        path = (run["path"] / filename).resolve()
        if path not in run["receipts"]:
            raise ValueError(f"Missing archived output receipt: {run_id}/{filename}")
        return self.record(path, run["receipts"][path])

    def json(self, run_id, filename):
        return json.loads(self.source(run_id, filename).read_text(encoding="utf-8-sig"))

    def table(self, run_id, filename, columns):
        return self.table_path(self.source(run_id, filename), columns)

    def log_entries(self, run_id):
        return [str(i) for i, line in enumerate(self.log, 1)
                if line.startswith("### ") and re.search(rf"\b{re.escape(run_id)}\b", line)]

    def cite(self, run_id, filename, selector, claim, kind="conclusion"):
        path = self.source(run_id, filename)
        evidence_id = f"E{len(self.evidence) + 1:04d}"
        self.evidence.append({"evidence_id": evidence_id, "kind": kind, "claim": claim,
                              "run_id": run_id, "source_file": path.relative_to(self.root).as_posix(),
                              "selector": json.dumps(selector, ensure_ascii=False, sort_keys=True),
                              "source_sha256": digest(path), "research_log": "research/研究记录.md",
                              "research_log_heading_lines": ";".join(self.log_entries(run_id))})
        return f"[{evidence_id}]"

    def figure(self, run_id, filename, csv_file, selector, caption, code, extra_sources=None):
        path = self.source(run_id, filename)
        ref = self.cite(run_id, csv_file, selector, caption, "figure")
        extra_sources = extra_sources or []
        for extra_file, extra_selector in extra_sources:
            ref += " " + self.cite(run_id, extra_file, extra_selector, caption, "figure")
        self.cite(run_id, filename, {"whole_file": True, "plotting_code": code,
                                    "source_files": [csv_file, *[item[0] for item in extra_sources]]}, caption, "figure_image")
        target = f"figures/{run_id}_{path.name}"
        self.figures.append((path, target))
        return f"![{caption}]({target})\n\n{caption} {ref}"


def selected_row(table, **filters):
    rows = table
    for key, value in filters.items():
        rows = rows.loc[rows[key].astype(str).eq(str(value))]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one result row for {filters}; found {len(rows)}")
    return rows.iloc[0]


def model_archive(archive, run_id, development=True):
    run = archive.run(run_id)
    model = archive.json(run_id, "model_manifest.json")
    if model.get("development_only") is not development:
        raise ValueError(f"Wrong development/final scope: {run_id}")
    if run["config"].get("mode") != ("development" if development else "final"):
        raise ValueError(f"Archived config has the wrong mode: {run_id}")
    for key in ["model_name", "model_type", "feature_group", "features", "categorical_features", "numeric_features"]:
        if key in run["config"] and run["config"][key] != model.get(key):
            raise ValueError(f"Archived typed model configuration differs: {run_id}/{key}")
    table = archive.table(run_id, "metrics.csv", METRIC_COLUMNS)
    folds = ["2026-05", "2026-06", "2026-07"] if development else ["2026-08"]
    if not model.get("evaluation_ids_sha256") or not model.get("split_sha256"):
        raise ValueError(f"Missing cohort/split identity: {run_id}")
    if [item.get("fold") for item in model.get("folds", [])] != folds:
        raise ValueError(f"Incorrect fixed forward folds: {run_id}")
    for fold in model["folds"]:
        start = pd.Timestamp(fold["validation_start"])
        stop = pd.Timestamp(fold["validation_end_exclusive"])
        if start != pd.Timestamp(fold["fold"] + "-01", tz="UTC") or stop != start + pd.offsets.MonthBegin(1):
            raise ValueError(f"Incorrect UTC validation boundary: {run_id}")
        if pd.Timestamp(fold["max_train_label_time"]) >= start or pd.Timestamp(fold["max_train_prediction_time"]) >= start:
            raise ValueError(f"Training boundary/label maturity violated: {run_id}")
    include_baselines = model.get("include_baselines", run["config"].get("include_baselines", True))
    for name in [model["model_name"], *(BASELINES if include_baselines or not development else [])]:
        pooled = selected_row(table, model=name, fold="overall", group="all", value="all")
        monthly = [selected_row(table, model=name, fold=fold, group="all", value="all") for fold in folds]
        if sum(int(row.n) for row in monthly) != int(pooled.n) or int(pooled.n) <= 0:
            raise ValueError(f"Metric counts do not reconcile: {run_id}/{name}")
        if not all(math.isfinite(float(row.mae)) and math.isfinite(float(row.rmse)) and row.mae >= 0 and row.rmse >= 0
                   and row.unit == "minutes" for row in [pooled, *monthly]):
            raise ValueError(f"Invalid error metric: {run_id}/{name}")
        mae = sum(row.n * row.mae for row in monthly) / pooled.n
        rmse = math.sqrt(sum(row.n * row.rmse ** 2 for row in monthly) / pooled.n)
        if not math.isclose(mae, pooled.mae, abs_tol=1e-8) or not math.isclose(rmse, pooled.rmse, abs_tol=1e-8):
            raise ValueError(f"Pooled metrics do not reconcile: {run_id}/{name}")
    return {**run, "model": model, "metrics": table, "id": run_id}


def same_cohort(left, right):
    for key in ["evaluation_ids_sha256", "split_sha256"]:
        if left["model"][key] != right["model"][key]:
            raise ValueError(f"Compared {key} differs: {left['id']}/{right['id']}")
    targets = [item.get("validation_target_sha256") for item in left["model"]["folds"]]
    if not all(targets) or targets != [item.get("validation_target_sha256") for item in right["model"]["folds"]]:
        raise ValueError("Compared target identities differ")


def comparison_text(archive, reference, candidate, folder="comparison"):
    receipt = archive.json(candidate["id"], f"{folder}/comparison.json")
    if receipt.get("reference_run") != reference["id"] or receipt.get("candidate_run") != candidate["id"]:
        raise ValueError("Paired comparison belongs to other run IDs")
    scores = archive.table(candidate["id"], f"{folder}/comparison_metrics.csv", COMPARISON_COLUMNS)
    row = selected_row(scores, scope="pooled", metric="mae")
    if folder == "comparison":
        same_cohort(reference, candidate)
        if receipt.get("sample_scope") != "full_evaluation" or int(row.n) != int(candidate["model"]["evaluation_rows"]):
            raise ValueError("Main paired comparison does not cover the complete evaluation population")
        r = selected_row(reference["metrics"], model=reference["model"]["model_name"], fold="overall", group="all", value="all")
        c = selected_row(candidate["metrics"], model=candidate["model"]["model_name"], fold="overall", group="all", value="all")
        if not math.isclose(float(row.difference_candidate_minus_reference), c.mae - r.mae, abs_tol=1e-8):
            raise ValueError("Paired MAE difference conflicts with archived metrics")
    ref = archive.cite(candidate["id"], f"{folder}/comparison_metrics.csv", {"scope": "pooled", "metric": "mae"}, "Paired MAE change and date-block interval")
    delta = float(row.difference_candidate_minus_reference)
    direction = "improved" if delta < 0 else "worsened" if delta > 0 else "was unchanged"
    magnitude = f" by {number(abs(delta))} min ({number(abs(delta) * 60, 2)} seconds)" if delta else ""
    return (f"Relative to {reference['id']}, {candidate['id']} MAE {direction}{magnitude}. "
            f"The signed candidate-minus-reference change was {number(delta)} min "
            f"(95% date-block interval {number(row.ci95_low)} to {number(row.ci95_high)}; n={int(row.n):,}). "
            f"Interval precision does not establish practical importance. {ref}")


def single_flight_prose(archive, run_id):
    """One scoped, auditable EDA digest shared by both report languages."""
    summary_file = "tables/eda_complete_summary.json"
    summary = archive.json(run_id, summary_file)
    if summary.get("mode") != "development":
        raise ValueError("Single-flight report EDA must come from development mode")
    counts = summary["single_flight_summary"]
    operations_file = "tables/single_flight_operations.csv"
    operations = archive.table(run_id, operations_file, ["dimension", "group_value", "flights", "completed", "cancelled", "diverted", "status_unknown"])
    monthly_ops = operations.loc[operations.dimension.eq("month")]
    months = ["2026-04", "2026-05", "2026-06", "2026-07"]
    if sorted(monthly_ops.group_value.astype(str)) != months:
        raise ValueError("Single-flight EDA must contain exactly April–July local dates")
    for key, column in [("scoped_flight_rows", "flights"), ("normal_completed_rows", "completed"), ("cancelled_rows", "cancelled"),
                        ("diverted_rows", "diverted"), ("unknown_operation_status_rows", "status_unknown")]:
        if int(monthly_ops[column].sum()) != int(counts[key]):
            raise ValueError(f"Single-flight operations and EDA summary differ: {key}")
    ref = archive.cite(run_id, operations_file, {"dimension": "month", "aggregation": "sum(flights,completed,cancelled,diverted,status_unknown)"}, "April–July single-flight operation population")
    archive.cite(run_id, summary_file, {"keys": ["mode", "scope", "single_flight_summary"]}, "Single-flight EDA scope and reconciled counts")
    en = [f"Single-flight EDA uses April–July local FlightDate, independently of the UTC connection windows. It includes {int(counts['scoped_flight_rows']):,} flights: "
          f"{int(counts['normal_completed_rows']):,} normally completed, {int(counts['cancelled_rows']):,} cancelled, {int(counts['diverted_rows']):,} diverted "
          f"and {int(counts['unknown_operation_status_rows']):,} with unknown operation status. Delay quantiles below describe normally completed flights only; "
          f"cancellation and diversion are not zero-delay outcomes. {ref}",
          "Raw DepDelay and ArrDelay preserve negative early departures/arrivals; DepDelayMinutes and ArrDelayMinutes are the corresponding nonnegative measures. "
          "Their means can therefore differ even for the same flights. Each table row reports its own observed-outcome n; missing outcomes are excluded from its quantiles, not filled with zero."]
    zh = ["## 单航段EDA与机型参考覆盖",
          f"单航段EDA按当地FlightDate取4–7月，与衔接的UTC预测窗分别定义。共{int(counts['scoped_flight_rows']):,}条航班："
          f"正常完成{int(counts['normal_completed_rows']):,}、取消{int(counts['cancelled_rows']):,}、改道{int(counts['diverted_rows']):,}、状态未知{int(counts['unknown_operation_status_rows']):,}。"
          f"下表延误分布只覆盖正常完成航班，取消和改道不能当作零延误。{ref}",
          "原始DepDelay/ArrDelay保留提前出发/到达的负值，DepDelayMinutes/ArrDelayMinutes为对应非负分钟，因此同批航班的两种均值并不相同。每行n是该结果字段实际非缺失数；缺失不填0，q50是中位数，q90/q99刻画右尾。"]
    stats_file = "tables/single_flight_delay_stats.csv"
    stats = archive.table(run_id, stats_file, ["dimension", "group_value", "outcome", "normal_completed_flights", "n", "missing", "negative", "zero", "mean", "q50", "q90", "q99"])
    table = ["| Month | Measure / evidence | n | Mean | q50 | q90 | q99 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    labels = {"DepDelay": "Departure raw", "ArrDelay": "Arrival raw", "DepDelayMinutes": "Departure nonnegative", "ArrDelayMinutes": "Arrival nonnegative"}
    for month in months:
        operation = selected_row(monthly_ops, group_value=month)
        for outcome, label in labels.items():
            selector = {"dimension": "month", "group_value": month, "outcome": outcome}
            row = selected_row(stats, **selector)
            if int(row.normal_completed_flights) != int(operation.completed) or int(row.n + row.missing) != int(operation.completed):
                raise ValueError("Single-flight observed/missing outcome counts do not reconcile")
            ref = archive.cite(run_id, stats_file, selector, "Monthly signed/nonnegative single-flight delay quantiles; minutes")
            table.append(f"| {month} | {label} {ref} | {int(row.n):,} | {number(row['mean'], 2)} | {number(row.q50, 2)} | {number(row.q90, 2)} | {number(row.q99, 2)} |")
    en.append("Monthly single-flight delay distributions (all summary values in minutes).\n\n" + "\n".join(table))
    zh.append("单航段月际分布，所有统计量均为分钟；raw表示保留负值，nonnegative表示非负分钟。\n\n" + "\n".join(table))
    coverage_file = "tables/airline_aircraft_coverage.csv"
    coverage = archive.table(run_id, coverage_file, ["month", "Reporting_Airline", "inventory_year", "flight_rows", "mapped_model_rows", "unknown_model_rows"])
    if (set(coverage.inventory_year.astype(int)) != {2024, 2025} or set(coverage.month.astype(str)) != set(months)
            or coverage.duplicated(["month", "Reporting_Airline", "inventory_year"]).any()):
        raise ValueError("Aircraft coverage requires unique airline/month keys for both 2024 and 2025 references")
    for year in [2024, 2025]:
        rows = coverage.loc[coverage.inventory_year.eq(year)]
        total, matched, unknown = int(rows.flight_rows.sum()), int(rows.mapped_model_rows.sum()), int(rows.unknown_model_rows.sum())
        if total != int(counts["scoped_flight_rows"]) or matched + unknown != total:
            raise ValueError("Aircraft coverage denominators differ from scoped flights")
        ref = archive.cite(run_id, coverage_file, {"inventory_year": year, "aggregation": "sum(mapped_model_rows)/sum(flight_rows); sum(unknown_model_rows)"}, "B43 reference-year flight-weighted mapping coverage")
        en.append(f"The {year} B43 reference mapped an aircraft model for {matched:,}/{total:,} flights ({number(100 * matched / total, 2)}%); "
                  f"{unknown:,} remained unmatched. {ref}")
        zh.append(f"{year}年B43参考表映射到原始机型的航班为{matched:,}/{total:,}（{number(100 * matched / total, 2)}%），未匹配{unknown:,}条。{ref}")
    en.append("The 2024 inventory remains the primary reference; 2025 is a coverage sensitivity comparison and does not replace it. "
              "Mapping to a reference model is not proof of contemporaneous fleet membership, and unmatched tails/models remain explicit unknowns rather than being dropped.")
    zh.append("主实验仍采用2024参考表，2025只作覆盖对照，覆盖较高也不自动替换主版本；这两份快照不能证明2026各预测时点的真实机队，未知机型保留为显式类别。")
    activity_file = "tables/airport_activity.csv"
    activity = archive.table(run_id, activity_file, ["airport_id", "origin_flights", "destination_flights", "total_flight_movements", "accepted_connection_count", "main_analysis_eligible_connection_count", "rq2_eligible_connection_count"])
    if activity.airport_id.duplicated().any() or not activity.total_flight_movements.eq(activity.origin_flights + activity.destination_flights).all():
        raise ValueError("Airport movement table has duplicate keys or inconsistent movement counts")
    ref = archive.cite(run_id, activity_file, {"rows": "all", "aggregation": "sum(total_flight_movements,accepted_connection_count,main_analysis_eligible_connection_count,rq2_eligible_connection_count)"}, "Airport movements versus distinct QA-cohort connection populations")
    movements, connections = int(activity.total_flight_movements.sum()), int(activity.accepted_connection_count.sum())
    main_n, rq2_n = int(activity.main_analysis_eligible_connection_count.sum()), int(activity.rq2_eligible_connection_count.sum())
    en.append(f"Airport activity totals {movements:,} arrival-plus-departure movements and {connections:,} accepted same-aircraft connections; "
              f"the QA-cohort eligibility flags count {main_n:,} main-analysis and {rq2_n:,} RQ2 connections. "
              f"A flight contributes an origin movement and a destination movement, whereas a connection is a validated pair at its transfer airport. "
              f"Connection activity here is grouped by QA cohort month, so it is not substituted for a UTC prediction-window model denominator. {ref}")
    zh.append(f"机场活动表合计{movements:,}次到达加出发movement，以及{connections:,}条有效同机衔接；QA-cohort条件下主分析标记{main_n:,}条、RQ2标记{rq2_n:,}条。"
              f"一条航班在起点/终点各贡献一次movement，同机衔接则是转接机场处核验后的航段对。该表衔接按QA队列月份计数，不能替代UTC预测窗的模型样本量。{ref}")
    return en, zh


def validate_inputs(archive, config):
    main_ids = config["mainline_run_ids"]
    if set(main_ids) != set(STAGES):
        raise ValueError("Require exactly the six registered mainline stages")
    main = [model_archive(archive, main_ids[group]) for group in STAGES]
    if len({run["id"] for run in main}) != 6:
        raise ValueError("Mainline stages must be distinct runs")
    for group, run in zip(STAGES, main):
        actual = run["model"]["feature_group"].upper().replace("G2_FAMILY", "G2")
        if actual != group or run["model"].get("model_type") != "linear" or run["config"].get("noaa_backfill", False):
            raise ValueError("Mainline must hold OLS fixed and use family/IEM")
        same_cohort(main[0], run)
        if any(run["model"].get(key) != main[0]["model"].get(key) for key in ["model_params", "effective_preprocessing"]):
            raise ValueError("Mainline OLS/preprocessing settings differ")
        if any(run["config"].get(key) != main[0]["config"].get(key)
               for key in ["data_id", "connection_path", "data_manifest_path"]):
            raise ValueError("Mainline scientific input versions differ")
        if "aircraft_variant" in run["model"]["features"]:
            raise ValueError("Variant cannot enter the family mainline")
    for left, right in zip(main, main[1:]):
        if right["config"].get("parent_run_id") != left["id"]:
            raise ValueError("Mainline parent relationship differs")
    variant = model_archive(archive, config["variant_run_id"])
    noaa = model_archive(archive, config["noaa_run_id"])
    if variant["model"]["feature_group"].upper() != "G2_VARIANT" or variant["config"].get("parent_run_id") != main[2]["id"]:
        raise ValueError("Variant must independently replace mainline family")
    if not noaa["config"].get("noaa_backfill") or noaa["config"].get("parent_run_id") != main[-1]["id"]:
        raise ValueError("NOAA must independently branch from IEM G4B")
    for branch in [variant, noaa]:
        same_cohort(main[0], branch)
        if any(branch["model"].get(key) != main[0]["model"].get(key) for key in ["model_type", "model_params", "effective_preprocessing"]):
            raise ValueError("Supplementary branch changed the fixed estimator")
    feature = archive.json(config["feature_selection_run_id"], "selection/feature_selection.json")
    choice = archive.json(config["model_selection_run_id"], "selection/model_selection_decision.json")
    for item in [feature, choice]:
        if item.get("selection_sha256") != canonical({k: v for k, v in item.items() if k != "selection_sha256"}):
            raise ValueError("Selection canonical receipt mismatch")
    if feature.get("selection_type") != "features" or choice.get("selection_type") != "model":
        raise ValueError("Incorrect completed selection type")
    selected = feature["selected_run_id"]
    if selected not in main_ids.values() or choice.get("feature_source_run_id") != selected:
        raise ValueError("Model selection does not use the selected mainline feature source")
    algorithms = [model_archive(archive, value) for value in config["algorithm_run_ids"]]
    if len({run["id"] for run in algorithms}) != 6:
        raise ValueError("Require six distinct DT4/RF2 algorithm candidates")
    feature_run = next(run for run in main if run["id"] == selected)
    actual_grid = []
    for run in algorithms:
        same_cohort(feature_run, run)
        params = run["model"]["model_params"]
        actual_grid.append((run["model"]["model_type"], params.get("max_depth"), params.get("min_samples_leaf")))
        if run["model"]["features"] != feature_run["model"]["features"] or params.get("random_state") != 42:
            raise ValueError("Algorithm candidate changed fixed features or seed")
        if run["model"].get("effective_preprocessing") != feature_run["model"].get("effective_preprocessing"):
            raise ValueError("Algorithm candidate changed fixed preprocessing")
        if run["config"].get("feature_source_run_id") != selected:
            raise ValueError("Algorithm feature-source lineage differs")
        if run["model"]["model_type"] == "random_forest" and (params.get("n_estimators") != 200 or params.get("max_features") != .5):
            raise ValueError("Forest differs from registered grid")
    expected = [("decision_tree", depth, leaf) for depth in [8, 16] for leaf in [50, 200]]
    expected += [("random_forest", 16, leaf) for leaf in [50, 200]]
    if sorted(actual_grid) != sorted(expected) or choice["selected_run_id"] not in [selected, *config["algorithm_run_ids"]]:
        raise ValueError("Model selection/candidate grid differs")
    # Selection output CSVs tie the passed runs to the sealed decision; no large
    # prediction/feature hashes are recomputed in a presentation-only step.
    for selection_id, expected_ids in [(config["feature_selection_run_id"], list(main_ids.values())),
                                       (config["model_selection_run_id"], [selected, *config["algorithm_run_ids"]])]:
        candidates = archive.table(selection_id, "selection/candidates.csv", ["run_id", "model", "pooled_mae", "pooled_rmse"])
        if sorted(candidates.run_id) != sorted(expected_ids):
            raise ValueError("Selection candidate table omits or substitutes requested runs")
        for _, row in candidates.iterrows():
            candidate = model_archive(archive, row.run_id)
            metric = selected_row(candidate["metrics"], model=candidate["model"]["model_name"], fold="overall", group="all", value="all")
            if (row.model != metric.model or not math.isclose(row.pooled_mae, metric.mae, abs_tol=1e-8)
                    or not math.isclose(row.pooled_rmse, metric.rmse, abs_tol=1e-8)):
                raise ValueError("Selection candidate metrics differ from archived scores")
        decision = feature if selection_id == config["feature_selection_run_id"] else choice
        pins = {str(item.get("path")): item for item in decision.get("candidate_files", [])}
        for run_id in expected_ids:
            for filename in ["manifest.json", "config.json", "model_manifest.json", "metrics.csv"]:
                source = archive.run(run_id)["path"] / filename
                relative = source.relative_to(archive.root).as_posix()
                pin = pins.get(relative) or pins.get(str(source))
                if pin is None:
                    raise ValueError("Selection lacks a lightweight candidate receipt")
                archive.record(source, pin)
    final_run = model_archive(archive, config["final_prediction_run_id"], development=False)
    frozen = archive.json(final_run["id"], "model_selection.json")
    final = archive.json(final_run["id"], "final_result.json")
    selected_model = model_archive(archive, choice["selected_run_id"])
    if (frozen.get("status") != "frozen" or frozen.get("payload_sha256") != canonical({k: v for k, v in frozen.items() if k != "payload_sha256"})
            or frozen.get("selection", {}).get("selection_sha256") != choice["selection_sha256"]
            or frozen["selection"].get("selected_run_id") != choice["selected_run_id"]
            or frozen.get("primary_model_name") != choice["primary_model_name"]):
        raise ValueError("Final freeze differs from the specified completed selection")
    if (final_run["model"]["model_name"] != selected_model["model"]["model_name"]
            or final_run["model"]["features"] != selected_model["model"]["features"]
            or final.get("primary_model_name") != choice["primary_model_name"]):
        raise ValueError("Final learned/primary predictors differ from the frozen choice")
    if final.get("freeze_payload_sha256") != frozen["payload_sha256"] or final.get("freeze_manifest_sha256") != digest(archive.source(final_run["id"], "model_selection.json")):
        raise ValueError("Final result does not pin the archived freeze")
    verification = archive.json(final_run["id"], "final_verification.json")
    if (not verification.get("verified") or not verification.get("metrics_recomputed")
            or verification.get("refitted") is not False or verification.get("raw_test_data_read") is not False):
        raise ValueError("Completed saved-prediction verification is required")
    ledger_path = archive.path("experiments/final_access/2026-08.json")
    ledger = archive.json_path(ledger_path)
    if ledger.get("status") != "completed" or ledger.get("run_id") != final_run["id"] or ledger.get("final_result_sha256") != digest(archive.source(final_run["id"], "final_result.json")):
        raise ValueError("Final result differs from the single-access ledger")
    development_id, final_id = config["development_rq1_run_id"], config["final_rq1_run_id"]
    development = archive.json(development_id, "rq1_complete_summary.json")
    final_rq1 = archive.json(final_id, "rq1_complete_summary.json")
    reference = archive.source(development_id, "rq1_standardized/frozen_reference.json")
    final_reference = archive.source(final_id, "rq1_standardized/frozen_reference.json")
    if (development.get("development") is not True or final_rq1.get("development") is not False
            or digest(reference) != digest(final_reference)
            or digest(reference) != final_rq1.get("frozen_reference_sha256")
            or archive.run(final_id)["config"].get("prediction_final_run_id") != final_run["id"]):
        raise ValueError("Final RQ1 must reuse the development reference after the specified final prediction")
    return main, variant, noaa, algorithms, feature, choice, final_run


def metric_sentence(archive, run, model=None, fold="overall", group="all", value="all"):
    model = model or run["model"]["model_name"]
    selector = dict(model=model, fold=fold, group=group, value=value)
    row = selected_row(run["metrics"], **selector)
    ref = archive.cite(run["id"], "metrics.csv", selector, f"{model}: MAE/RMSE and sample support")
    return f"{model} had MAE {number(row.mae)} min and RMSE {number(row.rmse)} min (n={int(row.n):,}). {ref}"


def noaa_prose(archive, run_id):
    """Report the effective treatment and observable overlap, not its label."""
    audit_file = "feature_audit.json"
    audit = archive.json(run_id, audit_file)
    counts = audit.get("noaa_fill_counts")
    model_columns = audit.get("weather_audit", {}).get("metar_model_columns")
    if not isinstance(counts, dict) or not counts or not isinstance(model_columns, list):
        raise ValueError("NOAA report requires archived fill counts and actual METAR predictor whitelist")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or int(value) != value for value in counts.values()):
        raise ValueError("NOAA archived fill counts must be nonnegative integers")
    primary = [name for name in counts if name in model_columns]
    all_fills = sum(int(value) for value in counts.values())
    predictor_fills = sum(int(counts[name]) for name in primary)
    ref = archive.cite(run_id, audit_file, {"keys": ["noaa_fill_counts", "weather_audit.metar_model_columns"]}, "Actual value-level NOAA backfills and predictor-eligible mappings")
    details = "; ".join(f"{name}={int(value):,}" for name, value in counts.items())
    en = [f"The NOAA audit tracks {len(counts)} mapped quantities, of which {len(primary)} are in the actual METAR predictor whitelist: "
          f"{', '.join(primary) or 'none'}. It records {all_fills:,} filled values in total and {predictor_fills:,} fills in predictor columns; "
          f"these are value-level counts rather than distinct-flight counts. Per-field counts were {details}. {ref}"]
    zh = ["## NOAA补缺实际改变了什么",
          f"归档记录{len(counts)}个映射量，但实际METAR预测变量白名单只包含其中{len(primary)}个：{', '.join(primary) or '无'}。"
          f"全部映射量合计填{all_fills:,}个值，入模变量填{predictor_fills:,}个值；这是逐值计数，不能当作不同航班数。各字段为{details}。{ref}"]
    if predictor_fills == 0:
        en.append("With zero effective predictor backfills, this branch provides no effective NOAA input contrast. "
                  "Unchanged errors reflect unchanged prediction inputs rather than evidence that otherwise usable NOAA information has no predictive value. "
                  "The experiment is retained as a zero-treatment result under the registered conservative rules. " + ref)
        zh.append("入模变量零有效补值，意味着本分支没有形成NOAA输入对照；误差不变反映预测输入未变，不能推断其他情况下可用的NOAA信息没有预测价值。"
                  "该轮作为既定保守规则下的零实际处理结果保留。" + ref)
    overlap_file = "tables/noaa_overlap_summary.csv"
    overlap = archive.table(run_id, overlap_file, ["feature", "overlap_count", "mean_signed_difference", "mean_absolute_difference", "metar_unit", "noaa_native_unit", "conversion", "unit_consistency"])
    pair_counts = pd.to_numeric(overlap.overlap_count, errors="raise")
    if (overlap.feature.duplicated().any() or set(overlap.feature) != set(counts)
            or any(not math.isfinite(value) or value < 0 or int(value) != value for value in pair_counts)):
        raise ValueError("NOAA overlap diagnostic has duplicate, omitted or invalid mapped fields")
    pairs = int(overlap.overlap_count.sum())
    ref = archive.cite(run_id, overlap_file, {"rows": "all", "aggregation": "sum(overlap_count); per-feature overlap_count and unit_consistency"}, "Observed NOAA/IEM overlap support and limits of unit comparison")
    if pairs == 0:
        en.append(f"Across {len(overlap)} mapped fields, the exact-airport/time diagnostic contains zero accepted numeric value pairs. "
                  "The no_numeric_overlap/no_exact_overlap labels indicate missing comparison support; they do not establish unit agreement. "
                  "Conversions are documented operations, but agreement after conversion was not empirically verified by these archived pairs. " + ref)
        zh.append(f"{len(overlap)}个映射字段在机场/时刻精确匹配诊断中均没有合格数值对。no_numeric_overlap/no_exact_overlap只表示缺乏对照支持，"
                  "不表示已验证单位一致。单位换算是代码中明确的操作，但此次归档没有数值对来实证核验换算后的一致程度。" + ref)
    else:
        en.append(f"The overlap table contains {pairs:,} numeric value pairs across mapped fields, counted per field rather than as unique observations. "
                  "Its converted differences are descriptive checks and do not certify global QC or historical delivery availability. " + ref)
        zh.append(f"重叠表各字段合计有{pairs:,}对数值，按字段计数而非唯一观测条数；换算后差异是描述性核对，不能认证全QC或历史送达可用性。" + ref)
    return en, zh


def rq1_reference_prose(archive, run_id):
    """Display the actual development A/C/w and every attempted fallback."""
    filename = "rq1_standardized/frozen_reference.json"
    frozen = archive.json(run_id, filename)
    if frozen.get("schema_version") != 1 or frozen.get("frozen_on") != "development_only":
        raise ValueError("RQ1 reference must be frozen on development data")
    complete = archive.json(run_id, "rq1_complete_summary.json")
    if complete.get("frozen_reference_sha256") != digest(archive.source(run_id, filename)):
        raise ValueError("RQ1 complete summary does not pin its frozen reference")
    protocol = frozen["protocol"]
    attempts = frozen["selection_attempts"]
    attempts_file = "rq1_standardized/selection_attempts.csv"
    saved = archive.table(run_id, attempts_file, ["attempt", "airport_limit", "airports", "passed", "common_master_cells", "airport_ids", "threshold_support"])
    limits = [None, *protocol["fallback_airport_limits"]]
    if not attempts or len(saved) != len(attempts) or len(attempts) > len(limits):
        raise ValueError("Incomplete RQ1 fallback-attempt evidence")
    minimum = float(protocol["minimum_retained_share"])
    protocol_ref = archive.cite(run_id, filename, {"keys": ["protocol"]}, "Registered airport, cell, date and retention eligibility gates")
    en = [f"The development reference required at least {int(protocol['airport_min_main_rows'])} main-threshold connections and "
          f"{int(protocol['airport_min_prediction_days'])} prediction dates per eligible airport. Each common cell required at least "
          f"{int(protocol['cell_min_rows'])} connections and {int(protocol['cell_min_prediction_days'])} dates; every selected airport had to retain "
          f"at least {number(100 * minimum, 1)}% of its threshold-specific population. Selection used covariates/counts, not net-delay outcomes. {protocol_ref}"]
    zh = ["## 冻结共同支持与预定回退",
          f"开发参考要求机场主阈值样本至少{int(protocol['airport_min_main_rows'])}条、至少{int(protocol['airport_min_prediction_days'])}个预测日期；"
          f"每个共同格至少{int(protocol['cell_min_rows'])}条、{int(protocol['cell_min_prediction_days'])}个日期，且每个机场在每个阈值下保留率均至少{number(100 * minimum, 1)}%。"
          f"选择只使用协变量和计数，不使用净延误结果。{protocol_ref}"]
    width = len(protocol["turnaround_bins"]) - 1
    for i, attempt in enumerate(attempts):
        row = selected_row(saved, attempt=i + 1)
        limit = None if pd.isna(row.airport_limit) else int(row.airport_limit)
        if (attempt["limit"] != limits[i] or limit != attempt["limit"] or int(row.airports) != len(attempt["airport_ids"])
                or bool(flag(pd.Series([row.passed])).iloc[0]) != attempt["passed"]
                or json.loads(row.common_master_cells) != attempt["common_master_cells"]
                or json.loads(row.airport_ids) != attempt["airport_ids"]
                or json.loads(row.threshold_support) != attempt["threshold_support"]):
            raise ValueError("RQ1 fallback CSV and frozen reference disagree")
        passed = len(attempt["airport_ids"]) >= 2
        details = []
        for threshold, threshold_minimum in [("gt0", 0), ("ge5", 5), ("ge15", 15)]:
            support = attempt["threshold_support"][threshold]
            cells = [cell for cell in attempt["common_master_cells"]
                     if float(protocol["master_delay_bins"][cell // width]) >= threshold_minimum]
            shares = support["retained_shares"]
            if cells != support["cells"] or set(shares) != {str(airport) for airport in attempt["airport_ids"]}:
                raise ValueError("RQ1 fallback conditional cells/airport shares disagree")
            values = list(shares.values())
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError("Invalid RQ1 fallback retained share")
            threshold_passed = bool(cells) and all(value >= minimum for value in values)
            if threshold_passed != support["passed"]:
                raise ValueError("RQ1 fallback pass flag contradicts retained support")
            passed &= threshold_passed
            details.append(f"{threshold}={number(min(values) if values else float('nan'), 4)}")
        if bool(passed) != attempt["passed"] or attempt["passed"] and i != len(attempts) - 1:
            raise ValueError("RQ1 fallback did not stop at its first passing set")
        ref = archive.cite(run_id, attempts_file, {"attempt": i + 1}, "Every registered common-support selection attempt and threshold-specific retained support")
        label = "all eligible airports" if limit is None else f"top {limit} airports by main-threshold count"
        en.append(f"Attempt {i + 1} used {label}: {len(attempt['airport_ids'])} airports and {len(attempt['common_master_cells'])} common master cells; "
                  f"minimum retained shares were {', '.join(details)}. The all-threshold decision was {'pass' if attempt['passed'] else 'fail'}. {ref}")
        zh.append(f"第{i + 1}次尝试{'全部合格机场' if limit is None else f'按主阈值样本量取前{limit}个机场'}："
                  f"实际{len(attempt['airport_ids'])}机场、{len(attempt['common_master_cells'])}个共同主格；各阈值最低保留率为{', '.join(details)}，"
                  f"全阈值判定{'通过' if attempt['passed'] else '未通过'}。{ref}")
    success = attempts[-1]["passed"]
    if not success and len(attempts) != len(limits):
        raise ValueError("Unsupported RQ1 reference omitted registered fallback attempts")
    if bool(frozen["supported"]) != success:
        raise ValueError("RQ1 frozen supported flag differs from fallback decision")
    if success and (frozen["airport_ids"] != attempts[-1]["airport_ids"] or frozen["master_cells"] != attempts[-1]["common_master_cells"]):
        raise ValueError("RQ1 frozen A/C differ from the first passing attempt")
    if not success and (frozen["airport_ids"] or frozen["master_cells"] or frozen["references"]):
        raise ValueError("Unsupported RQ1 reference must not invent A/C/weights")
    ref = archive.cite(run_id, filename, {"keys": ["airport_ids", "master_cells", "references", "protocol"]}, "Frozen airport set A, common master cells C and threshold-conditional reference weights w")
    nodes_file = "rq1_standardized/network_nodes.csv"
    nodes = archive.table(run_id, nodes_file, ["airport_id", "airport_code"])
    labels = []
    for airport in frozen["airport_ids"]:
        row = selected_row(nodes, airport_id=airport)
        labels.append(f"{row.airport_code} ({airport})")
    codes_ref = archive.cite(run_id, nodes_file, {"airport_id_in": frozen["airport_ids"], "columns": ["airport_id", "airport_code"]}, "Verified display codes for frozen airport set")
    en.append(f"The frozen set A contains {len(labels)} airports: {', '.join(labels) or 'none'}. "
              f"C contains {len(frozen['master_cells'])} master cells. The procedure stops at the first passing attempt; later fallback limits are not searched after success. {ref} {codes_ref}")
    zh.append(f"冻结A含{len(labels)}个机场：{', '.join(labels) or '无'}；C含{len(frozen['master_cells'])}个主格。"
              f"首次通过即停止，不再尝试后面的更小机场集合。{ref} {codes_ref}")
    maps = {}
    for threshold, threshold_minimum in [("gt0", 0), ("ge5", 5), ("ge15", 15)]:
        reference = frozen["references"].get(threshold)
        if not success:
            if reference:
                raise ValueError("Unsupported reference must not publish weights")
            maps[threshold] = {}
            continue
        if reference is None:
            raise ValueError("Supported reference lacks a threshold weight set")
        cells, weights, counts = reference["cells"], reference["weights"], reference["development_counts"]
        expected_cells = [cell for cell in frozen["master_cells"] if float(protocol["master_delay_bins"][cell // width]) >= threshold_minimum]
        if cells != expected_cells or not cells or len(cells) != len(weights) or len(cells) != len(counts):
            raise ValueError("RQ1 frozen reference conditional weight dimensions differ")
        if (not all(math.isfinite(weight) and weight > 0 for weight in weights)
                or not all(math.isfinite(count) and count > 0 for count in counts)
                or not math.isclose(sum(weights), 1, abs_tol=1e-12)
                or any(not math.isclose(weight, count / sum(counts), abs_tol=1e-12) for weight, count in zip(weights, counts))):
            raise ValueError("RQ1 frozen weights are not normalized pooled development counts")
        maps[threshold] = dict(zip(cells, weights))
    table = ["| Cell | Upstream delay (min) | Scheduled turnaround (min) | w: >0 | w: >=5 | w: >=15 |",
             "| ---: | --- | --- | ---: | ---: | ---: |"]
    edge = lambda value: format(float(value), ".12g")
    for cell in frozen["master_cells"]:
        delay = protocol["master_delay_bins"][cell // width: cell // width + 2]
        turn = protocol["turnaround_bins"][cell % width: cell % width + 2]
        values = [number(maps[threshold][cell], 6) if cell in maps[threshold] else "NA" for threshold in ["gt0", "ge5", "ge15"]]
        table.append(f"| {cell} | [{edge(delay[0])}, {edge(delay[1])}) | [{edge(turn[0])}, {edge(turn[1])}) | {' | '.join(values)} |")
    en.append("Threshold-specific weights condition one pooled development reference on its eligible cells; each weight column sums to one before display rounding. "
              "NA denotes a cell outside that threshold's reference, not a zero or an airport-specific renormalization. The >0 population excludes zero even in the [0,5) cell. "
              "These weights and A/C are reused for monthly and final RQ1, while an airport can lose support in an individual month. " + ref + "\n\n" + "\n".join(table))
    zh.append("三个阈值在同一开发期主参考上条件化，表中每列权重在显示舍入前合计1。NA表示该格不属于对应阈值参考，并非补零，也不是按机场重算权重。"
              "即使格区间从0开始，>0人群也排除零延误。逐月和最终RQ1复用同一A/C/w，但单月机场仍可能失去支持。" + ref + "\n\n" + "\n".join(table))
    return en, zh


def rq1_prose(archive, run_id, label):
    rawfile = "tables/rq1_airport_raw.csv"
    raw = archive.table(run_id, rawfile, ["airport_id", "airport_code", "n", "mean_delta", "s_plus", "s_minus", "s_net"])
    n = raw.n.sum()
    if n <= 0:
        raise ValueError("RQ1 raw sample is empty")
    means = (raw.n * raw.mean_delta).sum() / n
    ref = archive.cite(run_id, rawfile, {"rows": "all", "aggregation": "sum(n*mean_delta)/sum(n); sum(n,s_plus,s_minus,s_net)"}, "Raw RQ1 sample and cumulative minutes")
    raw_ref = ref
    text = [f"For {label}, the raw main population comprised {int(n):,} nonovernight connections with upstream delay at least 15 min. "
            f"Its mean net change was {number(means)} min per connection; cumulative increases, decreases and net change were "
            f"{number(raw.s_plus.sum(), 0)}, {number(raw.s_minus.sum(), 0)} and {number(raw.s_net.sum(), 0)} sample-minutes. "
            f"These totals depend on sample size and are distinct from airport means. {ref}"]
    name = "rq1_standardized/rq1_airports.csv"
    airports = archive.table(run_id, name, ["airport_id", "threshold", "supported", "retained_n", "retained_share", "standardized_mean", "ci_low", "ci_high", "valid_replicates"])
    supported = airports.loc[airports.threshold.eq("ge15") & flag(airports.supported)]
    ref = archive.cite(run_id, name, {"threshold": "ge15", "supported": True}, "Supported standardized airport results and retained coverage")
    text.append(f"The frozen common-support analysis supported {len(supported):,} airports at the main threshold. "
                f"The retained shares ranged from {number(supported.retained_share.min())} to {number(supported.retained_share.max())}; "
                f"standardized mean net changes ranged from {number(supported.standardized_mean.min())} to {number(supported.standardized_mean.max())} min per connection. "
                f"Unsupported estimates remain missing and are not ranked. {ref}")
    interval_present = supported.ci_low.notna() & supported.ci_high.notna()
    crosses_zero = interval_present & supported.ci_low.le(0) & supported.ci_high.ge(0)
    text.append(f"Supported common cells retain {int(supported.retained_n.sum()):,} connections, or {number(100 * supported.retained_n.sum() / n, 2)}% of the all-airport raw main population. "
                f"There are {int(supported.standardized_mean.lt(0).sum())} negative and {int(supported.standardized_mean.gt(0).sum())} positive supported point estimates; "
                f"{int(interval_present.sum())} have reportable intervals and {int(crosses_zero.sum())} of those intervals include zero. "
                f"Within-selected-airport retention is distinct from coverage of the whole raw population. {ref} {raw_ref}")
    for _, row in supported.sort_values(["retained_n", "airport_id"], ascending=[False, True]).head(3).iterrows():
        ref = archive.cite(run_id, name, {"threshold": "ge15", "airport_id": int(row.airport_id)}, "High-support airport estimate and interval")
        text.append(f"AirportID {int(row.airport_id)} had standardized net change {number(row.standardized_mean)} min "
                    f"(95% interval {number(row.ci_low)} to {number(row.ci_high)}; retained n={int(row.retained_n):,}; "
                    f"{int(row.valid_replicates)} valid bootstrap draws). {ref}")
    stability_file = "rq1_standardized/rq1_threshold_stability.csv"
    stability = archive.table(run_id, stability_file, ["threshold", "common_airports", "frozen_airports", "direction_agreement", "spearman"])
    for threshold in ["gt0", "ge5", "ge15"]:
        row = selected_row(stability, threshold=threshold)
        ref = archive.cite(run_id, stability_file, {"threshold": threshold}, "Threshold direction and ranking stability")
        text.append(f"At {threshold}, {int(row.common_airports)}/{int(row.frozen_airports)} frozen airports could be compared with the main threshold; "
                    f"direction agreement was {number(row.direction_agreement)} and Spearman correlation was {number(row.spearman)}. {ref}")
    monthly_file = "rq1_standardized/rq1_airports_monthly.csv"
    monthly = archive.table(run_id, monthly_file, ["month_utc", "threshold", "supported", "standardized_mean", "retained_share"])
    for month, rows in monthly.loc[monthly.threshold.eq("ge15")].groupby("month_utc", sort=True):
        rows = rows.loc[flag(rows.supported)]
        ref = archive.cite(run_id, monthly_file, {"month_utc": month, "threshold": "ge15", "supported": True}, "Monthly support under identical frozen reference")
        text.append(f"In {month}, {len(rows)} airports retained support under the same frozen reference; "
                    f"their standardized net changes ranged from {number(rows.standardized_mean.min())} to {number(rows.standardized_mean.max())} min. {ref}")
    text.append("The set of supported airports can differ by month. These cross-airport ranges are not a fixed-airport time trend; "
                "a monthly comparison must identify the same supported airports, and unsupported months remain NA.")
    nodes = archive.table(run_id, "rq1_standardized/network_nodes.csv", ["airport_id", "valid_connection_count", "standardized_mean"])
    edges = archive.table(run_id, "rq1_standardized/network_edges.csv", ["origin_airport_id", "destination_airport_id", "flight_count"])
    r1 = archive.cite(run_id, "rq1_standardized/network_nodes.csv", {"rows": "all"}, "Network node sample support")
    r2 = archive.cite(run_id, "rq1_standardized/network_edges.csv", {"rows": "all"}, "Network directed route traffic")
    text.append(f"The network tables contain {len(nodes)} airport nodes, {len(edges)} directed routes and {int(edges.flight_count.sum()):,} BTS flights. "
                f"Node size represents {int(nodes.valid_connection_count.sum()):,} accepted connections, node colour represents supported standardized mean change, "
                f"and edge width represents flight volume. Gray nodes are unsupported; route counts are not propagated-delay counts. {r1} {r2}")
    for filename, table, caption in [
        ("rq1_standardized_airports.png", name, "Frozen common-support airport estimates and date-block uncertainty"),
        ("rq1_threshold_stability.png", stability_file, "Direction and ranking sensitivity to upstream-delay thresholds"),
        ("rq1_network_geographic.png", "rq1_standardized/network_nodes.csv", "Airport connection performance and directed flight flow")]:
        extras = [("rq1_standardized/network_edges.csv", {"rows": "all", "plot_selection": "top 100 directed routes by flight_count; within-panel routes shown"})] if filename == "rq1_network_geographic.png" else None
        text.append(archive.figure(run_id, "figures/" + filename, table, {"rows": "all"}, caption, "scripts/complete_rq1.py", extra_sources=extras))
    return "\n\n".join(text)


def reproduction_prose(archive, config, final_run):
    """Check archived refit receipts only; never reopen row-level predictions."""
    run_id = config.get("reproduction_run_id")
    if run_id is None:
        return [], []
    run = archive.run(run_id)
    if run["config"].get("mode") != "reproduce_final":
        raise ValueError("Report reproduction run must have mode reproduce_final")
    formal_id = config["final_prediction_run_id"]
    original = archive.json(formal_id, "final_result.json")
    repeated = archive.json(run_id, "final_result.json")
    reference = archive.json(run_id, "reproduction_reference.json")
    verified = archive.json(run_id, "reproduction_verification.json")
    freeze = archive.source(formal_id, "model_selection.json")
    freeze_sha = digest(freeze)
    if digest(archive.source(run_id, "model_selection.json")) != freeze_sha:
        raise ValueError("Reproduction freeze differs from the formal archived copy")
    ledger = archive.path("experiments/final_access/2026-08.json")
    access = archive.json_path(ledger)
    if (access.get("status") != "completed" or access.get("run_id") != formal_id
            or access.get("freeze_manifest_sha256") != freeze_sha
            or access.get("final_result_sha256") != digest(archive.source(formal_id, "final_result.json"))):
        raise ValueError("Reproduction formal ledger identity differs")
    expected = {"mode": "reproduce_final", "claim_type": "reproduction", "formal_run_id": formal_id,
                "formal_final_result_sha256": access["final_result_sha256"], "freeze_manifest_sha256": freeze_sha,
                "ledger_sha256_before": digest(ledger), "creates_new_formal_claim": False,
                "refitted": True, "raw_test_data_read": True}
    for key, value in expected.items():
        if reference.get(key) != value or verified.get(key) != value or type(reference.get(key)) is not type(value) or type(verified.get(key)) is not type(value):
            raise ValueError(f"Reproduction provenance/verification differs: {key}")
    for name, expected_path in [("formal_run_dir", final_run["path"]), ("access_ledger_path", ledger)]:
        for source in [reference, verified]:
            actual = Path(source.get(name, ""))
            actual = actual if actual.is_absolute() else archive.root / actual
            if actual.resolve() != expected_path.resolve():
                raise ValueError("Reproduction formal path or ledger path differs")
    if (verified.get("verified") is not True or verified.get("formal_ledger_unchanged") is not True
            or verified.get("ledger_sha256_after") != digest(ledger)
            or original.get("mode") != "final" or original.get("claim_type") != "formal"
            or repeated.get("mode") != "reproduce_final" or repeated.get("claim_type") != "reproduction"
            or repeated.get("reproduction_reference") != reference):
        raise ValueError("Reproduction executed-refit/ledger verification is incomplete")
    identity_keys = ["config", "protocol", "models", "primary_model_name", "folds", "split_sha256",
                     "evaluation_rows", "evaluation_ids_sha256", "feature_fingerprints", "verified_inputs"]
    for key in identity_keys:
        if key not in original or not original[key] or original[key] != repeated.get(key):
            raise ValueError(f"Reproduction archived identity/config differs: {key}")
    models = original["models"]
    if (not isinstance(models, list) or len(models) != 4 or len(set(models)) != 4
            or set(models) != {final_run["model"]["model_name"], *BASELINES}):
        raise ValueError("Reproduction must compare the frozen learned model and all three baselines")
    frozen = archive.json(formal_id, "model_selection.json")
    for result in [original, repeated]:
        if (result.get("freeze_manifest_sha256") != freeze_sha or result.get("freeze_payload_sha256") != frozen["payload_sha256"]
                or result.get("config") != frozen.get("config") or result.get("protocol") != frozen.get("protocol")):
            raise ValueError("Reproduction result does not identify the original frozen configuration")
    n = original["evaluation_rows"]
    if (type(n) is not int or n <= 0 or n != final_run["model"]["evaluation_rows"]
            or original["evaluation_ids_sha256"] != final_run["model"]["evaluation_ids_sha256"]
            or original["split_sha256"] != final_run["model"]["split_sha256"]):
        raise ValueError("Reproduction formal evaluation identity/row count differs")
    comparison = repeated.get("reproduction_comparison", {})
    for key, expected_value in {"prediction_rows_compared": n * len(models), "all_feature_values_and_types_match": True,
                                "ids_and_targets_match": True, "prediction_atol": 1e-10, "metric_atol": 1e-10, "rtol": 0}.items():
        if (verified.get(key) != expected_value or comparison.get(key) != expected_value
                or isinstance(expected_value, bool) and (verified.get(key) is not expected_value or comparison.get(key) is not expected_value)):
            raise ValueError(f"Reproduction comparison declaration differs: {key}")
    filename = "reproduction_prediction_differences.csv"
    differences = archive.table(run_id, filename, ["model", "column", "rows", "max_absolute_difference", "atol", "rtol"])
    columns = ["predicted_delay_raw", "predicted_delay"]
    expected_pairs = {(model, column) for model in models for column in columns}
    if (len(differences) != 8 or differences.duplicated(["model", "column"]).any()
            or set(zip(differences.model, differences.column)) != expected_pairs):
        raise ValueError("Reproduction requires exactly eight complete model/raw/clipped comparison rows")
    for row in differences.itertuples():
        if (not all(math.isfinite(float(value)) for value in [row.rows, row.max_absolute_difference, row.atol, row.rtol])
                or row.rows != n or row.max_absolute_difference < 0 or row.max_absolute_difference > 1e-10
                or row.atol != 1e-10 or row.rtol != 0):
            raise ValueError("Reproduction prediction differences/rows exceed the fixed comparison protocol")
    metrics = archive.table(run_id, "metrics.csv", METRIC_COLUMNS)
    original_metrics = final_run["metrics"]
    keys = ["model", "fold", "group", "value", "n"]
    left, right = (frame.assign(value=frame.value.astype(str)).sort_values(keys).reset_index(drop=True)
                   for frame in [original_metrics, metrics])
    if list(left.columns) != list(right.columns) or len(left) != len(right) or not left[keys].equals(right[keys]):
        raise ValueError("Reproduction metric row identities/counts differ")
    if verified.get("metric_rows_compared") != len(left) or comparison.get("metric_rows_compared") != len(left):
        raise ValueError("Reproduction metric row-count declaration differs")
    for key in [column for column in left if column not in keys]:
        if key == "unit":
            if not left[key].equals(right[key]):
                raise ValueError("Reproduction metric units differ")
        elif not all(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(a - b) <= 1e-10 for a, b in zip(left[key], right[key])):
            raise ValueError("Reproduction metric differences exceed the fixed tolerance")
    refs = " ".join(archive.cite(run_id, source, selector, claim) for source, selector, claim in [
        ("reproduction_verification.json", {"keys": [*expected, "verified", "formal_ledger_unchanged", "ledger_sha256_after", *comparison]}, "Archived full-refit execution verification and fixed acceptance conditions"),
        ("reproduction_reference.json", {"keys": list(reference)}, "Original formal experiment and unchanged ledger identity"),
        ("model_selection.json", {"whole_file": True, "sha256_equals_formal_copy": freeze_sha}, "Identical formal/reproduction freeze copy"),
        (filename, {"rows": "all", "aggregation": "max(max_absolute_difference) separately for predicted_delay_raw and predicted_delay"}, "Observed raw and clipped prediction reproduction differences"),
        ("metrics.csv", {"rows": "all", "comparison": "same formal identities and absolute tolerance 1e-10"}, "Reproduced metric rows and fixed-tolerance agreement")])
    raw = differences.loc[differences.column.eq(columns[0]), "max_absolute_difference"].max()
    clipped = differences.loc[differences.column.eq(columns[1]), "max_absolute_difference"].max()
    en = [f"Completed reproduction {run_id} records feature rebuilding and refitting under the identical freeze used by formal run {formal_id}. "
          f"The archived execution verification reports matching feature values/types and IDs/targets, with the formal ledger unchanged. "
          f"It compared {n:,} evaluation rows per model across four models ({n * 4:,} model-prediction rows and eight raw/clipped comparison rows), "
          f"plus {len(metrics):,} metric rows. Maximum absolute raw/clipped prediction differences were {raw:.6g}/{clipped:.6g} min, "
          f"within fixed atol=1e-10 and rtol=0. This report checks archived JSON/CSV execution evidence only; it does not rebuild features, refit or reopen predictions. "
          f"This is reproduction on the already-seen test set, not a new independent test or formal claim. {refs}"]
    zh = [f"完整复现{run_id}的执行归档记录使用正式实验{formal_id}同一freeze重建特征并重新拟合；特征值/类型、ID/目标核对一致，正式账本前后不变。"
          f"四模型各核对{n:,}条评价记录，合计{n * 4:,}条模型预测记录及8行原始/截断差异汇总，另核对{len(metrics):,}行指标。"
          f"原始/截断预测最大绝对差为{raw:.6g}/{clipped:.6g}分钟，均在固定atol=1e-10、rtol=0内。"
          f"本报告仅读归档JSON/CSV执行证据，不重建、不重训、不打开预测文件；这是已见测试集上的复现，不是新增独立测试或第二项正式声明。{refs}"]
    return en, zh


def ols_coefficient_prose(archive, selection_run_id, feature_run):
    """Read the feature-source OLS, independently of any selected tree/forest."""
    run_id = feature_run["id"]
    model = feature_run["model"]
    if model.get("model_type") != "linear" or model.get("development_only") is not True:
        raise ValueError("OLS coefficient explanation requires the actual development linear feature source")
    if "StandardScaler(with_mean=False)" not in model.get("effective_preprocessing", {}).get("numeric_scaling", ""):
        raise ValueError("OLS coefficient explanation requires the recorded numeric scaling")
    base = f"explanations/{run_id}"
    summary_file = base + "/explain_summary.json"
    summary = archive.json(selection_run_id, summary_file)
    if (summary.get("source_run") != run_id or summary.get("fit_called") is not False
            or summary.get("model_name") != model["model_name"] or summary.get("features") != model["features"]):
        raise ValueError("OLS explanation source/model/features or no-refit declaration differs")
    folds = [item["fold"] for item in summary.get("folds", [])]
    if not folds or len(folds) != len(set(folds)) or not set(folds) <= {item["fold"] for item in model["folds"]}:
        raise ValueError("OLS explanation folds must identify saved development models")
    coefficient_file, intercept_file = base + "/explain_coefficients.csv", base + "/explain_intercepts.csv"
    coefficients = archive.table(selection_run_id, coefficient_file, ["fold", "encoded_feature", "coefficient"])
    intercepts = archive.table(selection_run_id, intercept_file, ["fold", "intercept"])
    if (coefficients.duplicated(["fold", "encoded_feature"]).any() or intercepts.fold.duplicated().any()
            or set(coefficients.fold) != set(folds) or set(intercepts.fold) != set(folds)
            or not all(math.isfinite(float(value)) for value in coefficients.coefficient)
            or not all(math.isfinite(float(value)) for value in intercepts.intercept)):
        raise ValueError("OLS coefficient/intercept rows are duplicated, nonfinite or have inconsistent folds")
    ref = archive.cite(selection_run_id, summary_file,
                       {"keys": ["source_run", "model_name", "features", "folds", "fit_called"]},
                       "Saved development OLS explanation provenance and no-refit scope")
    scaling_ref = archive.cite(run_id, "model_manifest.json",
                               {"keys": ["model_type", "features", "effective_preprocessing.numeric_scaling"]},
                               "Actual linear feature source and training-fitted numeric scaling")
    en = [f"For coefficient interpretation, the OLS benchmark is the selected feature-source run {run_id}, independently of the retained algorithm. "
          f"The saved development coefficient tables are read without refitting. {ref} {scaling_ref}"]
    zh = ["## OLS数值系数的含义",
          f"系数解释固定使用特征来源{run_id}的OLS基准，不因最终保留树或森林而改读其空系数表；只读取开发期已保存系数表，没有读取模型文件或重新拟合。{ref} {scaling_ref}"]
    specified = ["upstream_arr_delay_minutes", "scheduled_turnaround_minutes", "downstream_distance"]
    for fold in sorted(folds):
        values, chinese_values = [], []
        for feature in specified:
            encoded = "numeric__" + feature
            if feature not in model["features"]:
                if ((coefficients.fold == fold) & (coefficients.encoded_feature == encoded)).any():
                    raise ValueError("OLS coefficient exists for a feature absent from the source model")
                values.append(f"{feature} was not included")
                chinese_values.append(f"{feature}未纳入该模型")
                continue
            row = selected_row(coefficients, fold=fold, encoded_feature=encoded)
            row_ref = archive.cite(selection_run_id, coefficient_file, {"fold": fold, "encoded_feature": encoded},
                                   "Prespecified OLS numeric coefficient in transformed-input units")
            values.append(f"{feature}={number(row.coefficient, 6)} {row_ref}")
            chinese_values.append(f"{feature}={number(row.coefficient, 6)} {row_ref}")
        row = selected_row(intercepts, fold=fold)
        row_ref = archive.cite(selection_run_id, intercept_file, {"fold": fold}, "Saved OLS intercept in transformed coordinates")
        en.append(f"For the {fold} saved OLS fit, the prespecified numeric coefficients were {'; '.join(values)}; "
                  f"the intercept was {number(row.intercept, 6)} min. {row_ref}")
        zh.append(f"{fold}保存的OLS预指定数值系数为{'；'.join(chinese_values)}；截距为{number(row.intercept, 6)}分钟。{row_ref}")
    en.append("Under StandardScaler(with_mean=False), each listed coefficient is the change in the raw linear prediction, in minutes, "
              "for one unit of that transformed variable with other transformed inputs fixed: approximately one training standard deviation of the imputed original predictor. "
              "Constant predictors are an exception to this standard-deviation interpretation. Clipping predictions at zero means the deployed prediction need not have the same marginal change. "
              "The intercept describes transformed zero inputs and reference categories, which need not represent a feasible flight or population baseline. "
              "These conditional associations do not rank airport/route importance by absolute coefficient or identify causal effects. " + scaling_ref)
    zh.append("StandardScaler(with_mean=False)下，每个系数表示其他转换输入固定时，该转换变量增加1单位引起的原始线性预测变化，单位为分钟；"
              "约对应填补后原变量的1个训练期标准差，常量变量不适用这一标准差解释。预测截断为非负后不保证仍有相同边际变化。"
              "截距对应转换坐标为零及参考类别，未必是一条可行航班或总体基线；这些条件关联不能按系数绝对值给机场/航线重要性排名，也不识别因果。" + scaling_ref)
    return en, zh


def execution_prose(archive, development_runs, selected_run, final_run):
    """Disclose recorded resource budgets without inventing missing defaults."""
    def positive(value):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("Recorded execution budget must be a positive integer")
        return value

    def budget(run):
        values, refs = [], []
        for key in ["threads", "fold_workers"]:
            manifest_value = positive(run["model"].get(key))
            config_value = positive(run["config"].get(key))
            if manifest_value is not None and config_value is not None and manifest_value != config_value:
                raise ValueError(f"Execution manifest/config disagree: {run['id']}/{key}")
            values.append(manifest_value if manifest_value is not None else config_value)
            filename = "model_manifest.json" if manifest_value is not None else "config.json"
            refs.append(archive.cite(run["id"], filename, {"keys": [key], "recorded": values[-1]},
                                     "Recorded execution budget, not measured CPU utilization"))
        return tuple(values), " ".join(refs)

    def prediction_budget(run, threads):
        fit = positive(run["model"].get("fit_threads"))
        prediction = positive(run["model"].get("prediction_threads"))
        configured = positive(run["config"].get("prediction_threads"))
        if fit is not None and fit != threads:
            raise ValueError("Recorded fit_threads differ from the training thread budget")
        if configured is not None and (configured != 1 or run["model"].get("model_type") != "random_forest"):
            raise ValueError("Explicit prediction_threads must be 1 for random_forest")
        if prediction is not None and configured is not None and prediction != configured:
            raise ValueError("Prediction manifest/config disagree")
        effective = prediction if prediction is not None else configured if configured is not None else threads
        ref = archive.cite(run["id"], "model_manifest.json",
                           {"keys": ["threads", "fit_threads", "prediction_threads"], "legacy_prediction_default": "threads"},
                           "Separate recorded fit/prediction budgets; absent prediction setting retains the training budget")
        if configured is not None:
            ref += " " + archive.cite(run["id"], "config.json", {"keys": ["prediction_threads"]}, "Explicit prediction execution setting")
        return effective, ref

    groups, budgets = {}, {}
    for run in development_runs:
        pair, refs = budget(run)
        budgets[run["id"]] = pair
        groups.setdefault(pair, []).append((run["id"], refs))
    en, zh = [], []
    for (threads, workers), members in groups.items():
        ids = ", ".join(run_id for run_id, _ in members)
        refs = " ".join(ref for _, ref in members)
        if threads is None or workers is None:
            setting = f"threads per fold={threads if threads is not None else 'not recorded'}, fold workers={workers if workers is not None else 'not recorded'}"
            en.append(f"Development execution for {ids} recorded {setting}; missing fields are not inferred from other runs. {refs}")
            zh.append(f"开发运行{ids}的归档配置为{setting}；缺字段不能从其他运行推定。{refs}")
        else:
            en.append(f"Development runs {ids} used {threads} threads per fold and up to {workers} simultaneous fold workers "
                      f"(a configured budget of {threads * workers} worker threads, not measured CPU utilization). {refs}")
            zh.append(f"开发运行{ids}每折{threads}线程、最多{workers}折同时运行，配置预算为{threads * workers}个工作线程，"
                      f"并非实测CPU占用。{refs}")
    frozen = archive.json(final_run["id"], "model_selection.json")
    locked = frozen.get("config", {})
    locked_threads = positive(locked.get("threads"))
    locked_prediction = positive(locked.get("prediction_threads"))
    final_budget, final_refs = budget(final_run)
    final_prediction, prediction_ref = prediction_budget(final_run, final_budget[0])
    development_threads = budgets[selected_run["id"]][0]
    development_prediction, development_prediction_ref = prediction_budget(selected_run, development_threads)
    ref = archive.cite(final_run["id"], "model_selection.json",
                       {"keys": ["config.threads", "config.prediction_threads", "execution_delta", "source_files"], "selected_run_id": selected_run["id"]},
                       "Frozen final/reproduction execution configuration and any development-to-final override")
    delta = frozen.get("execution_delta")
    if locked_threads is not None:
        if final_budget[0] != locked_threads:
            raise ValueError("Final execution threads differ from the frozen configuration")
        if locked_prediction is not None and (locked_prediction != 1 or locked.get("model_type") != "random_forest"
                                               or final_run["model"].get("model_type") != "random_forest"
                                               or final_run["model"].get("prediction_threads") != locked_prediction):
            raise ValueError("Frozen RF prediction_threads disagree with the final model manifest")
        effective_prediction = locked_prediction if locked_prediction is not None else locked_threads
        if final_prediction != effective_prediction:
            raise ValueError("Final prediction threads differ from the frozen configuration")
        if delta is not None:
            if (delta.get("explicitly_requested") is not True or delta.get("learning_parameters_unchanged") is not True
                    or not any(key in delta for key in ["threads", "prediction_threads"])):
                raise ValueError("Frozen execution_delta disagrees with recorded development/final threads")
        if delta is not None and "threads" in delta:
            if (delta["threads"].get("final") != locked_threads or development_threads is None
                    or delta["threads"].get("development") != development_threads):
                raise ValueError("Frozen execution_delta training threads disagree")
            en.append(f"The freeze records an explicit execution-only change from {development_threads} development threads per fold "
                      f"to {locked_threads} final threads, with learning parameters unchanged. {ref}")
            zh.append(f"冻结execution_delta明确记录每折线程从开发{development_threads}改为最终{locked_threads}，学习参数不变。{ref}")
        elif development_threads is not None and development_threads != locked_threads:
            raise ValueError("Changed final thread budget lacks an explicit frozen execution_delta")
        else:
            en.append(f"The frozen final training-thread setting is {locked_threads}; "
                      f"{'no execution override is recorded' if delta is None else 'no training-thread override is recorded'}. {ref}")
            zh.append(f"冻结的最终训练线程数为{locked_threads}，未记录{'执行' if delta is None else '训练线程'}覆盖。{ref}")
        if delta is not None and "prediction_threads" in delta:
            if (delta["prediction_threads"].get("development") != development_prediction
                    or delta["prediction_threads"].get("final") != effective_prediction):
                raise ValueError("Frozen execution_delta prediction threads disagree")
            en.append(f"The execution delta separately records prediction threads changing from {development_prediction} to {effective_prediction}, "
                      f"with learning parameters unchanged. {development_prediction_ref} {ref}")
            zh.append(f"execution_delta另记录预测线程从{development_prediction}改为{effective_prediction}，学习参数不变。{development_prediction_ref} {ref}")
        elif development_prediction is not None and development_prediction != effective_prediction:
            # Legacy freezes that changed only threads also changed their default prediction budget.
            if locked_prediction is not None:
                raise ValueError("Changed prediction thread budget lacks a frozen execution_delta")
        en.append(f"The final run recorded {final_budget[0]} threads for training and {effective_prediction} for prediction, with "
                  f"{final_budget[1] if final_budget[1] is not None else 'an unrecorded number of'} fold workers. "
                  f"Full reproduction must reuse this same freeze, including its {locked_threads} training threads and {effective_prediction} prediction threads. "
                  f"Without an explicit prediction setting, prediction retains the recorded training budget. {final_refs} {prediction_ref} {ref}")
        zh.append(f"最终归档为训练{final_budget[0]}线程、预测{effective_prediction}线程、折间worker {final_budget[1] if final_budget[1] is not None else '未记录'}；"
                  f"完整复现必须使用同一freeze及相同训练/预测线程配置。未显式配置预测线程时，预测沿用归档训练预算。{final_refs} {prediction_ref} {ref}")
        if locked_prediction is not None:
            en.append("The frozen shared prediction entry temporarily sets RF prediction workers to the sequential budget so trees accumulate in a fixed order, "
                      "then restores the training n_jobs value before the joblib model is saved. Loading that model and calling bare predict does not preserve this prediction policy. "
                      "The policy targets the original absolute 1e-10 contract for all reported metrics, including unnormalised error sums; it changes execution, not learning parameters. " + ref)
            zh.append("冻结的共享预测入口临时将RF预测worker设为顺序预算，以固定树的累加次序，之后恢复训练n_jobs再保存joblib。"
                      "因此加载该模型后直接裸调用predict并不自动保持此预测策略，必须使用冻结的共享入口。"
                      "这项执行设置针对原有全部指标（含未归一化误差总和）的绝对1e-10契约，不改变学习参数。" + ref)
    else:
        en.append(f"This archive does not record frozen execution threads; the report does not infer a final or reproduction budget. {ref}")
        zh.append(f"该归档未记录冻结线程数，本报告不推定最终或复现线程预算。{ref}")
    en.append("Elapsed time is not used to compare algorithm quality because execution budgets can differ. Parallel prediction reductions can differ in the last floating-point digits; "
              "the fixed full-reproduction acceptance tolerances remain atol=1e-10 and rtol=0, without relaxation.")
    zh.append("执行预算可能不同，不能据运行时长判断算法优劣。并行预测归约可能产生浮点末位差异；完整复现仍使用既定atol=1e-10、rtol=0，不放宽容差。")
    return en, zh


def compose(archive, config, validated):
    main, variant, noaa, algorithms, feature, choice, final_run = validated
    lines = ["---\ntitle: Flight delay propagation and downstream prediction\nlang: en\nmainfont: Times New Roman\nfontsize: 11pt\n---",
             "# Flight delay propagation and downstream prediction",
             "## Research questions and principal findings",
             "This study asks which airports absorb or amplify delay within consecutive same-aircraft connections (RQ1), and whether downstream nonnegative departure delay can be predicted at upstream actual arrival (RQ2). "
             "The analysis is descriptive and retrospective, conditional on valid observed completed connections. It does not allocate causal responsibility or answer the separate cause-code attribution question.",
             metric_sentence(archive, final_run, choice["primary_model_name"]),
             "## Related work and adopted methods",
             "Li et al. (2024, pp. 4, 9–10) motivate distinguishing aircraft-chain propagation from airport-network summaries and auditing BTS/NOAA sources. Their distinction between minimum operational turnaround and buffer prevents treating scheduled turnaround as an observed buffer here. "
             "Chauhan et al. (2023, pp. 1–3) distinguish schedule connections and historical delay networks; this project adopts separate encodings for route traffic and airport connection outcomes, without implementing their percolation analysis. "
             "The UC Berkeley (2025) project motivates temporal validation, staged information experiments and severe-delay diagnosis. Its two-hours-before-departure horizon differs from upstream actual arrival, so its feature availability and performance figures are not transferred to this sample.",
             "## Data, preparation and exploratory analysis",
             "The project covers BTS Reporting US domestic flights from April through August 2026. Raw sources are preserved. Cleaning retains provenance and exclusion flags, handles local time zones, midnight and 2400, and forms candidates in same-tail scheduled order before checking endpoint continuity and observed timing. "
             "Cancelled, diverted, ambiguous and otherwise invalid chains define exclusions rather than zero outcomes. Month-prefix QA and future completed-connection selection mean the analysed population is conditional and is not a fully deployable arrival-time eligibility classifier. "
             "Aircraft family is the mainline representation, variant is an independent replacement, and the 2024/2025 B43 inventories are reference snapshots rather than historical fleet truth. Seats, aircraft age and airport infrastructure are excluded."]
    single_en, single_zh = single_flight_prose(archive, config["development_rq1_run_id"])
    reference_en, reference_zh = rq1_reference_prose(archive, config["development_rq1_run_id"])
    noaa_en, noaa_zh = noaa_prose(archive, noaa["id"])
    selected_run = next(run for run in [*main, *algorithms] if run["id"] == choice["selected_run_id"])
    execution_en, execution_zh = execution_prose(archive, [*main, variant, noaa, *algorithms], selected_run, final_run)
    feature_run = next(run for run in main if run["id"] == feature["selected_run_id"])
    coefficient_en, coefficient_zh = ols_coefficient_prose(archive, config["model_selection_run_id"], feature_run)
    reproduction_en, reproduction_zh = reproduction_prose(archive, config, final_run)
    lines.extend(single_en)
    eda_id = main[0]["id"]
    eda = archive.json(eda_id, "eda_summary.json")
    ref = archive.cite(eda_id, "eda_summary.json", {"keys": ["n", "airport_n", "route_n", "overnight_n", "scope"]}, "Development RQ2 population")
    lines.append(f"Development EDA contains {int(eda['n']):,} eligible connections, {int(eda['airport_n'])} airports and {int(eda['route_n'])} routes, "
                 f"including {int(eda['overnight_n']):,} overnight connections. RQ2 retains overnight and upstream-on-time connections; the RQ1 main sample is narrower. {ref}")
    monthly = archive.table(eda_id, "tables/eda_monthly_target.csv", ["month_utc", "n", "mean", "median", "p99", "target_ge60_share"])
    for _, row in monthly.iterrows():
        if str(row.month_utc) not in ["2026-04", "2026-05", "2026-06", "2026-07"]:
            raise ValueError("Development EDA contains a non-development month")
        ref = archive.cite(eda_id, "tables/eda_monthly_target.csv", {"month_utc": row.month_utc}, "Monthly target distribution")
        lines.append(f"In {row.month_utc}, n={int(row.n):,}; mean/median downstream delay was {number(row['mean'])}/{number(row['median'])} min, "
                     f"the 99th percentile was {number(row.p99)} min, and the severe-delay share was {number(row.target_ge60_share)}. {ref}")
    lines.append(archive.figure(eda_id, "figures/eda_target_distribution.png", "tables/eda_target_bins.csv", {"rows": "all"}, "Development target distribution", "scripts/analyze_eda.py"))
    lines += ["## Prediction timing, validation and new methods",
              "Prediction time is upstream actual arrival; the target is downstream nonnegative departure delay, with prediction strictly before label time. "
              "May, June and July are forward validation months, trained respectively on mature April, April–May and April–June labels. UTC month-start half-open windows govern membership. "
              "All learned imputation, category encoding and scaling are fitted inside each training fold, and every prediction is clipped at zero. The August test follows a committed frozen choice and mature April–July refitting; test performance never selects a model.",
              "The new methods are as-of availability matching, common-support standardization and paired UTC date-block uncertainty. METAR lacks actual receipt timestamps, so observation plus five minutes is assumed available, with observation age capped at 120 minutes. "
              "TAF uses the latest archive bulletin state available at prediction time and queries the later of scheduled departure and prediction time. Archive/issue timestamps are availability proxies, not verified historical delivery receipts; TEMPO, probability, amendment, cancellation and NIL states remain distinct. The registered conservative rule blocks unverified amended, corrected or delayed bulletin values without falling back to an older forecast. Thus locating a latest state does not imply an approved active forecast. "
              "NOAA is a separate retrospective supplement: exact quality-code whitelisting does not establish full official QC certification or contemporaneous delivery, and conflicting or structural missing values are not automatically filled. "
              "Weather absence retains the flight with explicit missingness rather than deleting it or imputing clear weather.",
              "\n\n".join(execution_en),
              "## RQ1: airport absorption and amplification",
              "For a connection, net change is downstream nonnegative departure delay minus upstream nonnegative arrival delay. A negative change means fewer delay minutes, not necessarily punctual departure. "
              "The main population is nonovernight with upstream delay at least 15 minutes. Airport comparison adjusts only for upstream-delay and scheduled-turnaround cells within frozen common support. "
              "Common airports, cells and reference weights are held fixed across months and final reporting; uncertainty uses common date blocks and is descriptive, not causal or multiplicity-adjusted.",
              "\n\n".join(reference_en),
              rq1_prose(archive, config["development_rq1_run_id"], "April–July development"),
              rq1_prose(archive, config["final_rq1_run_id"], "April–August final description"),
              "## RQ2: fixed-OLS feature development"]
    for baseline in BASELINES:
        lines.append(metric_sentence(archive, main[0], baseline))
    for i, run in enumerate(main):
        lines.append(f"The {STAGES[i]} stage used the fixed OLS estimator. " + metric_sentence(archive, run))
        if i:
            lines.append(comparison_text(archive, main[i - 1], run))
    lines += ["The sequence adds planned information, upstream and completed-tail history, family, recent airport operations and planned load, METAR, then TAF. "
              "Each adjacent comparison holds validation identities, targets and splits fixed. Incremental gains depend on this order and cannot be interpreted as intrinsic or causal variable importance.",
              "## Independent aircraft and weather checks",
              "Variant replaces family without becoming a cumulative mainline stage. " + metric_sentence(archive, variant),
              comparison_text(archive, main[2], variant),
              "NOAA supplements IEM in a separate branch and is ineligible for primary feature selection. " + metric_sentence(archive, noaa),
              comparison_text(archive, main[-1], noaa), "\n\n".join(noaa_en)]
    common = archive.json(main[-1]["id"], "common_weather/definition.json")
    if common.get("refitted") is not False or common.get("uses_target_for_selection") is not False:
        raise ValueError("Common-weather analysis must use a fixed availability subset without refitting")
    ref = archive.cite(main[-1]["id"], "common_weather/definition.json", {"keys": ["definition", "n", "full_validation_n", "refitted"]}, "Common weather sample definition")
    lines.append(f"The supplementary common source/prevailing-TAF availability subset contains {int(common['n']):,} of {int(common['full_validation_n']):,} validation rows; "
                 f"the rule is {common['definition']}. It compares saved predictions without refitting; source availability does not imply complete per-variable weather. {ref}")
    for reference, candidate, folder in [(main[3], main[4], "G3_to_G4a"), (main[4], main[5], "G4a_to_G4b"), (main[3], main[5], "G3_to_G4b")]:
        # All common-subset comparison files are archived under G4B.
        receipt = archive.json(main[-1]["id"], f"common_weather/{folder}/comparison.json")
        if receipt.get("reference_run") != reference["id"] or receipt.get("candidate_run") != candidate["id"]:
            raise ValueError("Common weather comparison lineage differs")
        filename = f"common_weather/{folder}/comparison_metrics.csv"
        table = archive.table(main[-1]["id"], filename, COMPARISON_COLUMNS)
        row = selected_row(table, scope="pooled", metric="mae")
        if int(row.n) != int(common["n"]):
            raise ValueError("Common weather comparison population differs")
        ref = archive.cite(main[-1]["id"], filename, {"scope": "pooled", "metric": "mae"}, "Common weather paired MAE comparison")
        lines.append(f"On this common subset, {folder} changed MAE by {number(row.difference_candidate_minus_reference)} min "
                     f"(95% interval {number(row.ci95_low)} to {number(row.ci95_high)}; candidate minus reference). {ref}")
    for run in [main[4], main[5], noaa]:
        coverage = archive.json(run["id"], "weather_coverage_summary.json")
        ref = archive.cite(run["id"], "weather_coverage_summary.json", {"keys": ["rows", "availability", "interpretation"]}, "Query-level weather availability")
        values = "; ".join(f"{key}: {int(value):,}" for key, value in coverage["availability"].items())
        lines.append(f"{run['id']} recorded {int(coverage['rows']):,} weather queries; matched availability counts were {values}. "
                     f"Counts represent source states, not complete weather vectors. {ref}")
    lines.append("## Fixed-feature algorithms and frozen model selection")
    ref = archive.cite(config["feature_selection_run_id"], "selection/feature_selection.json", {"keys": ["selected_run_id", "reason", "selected_pooled_mae"]}, "Development-only feature selection")
    lines.append(f"The development feature decision selected {feature['selected_run_id']}: {feature['reason']}. Variant and NOAA remained supplementary. {ref}")
    for run in algorithms:
        params = run["model"]["model_params"]
        lines.append(f"For {run['model']['model_type']}, depth={params['max_depth']} and minimum leaf={params['min_samples_leaf']}, " + metric_sentence(archive, run))
    ref = archive.cite(config["model_selection_run_id"], "selection/model_selection_decision.json", {"keys": ["selected_run_id", "primary_model_name", "reason", "promotion_decisions", "primary_candidates"]}, "Registered algorithm and baseline selection")
    lines.append(f"The retained learned-model run was {choice['selected_run_id']}; the frozen primary predictor was {choice['primary_model_name']}. "
                 f"The rule was: {choice['reason']}. A training-mean, training-median or carry-forward baseline was permitted to win; "
                 f"there was no requirement to select a complex estimator. {ref}")
    lines.append("The practical and monthly-stability promotion rule was committed after P3 and before weather or tree/forest performance was inspected; it was not registered before P0. "
                 "Feature selection is guided by OLS and depends on the fixed addition order; it does not establish the globally optimal tree feature set. The primary is the best of the rule-retained learned model and three baselines, which need not be the lowest-MAE estimator among every grid trial. "
                 "Paired date-block intervals resample the existing predictions with shared date draws and recompute summed absolute or squared error divided by the corresponding sample count. They do not repeat training, feature selection or hyperparameter selection and are not selection-adjusted confidence intervals. " + ref)
    for decision in choice["promotion_decisions"]:
        lines.append(f"The {decision['family']} challenger {decision['candidate_run_id']} had pooled improvement {number(decision['pooled_improvement'])} min, "
                     f"required improvement {number(decision['required_improvement'])} min, and {decision['improved_months']} improving months. "
                     f"Promotion was {bool(decision['promoted'])}; the recorded reasons were {decision.get('rejection_reasons') or 'all promotion criteria passed'}. {ref}")
    explanation_base = f"explanations/{choice['selected_run_id']}"
    explanation = archive.json(config["model_selection_run_id"], explanation_base + "/explain_summary.json")
    if explanation.get("source_run") != choice["selected_run_id"] or explanation.get("fit_called") is not False:
        raise ValueError("Explanation source or no-refit declaration differs")
    explanation_file = explanation_base + "/permutation_summary.csv"
    importance = archive.table(config["model_selection_run_id"], explanation_file,
                               ["fold", "group", "mean_delta_mae", "sd_delta_mae", "repeats"])
    lines.append("The retained learned model is explained separately from any winning baseline. Joint-group permutation preserves relationships within a feature group but can break relationships across groups; "
                 "repeat standard deviation is not a confidence interval. Encoded linear coefficients use reference categories and scaled inputs, and airport–route overlap prevents independent airport-effect interpretation. "
                 "PDP describes hypothetical input replacements and may cross correlated-feature support; neither technique establishes causation.")
    for fold, part in importance.groupby("fold", sort=True):
        row = part.sort_values(["mean_delta_mae", "group"], ascending=[False, True]).iloc[0]
        ref = archive.cite(config["model_selection_run_id"], explanation_file, {"fold": fold, "group": row.group}, "Learned-model grouped permutation sensitivity on a fixed validation subsample")
        lines.append(f"On the fixed {fold} validation explanation sample, the largest grouped permutation MAE increase was for {row.group}: "
                     f"{number(row.mean_delta_mae)} min (repeat SD {number(row.sd_delta_mae)}, {int(row.repeats)} repeats). "
                     f"This is model/sample sensitivity, not full-population temporal uncertainty. {ref}")
    lines.extend(coefficient_en)
    lines += ["## One-time August assessment and remaining errors",
              "The final metrics below come from the completed single-access ledger and saved-result verification. This report reads summary tables only and does not reopen test features or refit models."]
    for model in [final_run["model"]["model_name"], *BASELINES]:
        lines.append(metric_sentence(archive, final_run, model))
    for value in ["ordinary_lt60", "severe_ge60"]:
        lines.append(f"For the frozen primary predictor in {value}, " + metric_sentence(archive, final_run, choice["primary_model_name"], group="severity", value=value))
    final_comparison = archive.table(final_run["id"], "final_comparisons.csv", [*COMPARISON_COLUMNS, "reference_model", "candidate_model"])
    for baseline in BASELINES:
        row = selected_row(final_comparison, reference_model=baseline, candidate_model=choice["primary_model_name"], metric="mae", scope="pooled")
        ref = archive.cite(final_run["id"], "final_comparisons.csv", {"reference_model": baseline, "candidate_model": choice["primary_model_name"], "metric": "mae", "scope": "pooled"}, "Final paired comparison with baseline")
        lines.append(f"Relative to {baseline}, primary-minus-baseline August MAE changed by {number(row.difference_candidate_minus_reference)} min "
                     f"(95% date-block interval {number(row.ci95_low)} to {number(row.ci95_high)}). {ref}")
    residuals = archive.table(final_run["id"], "residual_summary.csv", ["model", "n", "mean_signed_error", "median_signed_error", "p95_absolute_error"])
    residual = selected_row(residuals, model=choice["primary_model_name"])
    ref = archive.cite(final_run["id"], "residual_summary.csv", {"model": choice["primary_model_name"]}, "Final signed residual bias and error tail")
    lines.append(f"The primary predictor's mean/median signed error (prediction minus target) was {number(residual.mean_signed_error)}/{number(residual.median_signed_error)} min; "
                 f"the 95th percentile absolute error was {number(residual.p95_absolute_error)} min (n={int(residual.n):,}). {ref}")
    for grouping in ["airport", "airline", "aircraft_family", "day_of_week", "scheduled_hour", "overnight"]:
        rows = final_run["metrics"].loc[final_run["metrics"].model.eq(choice["primary_model_name"])
                                      & final_run["metrics"].fold.eq("overall") & final_run["metrics"].group.eq(grouping)]
        if rows.empty:
            raise ValueError(f"Final grouped errors missing: {grouping}")
        # Support-led examples avoid choosing sparse groups for extreme errors.
        for _, row in rows.sort_values(["n", "value"], ascending=[False, True]).head(2).iterrows():
            lines.append(f"Among the largest observed {grouping} groups, {row.value}: " +
                         metric_sentence(archive, final_run, choice["primary_model_name"], group=grouping, value=row.value))
    lines.extend(reproduction_en)
    lines += ["## Research evolution, reproducibility and limitations",
              "The development chain preserves unfavourable and favourable results rather than skipping a stage after observing its MAE. Failed and invalidated runs remain in the experiment index and research journal; their metrics are excluded from current conclusions."]
    for _, row in archive.index.loc[archive.index.status.isin(["failed", "invalidated"])].iterrows():
        # Only status/change/decision are used; failed numerical outputs are never loaded.
        if not archive.log_entries(row.run_id):
            raise ValueError(f"Failed/invalidated history lacks journal entry: {row.run_id}")
        evidence_id = f"E{len(archive.evidence) + 1:04d}"
        archive.evidence.append({"evidence_id": evidence_id, "kind": "research_history", "claim": str(row.change),
                                 "run_id": row.run_id, "source_file": "experiments/index.csv", "selector": json.dumps({"run_id": row.run_id}),
                                 "source_sha256": digest(archive.root / "experiments/index.csv"), "research_log": "research/研究记录.md",
                                 "research_log_heading_lines": ";".join(archive.log_entries(row.run_id))})
        lines.append(f"{row.run_id} was {row.status}. Its recorded change was “{row.change}”; the decision was “{row.decision}”. [{evidence_id}]")
    lines += ["Shared scripts implement preparation, feature generation, training, evaluation, standardization and selection; Chinese notebooks call these same functions and expose archived checks. "
              "Each run holds its actual JSON configuration, manifest, logs, metrics, evaluation identities, predictions and applicable models/figures/notebooks. "
              "This report's evidence index ties every numerical claim and displayed figure to its original run, CSV/JSON selector, SHA-256 receipt and journal heading. "
              "Raw and large local derived/model files remain necessary for full computational reproduction even when they are excluded from version control.",
              "Limitations include one season and reporting coverage, conditional completed-connection selection, incomplete weather delivery provenance, retrospective NOAA supplementation, static aircraft inventories, and assumptions about advance availability of BTS planned schedules. "
              "Scheduled turnaround is not a measured operational buffer. Common support limits generalization and may disappear in individual months; missing intervals are not evidence of no effect. "
              "Date blocks quantify sampling uncertainty under the specified calendar, not causal identification or protection against all temporal dependence. "
              "Airport identity and route overlap, correlated weather and operating conditions prevent interpreting model coefficients, permutation/PDP patterns or airport rankings as responsibility. "
              "A final August result is one seasonal check, and any later method changes require a new unseen evaluation period.",
              "## References",
              "Li, C., Mao, J., Li, L., Wu, J., Zhang, L., Zhu, J., and Pan, Z. (2024). Flight delay propagation modeling: Data, Methods, and Future opportunities. Transportation Research Part E, 185, 103525. https://doi.org/10.1016/j.tre.2024.103525.",
              "Chauhan, V. K., Ledwoch, A., Brintrup, A., Herrera, M., Giannikas, V., Stojkovic, G., and Mcfarlane, D. (2023). Network science approach for identifying disruptive elements of an airline. arXiv:2211.08140v2. https://arxiv.org/abs/2211.08140.",
              "UC Berkeley School of Information (2025). Air Travel Delay Prediction: Feature Engineering and ML Approaches. https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches."]
    english = "\n\n".join(lines) + "\n"
    chinese = ["# 航班延误项目结果解读", "数值均来自已完成且未失效的正式归档；证据编号与英文报告共用。负的模型误差差值表示改善；负的RQ1净变化表示减少延误分钟，仍可能延误。",
               f"冻结主预测器为 {choice['primary_model_name']}，保留学习模型来自 {choice['selected_run_id']}，特征来源为 {feature['selected_run_id']}。基线允许获胜；8月结果不用于重新选择。",
               "## RQ2逐步特征、分支和最终结果"]
    chinese.insert(2, f"开发期EDA含{int(eda['n']):,}条有效RQ2衔接、{int(eda['airport_n'])}个机场和{int(eda['route_n'])}条航线，其中隔夜{int(eda['overnight_n']):,}条；其人群包括上游准点衔接，与RQ1主样本不同。")
    chinese[4:4] = single_zh
    for run in [*main, variant, noaa, *algorithms, final_run]:
        name = choice["primary_model_name"] if run is final_run else run["model"]["model_name"]
        row = selected_row(run["metrics"], model=name, fold="overall", group="all", value="all")
        ids = [item["evidence_id"] for item in archive.evidence if item["run_id"] == run["id"] and item["source_file"].endswith("/metrics.csv")]
        chinese.append(f"{run['id']} / {name}：MAE {number(row.mae)}分钟，RMSE {number(row.rmse)}分钟，样本{int(row.n):,}。" + " ".join(f"[{item}]" for item in ids[:1]))
    chinese.extend(noaa_zh)
    chinese += ["## 执行预算与复现", *execution_zh]
    chinese.extend(coefficient_zh)
    chinese.extend(reproduction_zh)
    chinese += ["## RQ1、研究演进和局限", "RQ1先报告原始均值与累计增加/减少/净变化，再按上游延误和计划周转共同支持标准化。共同支持机场、分层和权重由开发期冻结，逐月与4–8月最终描述重复使用；>0、≥5、≥15阈值不能另选有利机场。节点大小是衔接数、颜色是标准化每次净变化、有向边宽是航班量，不能解释为机场责任。"]
    chinese.extend(reference_zh)
    for run_id in [config["development_rq1_run_id"], config["final_rq1_run_id"]]:
        matching = [item for item in archive.evidence if item["run_id"] == run_id and item["kind"] == "conclusion"]
        chinese.append(f"{run_id} 的原始、共同支持、月际、阈值和网络证据见 " + " ".join(f"[{item['evidence_id']}]" for item in matching) + "；原始CSV筛选条件在evidence_index.csv。")
        raw = archive.table(run_id, "tables/rq1_airport_raw.csv", ["n", "mean_delta", "s_plus", "s_minus", "s_net"])
        adjusted = archive.table(run_id, "rq1_standardized/rq1_airports.csv", ["threshold", "supported", "standardized_mean", "retained_n"])
        adjusted = adjusted.loc[adjusted.threshold.eq("ge15") & flag(adjusted.supported)]
        chinese.append(f"{run_id} 原始主样本{int(raw.n.sum()):,}条，平均每次净变化{number((raw.n * raw.mean_delta).sum() / raw.n.sum())}分钟；"
                       f"累计增加{number(raw.s_plus.sum(), 0)}、累计减少{number(raw.s_minus.sum(), 0)}、累计净变化{number(raw.s_net.sum(), 0)}样本分钟。"
                       f"共同支持下可比较{len(adjusted)}个机场，标准化净变化范围{number(adjusted.standardized_mean.min())}至{number(adjusted.standardized_mean.max())}分钟。")
        chinese.append(f"共同支持格实际保留{int(adjusted.retained_n.sum()):,}条，占全机场原始主样本{number(100 * adjusted.retained_n.sum() / raw.n.sum(), 2)}%；"
                       "这与所选各机场内部保留率是不同分母。单月支持的机场集合可能变化，跨机场范围不等于同机场月际趋势；无支持月份仍为NA。")
    chinese += ["主线固定OLS并按预定顺序增量；variant替换和NOAA补缺独立比较。固定特征之后才比较4组决策树与2组随机森林。共同天气覆盖子集只比较已有预测，不重新训练；源报文可用不等于每个天气变量齐全。每轮无增益、失败和失效都保留，不能只讲最终最好模型。",
                "这是观察到有效、已完成、同机相邻衔接条件下的回顾性预测，不是全航班实时部署证明。METAR观测后5分钟和TAF归档/发布时间都是可用性假设/代理，NOAA精确质量码筛选不等于官方全QC认证。静态机队参考、计划信息事前可得假设、共同支持外推广、单季节和非因果解释都限制结论。",
                "新学方法包括as-of连接、共同支持标准化和共同日期块区间。Li等用于传播视角与buffer定义，Chauhan等用于网络口径区分，Berkeley用于预测时点与逐步实验组织；并未复现三项工作的结论。",
                "英文正文采用连续行文。Markdown只提供Times New Roman 11pt排版元数据；25页、字体、图表可读性仍须PDF渲染后逐页核验。任务书另要求约10分钟技术视频并避免AI配音，本构建器不生成视频或代替人声讲解。"]
    for _, row in archive.index.loc[archive.index.status.isin(["failed", "invalidated"])].iterrows():
        ids = [item["evidence_id"] for item in archive.evidence if item["run_id"] == row.run_id and item["kind"] == "research_history"]
        chinese.append(f"研究演进：{row.run_id}状态{row.status}，改动“{row.change}”，决定“{row.decision}”。" + " ".join(f"[{item}]" for item in ids))
    return english, "\n\n".join(chinese) + "\n"


def build_report(config, *, root=ROOT):
    """Validate everything before creating a new presentation-only output dir."""
    archive = Archive(root)
    output = (archive.root / config.get("output_dir", "reports")).resolve()
    if not output.is_relative_to(archive.root / "reports") or output.exists():
        raise ValueError("Report output must be a new directory within reports/")
    for parent in output.parents:
        manifest = parent / "manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8-sig")).get("status") == "completed":
            raise ValueError("Report must not modify a completed archive")
    validated = validate_inputs(archive, config)
    # Reference receipts supplement the already verified local literature record.
    for name in ["references/ind5003-2610_projects.pdf", "references/papers/1-s2.0-S1366554524001169-main.pdf", "references/papers/2211.08140v2.pdf"]:
        archive.record(name)
    english, chinese = compose(archive, config, validated)
    # Keep the prose deliberately compact; page count can only be certified after
    # actual rendering, including figures. Never claim a Markdown page estimate.
    words = len(english.split())
    if words > 7500:
        raise ValueError(f"Report exceeds the compact prose budget ({words} words)")
    # Recheck every consumed source before publication, catching concurrent edits.
    for item in list(archive.receipts.values()):
        archive.record(item["path"], item)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report_en.md").write_text(english, encoding="utf-8")
    (output / "结果解读_中文.md").write_text(chinese, encoding="utf-8")
    with (output / "evidence_index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(archive.evidence[0]))
        writer.writeheader()
        writer.writerows(archive.evidence)
    for source, destination in archive.figures:
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
               "sources": list(archive.receipts.values()), "english_word_count": words,
               "format_requirements": {"font": "Times New Roman", "minimum_points": 10,
                                       "body_page_limit": 25, "pdf_render_verified": False,
                                       "video_minutes": 10, "human_narration_required": True},
               "data_access": {"raw": False, "pickle": False, "prediction_rows": False, "fit": False},
               "outputs": [{"path": path.relative_to(output).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size}
                           for path in sorted(output.rglob("*")) if path.is_file()]}
    (output / "source_receipts.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": str(output), "evidence_items": len(archive.evidence), "english_words": words, "pdf_verified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    print(json.dumps(build_report(settings), ensure_ascii=False, indent=2))
