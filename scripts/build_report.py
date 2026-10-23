"""Build P7 prose and its evidence index from completed lightweight archives only.

No training, raw-data, pickle or prediction-file access occurs here. Invoke only
after P6: python scripts/build_report.py --config configs/report.json
The JSON keys are mainline_run_ids (G0/G1/G2/G3/G4A/G4B dictionary),
variant_run_id, noaa_run_id, algorithm_run_ids (the four DT and two RF runs),
development_rq1_run_id, feature_selection_run_id, model_selection_run_id,
final_prediction_run_id, final_rq1_run_id, and optional output_dir (reports).
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

    def figure(self, run_id, filename, csv_file, selector, caption, code):
        path = self.source(run_id, filename)
        ref = self.cite(run_id, csv_file, selector, caption, "figure")
        self.cite(run_id, filename, {"whole_file": True, "plotting_code": code}, caption, "figure_image")
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
    for name in [model["model_name"], *BASELINES]:
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
    table = ["| Month | Delay measure | n | Mean | q50 | q90 | q99 | Evidence |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    labels = {"DepDelay": "Departure raw", "ArrDelay": "Arrival raw", "DepDelayMinutes": "Departure nonnegative", "ArrDelayMinutes": "Arrival nonnegative"}
    for month in months:
        operation = selected_row(monthly_ops, group_value=month)
        for outcome, label in labels.items():
            selector = {"dimension": "month", "group_value": month, "outcome": outcome}
            row = selected_row(stats, **selector)
            if int(row.normal_completed_flights) != int(operation.completed) or int(row.n + row.missing) != int(operation.completed):
                raise ValueError("Single-flight observed/missing outcome counts do not reconcile")
            ref = archive.cite(run_id, stats_file, selector, "Monthly signed/nonnegative single-flight delay quantiles; minutes")
            table.append(f"| {month} | {label} | {int(row.n):,} | {number(row['mean'], 2)} | {number(row.q50, 2)} | {number(row.q90, 2)} | {number(row.q99, 2)} | {ref} |")
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


def rq1_prose(archive, run_id, label):
    rawfile = "tables/rq1_airport_raw.csv"
    raw = archive.table(run_id, rawfile, ["airport_id", "airport_code", "n", "mean_delta", "s_plus", "s_minus", "s_net"])
    n = raw.n.sum()
    if n <= 0:
        raise ValueError("RQ1 raw sample is empty")
    means = (raw.n * raw.mean_delta).sum() / n
    ref = archive.cite(run_id, rawfile, {"rows": "all", "aggregation": "sum(n*mean_delta)/sum(n); sum(n,s_plus,s_minus,s_net)"}, "Raw RQ1 sample and cumulative minutes")
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
        text.append(archive.figure(run_id, "figures/" + filename, table, {"rows": "all"}, caption, "scripts/complete_rq1.py"))
    return "\n\n".join(text)


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
              "TAF uses the latest archive bulletin state available at prediction time and queries the later of scheduled departure and prediction time. Archive/issue timestamps are availability proxies, not verified historical delivery receipts; TEMPO, probability, amendment, cancellation and NIL states must remain distinct. "
              "NOAA is a separate retrospective supplement: exact quality-code whitelisting does not establish full official QC certification or contemporaneous delivery, and conflicting or structural missing values are not automatically filled. "
              "Weather absence retains the flight with explicit missingness rather than deleting it or imputing clear weather.",
              "## RQ1: airport absorption and amplification",
              "For a connection, net change is downstream nonnegative departure delay minus upstream nonnegative arrival delay. A negative change means fewer delay minutes, not necessarily punctual departure. "
              "The main population is nonovernight with upstream delay at least 15 minutes. Airport comparison adjusts only for upstream-delay and scheduled-turnaround cells within frozen common support. "
              "Common airports, cells and reference weights are held fixed across months and final reporting; uncertainty uses common date blocks and is descriptive, not causal or multiplicity-adjusted.",
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
              comparison_text(archive, main[-1], noaa)]
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
    chinese += ["## RQ1、研究演进和局限", "RQ1先报告原始均值与累计增加/减少/净变化，再按上游延误和计划周转共同支持标准化。共同支持机场、分层和权重由开发期冻结，逐月与4–8月最终描述重复使用；>0、≥5、≥15阈值不能另选有利机场。节点大小是衔接数、颜色是标准化每次净变化、有向边宽是航班量，不能解释为机场责任。"]
    for run_id in [config["development_rq1_run_id"], config["final_rq1_run_id"]]:
        matching = [item for item in archive.evidence if item["run_id"] == run_id and item["kind"] == "conclusion"]
        chinese.append(f"{run_id} 的原始、共同支持、月际、阈值和网络证据见 " + " ".join(f"[{item['evidence_id']}]" for item in matching) + "；原始CSV筛选条件在evidence_index.csv。")
        raw = archive.table(run_id, "tables/rq1_airport_raw.csv", ["n", "mean_delta", "s_plus", "s_minus", "s_net"])
        adjusted = archive.table(run_id, "rq1_standardized/rq1_airports.csv", ["threshold", "supported", "standardized_mean"])
        adjusted = adjusted.loc[adjusted.threshold.eq("ge15") & flag(adjusted.supported)]
        chinese.append(f"{run_id} 原始主样本{int(raw.n.sum()):,}条，平均每次净变化{number((raw.n * raw.mean_delta).sum() / raw.n.sum())}分钟；"
                       f"累计增加{number(raw.s_plus.sum(), 0)}、累计减少{number(raw.s_minus.sum(), 0)}、累计净变化{number(raw.s_net.sum(), 0)}样本分钟。"
                       f"共同支持下可比较{len(adjusted)}个机场，标准化净变化范围{number(adjusted.standardized_mean.min())}至{number(adjusted.standardized_mean.max())}分钟。")
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
