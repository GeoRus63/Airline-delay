"""Compose the 600-second presentation from the report's verified small archives.

Usage: python scripts/build_presentation_content.py REPORT_CONFIG.json NEW_CONTENT.json
No raw inputs, processed data, predictions, pickles or model objects are read.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from scripts import build_report as report
except ModuleNotFoundError:
    import build_report as report


def build_content(config, output_path, *, root=report.ROOT):
    archive = report.Archive(root)
    validated = report.validate_inputs(archive, config)
    main, variant, noaa, algorithms, feature, choice, final = validated
    output = Path(output_path)
    output = (output if output.is_absolute() else archive.root / output).resolve()
    if not output.is_relative_to(archive.root) or output.suffix.lower() != ".json":
        raise ValueError("Content JSON must be inside the project")
    receipts_path = output.with_name(output.stem + ".sources.json")
    evidence_path = output.with_name(output.stem + ".evidence.csv")
    if any(p.exists() for p in [output, receipts_path, evidence_path]):
        raise FileExistsError("Use new content, source receipt and evidence paths")

    def cite(run_id, filename, selector, claim):
        archive.cite(run_id, filename, selector, claim)
        item = archive.evidence[-1]
        return (f"{run_id}: {item['source_file']}; selector={item['selector']}; "
                f"SHA256={item['source_sha256']}; research journal headings={item['research_log_heading_lines']}")

    def metric(run, model=None, **filters):
        selected = dict(model=model or run["model"]["model_name"], fold="overall", group="all", value="all")
        selected.update(filters)
        row = report.selected_row(run["metrics"], **selected)
        return row, cite(run["id"], "metrics.csv", selected, "Presentation error metric and support")

    def paired(reference, candidate):
        report.comparison_text(archive, reference, candidate)
        row = report.selected_row(archive.table(candidate["id"], "comparison/comparison_metrics.csv", report.COMPARISON_COLUMNS), scope="pooled", metric="mae")
        return row, cite(candidate["id"], "comparison/comparison_metrics.csv", {"scope": "pooled", "metric": "mae"}, "Paired development MAE difference")

    def fig(run_id, filename, table, selector, meaning):
        source = archive.source(run_id, filename)
        return {"path": source.relative_to(archive.root).as_posix(), "alt": meaning}, [
            cite(run_id, filename, {"whole_file": True}, meaning), cite(run_id, table, selector, meaning)]

    # Verify the frozen A/C/w and every fallback, rather than trusting a count alone.
    report.rq1_reference_prose(archive, config["development_rq1_run_id"])
    rqid = config["final_rq1_run_id"]
    frozen = archive.json(rqid, "rq1_standardized/frozen_reference.json")
    airport_table = archive.table(rqid, "rq1_standardized/rq1_airports.csv", ["airport_id", "threshold", "supported", "retained_n", "retained_share", "standardized_mean", "ci_low", "ci_high"])
    supported = airport_table.loc[airport_table.threshold.eq("ge15") & report.flag(airport_table.supported)]
    raw = archive.table(rqid, "tables/rq1_airport_raw.csv", ["airport_id", "n", "mean_delta", "s_net"])
    rq_n = int(raw.n.sum())
    if rq_n <= 0:
        raise ValueError("RQ1 main population must be nonempty")
    rq_mean = float((raw.n * raw.mean_delta).sum() / rq_n)
    rq_ref = cite(rqid, "rq1_standardized/rq1_airports.csv", {"threshold": "ge15", "supported": True}, "Final supported airport net-delay estimates")
    raw_ref = cite(rqid, "tables/rq1_airport_raw.csv", {"aggregation": "sum(n);sum(n*mean_delta)/sum(n);sum(s_net)"}, "Final raw connection support and net change")
    frozen_ref = cite(rqid, "rq1_standardized/frozen_reference.json", {"keys": ["airport_ids", "master_cells", "references"]}, "Unchanged development airport/cell/weight reference")
    primary, primary_ref = metric(final, choice["primary_model_name"])
    final_rows = [metric(final, name)[0] for name in [final["model"]["model_name"], *report.BASELINES]]
    final_metric_refs = [metric(final, name)[1] for name in [final["model"]["model_name"], *report.BASELINES]]
    selected_ref = cite(config["model_selection_run_id"], "selection/model_selection_decision.json", {"keys": ["selected_run_id", "primary_model_name", "reason", "promotion_decisions"]}, "Frozen learned model and primary predictor")
    fmt = report.number
    label = lambda name: {"training_mean": "Training mean", "training_median": "Training median", "carry_forward": "Upstream delay"}.get(name, name)
    primary_label = label(choice["primary_model_name"])
    range_text = (f"{fmt(supported.standardized_mean.min())} to {fmt(supported.standardized_mean.max())} minutes"
                  if len(supported) else "not estimable under the frozen support rule")
    slides = []

    def add(title, claims, notes, duration, citations, **evidence):
        slides.append(dict(title=title, claims=claims, notes=notes, duration=duration, citations=list(dict.fromkeys(citations)), **evidence))

    add(f"{primary_label}: August MAE {fmt(primary.mae)} min", [
        f"RQ2: {int(primary.n):,} final connections, RMSE {fmt(primary.rmse)} min",
        f"RQ1: {int(supported.standardized_mean.lt(0).sum())}/{len(supported)} supported estimates show fewer delay minutes",
        "Retrospective evidence, conditional on completed connections"],
        f"The main prediction result is an August mean absolute error of {fmt(primary.mae)} minutes for the frozen primary predictor, {primary_label}. "
        f"Its root mean squared error is {fmt(primary.rmse)} minutes over {int(primary.n):,} accepted connections. The airport analysis supports {len(supported)} estimates under the unchanged development reference. "
        "These are the two findings around which I will explain the project. A simple baseline was allowed to win, so model complexity is not a success criterion. "
        "The analysis is retrospective and describes valid observed completed connections. It does not assign causal responsibility to airports, and the final test result did not trigger another round of tuning.",
        50, [primary_ref, rq_ref, selected_ref])

    add("Prediction time and research questions", [
        "RQ1: downstream minus upstream delay, with upstream delay ≥15 min",
        "RQ2: downstream nonnegative departure delay, including upstream on-time flights",
        "UTC forward validation, then one frozen August assessment"],
        "The first question describes how delay changes across consecutive flights of the same aircraft. For the main airport analysis, upstream delay must reach fifteen minutes and the connection must be nonovernight. "
        "A negative net change means fewer delay minutes downstream, even if the aircraft remains late. The second question predicts nonnegative downstream departure delay at upstream actual arrival. "
        "Its population includes upstream on-time flights, and eligibility requires the prediction time to precede the downstream departure label. Training uses only labels mature before each UTC validation boundary. "
        "The forward folds train through April, May and June and validate the following month. August remains the final frozen assessment.",
        45, [cite(main[0]["id"], "model_manifest.json", {"keys": ["folds", "evaluation_rows"]}, "Forward validation boundaries and mature training labels"), raw_ref])

    eda = archive.json(main[0]["id"], "eda_summary.json")
    add("The prediction sample differs from all flights", [
        f"Development: {int(eda['n']):,} valid prediction connections",
        f"{int(eda['airport_n']):,} airports, {int(eda['route_n']):,} routes",
        f"{int(eda['overnight_n']):,} overnight connections reported separately"],
        f"The development prediction sample contains {int(eda['n']):,} valid connections across {int(eda['airport_n']):,} airports and {int(eda['route_n']):,} routes. "
        f"It includes {int(eda['overnight_n']):,} overnight connections, which receive separate error reporting. These counts describe connections, not the number of individual flights. "
        "Cleaning first follows scheduled same-tail order, then checks actual timing and endpoint continuity. Cancellations, diversions and ambiguous chains define exclusions rather than zero delay. "
        "This completed-connection selection limits operational interpretation because future completion is not guaranteed at prediction time. Planned schedule availability and static aircraft inventories also remain assumptions. "
        "Keeping these population definitions explicit prevents a lower error on a different sample from masquerading as an improvement.",
        50, [cite(main[0]["id"], "eda_summary.json", {"keys": ["n", "airport_n", "route_n", "overnight_n"]}, "Development EDA prediction population")])

    image, image_refs = fig(rqid, "figures/rq1_standardized_airports.png", "rq1_standardized/rq1_airports.csv", {"threshold": "ge15"}, "Frozen common-support airport estimates and date-block uncertainty")
    negative = int(supported.standardized_mean.lt(0).sum())
    add("Airport comparisons use one frozen reference", [
        f"{len(frozen['airport_ids'])} frozen airports, {len(supported)} supported final estimates",
        f"{negative} negative supported net-change estimates"],
        f"The development procedure froze {len(frozen['airport_ids'])} airports and {len(frozen['master_cells'])} common master cells. "
        f"The final main-threshold analysis supports {len(supported)} airport estimates, with standardized means spanning {range_text}. "
        f"Among supported point estimates, {negative} are negative. Across the raw main population of {rq_n:,} connections, the mean change is {fmt(rq_mean)} minutes, which answers a different question from standardized cross-airport comparison. "
        "The standardization reuses the same airport set, cells and threshold-conditional pooled development weights. It does not renormalize separately for a weakly supported airport. "
        "Unsupported estimates remain missing. An airport may lose support in a particular month, so missing months stay NA rather than becoming zero or entering a rank. "
        "The April–August RQ1 summary includes development months and is not an independent August validation.",
        55, [rq_ref, raw_ref, frozen_ref, *image_refs], image=image, caption="Negative change means fewer downstream delay minutes. Unsupported estimates remain NA.")

    nodes = archive.table(rqid, "rq1_standardized/network_nodes.csv", ["airport_id", "valid_connection_count", "standardized_mean"])
    edges = archive.table(rqid, "rq1_standardized/network_edges.csv", ["origin_airport_id", "destination_airport_id", "flight_count"])
    monthly = archive.table(rqid, "rq1_standardized/rq1_airports_monthly.csv", ["month_utc", "threshold", "supported", "standardized_mean"])
    counts = monthly.loc[monthly.threshold.eq("ge15")].groupby("month_utc", sort=True).supported.apply(lambda s: int(report.flag(s).sum()))
    monthly_text = ", ".join(f"{month}: {value}" for month, value in counts.items())
    image, image_refs = fig(rqid, "figures/rq1_network_geographic.png", "rq1_standardized/network_nodes.csv", {"rows": "all"}, "Airport connection performance and directed flight traffic")
    edge_ref = cite(rqid, "rq1_standardized/network_edges.csv", {"aggregation": "count;sum(flight_count)"}, "Directed traffic routes and flight volume")
    month_ref = cite(rqid, "rq1_standardized/rq1_airports_monthly.csv", {"threshold": "ge15", "aggregation": "count(supported) by month_utc"}, "Monthly support under identical frozen reference")
    add("Route traffic and connection outcomes are distinct", [
        f"{len(nodes):,} nodes, {len(edges):,} directed routes",
        f"{int(edges.flight_count.sum()):,} flights represented by route counts"],
        f"The network contains {len(nodes):,} airport nodes and {len(edges):,} directed routes, representing {int(edges.flight_count.sum()):,} flights in the route tables. "
        "Node size shows accepted connection support. Node colour shows the standardized net-delay change when support permits, while gray denotes an unsupported estimate. "
        "Edge width shows flight traffic, not a count of delay-propagation events. The geographic view separates traffic concentration from the connection outcome we actually measured. "
        f"Monthly supported-airport counts are {monthly_text}. These counts may change under the same frozen reference. "
        "Consequently, a range across available airports in one month is not a trend for a fixed airport panel, and the map cannot establish which airport caused a downstream delay.",
        55, [*image_refs, edge_ref, month_ref], image=image, caption="Node colour: supported standardized change. Edge width: traffic, not causal propagation.")

    main_rows, main_refs = zip(*(metric(run) for run in main))
    pair_rows = [paired(left, right) for left, right in zip(main, main[1:])]
    pair_text = " ".join(f"{report.STAGES[i+1]} changed pooled MAE by {fmt(row.difference_candidate_minus_reference)} minutes against {report.STAGES[i]}, with interval {fmt(row.ci95_low)} to {fmt(row.ci95_high)}." for i, (row, _) in enumerate(pair_rows))
    add("Feature increments share the same evaluation sample", [
        f"G0 MAE {fmt(main_rows[0].mae)} min; G4B {fmt(main_rows[-1].mae)} min",
        "OLS, preprocessing and evaluation identities stay fixed"],
        "This chart shows every cumulative feature stage, including steps that fail to improve error. G0 provides the baseline schedule context, followed by aircraft-chain history, aircraft family, airport operating history, observations and TAF. "
        f"G0 has MAE {fmt(main_rows[0].mae)} minutes and G4B has {fmt(main_rows[-1].mae)} minutes. "
        + pair_text + " All paired differences use the same evaluation identities and time splits. Negative candidate-minus-reference differences indicate improvement. "
        "The addition order determines the measured increment. This is not an order-independent ranking of information value, and scheduled turnaround already belongs to G0.",
        55, [*main_refs, *[ref for _, ref in pair_rows]], chart={"type": "bar", "categories": report.STAGES, "series": [{"name": "Development MAE", "values": [float(r.mae) for r in main_rows]}], "yLabel": "MAE (minutes)", "numberFormat": "0.00"})

    variant_row, variant_ref = metric(variant)
    noaa_row, noaa_ref = metric(noaa)
    variant_delta, variant_pair_ref = paired(main[2], variant)
    noaa_delta, noaa_pair_ref = paired(main[-1], noaa)
    coverage = archive.json(main[-1]["id"], "weather_coverage_summary.json")
    availability = "; ".join(f"{key}: {int(value):,}" for key, value in coverage["availability"].items())
    noaa_audit = archive.json(noaa["id"], "feature_audit.json")
    fills = noaa_audit.get("noaa_fill_counts")
    if not isinstance(fills, dict) or not fills or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                                                     or not math.isfinite(v) or v < 0 or int(v) != v for v in fills.values()):
        raise ValueError("NOAA audit requires nonnegative integer noaa_fill_counts")
    total_fills = int(sum(fills.values()))
    model_fills = int(sum(value for key, value in fills.items() if key in noaa["model"]["features"]))
    noaa_interpretation = (
        "No NOAA values entered the selected model columns in this branch. This unchanged value contribution cannot establish that NOAA information has no predictive value. "
        if model_fills == 0 else
        "The fill counts describe field values, not distinct connections. This result is a retrospective supplementation sensitivity, conditional on the selected fields and observed missingness. ")
    noaa_audit_ref = cite(noaa["id"], "feature_audit.json", {"keys": ["noaa_fill_counts", "noaa_fill_assumption", "noaa_qc_limitation", "weather_audit"]}, "NOAA inserted field counts and retrospective availability/QC limitations")
    add("Weather and aircraft detail have separate controls", [
        f"Variant versus family: ΔMAE {fmt(variant_delta.difference_candidate_minus_reference)} min",
        f"NOAA versus IEM G4B: ΔMAE {fmt(noaa_delta.difference_candidate_minus_reference)} min",
        "TAF blocks unverified revisions and delayed values",
        f"NOAA filled {total_fills:,} values, {model_fills:,} in selected model columns"],
        f"The variant replacement changes MAE by {fmt(variant_delta.difference_candidate_minus_reference)} minutes against family. "
        f"NOAA supplementation changes MAE by {fmt(noaa_delta.difference_candidate_minus_reference)} minutes against IEM G4B. "
        f"The archived audit records {total_fills:,} inserted field values, of which {model_fills:,} are in selected model columns. " + noaa_interpretation +
        "These remain separate branches. Weather joins respect the arrival prediction time and declared observation lag. "
        "TAF applies a conservative revision block, excluding unverified amended, corrected or delayed bulletin values. The missing-weather rows remain in the sample. "
        f"G4B records {int(coverage['rows']):,} weather queries. The saved availability counts describe source states, not complete weather vectors or proof of real-time delivery.",
        50, [variant_ref, noaa_ref, variant_pair_ref, noaa_pair_ref, noaa_audit_ref, cite(main[-1]["id"], "weather_coverage_summary.json", {"keys": ["rows", "availability"]}, "Weather query-level availability")])

    source_run = next(run for run in main if run["id"] == feature["selected_run_id"])
    algo_runs = [source_run, *algorithms]
    algo_rows, algo_refs = zip(*(metric(run) for run in algo_runs))
    def algorithm_name(run):
        if run is source_run:
            return "OLS"
        p = run["model"]["model_params"]
        prefix = "DT" if run["model"]["model_type"] == "decision_tree" else "RF"
        return f"{prefix} d{p['max_depth']} l{p['min_samples_leaf']}"
    add("Model selection allows a baseline to win", [
        f"Feature source {feature['selected_run_id']}; learned model {choice['selected_run_id']}",
        f"Frozen primary predictor: {primary_label}"],
        f"The feature-selection decision retained {feature['selected_run_id']}. The six tree and forest candidates then hold that feature set and preprocessing fixed. "
        "The chart includes four decision trees across two depths and two minimum-leaf sizes, plus two random forests at fixed depth. Forests use two hundred trees and one-half of features per split, with seed forty-two. "
        f"The retained learned model is {choice['selected_run_id']}, while the primary predictor is {primary_label}. The recorded reason is: {choice['reason']}. "
        "The practical and monthly-stability promotion rule was committed after P3, before weather and tree results were inspected. It was not registered before P0. "
        "The chosen primary need not be the single lowest-error trial among every exploratory candidate.",
        55, [*algo_refs, selected_ref, cite(config["feature_selection_run_id"], "selection/feature_selection.json", {"keys": ["selected_run_id", "reason"]}, "Feature decision before fixed-feature algorithms")], chart={"type": "bar", "categories": [algorithm_name(r) for r in algo_runs], "series": [{"name": "Development MAE", "values": [float(row.mae) for row in algo_rows]}], "yLabel": "MAE (minutes)", "numberFormat": "0.00"})

    add("August evaluates the frozen primary and comparators", [
        f"Primary MAE {fmt(primary.mae)} min, RMSE {fmt(primary.rmse)} min",
        f"{int(primary.n):,} connections in the UTC August test window"],
        f"This is the final August comparison after the rules, features, preprocessing and primary predictor were frozen. The primary, {primary_label}, records MAE {fmt(primary.mae)} minutes and RMSE {fmt(primary.rmse)} minutes. "
        "The table also preserves the learned model and all three simple comparators. The mean and median baselines are estimated from mature training labels, while carry-forward transfers upstream delay directly. "
        "MAE is the primary decision metric because it measures average absolute minutes. RMSE gives greater weight to large errors. Neither metric alone describes every delay regime. "
        "The final access ledger and saved-result verification identify this test. These slides read small saved summaries only. No test labels are reopened and no model is refitted for this presentation.",
        55, [*final_metric_refs, cite(final["id"], "final_verification.json", {"keys": ["verified", "metrics_recomputed", "refitted", "raw_test_data_read"]}, "Saved final-result verification"), cite(final["id"], "model_selection.json", {"keys": ["payload_sha256", "primary_model_name"]}, "Frozen final selection")], table=[["Predictor", "MAE (min)", "RMSE (min)"], *[[label(row.model), fmt(row.mae), fmt(row.rmse)] for row in final_rows]])

    ordinary, ordinary_ref = metric(final, choice["primary_model_name"], group="severity", value="ordinary_lt60")
    severe, severe_ref = metric(final, choice["primary_model_name"], group="severity", value="severe_ge60")
    residual = report.selected_row(archive.table(final["id"], "residual_summary.csv", ["model", "mean_signed_error", "median_signed_error", "p95_absolute_error"]), model=choice["primary_model_name"])
    add("Error differs across delay regimes", [
        f"Ordinary delay MAE {fmt(ordinary.mae)} min, n={int(ordinary.n):,}",
        f"Severe delay MAE {fmt(severe.mae)} min, n={int(severe.n):,}",
        f"95th percentile absolute error {fmt(residual.p95_absolute_error)} min"],
        f"For ordinary delays below sixty minutes, the primary predictor has MAE {fmt(ordinary.mae)} minutes across {int(ordinary.n):,} connections. "
        f"For severe delays of at least sixty minutes, its MAE is {fmt(severe.mae)} minutes across {int(severe.n):,} connections. These are outcome-defined diagnostic groups, not information used to choose a forecast at arrival. "
        f"The mean signed error, prediction minus target, is {fmt(residual.mean_signed_error)} minutes, and the ninety-fifth percentile absolute error is {fmt(residual.p95_absolute_error)} minutes. "
        "The signs and sample counts help distinguish systematic underprediction from rare large misses. Overall MAE therefore needs this diagnostic context before any operational use.",
        45, [ordinary_ref, severe_ref, cite(final["id"], "residual_summary.csv", {"model": choice["primary_model_name"]}, "Final residual bias and tail")])

    explanation_base = f"explanations/{choice['selected_run_id']}"
    explanation = archive.json(config["model_selection_run_id"], explanation_base + "/explain_summary.json")
    if explanation.get("source_run") != choice["selected_run_id"] or explanation.get("fit_called") is not False:
        raise ValueError("Explanation must use the saved selected learned model without refitting")
    importance = archive.table(config["model_selection_run_id"], explanation_base + "/permutation_summary.csv", ["fold", "group", "mean_delta_mae", "sd_delta_mae", "repeats"])
    top = importance.sort_values(["mean_delta_mae", "fold", "group"], ascending=[False, True, True]).iloc[0]
    add("Interpretation stays conditional on the fitted models", [
        f"Largest displayed permutation change: {top.group}, {fmt(top.mean_delta_mae)} min",
        "Date-block intervals condition on fitted predictions and selection",
        "Airport rankings, permutation and PDP are noncausal"],
        f"On the fixed explanation subsample for {top.fold}, group {top.group} has the largest archived permutation MAE increase, {fmt(top.mean_delta_mae)} minutes, with repeat standard deviation {fmt(top.sd_delta_mae)} over {int(top.repeats)} repeats. "
        "This explains the retained learned model separately from a potentially winning baseline. Joint-group permutation preserves within-group relationships but may break relationships across groups. Repeat standard deviation is not a confidence interval. "
        "Group permutation changes cannot be added together or substituted for staged feature increments. "
        "PDP uses hypothetical input replacements that may leave observed correlated-feature support. Coefficients, importance and airport rankings therefore do not identify causal effects. "
        "The date-block intervals resample saved predictions with common date draws. They do not repeat training or selection, and they are not selection-adjusted. "
        "The evidence also covers one season, conditional completed chains and incomplete weather delivery provenance. NOAA supplementation is retrospective. "
        "Any change made after seeing August requires a genuinely new unseen evaluation period, rather than relabeling the same test as independent validation.",
        65, [cite(config["model_selection_run_id"], explanation_base + "/permutation_summary.csv", {"fold": str(top.fold), "group": str(top.group)}, "Largest archived grouped permutation sensitivity"), selected_ref, frozen_ref])

    add("The final conclusions retain their scope", [
        f"RQ1: {len(supported)} supported estimates under frozen common support",
        f"RQ2: {primary_label}, final MAE {fmt(primary.mae)} min",
        "A new season is needed to test generalization"],
        f"The airport answer is conditional on {len(supported)} supported final estimates and the frozen common reference. "
        f"The prediction answer is the frozen {primary_label} predictor with August MAE {fmt(primary.mae)} minutes. "
        "The useful next step is an independently evaluated new season with stronger availability evidence and explicit operational eligibility, while preserving the archived decisions and unsuccessful experiments.",
        20, [rq_ref, primary_ref, selected_ref])

    content = {"title": "Flight delay propagation and downstream prediction", "duration": 600, "slides": slides}
    words = sum(len(slide["notes"].split()) for slide in slides)
    if len(slides) != 12 or sum(s["duration"] for s in slides) != 600:
        raise AssertionError("Presentation must contain twelve slides totaling 600 seconds")
    if not 1050 <= words <= 1600:
        raise AssertionError(f"Speaker script length outside ten-minute range: {words} words")
    payload = json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)
    source_receipt = {"schema_version": 1, "config_sha256": report.canonical(config), "duration": 600, "slides": len(slides), "speaker_words": words,
                      "raw_or_test_targets_read": False, "model_refit": False, "source_files": list(archive.receipts.values()), "report_config": config}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    receipts_path.write_text(json.dumps(source_receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with evidence_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(archive.evidence[0]))
        writer.writeheader()
        writer.writerows(archive.evidence)
    return {"content": str(output), "sources": str(receipts_path), "evidence": str(evidence_path), "slides": len(slides), "duration": 600, "speaker_words": words}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_config", type=Path)
    parser.add_argument("new_content_json", type=Path)
    args = parser.parse_args()
    config = json.loads(args.report_config.read_text(encoding="utf-8-sig"))
    print(json.dumps(build_content(config, args.new_content_json), indent=2))


if __name__ == "__main__":
    main()
