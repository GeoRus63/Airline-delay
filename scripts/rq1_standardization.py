"""Frozen common-support RQ1 estimates and paired UTC-date block uncertainty.

Consumes accepted connections, never selects airports using observed outcomes.
Not wired into the pipeline: freeze on development data before final evaluation.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = ("gt0", "ge5", "ge15")


def load_protocol(path=ROOT / "configs/rq1_protocol.json"):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def prepare(source, protocol, development=True, period=None):
    """Apply UTC window before outcomes; keep >0 to support all three thresholds."""
    time = pd.to_datetime(source.upstream_actual_arr_utc, utc=True, errors="coerce")
    start = pd.Timestamp(protocol["development_window"][0])
    end = pd.Timestamp(protocol["development_window"][1] if development else "2026-09-01T00:00:00Z")
    if period is not None:
        requested_start, requested_end = map(pd.Timestamp, period)
        if requested_start.tzinfo is None or requested_end.tzinfo is None:
            raise ValueError("Period bounds must have explicit timezones")
        requested_start, requested_end = requested_start.tz_convert("UTC"), requested_end.tz_convert("UTC")
        if not (start <= requested_start < requested_end <= end):
            raise ValueError("Requested period lies outside the permitted UTC scope")
        if requested_start != requested_start.floor("D") or requested_end != requested_end.floor("D"):
            raise ValueError("Bootstrap periods must use UTC midnight boundaries")
        start, end = requested_start, requested_end
    data = source.loc[time.ge(start) & time.lt(end)].copy()
    data["prediction_time"] = time.loc[data.index]
    if data.connection_id.isna().any() or not data.connection_id.is_unique:
        raise ValueError("Nonmissing unique connection IDs required")
    if "main_analysis_eligible" in data and not data.main_analysis_eligible.eq(True).all():
        raise ValueError("Pass accepted connections only")
    overnight = data.overnight_scheduled.astype("string").str.lower()
    if not overnight.isin(["true", "false"]).all():
        raise ValueError("Invalid overnight flag")
    for col in ["upstream_arr_delay_minutes", "downstream_dep_delay_minutes", "scheduled_turnaround_minutes", "actual_ground_minutes"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    valid = (~overnight.eq("true") & data.airport_id.notna()
             & np.isfinite(data.upstream_arr_delay_minutes) & data.upstream_arr_delay_minutes.gt(0)
             & np.isfinite(data.downstream_dep_delay_minutes) & data.downstream_dep_delay_minutes.ge(0)
             & np.isfinite(data.actual_ground_minutes) & data.actual_ground_minutes.ge(0))
    data = data.loc[valid].copy()
    delay_edges = [float(x) for x in protocol["master_delay_bins"]]
    turn_edges = [float(x) for x in protocol["turnaround_bins"]]
    delay_bin = pd.cut(data.upstream_arr_delay_minutes, delay_edges, right=False, labels=False)
    turn_bin = pd.cut(data.scheduled_turnaround_minutes, turn_edges, right=False, labels=False)
    data = data.loc[delay_bin.notna() & turn_bin.notna()].copy()
    data["cell"] = (delay_bin.loc[data.index] * (len(turn_edges) - 1) + turn_bin.loc[data.index]).astype(int)
    data["airport_id"] = data.airport_id.astype(int)
    data["date"] = data.prediction_time.dt.floor("D")
    data["delta"] = data.downstream_dep_delay_minutes - data.upstream_arr_delay_minutes
    return data, pd.date_range(start, end, freq="D", inclusive="left")


def threshold_rows(data, threshold):
    if threshold not in THRESHOLDS:
        raise ValueError(threshold)
    return data.loc[data.upstream_arr_delay_minutes.gt(0) if threshold == "gt0"
                    else data.upstream_arr_delay_minutes.ge(int(threshold[2:]))]


def _cells_for_threshold(cells, protocol, threshold):
    width = len(protocol["turnaround_bins"]) - 1
    minimum = {"gt0": 0, "ge5": 5, "ge15": 15}[threshold]
    return [cell for cell in cells if float(protocol["master_delay_bins"][cell // width]) >= minimum]


def freeze_reference(source, protocol=None):
    """Select common airports/cells using counts only and freeze pooled weights.

    All thresholds condition the same master reference, not three separately
    selected airport cohorts. Support must pass for every registered threshold.
    """
    protocol = load_protocol() if protocol is None else protocol
    data, _ = prepare(source, protocol, development=True)
    main = threshold_rows(data, "ge15")
    support = main.groupby("airport_id").agg(n=("connection_id", "size"), days=("date", "nunique"))
    eligible = support.loc[support.n.ge(protocol["airport_min_main_rows"])
                           & support.days.ge(protocol["airport_min_prediction_days"])]
    eligible = eligible.reset_index().sort_values(["n", "airport_id"], ascending=[False, True])
    airport_order = eligible.airport_id.tolist()
    counts = data.groupby(["airport_id", "cell"]).agg(n=("connection_id", "size"), days=("date", "nunique"))
    attempts, selected, selected_cells, references = [], [], [], {}
    seen = set()
    for limit in [None] + protocol["fallback_airport_limits"]:
        airports = airport_order if limit is None else airport_order[:limit]
        identity = tuple(airports)
        # Keep all prespecified attempts, even when a top-k equals the full set.
        duplicate_attempt = identity in seen
        seen.add(identity)
        cells = []
        if len(airports) >= 2:
            qualifying = counts.loc[counts.index.get_level_values("airport_id").isin(airports)]
            qualifying = qualifying.loc[qualifying.n.ge(protocol["cell_min_rows"])
                                        & qualifying.days.ge(protocol["cell_min_prediction_days"])]
            coverage = qualifying.reset_index().groupby("cell").airport_id.nunique()
            cells = sorted(int(c) for c in coverage.index[coverage.eq(len(airports))])
        threshold_support = {}
        ok = len(airports) >= 2
        for threshold in THRESHOLDS:
            chosen_cells = _cells_for_threshold(cells, protocol, threshold)
            subset = threshold_rows(data.loc[data.airport_id.isin(airports)], threshold)
            total = subset.groupby("airport_id").size().reindex(airports, fill_value=0)
            retained = subset.loc[subset.cell.isin(chosen_cells)].groupby("airport_id").size().reindex(airports, fill_value=0)
            shares = retained.div(total.replace(0, np.nan))
            passed = bool(chosen_cells) and shares.ge(protocol["minimum_retained_share"]).all()
            ok &= passed
            threshold_support[threshold] = {"cells": chosen_cells, "retained_shares": {str(k): float(v) for k, v in shares.items()}, "passed": bool(passed)}
        attempts.append({"limit": limit, "airport_ids": airports, "duplicate_airport_set": duplicate_attempt,
                         "common_master_cells": cells, "threshold_support": threshold_support, "passed": bool(ok)})
        if ok:
            selected, selected_cells = airports, cells
            pooled = data.loc[data.airport_id.isin(selected) & data.cell.isin(cells)].groupby("cell").size()
            for threshold in THRESHOLDS:
                conditional_cells = _cells_for_threshold(cells, protocol, threshold)
                conditioned = pooled.reindex(conditional_cells)
                references[threshold] = {"cells": conditional_cells, "weights": (conditioned / conditioned.sum()).tolist(),
                                         "development_counts": conditioned.astype(int).tolist()}
            break
    return {"schema_version": 1, "protocol": protocol, "frozen_on": "development_only",
            "airport_ids": selected, "master_cells": selected_cells, "references": references,
            "selection_attempts": attempts, "supported": bool(selected),
            "interpretation": "Descriptive common-support differences; not causal airport effects."}


def moving_block_indices(n_days, replicates=1000, block_days=7, seed=42):
    """Non-circular contiguous calendar blocks, truncated to original day count."""
    if n_days < block_days or block_days < 1:
        raise ValueError("Calendar window must span at least one full block")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_days - block_days + 1, size=(replicates, int(np.ceil(n_days / block_days))))
    return (starts[..., None] + np.arange(block_days)).reshape(replicates, -1)[:, :n_days]


def moving_block_counts(n_days, bootstrap):
    indices = moving_block_indices(n_days, bootstrap["replicates"], bootstrap["block_days"], bootstrap["seed"])
    counts = np.zeros((len(indices), n_days), dtype=np.int32)
    np.add.at(counts, (np.arange(len(indices))[:, None], indices), 1)
    return counts


def _interval(values, minimum):
    finite = values[np.isfinite(values)]
    return (float(np.quantile(finite, .025)), float(np.quantile(finite, .975)), len(finite)) if len(finite) >= minimum else (np.nan, np.nan, len(finite))


def evaluate(source, frozen, development=True, period=None):
    """Return tables plus paired replicate estimates; never refit frozen support."""
    protocol = frozen["protocol"]
    data, calendar = prepare(source, protocol, development, period=period)
    bootstrap = protocol["bootstrap"]
    draws = moving_block_counts(len(calendar), bootstrap)
    airports = frozen["airport_ids"]
    tables, differences, replicate_tables, cell_tables = [], [], {}, []
    for threshold in THRESHOLDS:
        subset = threshold_rows(data, threshold).copy()
        subset["plus"] = subset.delta.clip(lower=0)
        subset["minus"] = -subset.delta.clip(upper=0)
        for name, indicator in [("reduction", subset.delta.lt(0)), ("amplification", subset.delta.gt(0)), ("unchanged", subset.delta.eq(0))]:
            subset[name] = indicator.astype(float)
        cell_summary = subset.groupby(["airport_id", "cell"]).agg(
            n=("delta", "size"), prediction_days=("date", "nunique"), mean_delta=("delta", "mean"),
            reduction_share=("reduction", "mean"), amplification_share=("amplification", "mean"), unchanged_share=("unchanged", "mean"))
        master_size = (len(protocol["master_delay_bins"]) - 1) * (len(protocol["turnaround_bins"]) - 1)
        full_cells = _cells_for_threshold(list(range(master_size)), protocol, threshold)
        support_grid = pd.MultiIndex.from_product([sorted(set(subset.airport_id) | set(airports)), full_cells], names=["airport_id", "cell"])
        cell_summary = cell_summary.reindex(support_grid)
        cell_summary[["n", "prediction_days"]] = cell_summary[["n", "prediction_days"]].fillna(0).astype(int)
        cell_summary = cell_summary.reset_index()
        cell_summary["threshold"] = threshold
        cell_summary["selected_airport"] = cell_summary.airport_id.isin(airports)
        ref_cells = frozen["references"].get(threshold, {})
        weight_map = dict(zip(ref_cells.get("cells", []), ref_cells.get("weights", [])))
        cell_summary["reference_weight"] = cell_summary.cell.map(weight_map)
        cell_summary["in_frozen_common_support"] = cell_summary.selected_airport & cell_summary.cell.isin(weight_map)
        cell_summary["cell_support_passed"] = cell_summary.n.ge(protocol["cell_min_rows"]) & cell_summary.prediction_days.ge(protocol["cell_min_prediction_days"])
        width = len(protocol["turnaround_bins"]) - 1
        cell_summary["delay_bin_lower"] = cell_summary.cell.map(lambda c: protocol["master_delay_bins"][c // width])
        cell_summary["delay_bin_upper"] = cell_summary.cell.map(lambda c: protocol["master_delay_bins"][c // width + 1])
        cell_summary["turnaround_bin_lower"] = cell_summary.cell.map(lambda c: protocol["turnaround_bins"][c % width])
        cell_summary["turnaround_bin_upper"] = cell_summary.cell.map(lambda c: protocol["turnaround_bins"][c % width + 1])
        cell_tables.append(cell_summary)
        raw = subset.groupby("airport_id").agg(n=("connection_id", "size"), prediction_days=("date", "nunique"),
              mean_delta=("delta", "mean"), s_plus=("plus", "sum"), s_minus=("minus", "sum"), s_net=("delta", "sum"))
        raw = raw.reindex(sorted(set(raw.index) | set(airports)))
        raw["threshold"] = threshold
        raw["selected_airport"] = raw.index.isin(airports)
        raw["retained_n"] = 0
        raw["retained_share"] = np.nan
        raw["supported"] = False
        raw["standardized_mean"] = np.nan
        for name in ["reduction", "amplification", "unchanged"]:
            raw["standardized_" + name + "_share"] = np.nan
        raw["ci_low"] = np.nan
        raw["ci_high"] = np.nan
        raw["valid_replicates"] = 0
        raw["support_reason"] = "outside_frozen_airport_set"
        raw.loc[raw.selected_airport, "support_reason"] = "frozen_cell_support_insufficient"
        # Raw cumulative totals have their own units and calendar-block intervals.
        for metric in ["s_plus", "s_minus", "s_net"]:
            raw[metric + "_ci_low"] = np.nan
            raw[metric + "_ci_high"] = np.nan
        if len(raw):
            daily_raw = subset.groupby(["date", "airport_id"])[["plus", "minus", "delta"]].sum()
            grid_raw = pd.MultiIndex.from_product([calendar, raw.index], names=["date", "airport_id"])
            matrix_raw = daily_raw.reindex(grid_raw, fill_value=0).to_numpy().reshape(len(calendar), -1)
            sampled_raw = (draws @ matrix_raw).reshape(len(draws), len(raw), 3)
            for j, metric in enumerate(["s_plus", "s_minus", "s_net"]):
                raw[metric + "_ci_low"] = np.quantile(sampled_raw[:, :, j], .025, axis=0)
                raw[metric + "_ci_high"] = np.quantile(sampled_raw[:, :, j], .975, axis=0)
                raw.loc[raw.n.isna(), [metric + "_ci_low", metric + "_ci_high"]] = np.nan
        ref = frozen["references"].get(threshold)
        samples = np.full((len(draws), len(airports)), np.nan)
        if ref and airports:
            cells, weights = ref["cells"], np.asarray(ref["weights"])
            eligible = subset.loc[subset.airport_id.isin(airports) & subset.cell.isin(cells)]
            grid = pd.MultiIndex.from_product([airports, cells], names=["airport_id", "cell"])
            grouped = eligible.groupby(["airport_id", "cell"]).agg(n=("delta", "size"), days=("date", "nunique"), total=("delta", "sum"))
            grouped = grouped.reindex(grid, fill_value=0)
            cell_n = grouped.n.to_numpy().reshape(len(airports), len(cells))
            cell_days = grouped.days.to_numpy().reshape(cell_n.shape)
            means = np.divide(grouped.total.to_numpy().reshape(cell_n.shape), cell_n,
                              out=np.full(cell_n.shape, np.nan), where=cell_n > 0)
            retained = cell_n.sum(axis=1)
            shares = retained / raw.loc[airports, "n"].to_numpy()
            supported = ((cell_n >= protocol["cell_min_rows"]).all(axis=1)
                         & (cell_days >= protocol["cell_min_prediction_days"]).all(axis=1)
                         & (shares >= protocol["minimum_retained_share"]))
            raw.loc[airports, "retained_n"] = retained
            raw.loc[airports, "retained_share"] = shares
            raw.loc[airports, "supported"] = supported
            raw.loc[airports, "standardized_mean"] = np.where(supported, means @ weights, np.nan)
            cell_rates = eligible.groupby(["airport_id", "cell"])[["reduction", "amplification", "unchanged"]].mean().reindex(grid)
            for name in ["reduction", "amplification", "unchanged"]:
                values = cell_rates[name].to_numpy().reshape(len(airports), len(cells)) @ weights
                raw.loc[airports, "standardized_" + name + "_share"] = np.where(supported, values, np.nan)
            raw.loc[np.asarray(airports)[supported], "support_reason"] = "supported"
            daily = eligible.groupby(["date", "airport_id", "cell"]).delta.agg(["size", "sum"])
            day_grid = pd.MultiIndex.from_product([calendar, airports, cells], names=["date", "airport_id", "cell"])
            daily = daily.reindex(day_grid, fill_value=0)
            sampled_n = (draws @ daily["size"].to_numpy().reshape(len(calendar), -1)).reshape(len(draws), len(airports), len(cells))
            sampled_sum = (draws @ daily["sum"].to_numpy().reshape(len(calendar), -1)).reshape(sampled_n.shape)
            sampled_means = np.divide(sampled_sum, sampled_n, out=np.full(sampled_sum.shape, np.nan), where=sampled_n > 0)
            samples = sampled_means @ weights
            samples[:, ~supported] = np.nan
            for j, airport in enumerate(airports):
                low, high, valid = _interval(samples[:, j], bootstrap["min_valid_replicates"])
                raw.loc[airport, ["ci_low", "ci_high", "valid_replicates"]] = [low, high, valid]
            for left, right in combinations(range(len(airports)), 2):
                low, high, valid = _interval(samples[:, left] - samples[:, right], bootstrap["min_valid_replicates"])
                differences.append({"threshold": threshold, "airport_left": airports[left], "airport_right": airports[right],
                                    "mean_difference": raw.loc[airports[left], "standardized_mean"] - raw.loc[airports[right], "standardized_mean"],
                                    "ci_low": low, "ci_high": high, "valid_replicates": valid})
        raw["rank"] = raw.standardized_mean.rank(method="average")
        replicate_tables[threshold] = samples
        tables.append(raw.rename_axis("airport_id").reset_index())
    results = pd.concat(tables, ignore_index=True)
    stability = []
    main = results.loc[results.threshold.eq("ge15")].set_index("airport_id")
    for threshold in THRESHOLDS:
        other = results.loc[results.threshold.eq(threshold)].set_index("airport_id")
        joined = main[["standardized_mean"]].join(other[["standardized_mean"]], lsuffix="_main", rsuffix="_other", how="inner").dropna()
        rank_corr = joined.standardized_mean_main.rank().corr(joined.standardized_mean_other.rank()) if len(joined) >= 2 and joined.nunique().min() > 1 else np.nan
        stability.append({"threshold": threshold, "common_airports": len(joined), "frozen_airports": len(airports),
                          "common_airport_share": len(joined) / len(airports) if airports else np.nan,
                          "direction_agreement": float(np.sign(joined.standardized_mean_main).eq(np.sign(joined.standardized_mean_other)).mean()) if len(joined) else np.nan,
                          "spearman": rank_corr})
    difference_columns = ["threshold", "airport_left", "airport_right", "mean_difference", "ci_low", "ci_high", "valid_replicates"]
    return {"airports": results, "cells": pd.concat(cell_tables, ignore_index=True), "pairwise_differences": pd.DataFrame(differences, columns=difference_columns), "threshold_stability": pd.DataFrame(stability),
            "replicates": replicate_tables, "bootstrap_day_counts": draws, "calendar": calendar}


def network_tables(flights, accepted_connections, airport_results, development=True, reference_path=ROOT / "references/airports.csv"):
    """Edges count BTS rows; nodes count all accepted same-aircraft connections.

    Flight edge window uses scheduled departure UTC; node window uses upstream
    actual arrival UTC. Node color comes only from supported ge15 estimates.
    """
    start = pd.Timestamp("2026-04-01", tz="UTC")
    end = pd.Timestamp("2026-08-01" if development else "2026-09-01", tz="UTC")
    dep = pd.to_datetime(flights.scheduled_dep_utc, utc=True)
    scoped = flights.loc[dep.ge(start) & dep.lt(end)]
    edges = scoped.groupby(["OriginAirportID", "DestAirportID"]).size().reset_index(name="flight_count")
    edges = edges.rename(columns={"OriginAirportID": "origin_airport_id", "DestAirportID": "destination_airport_id"})
    prediction = pd.to_datetime(accepted_connections.upstream_actual_arr_utc, utc=True)
    node_counts = accepted_connections.loc[prediction.ge(start) & prediction.lt(end)].groupby("airport_id").size().rename("valid_connection_count")
    nodes = pd.read_csv(reference_path)[["airport_id", "airport_code", "latitude", "longitude"]]
    nodes = nodes.merge(node_counts, on="airport_id", how="left", validate="one_to_one")
    nodes["valid_connection_count"] = nodes.valid_connection_count.fillna(0).astype(int)
    main = airport_results.loc[airport_results.threshold.eq("ge15"), ["airport_id", "standardized_mean", "supported"]]
    nodes = nodes.merge(main, on="airport_id", how="left", validate="one_to_one")
    nodes["supported"] = nodes.supported.fillna(False).astype(bool)
    nodes.loc[~nodes.supported, "standardized_mean"] = np.nan
    return nodes, edges


def run(source, output_dir, frozen=None, development=True, protocol=None, period=None):
    """Archive tables/paired replicates; caller supplies accepted rows and manifest.

    Final evaluation requires a previously frozen reference. This function never
    loads or inspects August itself, nor modifies the supplied frozen reference.
    """
    output_dir = Path(output_dir)
    reference_reused = frozen is not None
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite RQ1 evidence: {output_dir}")
    if frozen is None:
        if not development:
            raise ValueError("Final evaluation requires a frozen development reference")
        frozen = freeze_reference(source, protocol)
    result = evaluate(source, frozen, development, period=period)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frozen_reference.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for name in ["airports", "cells", "pairwise_differences", "threshold_stability"]:
        result[name].to_csv(output_dir / f"rq1_{name}.csv", index=False)
    np.savez_compressed(output_dir / "paired_bootstrap.npz", **result["replicates"],
                        day_counts=result["bootstrap_day_counts"], airport_ids=np.asarray(frozen["airport_ids"]),
                        calendar_utc=result["calendar"].strftime("%Y-%m-%d").to_numpy(dtype=str))
    summary = {"development": development, "frozen_reference_reused": reference_reused,
               "period": [str(result["calendar"][0]), str(result["calendar"][-1] + pd.Timedelta(days=1))],
               "selected_airports": len(frozen["airport_ids"]), "supported_reference": frozen["supported"],
               "bootstrap": frozen["protocol"]["bootstrap"], "calendar_days": len(result["calendar"]),
               "sample_rule": "accepted, nonovernight, upstream positive; threshold-specific subsets",
               "uncertainty": "Common calendar moving blocks; fixed airports/cells/weights, no reselection per replicate; percentile intervals are not multiplicity adjusted.",
               "units": {"standardized_mean": "minutes per connection", "s_plus_s_minus_s_net": "sample cumulative minutes"}}
    (output_dir / "rq1_standardization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
