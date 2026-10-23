"""Paired comparisons on identical archived development predictions and UTC blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PREDICTION_COLUMNS = ["fold", "connection_id", "prediction_time", "target_delay", "predicted_delay", "model"]
KEYS = ["fold", "connection_id"]
STAT_COLUMNS = ["n", "reference_absolute_error_sum", "candidate_absolute_error_sum",
                "reference_squared_error_sum", "candidate_squared_error_sum"]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _completed_run(path, *, allow_running=False):
    path = Path(path).resolve()
    status = _json(path / "manifest.json").get("status")
    if status != "completed" and not (allow_running and status == "running"):
        raise ValueError(f"Comparison requires a completed run: {path.name}")
    if not (path / "metrics.csv").is_file():
        raise ValueError(f"Run model evaluation is not ready: {path.name}/metrics.csv")
    return path, _json(path / "model_manifest.json")


def _read_predictions(run, manifest, model):
    if model not in manifest["models"]:
        raise ValueError(f"Model {model!r} is absent from {run.name}")
    parts = []
    for fold in manifest["folds"]:
        name = fold["fold"]
        part = pd.read_csv(run / "predictions" / f"{name}_{model}.csv.gz", usecols=PREDICTION_COLUMNS,
                           dtype={"fold": str, "connection_id": str})
        if not part.fold.eq(name).all() or not part.model.eq(model).all():
            raise ValueError("Prediction file fold/model identity does not match its archive name")
        if "validation_rows" in fold and len(part) != fold["validation_rows"]:
            raise ValueError("Prediction file has missing/extra model samples relative to its fold manifest")
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if result.empty or result[KEYS].isna().any().any() or result.duplicated(KEYS).any():
        raise ValueError("Predictions must be nonempty with unique nonmissing fold+connection_id")
    if "evaluation_rows" in manifest and len(result) != manifest["evaluation_rows"]:
        raise ValueError("Prediction count differs from the run's full evaluation cohort")
    result["prediction_time"] = pd.to_datetime(result.prediction_time, utc=True, errors="raise").astype("datetime64[ns, UTC]")
    if not (result.prediction_time.ge(pd.Timestamp("2026-05-01", tz="UTC"))
            & result.prediction_time.lt(pd.Timestamp("2026-08-01", tz="UTC"))).all():
        raise ValueError("Comparison only accepts the declared May-July development validation scope")
    if not np.isfinite(result[["target_delay", "predicted_delay"]].to_numpy(dtype=float)).all() or result.target_delay.lt(0).any():
        raise ValueError("Predictions and nonnegative targets must be finite; no model rows may be dropped")
    return result


def align_predictions(reference, candidate, *, align_rows=False):
    """Require an exact cohort; optional alignment changes order, never membership."""
    left = pd.MultiIndex.from_frame(reference[KEYS])
    right = pd.MultiIndex.from_frame(candidate[KEYS])
    if left.has_duplicates or right.has_duplicates or len(left) != len(right) or set(left) != set(right):
        raise ValueError("Prediction cohort differs; missing model samples cannot be silently deleted")
    if not left.equals(right):
        if not align_rows:
            raise ValueError("Prediction row order differs; pass align_rows=True for explicit identity alignment")
        candidate = candidate.set_index(KEYS).reindex(left).reset_index()
    reference, candidate = reference.reset_index(drop=True), candidate.reset_index(drop=True)
    if not np.array_equal(reference.target_delay.to_numpy(), candidate.target_delay.to_numpy()):
        raise ValueError("Target labels differ on the same evaluation IDs")
    if not np.array_equal(reference.prediction_time.to_numpy(), candidate.prediction_time.to_numpy()):
        raise ValueError("UTC prediction_time differs on the same evaluation IDs")
    return reference, candidate


def validation_calendar(manifest):
    """Include every declared UTC date, including dates with no selected samples."""
    dates = []
    for fold in manifest["folds"]:
        start = pd.Timestamp(fold["validation_start"]).tz_convert("UTC").normalize()
        stop = pd.Timestamp(fold["validation_end_exclusive"]).tz_convert("UTC").normalize()
        dates.extend(pd.date_range(start, stop, inclusive="left", freq="D"))
    calendar = pd.DatetimeIndex(sorted(set(dates)))
    expected = pd.date_range("2026-05-01", "2026-08-01", inclusive="left", tz="UTC")
    if not calendar.equals(expected):
        raise ValueError("Comparison requires the full May-July UTC validation calendar")
    return calendar


def daily_statistics(reference, candidate, calendar):
    target = reference.target_delay.to_numpy(dtype=float)
    left = np.maximum(reference.predicted_delay.to_numpy(dtype=float), 0) - target
    right = np.maximum(candidate.predicted_delay.to_numpy(dtype=float), 0) - target
    values = pd.DataFrame({"date_utc": reference.prediction_time.dt.normalize(), "n": 1,
                           "reference_absolute_error_sum": np.abs(left), "candidate_absolute_error_sum": np.abs(right),
                           "reference_squared_error_sum": left ** 2, "candidate_squared_error_sum": right ** 2})
    daily = values.groupby("date_utc", observed=True)[STAT_COLUMNS].sum().reindex(calendar, fill_value=0)
    if int(daily.n.sum()) != len(reference):
        raise ValueError("Prediction dates are outside the declared validation calendar")
    return daily.rename_axis("date_utc")


def moving_block_draws(calendar, repeats, block_days, rng):
    """Non-circular moving blocks; no block crosses a gap in the date calendar."""
    if repeats < 1 or block_days < 1 or len(calendar) == 0:
        raise ValueError("Positive repeats/block_days and a nonempty calendar are required")
    calendar = pd.DatetimeIndex(calendar).as_unit("ns")
    splits = np.flatnonzero(np.diff(calendar.asi8) != pd.Timedelta(days=1).value) + 1
    draws = []
    for segment in np.split(np.arange(len(calendar)), splits):
        width = min(block_days, len(segment))
        starts = rng.integers(0, len(segment) - width + 1, size=(repeats, int(np.ceil(len(segment) / width))))
        sampled = (starts[:, :, None] + np.arange(width)).reshape(repeats, -1)[:, :len(segment)]
        draws.append(segment[sampled])
    return np.concatenate(draws, axis=1)


def _errors(sums):
    n = sums[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        reference_mae, candidate_mae = sums[..., 1] / n, sums[..., 2] / n
        reference_rmse, candidate_rmse = np.sqrt(sums[..., 3] / n), np.sqrt(sums[..., 4] / n)
    return {"mae": (reference_mae, candidate_mae), "rmse": (reference_rmse, candidate_rmse)}


def comparison_intervals(daily, *, bootstrap_repeats=1000, block_days=7, seed=42):
    """Pair both models on identical sampled dates; ratio-of-sums weights by rows.

    Each monthly interval resamples that month's full calendar. The pooled interval
    separately resamples the full continuous May-July calendar; neither crosses gaps.
    """
    rng = np.random.default_rng(seed)
    scopes = [(month, daily.loc[daily.index.strftime("%Y-%m") == month])
              for month in sorted(set(daily.index.strftime("%Y-%m")))] + [("pooled", daily)]
    rows = []
    for scope, dates in scopes:
        values = dates[STAT_COLUMNS].to_numpy(dtype=float)
        totals = values.sum(axis=0)
        draws = moving_block_draws(dates.index, bootstrap_repeats, block_days, rng)
        sampled = values[draws].sum(axis=1)
        observed, bootstrapped = _errors(totals), _errors(sampled)
        for metric in ["mae", "rmse"]:
            reference, candidate = observed[metric]
            differences = bootstrapped[metric][1] - bootstrapped[metric][0]
            valid = differences[np.isfinite(differences)]
            low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
            rows.append({"scope": scope, "metric": metric, "n": int(totals[0]), "calendar_dates": len(dates),
                         "reference_error": float(reference), "candidate_error": float(candidate),
                         "difference_candidate_minus_reference": float(candidate - reference),
                         "ci95_low": low, "ci95_high": high, "valid_bootstrap_repeats": len(valid), "unit": "minutes"})
    return pd.DataFrame(rows)


def _comparison_figure(metrics, output, reference_label, candidate_label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for ax, metric in zip(axes, ["mae", "rmse"]):
        data = metrics.loc[metrics.metric.eq(metric)].reset_index(drop=True)
        positions = np.arange(len(data))
        # CIs need not contain the observed estimate: draw independent interval bars.
        ax.hlines(positions, data.ci95_low, data.ci95_high, color="#356A91", linewidth=2)
        ax.scatter(data.difference_candidate_minus_reference, positions, color="#292929", marker="o", zorder=3)
        ax.axvline(0, color="#777777", linewidth=.8)
        ax.set(yticks=positions, yticklabels=[f"{r.scope} (n={r.n:,})" for r in data.itertuples()],
               xlabel=f"Candidate - reference {metric.upper()} (min)", title=f"{metric.upper()} difference with 95% block-bootstrap CI")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dddddd", linewidth=.5)
    axes[0].invert_yaxis()
    scope = metrics.sample_scope.iloc[0].replace("_", " ").upper()
    fig.suptitle(f"Reference {reference_label} → Candidate {candidate_label}\n{scope} | Positive difference = worse candidate", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def compare_runs(reference_run, candidate_run, output_dir, *, reference_model=None, candidate_model=None,
                 parent_run_id=None, comparison_purpose=None, subset_ids=None, common_sample_columns=None,
                 align_rows=False, bootstrap_repeats=1000, block_days=7, seed=42, allow_running_candidate=False):
    """Compare completed runs; explicit weather masks come from candidate features.

    subset_ids is a collection of connection_id strings. common_sample_columns is
    an explicit list of columns required nonmissing in candidate data/features.pkl.
    Their intersection is labeled a subset and cannot replace primary metrics.
    """
    reference_run, reference_manifest = _completed_run(reference_run)
    candidate_run, candidate_manifest = _completed_run(candidate_run, allow_running=allow_running_candidate)
    candidate_status = _json(candidate_run / "manifest.json")["status"]
    for name in ["evaluation_ids_sha256", "split_sha256"]:
        if not reference_manifest.get(name) or reference_manifest[name] != candidate_manifest.get(name):
            raise ValueError(f"Run {name} mismatch: comparison requires an identical evaluation protocol")
    config = _json(candidate_run / "config.json")
    parent_run_id = parent_run_id or reference_run.name
    declared_parent = config.get("parent_run_id")
    if (parent_run_id != reference_run.name or declared_parent and declared_parent != parent_run_id) and not comparison_purpose:
        raise ValueError("Comparison differs from candidate's declared parent; supply comparison_purpose explicitly")
    reference_model = reference_model or reference_manifest.get("model_name", "g0_linear")
    candidate_model = candidate_model or candidate_manifest.get("model_name", "g0_linear")
    reference = _read_predictions(reference_run, reference_manifest, reference_model)
    candidate = _read_predictions(candidate_run, candidate_manifest, candidate_model)
    reference, candidate = align_predictions(reference, candidate, align_rows=align_rows)
    calendar = validation_calendar(reference_manifest)
    if not calendar.equals(validation_calendar(candidate_manifest)):
        raise ValueError("Validation calendars differ")
    full_n = len(reference)
    keep = np.ones(full_n, dtype=bool)
    is_subset = subset_ids is not None or common_sample_columns is not None
    if subset_ids is not None:
        chosen = {str(value) for value in subset_ids}
        if chosen - set(reference.connection_id):
            raise ValueError("Explicit subset_ids contains IDs outside the full evaluation cohort")
        keep &= reference.connection_id.isin(chosen).to_numpy()
    if common_sample_columns is not None:
        columns = list(common_sample_columns)
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("common_sample_columns must be an explicit nonempty unique column list")
        features = pd.read_pickle(candidate_run / "data/features.pkl")
        if features.connection_id.isna().any() or features.connection_id.duplicated().any() or set(columns) - set(features):
            raise ValueError("Candidate feature archive has invalid IDs or missing common-sample columns")
        indexed = features.set_index("connection_id")
        if set(reference.connection_id) - set(indexed.index):
            raise ValueError("Candidate feature archive is missing evaluation IDs")
        keep &= indexed[columns].notna().all(axis=1).reindex(reference.connection_id).to_numpy(dtype=bool)
    reference, candidate = reference.loc[keep].copy(), candidate.loc[keep].copy()
    if reference.empty:
        raise ValueError("Explicit comparison subset is empty")
    daily = daily_statistics(reference, candidate, calendar)
    metrics = comparison_intervals(daily, bootstrap_repeats=bootstrap_repeats, block_days=block_days, seed=seed)
    metrics["sample_scope"] = "explicit_subset" if is_subset else "full_evaluation"
    output = Path(output_dir).resolve()
    protected_runs = [reference_run] + ([candidate_run] if candidate_status == "completed" else [])
    if any(output == run or output.is_relative_to(run) for run in protected_runs):
        raise ValueError("Comparison output must not modify either existing source run")
    output.mkdir(parents=True, exist_ok=True)
    filenames = ["comparison_daily_stats.csv", "comparison_metrics.csv", "comparison_error_differences.png", "comparison.json"]
    if is_subset:
        filenames.append("comparison_subset_ids.csv.gz")
    if any((output / name).exists() for name in filenames):
        raise FileExistsError("Comparison artifacts already exist; use a new output directory")
    daily.to_csv(output / filenames[0])
    metrics.to_csv(output / filenames[1], index=False)
    _comparison_figure(metrics, output / filenames[2], f"{reference_run.name}/{reference_model}",
                       f"{candidate_run.name}/{candidate_model}")
    identity = reference.fold.astype(str) + ":" + reference.connection_id.astype(str)
    subset_hash = hashlib.sha256(("\n".join(sorted(identity)) + "\n").encode()).hexdigest()
    if is_subset:
        reference[KEYS].to_csv(output / "comparison_subset_ids.csv.gz", index=False,
                               compression={"method": "gzip", "compresslevel": 3, "mtime": 0})
    summary = {"reference_run": reference_run.name, "candidate_run": candidate_run.name,
               "reference_model": reference_model, "candidate_model": candidate_model,
               "parent_run_id": parent_run_id, "candidate_declared_parent": declared_parent,
               "comparison_purpose": comparison_purpose, "align_rows": align_rows,
               "candidate_status_at_comparison": candidate_status, "allow_running_candidate": allow_running_candidate,
               "evaluation_ids_sha256": reference_manifest["evaluation_ids_sha256"], "split_sha256": reference_manifest["split_sha256"],
               "sample_scope": "explicit_subset" if is_subset else "full_evaluation", "full_evaluation_n": full_n,
               "compared_n": len(reference), "compared_identity_set_sha256": subset_hash,
               "common_sample_columns": list(common_sample_columns) if common_sample_columns is not None else None,
               "difference_definition": "candidate minus reference; positive is worse",
               "bootstrap": {"method": "paired non-circular moving UTC date blocks", "block_days": block_days,
                             "repeats": bootstrap_repeats, "seed": seed, "confidence_level": .95,
                             "full_calendar": "2026-05-01 through 2026-07-31 UTC, including zero-sample dates",
                             "monthly_intervals": "resample each monthly calendar separately",
                             "pooled_interval": "resample the full validation calendar; blocks never cross date gaps",
                             "empty_resamples": "Exclude only bootstrap replicates with sampled n=0; valid replicate counts are reported; no prediction rows are dropped",
                             "weighting": "sum AE / sum n; sqrt(sum SE / sum n); same sampled dates for both models"},
               "interpretation": "Paired sampling uncertainty, not causal evidence; explicit subsets are supplementary comparisons",
               "outputs": filenames}
    (output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-model")
    parser.add_argument("--candidate-model")
    parser.add_argument("--align-rows", action="store_true")
    parser.add_argument("--comparison-purpose")
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.reference_run, args.candidate_run, args.output_dir,
                                  reference_model=args.reference_model, candidate_model=args.candidate_model,
                                  align_rows=args.align_rows, comparison_purpose=args.comparison_purpose), indent=2))
