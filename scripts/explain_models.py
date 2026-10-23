"""Read-only development explanations of already fitted, archived pipelines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import export_text

try:
    from .build_features import G0_FEATURES, G1_ADDED_NUMERIC
    from .operational_features import AIRCRAFT_COLUMNS, AIRPORT_NUMERIC
except ImportError:
    from build_features import G0_FEATURES, G1_ADDED_NUMERIC
    from operational_features import AIRCRAFT_COLUMNS, AIRPORT_NUMERIC

GROUP_ORDER = ["G0_plan", "G1_upstream_delay", "G1_tail_history", "G2_aircraft",
               "G3_airport_operations", "G4a_METAR", "G4b_TAF"]
PDP_FEATURES = ["upstream_arr_delay_minutes", "scheduled_turnaround_minutes"]


def permutation_groups(features):
    """Each original predictor belongs to exactly one simultaneous permutation group."""
    groups = {
        "G0_plan": [name for name in G0_FEATURES if name in features],
        "G1_upstream_delay": ["upstream_arr_delay_minutes"] if "upstream_arr_delay_minutes" in features else [],
        "G1_tail_history": [name for name in G1_ADDED_NUMERIC if name != "upstream_arr_delay_minutes" and name in features],
        "G2_aircraft": [name for name in AIRCRAFT_COLUMNS if name in features],
        "G3_airport_operations": [name for name in AIRPORT_NUMERIC if name in features],
        "G4a_METAR": [name for name in features if name.startswith("metar_")],
        "G4b_TAF": [name for name in features if name.startswith("taf_")],
    }
    assigned = [name for values in groups.values() for name in values]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(features):
        raise ValueError(f"Unassigned or multiply assigned predictors: {sorted(set(features) - set(assigned))}")
    return groups


def permute_input_group(x, columns, permutation):
    """One row permutation is shared by all columns in the input group."""
    changed = x.copy()
    for column in columns:
        changed[column] = x[column].to_numpy()[permutation]
    return changed


def _predict(model, x, warning_audit):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        prediction = np.asarray(model.predict(x), dtype=float)
    for item in recorded:
        key = (item.category.__name__, str(item.message))
        warning_audit[key] = warning_audit.get(key, 0) + 1
    if prediction.ndim != 1 or not np.isfinite(prediction).all():
        raise ValueError("Archived pipeline returned invalid predictions")
    return np.maximum(prediction, 0)


def _coverage(full, sample, fold):
    dimensions = {"prediction_date_utc": (full.prediction_time.dt.strftime("%Y-%m-%d"),
                                             sample.prediction_time.dt.strftime("%Y-%m-%d")),
                  "severity": (pd.Series(np.where(full.target_delay < 60, "ordinary_lt60", "severe_ge60"), index=full.index),
                               pd.Series(np.where(sample.target_delay < 60, "ordinary_lt60", "severe_ge60"), index=sample.index))}
    for name in ["airport_id", "downstream_airline"]:
        if name in full:
            dimensions[name] = (full[name], sample[name])
    rows = []
    for dimension, (whole, chosen) in dimensions.items():
        total = whole.astype("string").fillna("__MISSING__").value_counts()
        counts = chosen.astype("string").fillna("__MISSING__").value_counts()
        for value, n in total.items():
            selected = int(counts.get(value, 0))
            rows.append({"fold": fold, "dimension": dimension, "value": str(value), "validation_n": int(n),
                         "sample_n": selected, "sample_share": selected / n})
    return rows


def _permutation_plot(summary, fold, model_name, sample_n, path):
    data = summary.loc[summary.fold.eq(fold)].sort_values("mean_delta_mae")
    fig, ax = plt.subplots(figsize=(9, max(3, .55 * len(data) + 1.5)))
    ax.errorbar(data.mean_delta_mae, np.arange(len(data)), xerr=data.sd_delta_mae.fillna(0),
                fmt="o", color="#356A91", ecolor="#777777", capsize=3)
    ax.axvline(0, color="#888888", linewidth=.8)
    ax.set(yticks=np.arange(len(data)), yticklabels=data.group,
           xlabel="Permuted - original MAE (min), mean ± one repeat SD",
           title=f"Grouped permutation | {fold} / {model_name}, sample n={sample_n:,}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _pdp_plot(points, support, fold, feature, path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(points.grid_value, points.mean_prediction, color="#356A91", marker="o", markersize=3)
    axes[0].set(xlabel=f"{feature} (min)", ylabel="Average clipped prediction (min)",
                title=f"Partial dependence | {fold} | 5th–95th percentile grid")
    axes[1].bar(support.bin_left, support.n, width=support.bin_right - support.bin_left,
                align="edge", color="#B8C5CF", edgecolor="white", linewidth=.4)
    axes[1].set(xlabel=f"{feature} (min), full finite sample range", ylabel="Sample rows",
                title=f"Support: finite n={int(support.n.sum()):,}; axes show different ranges")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def explain_run(source_run, output_dir, *, model_name=None, folds=("2026-07",), sample_n=10000, repeats=5, seed=42):
    """Explain archived validation predictions; never fit or alter a trained model.

    sample_n is per fold. PDP grids use only the selected development validation
    sample; August rows and labels are never inspected or used to select grids.
    """
    source, output = Path(source_run).resolve(), Path(output_dir).resolve()
    outer = json.loads((source / "manifest.json").read_text(encoding="utf-8-sig"))
    if outer.get("status") not in {"completed", "running"}:
        raise ValueError("Explanations require a completed run or a model-ready running run")
    if outer["status"] == "completed" and (output == source or output.is_relative_to(source)):
        raise ValueError("Cannot write explanations inside a completed source run")
    if (isinstance(sample_n, bool) or not isinstance(sample_n, int) or sample_n < 1
            or isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1):
        raise ValueError("sample_n and repeats must be positive integers")
    manifest = json.loads((source / "model_manifest.json").read_text(encoding="utf-8-sig"))
    model_name = model_name or manifest.get("model_name", "g0_linear")
    if model_name not in manifest["models"]:
        raise ValueError("Requested model is not in the source model manifest")
    features = manifest["features"]
    groups = permutation_groups(features)
    fold_specs = {item["fold"]: item for item in manifest["folds"]}
    folds = list(folds)
    if not folds or len(folds) != len(set(folds)) or set(folds) - set(fold_specs):
        raise ValueError("folds must be unique declared validation folds")
    frame = pd.read_pickle(source / "data/features.pkl")
    if set(features + ["prediction_time", "connection_id", "target_delay"]) - set(frame):
        raise ValueError("Source feature archive lacks required predictors or evaluation metadata")
    prediction = pd.to_datetime(frame.prediction_time, utc=True, errors="raise")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("explain_*")) or any(output.glob("permutation_*")) or any(output.glob("pdp_*")):
        raise FileExistsError("Explanation artifacts already exist; use a new output directory")
    permutation_rows, coverage_rows, sample_ids, sample_predictions, coefficients, intercepts = [], [], [], [], [], []
    pdp_rows, support_rows, fold_audit, skips, tree_paths = [], [], [], [], []
    pdp_audit = []
    prediction_warnings = {}
    for fold in folds:
        spec = fold_specs[fold]
        start, end = pd.Timestamp(spec["validation_start"]), pd.Timestamp(spec["validation_end_exclusive"])
        if start < pd.Timestamp("2026-05-01", tz="UTC") or end > pd.Timestamp("2026-08-01", tz="UTC"):
            raise ValueError("Only May-July development validation folds can be explained")
        positions = np.flatnonzero((prediction.ge(start) & prediction.lt(end)).to_numpy())
        full = frame.iloc[positions].copy()
        full["prediction_time"] = prediction.iloc[positions]
        full["target_delay"] = pd.to_numeric(full.target_delay, errors="coerce")
        if full.empty or full.connection_id.isna().any() or not full.connection_id.is_unique:
            raise ValueError("Validation sample must be nonempty with unique IDs")
        if "validation_rows" in spec and len(full) != spec["validation_rows"]:
            raise ValueError("Feature archive validation count differs from the trained fold manifest")
        if not (np.isfinite(full.target_delay) & full.target_delay.ge(0)).all():
            raise ValueError("Development validation labels must be finite and nonnegative")
        fold_seed = int(fold.replace("-", ""))
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold_seed, 0]))
        selected = np.sort(rng.choice(len(full), size=min(sample_n, len(full)), replace=False))
        sample = full.iloc[selected].reset_index(drop=True)
        ids = sample[[name for name in ["connection_id", "prediction_time", "label_time"] if name in sample]].copy()
        ids.insert(0, "fold", fold)
        ids["source_row_position"] = positions[selected]
        sample_ids.append(ids)
        coverage_rows.extend(_coverage(full, sample, fold))
        model_path = source / "models" / f"{fold}_{model_name}.joblib"
        model = joblib.load(model_path)
        if not hasattr(model, "named_steps") or not {"preprocess", "regression"}.issubset(model.named_steps):
            raise ValueError("Explanations require an archived fitted preprocessing/regression pipeline")
        x, target = sample[features].copy(), sample.target_delay.to_numpy(dtype=float)
        original_prediction = _predict(model, x, prediction_warnings)
        original_mae = float(np.abs(original_prediction - target).mean())
        original = ids.copy()
        original["target_delay"] = target
        original["original_predicted_delay"] = original_prediction
        original["absolute_error"] = np.abs(original_prediction - target)
        sample_predictions.append(original)
        fold_audit.append({"fold": fold, "validation_start": start.isoformat(), "validation_end_exclusive": end.isoformat(),
                           "validation_n": len(full), "sample_n": len(sample), "sampling_fraction": len(sample) / len(full),
                           "original_clipped_mae": original_mae, "model_path": str(model_path.relative_to(source))})
        for group_index, (name, columns) in enumerate(groups.items()):
            if not columns:
                skips.append({"fold": fold, "kind": "permutation", "group": name, "reason": "no group features in fitted model"})
                continue
            for repeat in range(repeats):
                repeat_seed = int(np.random.SeedSequence([seed, fold_seed, group_index, repeat, 1]).generate_state(1)[0])
                permutation = np.random.default_rng(repeat_seed).permutation(len(x))
                permuted = _predict(model, permute_input_group(x, columns, permutation), prediction_warnings)
                mae = float(np.abs(permuted - target).mean())
                permutation_rows.append({"fold": fold, "group": name, "features": "|".join(columns), "repeat": repeat,
                                         "permutation_seed": repeat_seed, "sample_n": len(sample), "original_mae": original_mae,
                                         "permuted_mae": mae, "delta_mae": mae - original_mae})
        encoded = model.named_steps["preprocess"].get_feature_names_out()
        estimator = model.named_steps["regression"]
        if hasattr(estimator, "coef_"):
            values = np.asarray(estimator.coef_).reshape(-1)
            if len(values) != len(encoded):
                raise ValueError("Coefficient count differs from encoded feature names")
            coefficients.extend({"fold": fold, "encoded_feature": str(name), "coefficient": float(value)}
                                for name, value in zip(encoded, values))
            intercepts.append({"fold": fold, "intercept": float(estimator.intercept_)})
        elif hasattr(estimator, "tree_") or hasattr(estimator, "estimators_"):
            trees = [estimator] if hasattr(estimator, "tree_") else estimator.estimators_[:3]
            text = "\n\n".join(f"Tree {i}; encoded/scaled inputs; truncated at depth 3\n" +
                                  export_text(tree, feature_names=list(encoded), max_depth=3, decimals=3)
                                  for i, tree in enumerate(trees))
            filename = f"explain_{fold}_tree_paths.txt"
            (output / filename).write_text(text, encoding="utf-8")
            tree_paths.append(filename)
        for feature in PDP_FEATURES:
            if feature not in features:
                skips.append({"fold": fold, "kind": "pdp", "feature": feature, "reason": "feature absent from fitted model"})
                continue
            values = pd.to_numeric(x[feature], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            finite = values[np.isfinite(values)]
            if not len(finite):
                skips.append({"fold": fold, "kind": "pdp", "feature": feature, "reason": "no finite values in fixed sample"})
                continue
            low, high = np.quantile(finite, [.05, .95])
            grid = np.unique(np.linspace(low, high, 11))
            points = []
            for value in grid:
                changed = x.copy()
                changed[feature] = float(value)
                points.append({"fold": fold, "feature": feature, "grid_value": float(value),
                               "mean_prediction": float(_predict(model, changed, prediction_warnings).mean()), "sample_n": len(sample)})
            n, bins = np.histogram(finite, bins=min(20, len(np.unique(finite))))
            support = [{"fold": fold, "feature": feature, "bin_left": float(left), "bin_right": float(right),
                        "n": int(count), "share_finite": count / len(finite)} for left, right, count in zip(bins[:-1], bins[1:], n)]
            pdp_rows.extend(points)
            support_rows.extend(support)
            pdp_audit.append({"fold": fold, "feature": feature, "finite_sample_n": len(finite),
                              "missing_or_nonfinite_n": int(len(values) - len(finite)), "q05": float(low), "q95": float(high), "grid_n": len(grid)})
            _pdp_plot(pd.DataFrame(points), pd.DataFrame(support), fold, feature, output / f"pdp_{fold}_{feature}.png")
    permutation = pd.DataFrame(permutation_rows)
    summary = permutation.groupby(["fold", "group", "features"], sort=False).delta_mae.agg(
        mean_delta_mae="mean", sd_delta_mae="std", repeats="size").reset_index()
    permutation.to_csv(output / "permutation_repeats.csv", index=False)
    summary.to_csv(output / "permutation_summary.csv", index=False)
    pd.concat(sample_ids, ignore_index=True).to_csv(output / "explain_sample_ids.csv.gz", index=False,
                                                   compression={"method": "gzip", "compresslevel": 3, "mtime": 0})
    pd.concat(sample_predictions, ignore_index=True).to_csv(output / "explain_sample_predictions.csv.gz", index=False,
                                                           compression={"method": "gzip", "compresslevel": 3, "mtime": 0})
    pd.DataFrame(coverage_rows).to_csv(output / "explain_sample_coverage.csv", index=False)
    pd.DataFrame(coefficients, columns=["fold", "encoded_feature", "coefficient"]).to_csv(output / "explain_coefficients.csv", index=False)
    pd.DataFrame(intercepts, columns=["fold", "intercept"]).to_csv(output / "explain_intercepts.csv", index=False)
    pd.DataFrame(pdp_rows, columns=["fold", "feature", "grid_value", "mean_prediction", "sample_n"]).to_csv(output / "pdp_values.csv", index=False)
    pd.DataFrame(support_rows, columns=["fold", "feature", "bin_left", "bin_right", "n", "share_finite"]).to_csv(output / "pdp_support.csv", index=False)
    for item in fold_audit:
        _permutation_plot(summary, item["fold"], model_name, item["sample_n"], output / f"permutation_{item['fold']}.png")
    result = {"source_run": source.name, "source_status_at_explanation": outer["status"], "model_name": model_name,
              "source_files": {"features": "data/features.pkl", "model_manifest": "model_manifest.json",
                               "models": [item["model_path"] for item in fold_audit]},
              "features": features, "folds": fold_audit, "sample_n_requested_per_fold": sample_n, "repeats": repeats, "seed": seed,
              "sampling": "Uniform without replacement within each validation fold; fixed seed; chronological source positions retained",
              "fit_called": False, "prediction_rule": "clip predictions below zero before all error/PDP calculations",
              "permutation_groups": groups, "permutation_sd": "Sample standard deviation across repeats; not a confidence interval",
              "pdp": {"features": PDP_FEATURES, "grid": "11 evenly spaced points from fixed development sample q05 to q95; unique points retained",
                      "audit": pdp_audit}, "skips": skips, "tree_path_files": tree_paths,
              "prediction_warnings": [{"category": category, "message": message, "count": count}
                                      for (category, message), count in prediction_warnings.items()],
              "limitations": ["Permutation may create unrealistic combinations with other correlated input groups; importance is conditional on model and sample, not causal",
                              "Joint G0 permutation keeps airport, route and other plan inputs together; individual airport/route importance is not estimated",
                              "Encoded linear coefficients use reference categories and scaled numeric inputs; airport-route collinearity prevents interpreting individual coefficients as independent airport effects",
                              "PDP averages hypothetical input replacements, can cross correlated-feature support, and is descriptive rather than causal",
                              "Fixed validation subsampling and repeat SD do not quantify full population or temporal uncertainty; no August grid selection or explanation"]}
    (output / "explain_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--sample-n", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(explain_run(args.source_run, args.output_dir, model_name=args.model_name, sample_n=args.sample_n), indent=2))
