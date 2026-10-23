"""Fixed UTC development folds with train-only preprocessing and mature labels."""
from __future__ import annotations

import hashlib
import json
import contextlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import re
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

try:
    from .build_features import DEVELOPMENT_END, DEVELOPMENT_START, G0_CATEGORICAL, G0_FEATURES, G0_NUMERIC, G1_NUMERIC
    from .evaluate_models import metrics_from_summaries, summarize_predictions
    from .operational_features import AIRCRAFT_COLUMNS, AIRPORT_NUMERIC
except ImportError:
    from build_features import DEVELOPMENT_END, DEVELOPMENT_START, G0_CATEGORICAL, G0_FEATURES, G0_NUMERIC, G1_NUMERIC
    from evaluate_models import metrics_from_summaries, summarize_predictions
    from operational_features import AIRCRAFT_COLUMNS, AIRPORT_NUMERIC

DEVELOPMENT_FOLDS = [
    ("2026-05", "2026-05-01", "2026-06-01"),
    ("2026-06", "2026-06-01", "2026-07-01"),
    ("2026-07", "2026-07-01", "2026-08-01"),
]
GZIP = {"method": "gzip", "compresslevel": 3, "mtime": 0}
MODEL_NAMES = ["training_mean", "training_median", "carry_forward", "g0_linear"]
PREDICTION_METADATA = ["connection_id", "prediction_time", "label_time", "target_delay", "airport_id",
                       "downstream_airline", "downstream_day_of_week", "downstream_sched_hour", "overnight_scheduled"]
WEATHER_NUMERIC_SUFFIXES = ["wind_knots", "gust_knots", "visibility_miles", "ceiling_ft", "precip_amount_in",
                            "precip_flag", "rain_flag", "snow_flag", "thunder_flag", "age_minutes", "lead_minutes",
                            "missing", "missing_flag"]
WEATHER_NUMERIC = [prefix + suffix for prefix in ["metar_", "taf_"] for suffix in WEATHER_NUMERIC_SUFFIXES]


def resolve_features(config: dict) -> dict:
    """Resolve an explicit typed whitelist; arbitrary prefixes never authorize a field."""
    group = str(config.get("feature_group", "G0")).upper()
    if group not in {"G0", "G1", "G2", "G2_FAMILY", "G2_VARIANT", "G3", "G3_VARIANT", "G4A", "G4B"}:
        raise ValueError(f"Unsupported feature_group: {group}")
    default_cat = G0_CATEGORICAL.copy()
    default_num = G0_NUMERIC.copy() if group == "G0" else G1_NUMERIC.copy()
    if group not in {"G0", "G1"}:
        default_cat += ["aircraft_variant" if "VARIANT" in group else "aircraft_family"]
    if group in {"G3", "G3_VARIANT", "G4A", "G4B"}:
        default_num += AIRPORT_NUMERIC
    allowed_cat = set(default_cat)
    # Explicit family/variant replacement is allowed, but never both at once.
    if group not in {"G0", "G1"}:
        allowed_cat.update(AIRCRAFT_COLUMNS)
    allowed_num = set(default_num)
    if group == "G4A":
        allowed_num.update(name for name in WEATHER_NUMERIC if name.startswith("metar_"))
    elif group == "G4B":
        allowed_num.update(WEATHER_NUMERIC)
    explicit_features = config.get("features")
    if explicit_features is None:
        categorical = list(config.get("categorical_features", default_cat))
        numeric = list(config.get("numeric_features", default_num))
        features = categorical + numeric
    else:
        features = list(explicit_features)
        categorical = list(config.get("categorical_features", [name for name in features if name in allowed_cat]))
        numeric = list(config.get("numeric_features", [name for name in features if name in allowed_num]))
    if not features or len(features) != len(set(features)):
        raise ValueError("features must be nonempty and unique")
    if (len(categorical) != len(set(categorical)) or len(numeric) != len(set(numeric))
            or set(categorical) & set(numeric) or set(features) != set(categorical + numeric)):
        raise ValueError("features must exactly match disjoint categorical_features/numeric_features")
    if set(categorical) - allowed_cat or set(numeric) - allowed_num:
        raise ValueError("Feature list contains fields outside the declared predictor whitelist")
    if set(AIRCRAFT_COLUMNS).issubset(features):
        raise ValueError("Use aircraft_family OR aircraft_variant, never both nested categories")
    return {"feature_group": group, "features": features, "categorical_features": categorical, "numeric_features": numeric}


def model_names(config: dict) -> list[str]:
    name = config.get("model_name", "g0_linear")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in MODEL_NAMES[:3]:
        raise ValueError("model_name must be a safe unique non-baseline name")
    include = config.get("include_baselines", True)
    if not isinstance(include, bool):
        raise ValueError("include_baselines must be boolean")
    return (MODEL_NAMES[:3] if include else []) + [name]


def _regressor(config: dict):
    """Only the course plan's limited parameter comparisons are exposed."""
    kind = config.get("model_type", "linear")
    params = dict(config.get("model_params", {}))
    threads = int(config.get("threads", 8))
    if kind == "linear":
        if set(params) - {"tol", "fit_intercept"} or params.get("fit_intercept", True) is not True:
            raise ValueError("Linear parameters allow tol and fixed fit_intercept=True only")
        tolerance = float(params.get("tol", config.get("linear_tolerance", 1e-6)))
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Linear tolerance must be finite and positive")
        return LinearRegression(n_jobs=threads, tol=tolerance, fit_intercept=True)
    if kind not in {"decision_tree", "random_forest"}:
        raise ValueError("model_type must be linear, decision_tree or random_forest")
    allowed = {"max_depth", "min_samples_leaf", "random_state"}
    if kind == "random_forest":
        allowed.update({"n_estimators", "max_features"})
    if set(params) - allowed:
        raise ValueError("Unplanned model parameter")
    depth = params.get("max_depth", 8 if kind == "decision_tree" else 16)
    leaf = params.get("min_samples_leaf", 50)
    if depth not in ({8, 16} if kind == "decision_tree" else {16}) or leaf not in {50, 200}:
        raise ValueError("Tree plan permits depth 8/16 and leaf 50/200; forest depth is fixed at 16")
    if params.get("random_state", 42) != 42:
        raise ValueError("random_state is fixed at 42")
    common = {"max_depth": depth, "min_samples_leaf": leaf, "random_state": 42}
    if kind == "decision_tree":
        return DecisionTreeRegressor(**common)
    trees = params.get("n_estimators", 200)
    if isinstance(trees, bool) or not isinstance(trees, int) or not 1 <= trees <= 200 or params.get("max_features", .5) != .5:
        raise ValueError("Forest uses max_features=0.5 and 1..200 trees; formal runs must lock 200 externally")
    return RandomForestRegressor(**common, n_estimators=trees, max_features=.5, n_jobs=threads)


def development_splits(frame: pd.DataFrame):
    """Yield masks; UTC half-open prediction windows and strict mature training labels."""
    prediction = pd.to_datetime(frame.prediction_time, utc=True, errors="raise")
    label = pd.to_datetime(frame.label_time, utc=True, errors="raise")
    for name, start, end in DEVELOPMENT_FOLDS:
        cutoff = pd.Timestamp(start, tz="UTC")
        stop = pd.Timestamp(end, tz="UTC")
        train = prediction.ge(DEVELOPMENT_START) & prediction.lt(cutoff) & label.lt(cutoff)
        validation = prediction.ge(cutoff) & prediction.lt(stop)
        yield name, cutoff, stop, train, validation


def hash_ids(ids) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_targets(frame) -> str:
    digest = hashlib.sha256()
    digest.update(hash_ids(frame.connection_id).encode('ascii'))
    digest.update(np.asarray(frame.target_delay, dtype='<f8').tobytes())
    return digest.hexdigest()


def make_pipeline(config: dict) -> Pipeline:
    """All learned operations are fitted within each fold; OHE always stays sparse."""
    preprocess = config.get("preprocessing", {})
    features = resolve_features(config)
    if preprocess.get("drop", "first") != "first":
        raise ValueError("P0 linear regression uses fixed drop='first' reference coding")
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", drop="first", sparse_output=True,
                                  min_frequency=preprocess.get("min_frequency", 20), dtype=np.float64)),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler(with_mean=False)),
    ])
    transform = ColumnTransformer([("categorical", categorical, features["categorical_features"]),
                                   ("numeric", numeric, features["numeric_features"])], sparse_threshold=1.0)
    return Pipeline([("preprocess", transform), ("regression", _regressor(config))])


def make_g0_pipeline(config: dict) -> Pipeline:
    """Backward-compatible G0 entry point delegating to the shared pipeline."""
    return make_pipeline(config)


def _run_fold(fold: str, train: pd.DataFrame, validation: pd.DataFrame, config: dict,
              output_dir: str, audit: dict) -> tuple[pd.DataFrame, dict]:
    """Picklable Windows-spawn worker; owns only this fold's files and mutable data."""
    output = Path(output_dir)
    threads = int(config.get("threads", 8))
    started = time.perf_counter()
    timing = {"fold": fold, "pid": os.getpid(), "threads": threads, "train_rows": len(train),
              "validation_rows": len(validation), "models_seconds": {}, "status": "running"}
    summaries = []
    with (output / f"fold_{fold}.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                print(json.dumps(audit, sort_keys=True), flush=True)
                print(f"pid={os.getpid()}, threads={threads}", flush=True)
                y = train.target_delay.to_numpy(dtype=float)
                # Only these named predictors enter sklearn; outcomes remain metadata.
                features = resolve_features(config)["features"]
                x_train, x_val = train[features], validation[features]
                for name in model_names(config):
                    model_started = time.perf_counter()
                    with threadpool_limits(limits=threads):
                        if name == config.get("model_name", "g0_linear"):
                            model = make_pipeline(config)
                            model.fit(x_train, y)
                            predicted = model.predict(x_val)
                        elif name == "carry_forward":
                            model = {"rule": "max(upstream_arr_delay_minutes, 0)"}
                            predicted = validation.upstream_arr_delay_minutes.to_numpy(dtype=float)
                        else:
                            model = DummyRegressor(strategy="mean" if name == "training_mean" else "median")
                            model.fit(np.zeros((len(train), 1)), y)
                            predicted = model.predict(np.zeros((len(validation), 1)))
                    result = validation[PREDICTION_METADATA].copy()
                    result.insert(0, "fold", fold)
                    result.insert(0, "model", name)
                    result["predicted_delay_raw"] = predicted
                    result["predicted_delay"] = np.maximum(predicted, 0)
                    summaries.append(summarize_predictions(result))
                    result.to_csv(output / "predictions" / f"{fold}_{name}.csv.gz", index=False, compression=GZIP)
                    joblib.dump(model, output / "models" / f"{fold}_{name}.joblib", compress=3)
                    timing["models_seconds"][name] = time.perf_counter() - model_started
                    print(f"{name} saved; elapsed={timing['models_seconds'][name]:.3f}s", flush=True)
                summary = pd.concat(summaries, ignore_index=True)
                summary.to_csv(output / f"fold_{fold}_metrics.csv", index=False)
                timing["status"] = "completed"
            except BaseException as error:
                timing.update(status="failed", error=repr(error))
                traceback.print_exc()
                raise
            finally:
                timing["elapsed_seconds"] = time.perf_counter() - started
                (output / f"fold_{fold}_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
                print(f"status={timing['status']}; elapsed={timing['elapsed_seconds']:.3f}s", flush=True)
    return summary, timing


def run_development(frame: pd.DataFrame, config: dict, output_dir: str | Path) -> dict:
    """Run the three declared development folds without inspecting August targets.

    Config P0 keys: fold_workers (1 or 3, default 1), threads per fold (8),
    linear_tolerance (1e-6), preprocessing
    {min_frequency:20, drop:'first'}. Typed predictors must match an explicit group whitelist.
    The caller owns run_id, provenance, outer manifest and logging.
    """
    threads = int(config.get("threads", 8))
    workers = config.get("fold_workers", 1)
    if isinstance(workers, bool) or workers not in (1, 3):
        raise ValueError("fold_workers must be 1 or 3")
    if threads < 1:
        raise ValueError("threads must be positive")
    feature_spec = resolve_features(config)
    features = feature_spec["features"]
    names = model_names(config)
    prediction = pd.to_datetime(frame.prediction_time, utc=True, errors="raise")
    data = frame.loc[prediction.ge(DEVELOPMENT_START) & prediction.lt(DEVELOPMENT_END)].copy()
    required = set(features + PREDICTION_METADATA + ["upstream_arr_delay_minutes"])
    if required - set(data):
        raise ValueError(f"Missing modeling fields: {sorted(required - set(data))}")
    if data.empty or not data.connection_id.is_unique:
        raise ValueError("Development cohort must be nonempty with unique IDs")
    if not (data.prediction_time < data.label_time).all():
        raise ValueError("RQ2 requires prediction_time < label_time for every sample")
    if not (np.isfinite(data.target_delay) & data.target_delay.ge(0)).all():
        raise ValueError("Targets must be finite and nonnegative")
    data = data.sort_values(["prediction_time", "connection_id"], kind="stable").reset_index(drop=True)
    output = Path(output_dir)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    summaries, audits, evaluation_ids = [], [], []
    hash_payload = []
    fold_timings = []
    # Spawn is explicit for equivalent Windows/Linux behavior. Submission sends only
    # the minimum train/validation columns, never three copies of the whole frame.
    pool = (ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn"))
            if workers == 3 else contextlib.nullcontext(None))
    jobs = []
    with pool as executor:
        for fold, cutoff, end, train_mask, val_mask in development_splits(data):
            train = data.loc[train_mask]
            validation = data.loc[val_mask]
            if train.empty or validation.empty:
                raise ValueError(f"{fold}: empty training or validation cohort")
            train_hash = hash_ids(train.connection_id)
            validation_hash = hash_ids(validation.connection_id)
            mature_exclusions = (data.prediction_time.ge(DEVELOPMENT_START) & data.prediction_time.lt(cutoff)
                                 & ~data.label_time.lt(cutoff)).sum()
            audit = {"fold": fold, "train_start": DEVELOPMENT_START.isoformat(), "train_end_exclusive": cutoff.isoformat(),
                     "validation_start": cutoff.isoformat(), "validation_end_exclusive": end.isoformat(),
                     "train_rows": len(train), "validation_rows": len(validation),
                     "excluded_immature_training_labels": int(mature_exclusions),
                     "max_train_prediction_time": train.prediction_time.max().isoformat(),
                     "max_train_label_time": train.label_time.max().isoformat(),
                     "min_validation_prediction_time": validation.prediction_time.min().isoformat(),
                     "max_validation_prediction_time": validation.prediction_time.max().isoformat(),
                     "train_id_sha256": train_hash, "validation_id_sha256": validation_hash,
                     "train_target_sha256": hash_targets(train), "validation_target_sha256": hash_targets(validation)}
            audits.append(audit)
            hash_payload.append(audit)
            ids = validation[["connection_id", "prediction_time", "label_time"]].copy()
            ids.insert(0, "fold", fold)
            evaluation_ids.append(ids)
            print(f"[{fold}] train={len(train):,}, validation={len(validation):,}, immature={mature_exclusions:,}", flush=True)
            worker_train = train[features + ["target_delay"]].copy()
            validation_columns = list(dict.fromkeys(features + PREDICTION_METADATA + ["upstream_arr_delay_minutes"]))
            worker_validation = validation[validation_columns].copy()
            # Persist the assigned split before fitting, including on a failed run.
            pd.DataFrame(audits).to_csv(output / "split_audit.csv", index=False)
            if executor is None:
                jobs.append(_run_fold(fold, worker_train, worker_validation, config, str(output), audit))
            else:
                jobs.append(executor.submit(_run_fold, fold, worker_train, worker_validation, config, str(output), audit))
        # Fixed submission order preserves pooled rows, metrics and hashes regardless
        # of the order in which worker processes finish. Exceptions propagate here.
        for job in jobs:
            summary, timing = job.result() if executor is not None else job
            summaries.append(summary)
            fold_timings.append(timing)
            print(f"[{timing['fold']}] completed in {timing['elapsed_seconds']:.2f}s", flush=True)
    ids = pd.concat(evaluation_ids, ignore_index=True)
    ids.to_csv(output / "evaluation_ids.csv.gz", index=False, compression=GZIP)
    evaluation_hash = hash_ids(ids.fold.astype(str) + ":" + ids.connection_id.astype(str))
    split_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()
    pd.DataFrame(audits).to_csv(output / "split_audit.csv", index=False)
    metrics = metrics_from_summaries(pd.concat(summaries, ignore_index=True))
    metrics.to_csv(output / "metrics.csv", index=False)
    manifest = {"development_only": True, "evaluation_rows": len(ids), "folds": audits,
                "evaluation_ids_sha256": evaluation_hash, "split_sha256": split_hash,
                **feature_spec, "models": names, "model_name": config.get("model_name", "g0_linear"),
                "model_type": config.get("model_type", "linear"),
                "model_params": _regressor(config).get_params(deep=False),
                "include_baselines": config.get("include_baselines", True),
                "effective_preprocessing": {"min_frequency": config.get("preprocessing", {}).get("min_frequency", 20),
                                            "drop": "first", "sparse": True,
                                            "categorical_imputation": "constant __MISSING__; keep_empty_features=True",
                                            "numeric_imputation": "median; add_indicator=True; keep_empty_features=True",
                                            "numeric_scaling": "StandardScaler(with_mean=False), fixed across algorithms",
                                            "unknown_categories": "infrequent_if_exist"},
                "threads": threads, "fold_workers": int(workers), "fold_timings": fold_timings,
                "linear_tolerance": float(config.get("model_params", {}).get("tol", config.get("linear_tolerance", 1e-6))),
                "linear_fit_intercept": True if config.get("model_type", "linear") == "linear" else None,
                "sparse_solver": "scipy.sparse.linalg.lsqr via sklearn LinearRegression" if config.get("model_type", "linear") == "linear" else None,
                "coefficient_limit": "Airport and route columns are related; individual coefficients are not independent airport effects" if config.get("model_type", "linear") == "linear" else None,
                "metric_path": str(output / "metrics.csv")}
    (output / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
