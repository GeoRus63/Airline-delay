"""Fixed UTC development folds with train-only preprocessing and mature labels."""
from __future__ import annotations

import hashlib
import json
import contextlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

try:
    from .build_features import DEVELOPMENT_END, DEVELOPMENT_START, G0_CATEGORICAL, G0_FEATURES, G0_NUMERIC
    from .evaluate_models import metrics_from_summaries, summarize_predictions
except ImportError:
    from build_features import DEVELOPMENT_END, DEVELOPMENT_START, G0_CATEGORICAL, G0_FEATURES, G0_NUMERIC
    from evaluate_models import metrics_from_summaries, summarize_predictions

DEVELOPMENT_FOLDS = [
    ("2026-05", "2026-05-01", "2026-06-01"),
    ("2026-06", "2026-06-01", "2026-07-01"),
    ("2026-07", "2026-07-01", "2026-08-01"),
]
GZIP = {"method": "gzip", "compresslevel": 3, "mtime": 0}
MODEL_NAMES = ["training_mean", "training_median", "carry_forward", "g0_linear"]
PREDICTION_METADATA = ["connection_id", "prediction_time", "label_time", "target_delay", "airport_id",
                       "downstream_airline", "downstream_day_of_week", "downstream_sched_hour", "overnight_scheduled"]


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


def make_g0_pipeline(config: dict) -> Pipeline:
    """All learned operations are fitted within each fold; OHE always stays sparse."""
    preprocess = config.get("preprocessing", {})
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
    transform = ColumnTransformer([("categorical", categorical, G0_CATEGORICAL),
                                   ("numeric", numeric, G0_NUMERIC)], sparse_threshold=1.0)
    return Pipeline([("preprocess", transform), ("regression", LinearRegression(
        n_jobs=int(config.get("threads", 8)), tol=float(config.get("linear_tolerance", 1e-6))))])


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
                x_train, x_val = train[G0_FEATURES], validation[G0_FEATURES]
                for name in MODEL_NAMES:
                    model_started = time.perf_counter()
                    with threadpool_limits(limits=threads):
                        if name == "g0_linear":
                            model = make_g0_pipeline(config)
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
    {min_frequency:20, drop:'first'}. Additional feature lists must match G0 exactly.
    The caller owns run_id, provenance, outer manifest and logging.
    """
    threads = int(config.get("threads", 8))
    workers = config.get("fold_workers", 1)
    if isinstance(workers, bool) or workers not in (1, 3):
        raise ValueError("fold_workers must be 1 or 3")
    if threads < 1:
        raise ValueError("threads must be positive")
    if list(config.get("features", G0_FEATURES)) != G0_FEATURES:
        raise ValueError("P0 feature list must exactly match the declared G0 whitelist")
    prediction = pd.to_datetime(frame.prediction_time, utc=True, errors="raise")
    data = frame.loc[prediction.ge(DEVELOPMENT_START) & prediction.lt(DEVELOPMENT_END)].copy()
    required = set(G0_FEATURES + ["connection_id", "label_time", "target_delay", "upstream_arr_delay_minutes",
                                  "overnight_scheduled"])
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
            worker_train = train[G0_FEATURES + ["target_delay"]].copy()
            validation_columns = list(dict.fromkeys(G0_FEATURES + PREDICTION_METADATA + ["upstream_arr_delay_minutes"]))
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
                "features": G0_FEATURES, "categorical_features": G0_CATEGORICAL, "numeric_features": G0_NUMERIC,
                "models": MODEL_NAMES,
                "effective_preprocessing": {"min_frequency": config.get("preprocessing", {}).get("min_frequency", 20),
                                            "drop": "first", "sparse": True},
                "threads": threads, "fold_workers": int(workers), "fold_timings": fold_timings,
                "linear_tolerance": float(config.get("linear_tolerance", 1e-6)),
                "linear_fit_intercept": True, "sparse_solver": "scipy.sparse.linalg.lsqr via sklearn LinearRegression",
                "coefficient_limit": "Airport and route columns are related; individual coefficients are not independent airport effects",
                "metric_path": str(output / "metrics.csv")}
    (output / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
