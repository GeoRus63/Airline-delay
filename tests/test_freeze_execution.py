"""Explicit final thread overrides and seeded RF scheduling equivalence."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import _forest

from scripts import select_models as selection
from tests.test_model_selection import write_run, algorithm_grid


class FreezeExecutionTests(unittest.TestCase):
    def test_explicit_threads_only_default_identity_and_source_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            linear = write_run(root, "LINEAR")
            decision = selection.select_model(linear, algorithm_grid(root, linear), root / "choice")
            original = copy.deepcopy(decision)
            archived = {path: path.read_bytes() for path in [
                root / "choice/model_selection_decision.json",
                Path(decision["selected_run_dir"]) / "config.json",
                Path(decision["selected_run_dir"]) / "model_manifest.json"]}
            with patch.object(selection, "_assert_committed", return_value="synthetic-verified-source-commit"), \
                    patch.object(selection.final, "_required_data", return_value={root / "connections.csv"}), \
                    patch.object(selection.os, "cpu_count", return_value=32), \
                    patch.object(selection.final, "load_configured_features", side_effect=AssertionError("no August loader")):
                selection.create_freeze(decision, root / "default/freeze.json")
                default = json.loads((root / "default/freeze.json").read_text())
                self.assertEqual(default["config"], decision["config"])
                self.assertNotIn("execution_delta", default)
                selection.create_freeze(decision, root / "parallel/freeze.json", final_threads=24)
                parallel = json.loads((root / "parallel/freeze.json").read_text())
                expected = {**decision["config"], "threads": 24}
                self.assertEqual(parallel["config"], expected)
                self.assertEqual(parallel["execution_delta"]["threads"],
                                 {"development": decision["config"]["threads"], "final": 24})
                self.assertTrue(parallel["execution_delta"]["learning_parameters_unchanged"])
                self.assertEqual(parallel["selection"]["actual_git_commit"], "synthetic-verified-source-commit")
                self.assertEqual(parallel["config"]["model_params"], decision["config"]["model_params"])
                frozen_path = root / "parallel/freeze.json"
                for mode in ["final", "reproduce_final"]:
                    validated = selection.final.validate_freeze({"mode": mode, "freeze_manifest_path": str(frozen_path),
                        "freeze_manifest_sha256": selection.final.sha256_file(frozen_path)})[0]
                    self.assertEqual(validated["config"]["threads"], 24)
                for invalid in [True, False, 0, -1, 33, 2.0, "24"]:
                    with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "final_threads"):
                        selection.create_freeze(decision, root / "invalid/freeze.json", final_threads=invalid)
                changed = copy.deepcopy(decision)
                changed["config"]["threads"] = 24
                changed["selection_sha256"] = selection.final.canonical_sha256(
                    {k: v for k, v in changed.items() if k != "selection_sha256"})
                with self.assertRaisesRegex(ValueError, "source/config"):
                    selection.create_freeze(changed, root / "tampered/freeze.json", final_threads=24)
            self.assertEqual(decision, original)
            for path, contents in archived.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_seeded_200_tree_forest_is_unchanged_by_parallel_worker_count(self):
        rng = np.random.default_rng(42)
        config = {"feature_group": "G0", "preprocessing": {"drop": "first", "min_frequency": 20}}
        spec = selection.training.resolve_features(config)
        frame = pd.DataFrame({name: rng.normal(size=6000) for name in spec["numeric_features"]})
        labels = np.maximum(0, 25 + 35 * frame.iloc[:, 0].to_numpy() + 10 * frame.iloc[:, 1].to_numpy())
        for index, name in enumerate(spec["categorical_features"]):
            frame[name] = pd.Series(rng.integers(0, 6 + index * 2, len(frame))).astype(str)
        frame.loc[::29, spec["numeric_features"][1]] = np.nan
        # Use the actual production imputation/OHE/scaling composition, fitted
        # only on synthetic training rows; no project flight table is opened.
        preprocess = selection.training.make_pipeline(config).named_steps["preprocess"]
        features = preprocess.fit_transform(frame.iloc[:5400])
        query = preprocess.transform(frame.iloc[5400:])
        self.assertTrue(sparse.isspmatrix_csr(features))
        self.assertEqual(features.dtype, np.dtype("float64"))
        backends = []

        class RecordedParallel(_forest.Parallel):
            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                backends.append({"n_jobs": self.n_jobs, "backend": type(self._backend).__name__})
                return result

        checks = []
        for leaf in [50, 200]:
            with self.subTest(min_samples_leaf=leaf):
                params = dict(n_estimators=200, max_depth=16, max_features=.5,
                              min_samples_leaf=leaf, random_state=42)
                with patch.object(_forest, "Parallel", RecordedParallel):
                    serial = RandomForestRegressor(**params, n_jobs=1).fit(features, labels[:5400])
                    parallel = RandomForestRegressor(**params, n_jobs=24).fit(features, labels[:5400])
                    raw_serial, raw_parallel = serial.predict(query), parallel.predict(query)
                self.assertEqual([tree.random_state for tree in serial.estimators_],
                                 [tree.random_state for tree in parallel.estimators_])
                nodes = []
                for left, right in zip(serial.estimators_, parallel.estimators_):
                    left_state, right_state = left.tree_.__getstate__(), right.tree_.__getstate__()
                    np.testing.assert_array_equal(left_state["nodes"], right_state["nodes"])
                    np.testing.assert_array_equal(left_state["values"], right_state["values"])
                    nodes.append(left.tree_.node_count)
                self.assertGreater(min(nodes), 1, "Even leaf=200 must exercise tree splitting")
                np.testing.assert_allclose(raw_serial, raw_parallel, rtol=0, atol=1e-10)
                clipped_serial, clipped_parallel = np.maximum(0, raw_serial), np.maximum(0, raw_parallel)
                np.testing.assert_allclose(clipped_serial, clipped_parallel, rtol=0, atol=1e-10)
                checks.append({"parameters": params, "threads_compared": [1, 24],
                    "all_tree_seeds_equal": True, "all_tree_nodes_and_values_exactly_equal": True,
                    "node_count_min": int(min(nodes)), "node_count_max": int(max(nodes)),
                    "total_nodes": int(sum(nodes)), "prediction_rows": query.shape[0],
                    "raw_prediction_max_absolute_difference": float(np.max(np.abs(raw_serial - raw_parallel))),
                    "clipped_prediction_max_absolute_difference": float(np.max(np.abs(clipped_serial - clipped_parallel))),
                    "atol": 1e-10, "rtol": 0, "passed": True})
        self.assertTrue(any(b["n_jobs"] == 24 and b["backend"] == "ThreadingBackend" for b in backends))
        self.assertEqual(len(checks), 2, "Both leaf specifications must pass before evidence is written")
        evidence = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Synthetic data only; actual production preprocessing, no final freeze or August access",
            "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
            "preprocessing": config, "input_matrix": {"format": features.format, "dtype": str(features.dtype),
                "shape": list(features.shape), "nnz": features.nnz, "training_rows": 5400, "query_rows": query.shape[0]},
            "observed_parallel_backends": backends, "comparisons": checks,
            "source_evidence": {"installed_forest_file": str(Path(_forest.__file__).resolve()),
                "installed_forest_sha256": selection.final.sha256_file(Path(_forest.__file__)),
                "seed_order": "Forest.fit constructs seeded estimators sequentially before Parallel enumerates those trees",
                "prediction_order": "Threaded predict accumulates tree predictions under a lock; last-bit summation differences are possible"},
            "limitation": "Small synthetic equivalence is not a full-data guarantee; final and reproduction use the identical frozen thread count and unchanged 1e-10 tolerance"}
        receipt_path = os.environ.get("AIRLINE_THREAD_EQUIVALENCE_RECEIPT")
        if receipt_path:
            destination = Path(receipt_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as receipt:
                json.dump(evidence, receipt, indent=2)


if __name__ == "__main__":
    unittest.main()
