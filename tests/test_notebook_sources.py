"""Execute tiny cross-run display notebooks, without targets, estimators or old outputs."""
import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import unquote

import nbformat

from scripts import render_research_notebooks as notebooks


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value), encoding="utf-8")
    elif isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


GUARD = r'''import builtins, io
from sklearn.pipeline import Pipeline
import joblib
READ_AUDIT = []
def forbidden(*args, **kwargs):
    raise AssertionError("Display notebooks must not train or load model/target objects")
Pipeline.fit = forbidden
pd.read_pickle = forbidden
joblib.load = forbidden
original_open, original_io_open = builtins.open, io.open
def checked_open(original, file, mode="r", *args, **kwargs):
    if isinstance(file, (str, bytes, Path)):
        candidate = Path(file).resolve()
        if candidate.is_relative_to(ROOT):
            relative = candidate.relative_to(ROOT)
            if set(relative.parts) & {"raw", "processed", "predictions", "models", "data", "residuals"} or candidate.suffix in {".pkl", ".pickle", ".joblib"}:
                raise AssertionError("Forbidden data read: " + str(relative))
            if "r" in mode:
                READ_AUDIT.append(relative.as_posix())
    return original(file, mode, *args, **kwargs)
builtins.open = lambda file, mode="r", *args, **kwargs: checked_open(original_open, file, mode, *args, **kwargs)
io.open = lambda file, mode="r", *args, **kwargs: checked_open(original_io_open, file, mode, *args, **kwargs)
'''


def guarded(template):
    template.cells.insert(2, nbformat.v4.new_code_cell(GUARD))
    template.cells.append(nbformat.v4.new_code_cell('print("READ_AUDIT_JSON=" + json.dumps(sorted(set(READ_AUDIT))))'))
    return template


def output_text(notebook):
    return "\n".join(output.get("text", "") + "\n" + output.get("data", {}).get("text/markdown", "")
                     + "\n" + output.get("data", {}).get("text/plain", "")
                     for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []))


def markdown_links(notebook, notebook_dir):
    links = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            text = output.get("data", {}).get("text/markdown", "")
            for name, target in re.findall(r"\[([^]]+)\]\(([^)]+)\)", text):
                links.append((name, (notebook_dir / unquote(target)).resolve()))
    return links


def single_eda(run, marker):
    write(run / "tables/eda_complete_summary.json", {"marker": marker})
    for filename in ["single_flight_delay_stats.csv", "single_flight_operations.csv", "airline_aircraft_coverage.csv",
                     "airline_aircraft_models.csv", "airport_activity.csv", "rq2_weather_missing_coverage.csv"]:
        write(run / "tables" / filename, "marker,n\n" + marker + ",1\n")
    for filename in ["monthly_delay_quantiles.png", "delay_sign_counts.png"]:
        write(run / "figures" / filename, PNG)


class NotebookSources(unittest.TestCase):
    def test_selection_notebook_reads_explicit_eda_and_explanation_sources(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rq1, rq2, selection = [root / "experiments/runs" / rid for rid in ["R0018", "R0017", "R0027"]]
            single_eda(rq1, "SINGLE_FLIGHT_FROM_R18")
            write(rq1 / "config.json", {"marker": "RQ1_CONFIG_R18"})
            write(rq1 / "manifest.json", {"status": "completed"})
            # A wrong-source decoy must never appear in the selection explanation section.
            write(rq1 / "explanations/old/explain_summary.json", {"marker": "WRONG_OLD_EXPLANATION"})
            write(rq2 / "eda_summary.json", {"marker": "RQ2_FROM_R17"})
            for filename in ["eda_monthly_target.csv", "eda_target_bins.csv", "eda_feature_missing.csv",
                             "eda_support_airport.csv", "eda_support_route.csv", "eda_support_airline.csv"]:
                write(rq2 / "tables" / filename, "marker,n\nRQ2_FROM_R17,1\n")
            for filename in ["eda_target_distribution.png", "eda_monthly_target.png", "eda_airport_support.png",
                             "eda_turnaround_count.png", "eda_turnaround_mean_delta.png"]:
                write(rq2 / "figures" / filename, PNG)
            write(selection / "manifest.json", {"status": "running"})
            write(selection / "selection/model_selection_decision.json", {"marker": "CHOICE_FROM_R27", "primary_model_name": "training_median"})
            write(selection / "selection/candidates.csv", "run_id,mae\nR0017,1\n")
            write(selection / "explanations/R0017/explain_summary.json", {"marker": "EXPLANATION_FROM_R27"})
            explanation_image = selection / "explanations/R0017/permutation_2026-07.png"
            write(explanation_image, PNG)
            write(selection / "explanations/R0017/permutation_summary.csv", "fold,group,mean_delta_mae\n2026-07,G0,1\n")
            result = notebooks._render(guarded(notebooks.analysis_template()), "06_test.ipynb", rq1, root,
                                       rq2_eda_run_dir=rq2, output_dir=selection, update_main=True)
            executed = nbformat.read(result["executed_notebook"], as_version=4)
            text = output_text(executed)
            for marker in ["SINGLE_FLIGHT_FROM_R18", "RQ2_FROM_R17", "CHOICE_FROM_R27", "EXPLANATION_FROM_R27"]:
                self.assertIn(marker, text)
            self.assertNotIn("WRONG_OLD_EXPLANATION", text)
            links = markdown_links(executed, selection / "notebooks")
            for source in [rq1 / "config.json", rq1 / "tables/eda_complete_summary.json", rq2 / "eda_summary.json",
                           rq2 / "tables/eda_monthly_target.csv", rq2 / "figures/eda_target_distribution.png",
                           selection / "selection/model_selection_decision.json", explanation_image]:
                self.assertIn((source.name, source.resolve()), links)
            self.assertTrue(all(path.is_file() for _, path in links))
            self.assertFalse((rq1 / "notebooks").exists())
            self.assertFalse(result["raw_data_read"])
            self.assertFalse(result["trained"])
            self.assertIn("READ_AUDIT_JSON=", text)
            self.assertEqual(result["source_runs"]["rq2_eda"], str(rq2.resolve()))
            display_path = root / "notebooks/06_test.ipynb"
            displayed = nbformat.read(display_path, as_version=4)
            self.assertEqual(markdown_links(displayed, display_path.parent), links)
            source_binding = next(cell for cell in executed.cells if "run_binding" in cell.metadata.get("tags", []))
            display_binding = next(cell for cell in displayed.cells if "run_binding" in cell.metadata.get("tags", []))
            self.assertIn("experiments/runs/R0027/notebooks", source_binding.source)
            self.assertIn("NOTEBOOK_DIR = ROOT / 'notebooks'", display_binding.source)
            namespace = {}
            with patch.object(Path, "cwd", return_value=root), redirect_stdout(io.StringIO()):
                exec(display_binding.source, namespace)
                exec(notebooks.SETUP, namespace)
            self.assertEqual(namespace["NOTEBOOK_DIR"], display_path.parent)
            fresh_link = namespace["link_file"](rq2 / "eda_summary.json")
            relative = re.search(r"\]\(([^)]+)\)", fresh_link).group(1)
            self.assertEqual((display_path.parent / unquote(relative)).resolve(), (rq2 / "eda_summary.json").resolve())
            archived_images = [o["data"]["image/png"] for c in executed.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
            display_images = [o["data"]["image/png"] for c in displayed.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
            self.assertEqual(archived_images, display_images)

    def test_final_notebook_links_prediction_and_rq1_eda_from_actual_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "experiments/runs/R0101"
            rq1 = root / "experiments/runs/R0102"
            output = root / "experiments/runs/R0103"
            write(prediction / "model_selection.json", {"marker": "FROZEN_PREDICTION_SOURCE", "status": "frozen"})
            write(prediction / "metrics.csv", "model,fold,group,value,n,mae,rmse\ntraining_median,overall,all,all,1,1,2\n")
            single_eda(rq1, "FINAL_COMPLETE_EDA_FROM_RQ1")
            write(rq1 / "config.json", {})
            write(rq1 / "rq1_complete_summary.json", {"marker": "FINAL_RQ1_SOURCE"})
            write(rq1 / "rq1_standardized/frozen_reference.json", {"supported": False})
            result = notebooks._render(guarded(notebooks.final_template()), "07_test.ipynb", prediction, root,
                                       rq1_run_dir=rq1, output_dir=output, update_main=True)
            executed = nbformat.read(result["executed_notebook"], as_version=4)
            text = output_text(executed)
            self.assertIn("FINAL_COMPLETE_EDA_FROM_RQ1", text)
            self.assertIn("FINAL_RQ1_SOURCE", text)
            links = markdown_links(executed, output / "notebooks")
            for source in [prediction / "metrics.csv", rq1 / "tables/eda_complete_summary.json",
                           rq1 / "figures/monthly_delay_quantiles.png", rq1 / "rq1_complete_summary.json"]:
                self.assertIn((source.name, source.resolve()), links)
            self.assertTrue(all(path.is_file() for _, path in links))
            displayed = nbformat.read(root / "notebooks/07_test.ipynb", as_version=4)
            self.assertEqual(markdown_links(displayed, root / "notebooks"), links)
            self.assertFalse((prediction / "notebooks").exists())
            self.assertFalse((rq1 / "notebooks").exists())
            self.assertIn("READ_AUDIT_JSON=", text)

    def test_display_copy_preserves_external_fragment_and_embedded_images(self):
        root = Path("D:/synthetic-root")
        source = root / "experiments/runs/R0001/notebooks"
        target = root / "notebooks"
        external = "**[website](https://example.com/report)** **[section](#details)** ![figure](../unchanged.png)"
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("pass", outputs=[
            nbformat.v4.new_output("display_data", data={"text/markdown": external}),
            nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(PNG).decode()}),
        ])])
        displayed = notebooks._display_copy(notebook, source, target, root)
        self.assertEqual(displayed.cells[0].outputs, notebook.cells[0].outputs)


if __name__ == "__main__":
    unittest.main()
