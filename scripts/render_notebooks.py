"""Create and top-to-bottom execute Chinese P0 notebooks from archived artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import AsyncKernelManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAMES = ["03_EDA.ipynb", "04_RQ1机场分析.ipynb", "05_RQ2预测与解释.ipynb"]

SETUP = '''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown

if "ROOT" not in globals():
    ROOT = Path.cwd()
    while not (ROOT / "scripts/build_features.py").exists():
        if ROOT.parent == ROOT:
            raise RuntimeError("请在 Airline-delay 仓库中运行 Notebook")
        ROOT = ROOT.parent
if "RUN_DIR" not in globals():
    candidates = [p for p in (ROOT / "experiments/runs").glob("R*")
                  if (p / "metrics.csv").exists() and (p / "eda_summary.json").exists()
                  and (p / "rq1_summary.json").exists()]
    if not candidates:
        raise RuntimeError("请先运行 P0 开发分析，或手动设置 RUN_DIR")
    RUN_DIR = max(candidates, key=lambda p: int(p.name[1:]))
TABLES = RUN_DIR / "tables"
FIGURES = RUN_DIR / "figures"
def table(name):
    return pd.read_csv(TABLES / name)
def figure(name):
    display(Image(filename=str(FIGURES / name)))
def receipt(name):
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))
display(Markdown(f"**归档运行：{RUN_DIR.name}**。本 Notebook 读取该轮表格与图，不重新拟合模型。"))'''


def _notebook(cells):
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Airline delay (.venv)", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}
    notebook.cells = [nbformat.v4.new_markdown_cell(text) if kind == "md" else nbformat.v4.new_code_cell(text)
                      for kind, text in cells]
    return notebook


def notebook_templates():
    eda = [
        ("md", "# P0：开发期探索性分析\n\n分析窗口为 2026 年 4–7 月，以 **prediction_time 的 UTC 半开区间**界定。8 月作为最终测试月，开发期不查看其目标分布。这里的分析单位是符合 RQ2 条件的同机相邻航段衔接；不是所有 BTS 航班。"),
        ("code", SETUP),
        ("md", "## 1. 样本定义与排除\n\n预测时点为上游实际到达，标签时点为下游实际出发。保留有限且非负的下游出发延误，要求 prediction_time < label_time；同刻衔接因为下游已经出发而排除。上游准点、非隔夜与隔夜记录均可进入 RQ2，隔夜误差另外评价。"),
        ("code", 'summary = receipt("eda_summary.json")\ndisplay(pd.DataFrame([{"指标": k, "值": v} for k, v in summary.items() if k not in ["feature_audit", "heatmap_note"]]))\naudit = summary.get("feature_audit", {}).get("counts", {})\ndisplay(pd.DataFrame(list(audit.items()), columns=["顺序样本流", "数量"]))'),
        ("md", "## 2. 目标分布与月份漂移\n\n目标使用下游**非负出发延误**，缺失标签不填零。直方分组覆盖全部右尾，最后一档保留极端延误；分位数表同时给出样本量、均值、中位数和 p90/p95/p99，避免均值掩盖偏态。月份只用于漂移和分层，不进入 G0 默认模型。"),
        ("code", 'display(table("eda_monthly_target.csv"))\ndisplay(table("eda_target_bins.csv"))\nfigure("eda_target_distribution.png")\nfigure("eda_monthly_target.png")'),
        ("md", r"## 3. 上游延误 × 计划周转" + "\n\n" + r"计划周转是计划到达至计划出发的间隔，不等于已知作业缓冲。色图分开呈现支持量与平均 \(\Delta = D_{down}^{dep} - D_{up}^{arr}\)（分钟）。空样本均值格保留缺失；计数格明确为零。负值表示样本衔接中的净延误减少，不自动代表机场造成了减少。"),
        ("code", 'figure("eda_turnaround_count.png")\nfigure("eda_turnaround_mean_delta.png")\ndisplay(table("eda_turnaround_heatmap.csv").head(12))'),
        ("md", "## 4. 机场、航线、航司支持与字段缺失\n\n图按衔接量选出最多的 20 个机场，完整支持表保留所有机场。BTS 航班量与有效同机衔接量是不同分母。稀疏机场或类别可能导致估计不稳定；编码与填补只在每折训练段拟合。此处只描述组成，不作因果判断。"),
        ("code", 'figure("eda_airport_support.png")\nfor name in ["airport", "route", "airline"]:\n    display(Markdown(f"**{name} 支持（前20行）**"))\n    display(table(f"eda_support_{name}.csv").head(20))\ndisplay(table("eda_feature_missing.csv"))'),
        ("md", "## 5. P0 的解释边界\n\n这轮用于建立统一样本、时间规则和初始参照。天气、飞机家族、近期机场运行等信息尚未加入；后续每次只改变一个研究因素，继续使用相同评估 ID。RQ1 主样本与本页 RQ2 样本不同，请参阅下一本 Notebook。"),
    ]
    rq1 = [
        ("md", "# P0：RQ1 机场延误净变化（原始描述）\n\n研究问题：哪些机场的同机相邻航段衔接平均吸收或放大延误？本阶段先报告开发期原始汇总；共同支持标准化、日期块区间和阈值敏感性留待 P5。"),
        ("code", SETUP),
        ("md", r"## 1. 主样本与三个不同的量" + "\n\n" + r"主样本来自有效衔接，要求上游非负到达延误至少 15 分钟、计划非隔夜。RQ1 允许实际地面时间为 0；不套用 RQ2 的严格预测早于出发条件。开发期先按上游实际到达 UTC 4–7 月筛选，再分析目标。" + "\n\n" + r"\[\Delta_i = D_{down,i}^{dep} - D_{up,i}^{arr}\]" + "\n\n" + r"**每次平均净变化**的单位是分钟/次衔接；**累计增加** \(S_+=\sum \max(\Delta_i,0)\)、**累计减少** \(S_-=\sum\max(-\Delta_i,0)\) 和 **累计净变化** \(S_{net}=\sum\Delta_i=S_+-S_-\) 的单位是样本总分钟。累计量受样本量影响，不能直接当作机场表现。"),
        ("code", 'summary = receipt("rq1_summary.json")\ndisplay(pd.DataFrame(list(summary["counts"].items()), columns=["顺序样本流", "数量"]))\ndisplay(pd.DataFrame([{k: v for k, v in summary.items() if k not in ["counts", "interpretation"]}]))\nassert summary["delta_identity_checked"]'),
        ("md", "## 2. 机场原始均值、分布与支持\n\n按样本量选择最大的 20 个机场展示；图内顺序不构成标准化排名。表中 mean/median/p10/p90、减少/增加/不变比例均使用同一主样本。完整机场表归档于 tables/rq1_airport_raw.csv。"),
        ("code", 'raw = table("rq1_airport_raw.csv")\ndisplay(raw.head(20))\nfigure("rq1_raw_airport_mean.png")'),
        ("md", "## 3. 累计增加、累计减少与净值\n\n较大机场的累计分钟可能很高，即使其每次平均净变化较小。必须将累计量与样本量及每次平均量一起阅读。逐连接与逐机场均验证 Snet = S+ − S−，避免正负变化混算。"),
        ("code", 'figure("rq1_cumulative_delay.png")\ndisplay(raw[["airport_id", "airport_code", "n", "mean_delta", "s_plus", "s_minus", "s_net"]].head(20))'),
        ("md", "## 4. 分月支持及上游延误 × 计划周转\n\n逐月表帮助识别月份组成变化。热图分别列计数与平均净变化：低支持格不足以支持稳定结论。没有正式最低周转时间，本项目只使用计划周转，不把它直接命名为可吸收缓冲。"),
        ("code", 'monthly = table("rq1_airport_monthly.csv")\ndisplay(monthly.groupby("month_utc")["n"].sum().rename("connections").to_frame())\ndisplay(monthly.head(20))\nfigure("rq1_turnaround_count.png")\nfigure("rq1_turnaround_mean_delta.png")'),
        ("md", "## 5. 当前可以与尚不能得出的结论\n\n负 Δ 表示这次衔接保留的下游延误少于上游晚到；下游仍可能延误。当前机场差异尚未按共同上游延误和计划周转分布标准化，也没有抽样不确定性区间，因此只能描述本样本的原始表现，不能解释为机场因果责任或正式优劣排名。"),
    ]
    rq2 = [
        ("md", "# P0：RQ2 下游出发延误预测与解释\n\n研究问题：上游飞机实际到达后，能否预测下游出发延误？P0 使用三个朴素基线与计划信息 G0 线性回归。此页从该轮完整预测与指标归档读取结果，不重新训练。"),
        ("code", SETUP),
        ("md", "## 1. 预测时点、时间折与标签成熟\n\n固定 UTC 前向验证：4 月训练→5 月验证；4–5 月训练→6 月验证；4–6 月训练→7 月验证。预测时间窗为半开区间。训练记录不仅预测时点须早于截止，实际出发标签也须严格早于该折截止；在月初同刻才成熟的标签不能用于训练。8 月不参与本轮开发。"),
        ("code", 'splits = pd.read_csv(RUN_DIR / "split_audit.csv")\ndisplay(splits)\nmodel_manifest = receipt("model_manifest.json")\ndisplay(pd.DataFrame([{k: model_manifest[k] for k in ["evaluation_rows", "evaluation_ids_sha256", "split_sha256"]}]))'),
        ("md", "## 2. G0 特征和朴素基线\n\nG0 仅包含机场、航线、航司、航程、计划飞行时长、星期、计划出发小时、计划周转。上游到达延误仅用于本轮 carry-forward 业务基线，还未加入 G0。实际地面时间、下游实际出发和结果字段禁止入模。训练均值/中位数在每折训练段估计；填补、类别合并、编码和缩放均在该训练段拟合。"),
        ("code", 'display(pd.DataFrame({"G0特征": model_manifest["features"]}))\ndisplay(pd.DataFrame([model_manifest["effective_preprocessing"]]))'),
        ("md", "## 3. 每折与加权整体误差\n\n主要指标为 MAE，RMSE 反映大误差的影响。所有模型统一先将负预测截断为零。整体 MAE 使用所有验证记录的绝对误差总量除以总样本数，整体 RMSE 也由平方误差总量计算；不是三个折指标的简单平均。以下结果只用于建立初始参照。"),
        ("code", 'metrics = pd.read_csv(RUN_DIR / "metrics.csv")\nmain = metrics.loc[metrics["group"].eq("all")]\ndisplay(main[["model", "fold", "n", "mae", "rmse", "target_ge15_share"]])\nfigure("rq2_metric_comparison.png")\npooled = main.loc[main.fold.eq("overall")].set_index("model")\ncomparison = pooled[["n", "mae", "rmse"]].copy()\ncomparison["MAE改善_vs_carry_forward_分钟"] = pooled.loc["carry_forward", "mae"] - comparison.mae\ndisplay(comparison)'),
        ("md", "## 4. 严重延误、隔夜及运行分组\n\n普通/严重组按真实非负目标预定义为 <60 / ≥60 分钟。这些是结果人群的误差分层，不能作为训练后筛选有利人群的方法。隔夜、机场、航司、星期和计划小时同时给出样本数；小样本组误差应谨慎解读。"),
        ("code", 'overall = metrics.loc[metrics.fold.eq("overall")]\nfor group in ["severity", "overnight", "day_of_week", "scheduled_hour"]:\n    display(Markdown(f"**{group}**"))\n    display(overall.loc[overall["group"].eq(group), ["model", "value", "n", "mae", "rmse"]])\nfor group in ["airport", "airline"]:\n    display(Markdown(f"**{group}（G0样本最多20组）**"))\n    display(overall.loc[overall["group"].eq(group) & overall.model.eq("g0_linear"), ["value", "n", "mae", "rmse"]].sort_values("n", ascending=False).head(20))'),
        ("md", "## 5. 下一轮比较与证据\n\n每折完整预测在 predictions/，模型在 models/，实际评估 connection_id 在 evaluation_ids.csv.gz。后续添加 G1、机型、机场运行与天气时使用同一评估 ID、相同时间折和线性模型设置，逐轮只改变一项研究因素。当前没有天气增益、复杂模型增益或最终 8 月测试结论。"),
    ]
    return dict(zip(NOTEBOOK_NAMES, [_notebook(eda), _notebook(rq1), _notebook(rq2)]))


def write_templates(root: Path = ROOT):
    destination = Path(root) / "notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    for name, notebook in notebook_templates().items():
        nbformat.write(notebook, destination / name)


def _rq2_figure(run_dir):
    import numpy as np
    import pandas as pd
    try:
        from .analyze_eda import BLUE, GOLD, plot_style, save_figure
    except ImportError:
        from analyze_eda import BLUE, GOLD, plot_style, save_figure
    import matplotlib.pyplot as plt
    metrics = pd.read_csv(run_dir / "metrics.csv")
    selected = metrics.loc[metrics["group"].eq("all") & metrics.fold.eq("overall")].set_index("model")
    order = ["training_mean", "training_median", "carry_forward", "g0_linear"]
    selected = selected.loc[order]
    labels = ["Training mean", "Training median", "Carry-forward", "G0 linear"]
    positions = np.arange(4)
    plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(positions - .18, selected.mae, width=.36, label="MAE", color=BLUE)
    ax.bar(positions + .18, selected.rmse, width=.36, label="RMSE", color=GOLD)
    ax.set(xticks=positions, xticklabels=labels, ylabel="Prediction error (min)",
           title=f"P0 development errors | pooled validation n={int(selected.n.iloc[0]):,}")
    ax.legend(frameon=False)
    save_figure(fig, run_dir / "figures/rq2_metric_comparison.png")


def render_notebooks(run_dir: str | Path, root: str | Path = ROOT, include_preparation: bool = True) -> dict:
    """Execute fresh notebook sources using this Python, archive outputs and HTML.

    Uses nbclient with an explicit interpreter command, so the kernel cannot silently
    select a global Python. Source templates are read from root/notebooks; executed
    copies are saved under the run directory, then copied to the main notebooks.
    """
    run_dir, root = Path(run_dir).resolve(), Path(root).resolve()
    for filename in ["eda_summary.json", "rq1_summary.json", "metrics.csv", "split_audit.csv", "model_manifest.json"]:
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(run_dir / filename)
    destination = run_dir / "notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    _rq2_figure(run_dir)
    exporter = HTMLExporter()
    paths = []
    names = (["01_航班清洗与衔接.ipynb"] if include_preparation and
             (root / "notebooks/01_航班清洗与衔接.ipynb").is_file() else []) + NOTEBOOK_NAMES
    for name in names:
        source = root / "notebooks" / name
        if not source.exists():
            write_templates(root)
        notebook = nbformat.read(source, as_version=4)
        if name.startswith("01_") and any(
                "from prepare_flights import run" in cell.source or "prepare_flights.run(" in cell.source
                for cell in notebook.cells if cell.cell_type == "code"):
            raise ValueError("Preparation notebook must read archived receipts; refusing to rerun flight preparation")
        # Erase outputs and old injected bindings when rerendering an executed copy.
        notebook.cells = [cell for cell in notebook.cells if "run_binding" not in cell.metadata.get("tags", [])]
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
        try:
            relative = run_dir.relative_to(root).as_posix()
            binding = f"ROOT = __import__('pathlib').Path.cwd()\nRUN_DIR = ROOT / {relative!r}"
        except ValueError:
            binding = f"ROOT = __import__('pathlib').Path.cwd()\nRUN_DIR = __import__('pathlib').Path({str(run_dir)!r})"
        parameter_cell = nbformat.v4.new_code_cell(binding, metadata={"tags": ["run_binding"]})
        notebook.cells.insert(1, parameter_cell)
        manager = AsyncKernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=300, resources={"metadata": {"path": str(root)}},
                                allow_errors=False, kernel_name="python3")
        client.execute(cleanup_kc=True)
        if any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs):
            raise RuntimeError(f"Notebook contains an error: {name}")
        executed = destination / name
        nbformat.write(notebook, executed)
        html, _ = exporter.from_notebook_node(notebook)
        executed.with_suffix(".html").write_text(html, encoding="utf-8")
        shutil.copy2(executed, source)
        paths.append(str(executed))
        print(f"Executed and exported {name}", flush=True)
    return {"executed_notebooks": paths, "interpreter": sys.executable}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--write-templates", action="store_true")
    args = parser.parse_args()
    if args.write_templates:
        write_templates()
    if args.run_dir:
        print(render_notebooks(args.run_dir))
    elif not args.write_templates:
        parser.error("Provide --run-dir or --write-templates")
