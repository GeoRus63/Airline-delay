"""Execute Chinese research notebooks from each run's archived artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import AsyncKernelManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAMES = ["02_天气与特征.ipynb", "03_EDA.ipynb", "04_RQ1机场分析.ipynb", "05_RQ2预测与解释.ipynb"]

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
                  if (p / "metrics.csv").exists() and (p / "model_manifest.json").exists()]
    if not candidates:
        raise RuntimeError("请先归档预测实验，或手动设置 RUN_DIR")
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
    templates = progressive_templates()
    templates.update({"03_EDA.ipynb": _notebook(eda), "04_RQ1机场分析.ipynb": _notebook(rq1)})
    return templates


def progressive_templates():
    shared = '''config = receipt("config.json") if (RUN_DIR / "config.json").exists() else {}
model_manifest = receipt("model_manifest.json")
features = model_manifest.get("features", config.get("features", []))
group_name = model_manifest.get("feature_group", config.get("feature_group", "未登记"))
models = model_manifest.get("models", [])
display(Markdown(f"本轮信息组：**{group_name}**；阶段：**{config.get('stage', '未登记')}**；模型：**{', '.join(models)}**。"))
display(pd.DataFrame([{"development_only": model_manifest.get("development_only"),
                       "parent_run_id": config.get("parent_run_id"),
                       "change": config.get("change"), "include_baselines": model_manifest.get("include_baselines")}]))'''
    feature_notebook = [
        ("md", "# 天气与特征：本轮生成、可用时点及覆盖\n\n本页读取已归档特征审计与字典。以下方法契约解释不同信息组；是否实际入模以本轮 manifest 的特征清单为准，不把尚未启用的信息组说成已完成实验。"),
        ("code", SETUP), ("code", shared),
        ("md", "## 1. 实际特征及生成审计\n\n目标与标签时刻是评价元数据，不是预测变量。只在训练折拟合插补、类别处理和缩放；缺失不能按无延误、晴天或已知零填充。"),
        ("code", '''display(pd.DataFrame({"实际入模字段": features}))
if (RUN_DIR / "feature_audit.json").exists():
    audits = receipt("feature_audit.json")
    for section, values in audits.items():
        display(Markdown(f"**{section}**"))
        display(pd.json_normalize(values) if isinstance(values, dict) else values)
else:
    display(Markdown("本轮未归档 feature_audit.json。"))
dictionary_path = RUN_DIR / "feature_dictionary.csv"
if dictionary_path.exists():
    display(pd.read_csv(dictionary_path))
else:
    display(Markdown("本轮未归档特征字典。"))'''),
        ("md", "## 2. 航班链、飞机映射及机场运行\n\n航班链历史默认使用预测前 24 小时已完成的航段，不重复计入当前上游；其已知到达延误单独作为字段。已知尾号没有观察到有效历史时，计数为零、均值缺失；未知尾号保留缺失。history_window_incomplete 是**已知左截断指示**，不是 flag=0 时的完整覆盖保证。源数据按当地 4 月起飞日期取样，前月起飞而延迟到达/出发的 carry-in 航班可能缺失。\n\nB43 是静态参考清单，以航司＋尾号连接，不能当作逐预测时点的历史机队快照。family 与 variant 单独比较，含截距线性模型不同时冗余展开二者。\n\n机场历史窗口是 \\( [t-60\\mathrm{min},t) \\)，只计已实际出发事件；计划负荷窗口是 \\( [t,t+60\\mathrm{min}) \\)。最终取消状态不能用于事后删去计划负荷。计划快照事前已知是研究假设，仓库没有逐时计划修订版本。"),
        ("md", "## 3. 天气时点契约\n\nMETAR 假设观测后 5 分钟可用，观测年龄最多 120 分钟；这是建模假设，不是源数据提供的接收时间。TAF 先选预测时点已可用的最新公告，再匹配查询时刻 max(下游计划出发, prediction_time)，并处理有效期、修订、取消/NIL 及临时/概率段。NOAA 仅作独立补缺分支，保留质量码、单位核对和逐值来源。缺天气保留样本及缺失标记，覆盖对照须核对评估 ID。"),
        ("md", "## 4. 实际缺失、映射与覆盖表\n\n以下仅展示本轮确实归档的表。没有对应表不代表该信息源覆盖完整，也不代表该信息组已经启用。"),
        ("code", '''coverage_paths = sorted({p for folder in [RUN_DIR, TABLES] if folder.exists()
                         for p in folder.glob("*.csv")
                         if any(word in p.stem.lower() for word in ["missing", "coverage", "match_rate", "unmatched"])})
for path in coverage_paths:
    display(Markdown(f"**{path.relative_to(RUN_DIR)}**"))
    data = pd.read_csv(path)
    display(Markdown(f"归档表共 {len(data):,} 行；下方展示前 50 行。"))
    display(data.head(50))
if not coverage_paths:
    display(Markdown("本轮没有单独归档缺失/覆盖表；请结合上方审计阅读。"))'''),
    ]
    rq2 = [
        ("md", "# RQ2：下游出发延误预测与解释\n\n研究问题：上游飞机实际到达后，能否预测下游出发延误？本页依据本轮配置、模型 manifest 与实际指标解释结果，不重新训练，也不预设使用哪一组特征或算法。"),
        ("code", SETUP), ("code", shared),
        ("md", "## 1. 时间折与成熟标签\n\n开发使用 4→5、4–5→6、4–6→7 月 UTC 前向验证。训练预测时点及标签时点均须严格早于训练截止；最终测试只能在规则冻结后进行。下表记录本轮实际切分和评估 ID 哈希。"),
        ("code", '''display(pd.read_csv(RUN_DIR / "split_audit.csv"))
display(pd.DataFrame([{key: model_manifest.get(key) for key in ["evaluation_rows", "evaluation_ids_sha256", "split_sha256", "development_only"]}]))'''),
        ("md", "## 2. 本轮信息、算法与预处理\n\n实际地面周转、下游实际出发及未来运营结果不能进入特征。朴素参照包括每折训练均值、中位数，以及上游非负到达延误原样传递；仅在本轮确实归档它们时展示比较。所有数据依赖的预处理均在训练折拟合。"),
        ("code", '''categorical = set(model_manifest.get("categorical_features", []))
display(pd.DataFrame({"字段": features, "类型": ["类别" if name in categorical else "数值" for name in features]}))
display(pd.json_normalize(model_manifest.get("effective_preprocessing", {})))
display(pd.json_normalize({key: model_manifest.get(key) for key in ["models", "model_name", "model_type", "model_params"]}))
display(Markdown("上游到达延误：" + ("**已作为预测变量加入**。" if "upstream_arr_delay_minutes" in features else "**不在本轮模型特征清单中**。")))'''),
        ("md", "## 3. 各月与整体预测误差\n\nMAE 为主要指标；RMSE 对大误差更敏感。预测统一非负截断。整体指标按全部评估行聚合绝对/平方误差，不能简单平均月折指标。是否改善还须检查评估 ID、切分和目标一致。"),
        ("code", '''metrics = pd.read_csv(RUN_DIR / "metrics.csv")
main = metrics.loc[metrics["group"].eq("all")]
columns = [name for name in ["model", "fold", "n", "mae", "rmse", "target_ge15_share"] if name in main]
display(main[columns])
if (FIGURES / "rq2_metric_comparison.png").exists():
    figure("rq2_metric_comparison.png")
pooled = main.loc[main.fold.eq("overall")].set_index("model")
comparison = pooled[["n", "mae", "rmse"]].copy()
for baseline in ["training_mean", "training_median", "carry_forward"]:
    if baseline in pooled.index:
        comparison[f"MAE改善_vs_{baseline}_分钟"] = pooled.loc[baseline, "mae"] - comparison.mae
display(comparison)
if not any(name in pooled.index for name in ["training_mean", "training_median", "carry_forward"]):
    display(Markdown("本轮未重算朴素基线。跨轮参照须使用下方已有比较文件，并确认相同评估 ID/切分。"))'''),
        ("md", "## 4. 严重延误、隔夜与运行分组\n\n普通/严重组按真实非负延误 <60 / ≥60 分钟预定义，是结果人群误差，不能用于事后挑选有利人群。各组同时列样本量；缺失或很小的组不强行解释。"),
        ("code", '''overall = metrics.loc[metrics.fold.eq("overall")]
for group in ["severity", "overnight", "day_of_week", "scheduled_hour"]:
    display(Markdown(f"**{group}**"))
    display(overall.loc[overall["group"].eq(group), ["model", "value", "n", "mae", "rmse"]])
for group in ["airport", "airline", "aircraft_family", "aircraft_variant"]:
    grouped = overall.loc[overall["group"].eq(group)]
    for model, rows in grouped.groupby("model", sort=False):
        display(Markdown(f"**{group} / {model}（样本最多的 20 组）**"))
        display(rows[["value", "n", "mae", "rmse"]].sort_values("n", ascending=False).head(20))'''),
        ("md", "## 5. 直接前轮比较与证据\n\n特征增量与算法差异分开比较，每轮只改变一个研究因素。只有同一评估样本、目标及切分下的差值才能作为直接改善证据。预测、模型、评估 ID 与配置保存在本轮归档；下方只显示已有比较文件，不自动重算或更改前轮结果。"),
        ("code", '''comparison_paths = {p for folder in [RUN_DIR, TABLES] if folder.exists()
                    for pattern in ["*comparison*.csv", "*parent*.csv", "*comparison*.json", "*parent*.json"]
                    for p in folder.glob(pattern)}
for key in ["parent_comparison_path", "comparison_path", "parent_comparison_file"]:
    value = config.get(key)
    if isinstance(value, str):
        candidate = Path(value)
        for path in ([candidate] if candidate.is_absolute() else [RUN_DIR / candidate, ROOT / candidate]):
            if path.is_file():
                comparison_paths.add(path)
for path in sorted(comparison_paths):
    display(Markdown(f"**{path.name}**"))
    display(pd.read_csv(path) if path.suffix == ".csv" else pd.json_normalize(json.loads(path.read_text(encoding="utf-8"))))
if not comparison_paths:
    display(Markdown("本轮尚无单独前轮比较文件；不从不同样本的指标直接推断增益。"))'''),
    ]
    return {"02_天气与特征.ipynb": _notebook(feature_notebook), "05_RQ2预测与解释.ipynb": _notebook(rq2)}


def write_templates(root: Path = ROOT, names=None):
    destination = Path(root) / "notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    selected = {"02_天气与特征.ipynb", "05_RQ2预测与解释.ipynb"} if names is None else set(names)
    for name, notebook in notebook_templates().items():
        if name in selected:
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
    manifest = json.loads((run_dir / "model_manifest.json").read_text(encoding="utf-8"))
    order = [name for name in manifest.get("models", []) if name in selected.index]
    order += [name for name in selected.index if name not in order]
    if not order:
        raise ValueError("No overall model metrics available for the comparison figure")
    selected = selected.loc[order]
    friendly = {"training_mean": "Training mean", "training_median": "Training median", "carry_forward": "Carry-forward"}
    labels = [friendly.get(name, name.replace('_', ' ')) for name in order]
    positions = np.arange(len(order))
    plot_style()
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.5), 4.8))
    ax.bar(positions - .18, selected.mae, width=.36, label="MAE", color=BLUE)
    ax.bar(positions + .18, selected.rmse, width=.36, label="RMSE", color=GOLD)
    ax.set(xticks=positions, xticklabels=labels, ylabel="Prediction error (min)",
           title=f"{manifest.get('feature_group', 'RQ2')} {'development' if manifest.get('development_only', True) else 'final'} errors | n={int(selected.n.iloc[0]):,}")
    ax.legend(frameon=False)
    save_figure(fig, run_dir / "figures/rq2_metric_comparison.png")


def render_notebooks(run_dir: str | Path, root: str | Path = ROOT, include_preparation: bool = True,
                     notebook_names=None, update_main: bool = True) -> dict:
    """Execute fresh notebook sources using this Python, archive outputs and HTML.

    Uses nbclient with an explicit interpreter command, so the kernel cannot silently
    select a global Python. Source templates are read from root/notebooks; executed
    copies are saved under the run directory, then copied to the main notebooks.
    """
    run_dir, root = Path(run_dir).resolve(), Path(root).resolve()
    for filename in ["metrics.csv", "split_audit.csv", "model_manifest.json"]:
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(run_dir / filename)
    destination = run_dir / "notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    _rq2_figure(run_dir)
    exporter = HTMLExporter()
    paths = []
    names = list(notebook_names) if notebook_names is not None else (
        (["01_航班清洗与衔接.ipynb"] if include_preparation and (run_dir / "cleaning_summary.json").is_file() and
         (root / "notebooks/01_航班清洗与衔接.ipynb").is_file() else []) +
        [name for name in NOTEBOOK_NAMES if not (
            (name.startswith("03_") and not (run_dir / "eda_summary.json").is_file()) or
            (name.startswith("04_") and not (run_dir / "rq1_summary.json").is_file()))])
    for name in names:
        source = root / "notebooks" / name
        if not source.exists():
            write_templates(root, names=[name])
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
        if update_main:
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
