"""Read-only Chinese P5/P6 notebooks from archived research tables and figures."""
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

SETUP = r'''from pathlib import Path
import hashlib
import json
import os
from urllib.parse import quote
import pandas as pd
from IPython.display import display, Markdown, Image

def notice(text):
    display(Markdown(text))

def link_file(path):
    relative = os.path.relpath(path, RUN_DIR / "notebooks").replace(chr(92), "/")
    return f"[{path.name}]({quote(relative, safe='/.:')})"

def archived_json(name, base=None, show=True, keys=None):
    path = (RUN_DIR if base is None else base) / name
    if not path.is_file():
        notice(f"**未归档：{path.name}**（{path.parent.name}）。")
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if show:
        notice(f"**{link_file(path)}**")
        selected = value if keys is None else {key: value.get(key, "未归档该字段") for key in keys}
        notice("```json\n" + json.dumps(selected, ensure_ascii=False, indent=2) + "\n```")
    return value

def archived_table(name, base=None, limit=100, filters=None):
    path = (RUN_DIR if base is None else base) / name
    if not path.is_file():
        notice(f"**未归档：{path.name}**（{path.parent.name}）。")
        return None
    value = pd.read_csv(path, float_precision="round_trip")
    full_n = len(value)
    if filters:
        for column, selected in filters.items():
            if column not in value:
                notice(f"**{path.name} 未归档字段：{column}；未显示该筛选。**")
                return value
            value = value.loc[value[column].eq(selected)]
    notice(f"**{link_file(path)}**：完整表 {full_n:,} 行；当前条件 {len(value):,} 行。")
    shown = value if limit is None else value.head(limit)
    with pd.option_context("display.max_columns", None, "display.max_rows", None if limit is None else max(limit, 10)):
        display(shown)
    if len(shown) < len(value):
        notice(f"上方仅展示前 {len(shown):,} 行，完整结果见该归档 CSV。")
    return value

def archived_image(name, base=None):
    path = (RUN_DIR if base is None else base) / name
    if not path.is_file():
        notice(f"**未归档图：{path.name}**。")
        return False
    notice(f"**{path.name}**")
    display(Image(filename=str(path)))
    return True

def archive_matches(pattern, base=None, kind="table", limit=60):
    base = RUN_DIR if base is None else base
    paths = sorted(base.glob(pattern))
    if not paths:
        notice(f"**未归档：{base.name}/{pattern}**。")
    for path in paths:
        if kind == "json":
            archived_json(str(path.relative_to(base)), base=base)
        elif kind == "image":
            archived_image(str(path.relative_to(base)), base=base)
        else:
            archived_table(str(path.relative_to(base)), base=base, limit=limit)

def rq1_reference(base):
    value = archived_json("rq1_standardized/frozen_reference.json", base=base,
                          keys=["schema_version", "frozen_on", "supported", "protocol", "interpretation"])
    if value is None:
        return
    notice("**冻结的机场集合 A 与共同主分层 C**")
    display(pd.DataFrame({"AirportID_in_A": value.get("airport_ids", [])}))
    display(pd.DataFrame({"master_cell_in_C": value.get("master_cells", [])}))
    rows = []
    for threshold, ref in value.get("references", {}).items():
        cells, weights = ref.get("cells", []), ref.get("weights", [])
        counts = ref.get("development_counts", [None] * len(cells))
        if len(cells) != len(weights) or len(cells) != len(counts):
            notice(f"**{threshold} 的冻结 cells/weights/counts 长度不一致，不能构造权重表。**")
            continue
        rows.extend({"threshold": threshold, "cell": cell, "weight_w": weight, "development_count": count}
                    for cell, weight, count in zip(cells, weights, counts))
    notice("**分阈值条件化的固定参考权重 w**")
    display(pd.DataFrame(rows))
    if not value.get("supported", False):
        notice("**该冻结参考未形成足够共同支持；不补零，不强行排序。**")

def model_comparison_runs():
    records, protocols = [], []
    if not MODEL_RUNS:
        notice("**未归档模型运行列表：本次未指定 model_runs，不自动选择其他实验。**")
        return
    for source in MODEL_RUNS:
        notice(f"### 模型归档：{source.name}")
        outer = archived_json("manifest.json", base=source, show=False)
        if outer is not None:
            notice(f"运行状态：**{outer.get('status', '未归档状态')}**。")
            if outer.get("config", {}).get("mode") == "final":
                notice("**这是最终测试归档，不纳入开发模型比较。**")
                continue
        manifest = archived_json("model_manifest.json", base=source, show=False)
        if manifest is None:
            continue
        if manifest.get("development_only") is False:
            notice("**这是最终测试模型，不纳入 P5 开发比较。**")
            continue
        notice("```json\n" + json.dumps({key: manifest.get(key, "未归档该字段") for key in
            ["feature_group", "model_name", "model_type", "models", "features", "categorical_features", "numeric_features",
             "model_params", "effective_preprocessing", "evaluation_rows", "evaluation_ids_sha256", "split_sha256"]}, ensure_ascii=False, indent=2) + "\n```")
        metrics = archived_table("metrics.csv", base=source, filters={"group": "all"}, limit=None)
        if metrics is not None and {"fold", "model", "mae", "rmse", "n"}.issubset(metrics):
            for row in metrics.loc[metrics.fold.eq("overall")].to_dict("records"):
                records.append({"run_id": source.name, "model": row["model"], "n": row["n"], "mae": row["mae"], "rmse": row["rmse"],
                                "feature_group": manifest.get("feature_group"), "model_type": manifest.get("model_type"),
                                "evaluation_ids_sha256": manifest.get("evaluation_ids_sha256"), "split_sha256": manifest.get("split_sha256")})
            protocols.append((manifest.get("evaluation_ids_sha256"), manifest.get("split_sha256")))
        archive_matches("**/comparison_metrics.csv", base=source, limit=None)
        archive_matches("**/comparison.json", base=source, kind="json")
        archive_matches("**/comparison_error_differences.png", base=source, kind="image")
    if records:
        notice("### 所指定开发运行的整体指标")
        combined = pd.DataFrame(records)
        display(combined)
        comparable = bool(protocols) and all(a and b for a, b in protocols) and len(set(protocols)) == 1
        if comparable:
            notice("这些归档声明相同的评估 ID 与切分哈希；完整逐行标签/时间一致性及配对区间仍以各 comparison.json/CSV 为证据。")
            for baseline in ["training_mean", "training_median", "carry_forward"]:
                values = combined.loc[combined.model.eq(baseline), "mae"]
                if len(values) and values.max() - values.min() <= 1e-10:
                    combined[f"delta_MAE_vs_{baseline}"] = combined.mae - values.iloc[0]
                elif len(values):
                    notice(f"**同名 {baseline} 的 MAE 在归档间不一致；不合并为单一参照。**")
            display(combined)
            notice("增量表为候选 MAE 减参照 MAE：正值表示更差；不是配对置信区间，也不是因果效应。")
        else:
            notice("**评估 ID/切分哈希缺失或不一致：上表仅陈列各轮结果，不据此判断模型改善。**")

def rq1_reference_identity(base):
    copied = base / "rq1_standardized/frozen_reference.json"
    if not copied.is_file():
        notice("**未归档最终 RQ1 的冻结参考副本，无法核对参考身份。**")
        return
    observed = hashlib.sha256(copied.read_bytes()).hexdigest()
    complete = archived_json("rq1_complete_summary.json", base=base, show=False)
    expected = None if complete is None else complete.get("frozen_reference_sha256")
    evidence = {"copied_reference_sha256": observed, "complete_summary_sha256": expected,
                "matches_complete_summary": observed == expected if expected else "未归档待比较 SHA"}
    config = archived_json("config.json", base=base, show=False)
    origin = None if config is None else config.get("rq1_frozen_reference_path")
    if origin:
        original = Path(origin)
        original = original if original.is_absolute() else ROOT / original
        if original.is_file():
            original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
            evidence.update(development_reference_path=origin, development_reference_sha256=original_sha,
                            matches_development_reference=original_sha == observed)
        else:
            evidence["development_reference_identity"] = "源参考路径未归档/不可用，未额外声明相同"
    else:
        evidence["development_reference_identity"] = "未归档开发参考源路径"
    display(pd.json_normalize(evidence))
    if evidence.get("matches_complete_summary") is False or evidence.get("matches_development_reference") is False:
        notice("**参考 SHA 不一致：不能据此宣称最终沿用了相同冻结参考。**")

notice(f"**阅读归档：{RUN_DIR.name}**。所有单元仅读取已保存结果，不训练、不重算研究样本或选择参数。")
'''


def _cells(items):
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Airline delay (.venv)", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}
    notebook.cells = [nbformat.v4.new_markdown_cell(source) if kind == "md" else nbformat.v4.new_code_cell(source)
                      for kind, source in items]
    return notebook


def analysis_template():
    return _cells([
        ("md", "# P5：共同支持、完整 EDA 与开发模型比较\n\n本 Notebook 整理开发阶段 2026 年 4–7 月的既有证据。RQ1 分析同机衔接中的延误净变化；RQ2 评价预测误差。两者的样本、分母和解释目标不同。未归档的文件会明确标出，不用文字代替实际结果。"),
        ("code", SETUP),
        ("md", "## 1. 运行范围与完整探索性分析\n\n单航段原始有符号延误与衔接预测的非负目标分别展示。月份、机场、星期/小时、航司和机型覆盖用来识别组成与缺失；B43 2024/2025 是参考快照，不是逐预测时刻的历史机队。所有开发目标统计应来自 4–7 月。"),
        ("code", r'''archived_json("config.json")
archived_json("tables/eda_complete_summary.json")
archived_json("eda_summary.json")
for name in ["single_flight_delay_stats.csv", "single_flight_operations.csv", "airline_aircraft_coverage.csv",
             "airline_aircraft_models.csv", "airport_activity.csv", "rq2_weather_missing_coverage.csv",
             "eda_monthly_target.csv", "eda_target_bins.csv", "eda_feature_missing.csv",
             "eda_support_airport.csv", "eda_support_route.csv", "eda_support_airline.csv"]:
    archived_table("tables/" + name)
for name in ["monthly_delay_quantiles.png", "delay_sign_counts.png", "eda_target_distribution.png",
             "eda_monthly_target.png", "eda_airport_support.png", "eda_turnaround_count.png", "eda_turnaround_mean_delta.png"]:
    archived_image("figures/" + name)'''),
        ("md", r"## 2. RQ1 定义与原始累计量" + "\n\n" + r"主分析为上游非负到达延误至少 15 分钟、计划非隔夜的有效同机衔接。\(\Delta=D^{dep}_{down}-D^{arr}_{up}\)。每次平均净变化以分钟/衔接计；\(S_+=\sum\max(\Delta,0)\)、\(S_-=\sum\max(-\Delta,0)\)、\(S_{net}=S_+-S_-\) 为样本累计分钟，受流量影响。负 Δ 不保证下游准点，更不证明机场造成了恢复。"),
        ("code", r'''archived_json("rq1_complete_summary.json")
archived_json("rq1_summary.json")
archived_table("tables/rq1_airport_raw.csv")
archived_table("tables/rq1_airport_monthly.csv")
archived_image("figures/rq1_raw_airport_mean.png")
archived_image("figures/rq1_cumulative_delay.png")'''),
        ("md", r"## 3. 冻结共同支持：机场 A、分层 C 与权重 w" + "\n\n" + r"主比较只对上游到达延误和计划周转做标准化，\(\hat\mu_a=\sum_{c\in C}w_c\bar\Delta_{a,c}\)。机场集合、共同分层与参考权重在开发期冻结；每个阈值条件化同一主参考，不能分别挑选一套有利机场。天气与航司只作辅助分层。计划周转不是已知最低作业缓冲，实际地面时间不作为调整变量。"),
        ("code", r'''rq1_reference(RUN_DIR)
archived_json("rq1_standardized/rq1_standardization_summary.json")
archived_table("rq1_standardized/selection_attempts.csv", limit=None)
archived_table("rq1_standardized/rq1_cells.csv", limit=100)'''),
        ("md", "## 4. 机场估计、日期块区间与阈值敏感性\n\n机场估计同时给出保留样本、保留率、支持原因和区间。不支持的结果保留缺失，不填零或强行排名。以同一 UTC 日期块共同重抽样，区间表示抽样不确定性；成对机场差异使用同一重复样本，未做多重比较校正。>0、≥5、≥15 分钟的方向一致率与 Spearman 只在共同可比机场上计算。"),
        ("code", r'''archived_table("rq1_standardized/rq1_airports.csv", limit=None, filters={"threshold": "ge15"})
archived_table("rq1_standardized/rq1_pairwise_differences.csv")
archived_table("rq1_standardized/rq1_threshold_stability.csv", limit=None)
archived_image("figures/rq1_standardized_airports.png")
archived_image("figures/rq1_threshold_stability.png")'''),
        ("md", "## 5. 按月稳定性与网络导航\n\n逐月复用同一冻结 A/C/w；低支持月份仍保留缺失。经纬度布局图的节点大小表示有效衔接数，节点颜色表示共同支持中的每次标准化净变化，灰色表示不支持；有向边宽表示 BTS 航班量，不表示传播强度。图仅显示有限主要航线及面板内连线，完整节点/边表保留全部范围。"),
        ("code", r'''archived_table("rq1_standardized/rq1_airports_monthly.csv")
archive_matches("rq1_standardized/monthly/*/rq1_standardization_summary.json", kind="json")
archived_table("rq1_standardized/network_nodes.csv")
archived_table("rq1_standardized/network_edges.csv")
archived_image("figures/rq1_network_geographic.png")'''),
        ("md", "## 6. 动态模型与基线比较\n\n每轮实际算法、特征、参数和误差来自该轮 manifest/metrics。训练中位数直接对应绝对误差目标，训练均值对应平方误差目标；目标偏态时两种常数基线的 MAE 可以明显不同。carry-forward 是已知上游延误原样传递，也不应与常数基线混称。学习模型若没有优于中位数，应如实保留该结果。特征增益和算法增益分开比较，同样本差值才有直接意义。"),
        ("code", "model_comparison_runs()"),
        ("md", "## 7. 解释分析与待完成证据\n\n分组置换把机场/航线与其他计划信息一同置换，保留组内结构，但仍可能打破与其他组的相关性。置换差值、编码系数和 PDP 都是模型与样本下的描述，不是因果效应。PDP 仅在固定开发样本支持范围选网格；线性机场/航线共线时不能把单独系数解释为机场作用。"),
        ("code", r'''archive_matches("**/explain_summary.json", kind="json")
archive_matches("**/permutation_summary.csv", limit=None)
archive_matches("**/permutation_2026-07.png", kind="image")
archive_matches("**/pdp_values.csv")
archive_matches("**/pdp_support.csv")
archive_matches("**/pdp_2026-07_*.png", kind="image")'''),
    ])


def final_template():
    return _cells([
        ("md", "# P6：冻结方案的最终评估\n\n本 Notebook 只读取已归档的最终预测、指标及研究凭据，不重新训练、不重新访问原始 8 月目标，也不根据测试结果返回调参。主指标候选可能是朴素基线；模型选择必须忠于开发结果。"),
        ("code", SETUP),
        ("md", "## 1. 冻结选择与执行配置\n\n冻结文件包含开发选择理由、预先指定的主模型、最佳学习模型的特征/算法/参数、预处理和评估协议。最终一次同时报告冻结学习模型与均值/中位数/carry-forward；主模型字段与学习模型名称分开记录。冻结 SHA 与源码/数据 SHA 在正式访问前核对。"),
        ("code", r'''freeze = archived_json("model_selection.json", show=False)
if freeze is not None:
    for key in ["schema_version", "status", "frozen_at_utc", "selection", "primary_model_name", "payload_sha256"]:
        notice("```json\n" + json.dumps({key: freeze.get(key, "未归档该字段")}, ensure_ascii=False, indent=2) + "\n```")
    notice("**冻结科学配置**")
    display(pd.json_normalize(freeze.get("config", {})))
    notice("**冻结评估协议**")
    notice("```json\n" + json.dumps(freeze.get("protocol", {}), ensure_ascii=False, indent=2) + "\n```")
archived_json("final_effective_config.json")
archived_json("final_result.json", keys=["mode", "primary_model_name", "model_name", "models", "evaluation_rows",
              "freeze_manifest_sha256", "freeze_payload_sha256", "access_ledger_path", "utc_boundary_audit", "primary_metric"])'''),
        ("md", "## 2. 源码和数据冻结凭据\n\n下列记录是访问前实际验证的输入凭据；天气版本 manifest 及其输入/输出也应被固定。归档中缺失某一凭据时只显示未归档，不宣称已经验证。Notebook 不重新读取原始测试数据来核对这些条目。"),
        ("code", r'''if freeze is not None:
    for kind in ["source_files", "data_files"]:
        notice(f"**冻结 {kind}**")
        if kind in freeze:
            display(pd.DataFrame(freeze[kind]))
        else:
            notice("**未归档该冻结凭据列表。**")
archived_json("final_verified_inputs.json")'''),
        ("md", r"## 3. UTC 时间窗与成熟标签" + "\n\n" + r"训练使用 \(2026\text{-}04\text{-}01\le prediction\_time<2026\text{-}08\text{-}01\)，并严格要求 \(label\_time<2026\text{-}08\text{-}01\)。测试预测时点属于 UTC 8 月半开窗 [8 月 1 日, 9 月 1 日)。月初同刻才成熟的训练标签排除，测试中缺失结果不补零。当地 8 月但预测时点已跨入 UTC 9 月的记录单列，不并入主测试。"),
        ("code", r'''archived_table("split_audit.csv", limit=None)
archived_json("model_manifest.json", keys=["development_only", "primary_model_name", "model_name", "model_type", "model_params",
              "features", "categorical_features", "numeric_features", "effective_preprocessing", "evaluation_rows", "evaluation_ids_sha256", "split_sha256"])
result = archived_json("final_result.json", show=False)
if result is not None:
    notice("**单次正式测试的访问账本（归档字段）**")
    notice("```json\n" + json.dumps({"access_ledger_path": result.get("access_ledger_path"),
                                    "utc_boundary_audit": result.get("utc_boundary_audit")}, ensure_ascii=False, indent=2) + "\n```")'''),
        ("md", "## 4. 最终整体指标与分组误差\n\n预测统一非负截断。整体 MAE/RMSE 来自完整测试行的误差总量；不能简单平均组均值。严重组按真实非负目标 ≥60 分钟定义，是结果人群的误差分层。机场、航司和 family 的样本数应与误差一起阅读，小样本不作过度解释。"),
        ("code", r'''archived_table("metrics.csv", filters={"group": "all"}, limit=None)
for group in ["severity", "overnight", "airport", "airline", "aircraft_family", "day_of_week", "scheduled_hour"]:
    notice(f"### 分组：{group}")
    table = archived_table("metrics.csv", filters={"fold": "overall", "group": group}, limit=100)
    if table is not None and table.empty:
        notice("**此组在实际指标中未归档；不生成替代结果。**")'''),
        ("md", "## 5. 残差与主模型的配对比较\n\n残差为预测减真实目标，正值表示高估。原始完整残差在 residuals/，图包含尾部档位。主模型分别对均值、中位数、carry-forward 使用同一 UTC 日期 7 日块、1000 次配对重抽样；误差按每日样本数加权。差值为候选减参照，正值表示主模型更差。若主模型本身就是某基线，与自身的差值和区间应为零。"),
        ("code", r'''archived_table("residual_summary.csv", limit=None)
archived_image("figures/final_residual_distribution.png")
archived_table("final_comparisons.csv", limit=None)
for baseline in ["training_mean", "training_median", "carry_forward"]:
    archived_table(f"final_comparison_{baseline}.csv", limit=None)
    archived_table(f"final_comparison_{baseline}_daily.csv", limit=None)
    archived_image(f"figures/final_comparison_{baseline}.png")'''),
        ("md", r"## 6. 最终 RQ1：固定开发参考的机场描述" + "\n\n" + r"最终机场分析应复用开发期冻结的 A/C/w，不根据 8 月重新选择共同支持。每次标准化平均 \(\Delta\) 与累计 \(S_+,S_-,S_{net}\) 保持不同单位。这里的机场差异和网络流量是描述性结果，不证明机场造成了延误或恢复。"),
        ("code", r'''notice(f"**RQ1 结果归档：{RQ1_RUN_DIR.name}**")
archived_json("rq1_complete_summary.json", base=RQ1_RUN_DIR)
standard = archived_json("rq1_standardized/rq1_standardization_summary.json", base=RQ1_RUN_DIR)
if standard is not None:
    notice("冻结参考复用状态：**" + ("归档声明已复用" if standard.get("frozen_reference_reused") else "未归档为已复用") + "**。")
rq1_reference(RQ1_RUN_DIR)
rq1_reference_identity(RQ1_RUN_DIR)
archived_table("rq1_standardized/selection_attempts.csv", base=RQ1_RUN_DIR, limit=None)
archived_table("rq1_standardized/rq1_airports.csv", base=RQ1_RUN_DIR, limit=None, filters={"threshold": "ge15"})
archived_table("rq1_standardized/rq1_threshold_stability.csv", base=RQ1_RUN_DIR, limit=None)
archived_table("rq1_standardized/rq1_airports_monthly.csv", base=RQ1_RUN_DIR)
archived_table("rq1_standardized/network_nodes.csv", base=RQ1_RUN_DIR)
archived_table("rq1_standardized/network_edges.csv", base=RQ1_RUN_DIR)
for name in ["rq1_standardized_airports.png", "rq1_threshold_stability.png", "rq1_network_geographic.png", "rq1_cumulative_delay.png"]:
    archived_image("figures/" + name, base=RQ1_RUN_DIR)'''),
        ("md", "## 7. 可重复核对与结论边界\n\n共享工具 `scripts/final_evaluation.py --verify-run <run目录>` 可以从已保存预测重算指标、核对产物哈希及可用访问账本，不重新训练或读取原始测试数据。本 Notebook 不自动调用该工具、不把未经运行的核对写成已通过。测试结果只能用于报告冻结候选的表现；若之后修改方法，应明确记为测试后探索，不能继续把 8 月当作未见数据。"),
    ])


def _resolve(value, root):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if len(path.parts) == 1 and path.name.startswith("R"):
        return (root / "experiments/runs" / path).resolve()
    return (root / path).resolve()


def _expression(path, root):
    try:
        return f"ROOT / {path.relative_to(root).as_posix()!r}"
    except ValueError:
        return f"Path({str(path)!r})"


def _render(notebook, filename, run_dir, root, model_runs=(), rq1_run_dir=None, update_main=True, output_dir=None):
    root = Path(root).resolve()
    run_dir = _resolve(run_dir, root)
    model_runs = [_resolve(value, root) for value in model_runs]
    rq1_run_dir = run_dir if rq1_run_dir is None else _resolve(rq1_run_dir, root)
    archive_dir = run_dir if output_dir is None else _resolve(output_dir, root)
    manifest_path = archive_dir / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8-sig")).get("status") == "completed":
        raise ValueError("Research notebooks must be archived outside an already completed run")
    destination = archive_dir / "notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    if target.exists() or target.with_suffix(".html").exists():
        raise FileExistsError(f"Refusing to overwrite executed research notebook: {target}")
    binding = ("from pathlib import Path\n"
               "cwd = Path.cwd().resolve()\n"
               f"ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'scripts/build_features.py').is_file()), Path({str(root)!r}))\n"
               f"RUN_DIR = {_expression(run_dir, root)}\n"
               f"MODEL_RUNS = [{', '.join(_expression(path, root) for path in model_runs)}]\n"
               f"RQ1_RUN_DIR = {_expression(rq1_run_dir, root)}")
    notebook.cells.insert(1, nbformat.v4.new_code_cell(binding, metadata={"tags": ["run_binding"]}))
    manager = AsyncKernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=300, allow_errors=False,
                            resources={"metadata": {"path": str(root)}}, kernel_name="python3")
    try:
        client.execute(cleanup_kc=True)
    except BaseException:
        nbformat.write(notebook, target.with_name(target.stem + ".failed.ipynb"))
        raise
    nbformat.write(notebook, target)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    target.with_suffix(".html").write_text(html, encoding="utf-8")
    display_warnings = []
    if update_main:
        (root / "notebooks").mkdir(exist_ok=True)
        try:
            shutil.copy2(target, root / "notebooks" / filename)
        except OSError as error:
            if getattr(error, "winerror", None) not in {32, 33, 1224}:
                raise
            warning = {"notebook": filename, "winerror": error.winerror, "message": str(error),
                       "executed_archive": str(target), "display_copy_updated": False}
            display_warnings.append(warning)
            warning_path = archive_dir / "notebook_display_warnings.json"
            existing = json.loads(warning_path.read_text(encoding="utf-8")) if warning_path.is_file() else []
            if not isinstance(existing, list):
                raise ValueError("notebook_display_warnings.json must contain a list")
            warning_path.write_text(json.dumps(existing + display_warnings, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"executed_notebook": str(target), "html": str(target.with_suffix(".html")),
            "interpreter": sys.executable, "update_main": update_main, "display_copy_warnings": display_warnings,
            "raw_data_read": False, "trained": False}


def render_analysis(run_dir, model_runs=(), *, root=ROOT, update_main=True):
    return _render(analysis_template(), "06_共同支持与模型比较.ipynb", run_dir, root,
                   model_runs=model_runs, update_main=update_main)


def render_final(run_dir, rq1_run_dir=None, *, output_dir=None, root=ROOT, update_main=True):
    """Read the model archive and optionally write 07 into a separate active run."""
    return _render(final_template(), "07_最终评估.ipynb", run_dir, root,
                   rq1_run_dir=rq1_run_dir, update_main=update_main, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", choices=["analysis", "final"], default="analysis")
    parser.add_argument("--model-runs", nargs="*", default=[])
    parser.add_argument("--rq1-run-dir")
    parser.add_argument("--output-dir", help="Separate active archive for the final research notebook")
    parser.add_argument("--no-update-main", action="store_true")
    args = parser.parse_args()
    result = (render_analysis(args.run_dir, args.model_runs, update_main=not args.no_update_main)
              if args.mode == "analysis" else render_final(args.run_dir, args.rq1_run_dir, output_dir=args.output_dir, update_main=not args.no_update_main))
    print(result)
