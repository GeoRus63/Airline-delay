# Airline Delay A 部分 / Part A

本文件夹包含数据清洗、同机航班衔接和整体 EDA。两本 Notebook 按中文、英文对照说明每一步，HTML 可直接用浏览器阅读。

This folder contains data cleaning, same-aircraft flight connections and overall EDA. Both notebooks explain each step in Chinese and English. The HTML versions can be read directly in a browser.

## 从哪里开始 / Where to start

| 内容 / Content | 文件 / File |
| --- | --- |
| 数据清洗与衔接 / Cleaning and connections | [01 Notebook](01_数据清洗与航班衔接.ipynb) · [HTML](01_数据清洗与航班衔接.html) |
| 整体 EDA / Overall EDA | [02 Notebook](02_整体EDA.ipynb) · [HTML](02_整体EDA.html) |
| 字段说明 / Field definitions | [数据字典 / Data dictionary](data_dictionary.md) |
| 图表 / Figures | `figures/` |
| 汇总表 / Summary tables | `tables/` |
| 原始字段与机场参照 / Source fields and airport reference | [资料说明 / Reference guide](references/README_资料说明.md) |

## 三张共享数据表 / Three shared datasets

| 文件 / File | 行的含义 / Row meaning | 用途 / Use |
| --- | --- | --- |
| `data/flights_clean.csv.gz` | 一班航班 / One flight | 全量航班与处理标记 / All flights with processing flags |
| `data/connection_candidates.csv.gz` | 一对相邻候选 / One adjacent candidate pair | 查看衔接筛选 / Inspect connection screening |
| `data/flight_connections.csv.gz` | 一对有效衔接 / One eligible pair | B、C、D 的主要输入 / Main input for B, C and D |

本次包含 1,209,654 条航班记录和 1,131,645 对有效衔接。数据覆盖 2026 年 4—5 月。有效衔接的两端均完成且未备降、机场连续、时间一致，计划周转为 0—24 小时。

The dataset contains 1,209,654 flight records and 1,131,645 eligible connections for April–May 2026. Eligible connections have two completed, non-diverted legs, matching airports, consistent timestamps and a scheduled turnaround of 0–24 hours.

## 阅读与运行 / Reading and running

只查看结果：打开 HTML 或直接阅读 Notebook 已保存的输出。

To view results, open the HTML files or read the saved notebook outputs.

完整运行：两份原始 ZIP 保持在 `output_A` 上一级，选择自己的 Python 内核，依次运行 01、02。依赖版本见 `requirements.txt`；中文图表使用本机中文字体。重跑会更新派生数据和图表。

To rerun, keep both original ZIP files one level above `output_A`, select your Python kernel, and run 01 followed by 02. Package versions are listed in `requirements.txt`; Chinese chart labels use a locally installed Chinese font. Rerunning updates the derived data and figures.

```text
airline_delay/
├── 2026_04.zip
├── 2026_05.zip
└── output_A/
    ├── 01_数据清洗与航班衔接.ipynb
    ├── 02_整体EDA.ipynb
    ├── data/
    ├── tables/
    ├── figures/
    └── references/
```

从 `output_A` 启动 Notebook，可直接读取压缩 CSV：

From a notebook started in `output_A`, read a compressed CSV directly:

```python
import pandas as pd
connections = pd.read_csv("data/flight_connections.csv.gz")
```

数据列名使用英文，中文和英文含义见数据字典。仓库保存 `.csv.gz`；可直接读取，也可解压为普通 CSV。

Dataset column names remain in English; the dictionary provides Chinese and English definitions. The repository stores `.csv.gz` files, which can be read directly or decompressed into ordinary CSV files.

## 组员接续 / Team handoff

| 组员 / Member | 工作 / Work |
| --- | --- |
| B | 按衔接机场比较延误恢复与增加 / Compare delay recovery and increases by connection airport |
| C | 固定特征、时间划分，建立基准和线性回归 / Define features and time splits; fit baselines and linear regression |
| D | 使用同一样本训练随机森林并比较模型 / Fit random forests on the same samples and compare models |

预测时点为上游航班实际到达时，目标为下游出发延误。上游延误、周转和计划信息是输入；延误变化与实际地面时间用于事后解释。

The prediction time is the upstream flight’s actual arrival; the target is downstream departure delay. Inputs include upstream delay, turnaround and schedule information. Delay change and actual ground time support retrospective interpretation.

`_build/` 保存 Notebook 生成、HTML 导出及样本复算脚本。日常分析从两本 Notebook 开始。

`_build/` contains notebook generators, HTML export and sample-recalculation scripts. Start routine analysis with the two notebooks.
