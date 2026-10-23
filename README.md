# Airline Delay — IND5003

美国国内航班延误传播四人项目。本次提交包含 2026 年 4—5 月原始数据、项目要求、参考论文，以及 A 部分的数据清洗、同机航班衔接和整体 EDA。

A four-person IND5003 project on delay propagation in US domestic flights. This initial upload includes April–May 2026 raw data, the project brief, reference papers, and Part A: cleaning, same-aircraft connections and overall EDA.

## 文件入口 / Start here

| 内容 / Content | 路径 / Path |
| --- | --- |
| 原始 BTS 数据 / Raw BTS data | [2026_04.zip](2026_04.zip), [2026_05.zip](2026_05.zip) |
| 项目要求 / Project brief | [ind5003-2610_projects.pdf](ind5003-2610_projects.pdf) |
| 参考论文 / Reference papers | [papers/](papers/) |
| A 部分说明 / Part A guide | [output_A/README_先读这里.md](output_A/README_先读这里.md) |
| 数据清洗与衔接 / Cleaning and connections | [01 Notebook](output_A/01_数据清洗与航班衔接.ipynb) |
| 整体探索性分析 / Overall EDA | [02 Notebook](output_A/02_整体EDA.ipynb) |
| 字段定义 / Data dictionary | [output_A/data_dictionary.md](output_A/data_dictionary.md) |
| 共享数据 / Shared datasets | [output_A/data/](output_A/data/) |
| 图表与汇总 / Figures and summaries | [figures/](output_A/figures/), [tables/](output_A/tables/) |
| 机场及字段来源 / Airport and field references | [output_A/references/](output_A/references/) |

## 获取与运行 / Get and run

```bash
git clone https://github.com/GeoRus63/Airline-delay.git
cd Airline-delay
```

使用 Python/Jupyter 打开 `output_A` 中的两本 Notebook，依次运行 01、02。原始 ZIP 与 `output_A` 保持当前相对位置。依赖版本见 [requirements.txt](output_A/requirements.txt)。Notebook 保存了已有输出；HTML 阅读版下载后可在浏览器打开。

Open the two notebooks under `output_A` in Python/Jupyter and run 01, then 02. Keep the raw ZIP files next to `output_A`. Package versions are recorded in [requirements.txt](output_A/requirements.txt). Saved notebook outputs are included; download the HTML previews to read them in a browser.

三张共享数据表采用 `.csv.gz`，pandas 可直接读取，内容与解压后的 CSV 相同。组员继续研究主要使用 `flight_connections.csv.gz`。

The three shared datasets use `.csv.gz`. Pandas reads these directly; their contents are identical to decompressed CSVs. Use `flight_connections.csv.gz` as the main input for subsequent research.

```python
import pandas as pd
connections = pd.read_csv("output_A/data/flight_connections.csv.gz")
```

## 协作 / Collaboration

每位组员在自己的分支开展工作，完成后提交 Pull Request 合并。提交代码、说明和必要结果；共享数据已有压缩版本，无需重复提交解压 CSV。

Work on a personal branch and open a pull request when ready to merge. Commit code, documentation and relevant results; use the existing compressed datasets instead of committing duplicate uncompressed CSVs.

## 数据与参考 / Data and references

- 航班数据 / Flight data: [BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/ontime/).
- 论文 / Paper: [Flight delay propagation modeling: Data, Methods, and Future opportunities](https://doi.org/10.1016/j.tre.2024.103525).
- 论文 / Paper: [Network science approach for identifying disruptive elements of an airline](https://arxiv.org/abs/2211.08140).
- 机场数据的版本、来源及许可证保存在 `output_A/references/`。 / Airport reference versions, sources and licence are retained in `output_A/references/`.
