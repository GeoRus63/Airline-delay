# Airline delay：项目与开工入口

研究范围：2026 年 4–8 月，3,057,590 条 BTS 航班，355 个机场。研究同机航班衔接中的延误吸收与放大，并预测下一班的出发延误。

本次交付原始数据、下载核对记录、完整项目方案和准备脚本。全量清洗、特征连接、EDA 与建模在新电脑执行。

## 新电脑 Codex 开工入口

克隆后在 Codex 中打开此项目，发送：

> 请按照 AGENTS.md 和 docs/Codex完整执行流程.md，在这台电脑完整完成项目：实际运行数据准备、天气与特征连接、EDA、RQ1、RQ2、模型比较和最终报告，直到完成标准满足。先跑通基础分析再逐组扩展，每轮保留研究理由、配置、实际结果和决定，包括无增益与失败实验。

[Codex 完整执行流程](docs/Codex完整执行流程.md) 列出已有入口、待实现工作、执行顺序、产物和完成检查。其他说明：

- [完整项目计划](docs/完整项目计划.md)：研究问题、变量、方法、模型、验证与最终成品。
- [迁移与运行](docs/迁移与运行.md)：Git 克隆和环境安装。
- [下载完成说明](下载完成说明.md)：数据清单、校验结果与源覆盖。
- [研究记录](research/研究记录.md)与[实验索引](experiments/index.csv)：从本次重建版开始保留研究与迭代证据。

## 文件在哪里

| 目录/文件 | 内容 |
|---|---|
| `raw/flights/` | 五个月 BTS 原始航班 ZIP |
| `raw/aircraft/` | BTS B-43 原始飞机清单 ZIP |
| `raw/weather/metar/` | METAR/SPECI 实况原始 CSV，使用 gzip 压缩 |
| `raw/weather/taf/` | TAF 分段预报原始 CSV，使用 gzip 压缩 |
| `raw/weather/taf_text/` | TAF 完整原文、原始接口响应及补查记录 |
| `raw/weather/noaa/` | 25 个缺测站的 NOAA GHCNh 原始 PSV 快照，使用 gzip 压缩 |
| `references/` | 机场/时区、机型参考表、字段文档、任务书与论文 |
| `tables/` | 月份清单、下载来源、机场天气覆盖、文件大小与 SHA-256 |
| `notebooks/` | 数据清单和航班准备 Notebook，附中文说明 |
| `scripts/` | 下载、完整性检查及目标电脑准备脚本 |
| `processed/` | 新电脑运行后生成的可重建中间数据；当前只有说明文件 |
| `research/`、`experiments/` | 研究日志、实验索引、按 run_id 保存的运行证据 |
| `sources.json` | 原始数据来源和下载地址 |
| `requirements.txt` | Python 环境版本 |

## 在新电脑克隆并核对

```bash
git clone --depth 1 https://github.com/GeoRus63/Airline-delay.git airline_delay
cd airline_delay
python scripts/verify_downloads.py
```

仓库直接保存原始数据文件。`.gitattributes` 保留文件原始字节，Windows 克隆时也不会因换行转换改变数据校验值。

命令只核对文件，不联网、不清洗数据。检查通过后，按照[迁移与运行](docs/迁移与运行.md)安装环境并执行准备步骤。

机型只用系列与具体型号两层；机场比较围绕 RQ1，不加入机场设施参数。8 月作为最终时间外测试，开发比较使用 4–7 月。网络图、周转热力图、小延误敏感性分析和变量增量实验已纳入方案。

<details><summary>English</summary>

This portable project covers 3,057,590 BTS flights from April–August 2026 across 355 airport IDs. It contains raw flight, aircraft and weather sources, integrity/coverage records, the complete research plan, and preparation scripts. Full cleaning, feature joins, EDA and modeling are to run on the destination computer. Clone the repository, read the download report, project plan and migration guide, then run the offline verifier. RQ1 studies airport delay absorption/amplification; RQ2 predicts downstream departure delay at upstream arrival. August is reserved for the final temporal test.

</details>
