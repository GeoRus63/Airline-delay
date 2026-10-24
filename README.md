# Airline delay：项目与运行入口

研究范围：2026 年 4–8 月，3,057,590 条 BTS 航班，355 个机场。研究同机航班衔接中的延误吸收与放大，并预测下一班的出发延误。

本次 4–8 月重建已完成五个月清洗、D0003 衔接、W0002 天气、G0–G4B 主线与 variant/NOAA 分支、开发 EDA、完整 RQ1 和全部 DT/RF 开发网格。R0027 已按登记规则选中 R0025：G4B 65 个输入、200 树随机森林、depth16/leaf50、max_features=0.5、seed42，开发期 pooled MAE 12.1643 分钟、RMSE 41.7654 分钟，胜过三个基线。

**项目已进入 P6：第九轮 Pro 最终选模审查完成，F0001 已冻结；唯一正式 8 月评估 R0028 于 2026-10-24 03:08:45Z 左右启动，目前仍在运行，尚无已完成的测试结果。** 不得重新启动第二次 formal 或删除访问账本。最终 RQ1/R0029、完整复现/R0030 的配置已提交但未执行；最终中英文报告、PDF、PPTX、字幕视频和 P7 交付审计仍待完成。冻结 SHA 与后续条件性命令见[最终运行说明](docs/最终运行说明.md)，当前状态见[执行进度](docs/执行进度.md)及[实验索引](experiments/index.csv)。

## 新电脑 Codex 开工入口

克隆后在 Codex 中打开此项目，发送：

> 请按照 AGENTS.md 和 docs/Codex完整执行流程.md，在这台电脑完整完成项目：实际运行数据准备、天气与特征连接、EDA、RQ1、RQ2、模型比较和最终报告，直到完成标准满足。先跑通基础分析再逐组扩展，每轮保留研究理由、配置、实际结果和决定，包括无增益与失败实验。

[Codex 完整执行流程](docs/Codex完整执行流程.md) 保留阶段要求、执行顺序、产物和完成检查；其中初始交付状态是历史背景，实际完成状态以研究日志和实验索引为准。继续运行和迁移优先阅读[最终运行说明](docs/最终运行说明.md)。其他说明：

- [完整项目计划](docs/完整项目计划.md)：研究问题、变量、方法、模型、验证与最终成品。
- [最终运行说明](docs/最终运行说明.md)：实际 Python 3.13.15 环境、数据版本、run/config 对应、最终冻结与复现、报告及交付命令。
- [迁移与运行](docs/迁移与运行.md)：早期克隆说明；环境版本和当前运行办法以最终运行说明为准。
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
| `notebooks/` | 中文数据准备、特征、EDA、RQ1/RQ2 与研究结果 Notebook；逐轮执行版本在 run 档案 |
| `scripts/` | 数据准备、as-of 特征、训练/评估、分析、选模、冻结/复现、报告构建及只读审计 |
| `processed/` | 本机 D0003 衔接、W0001/W0002 天气和基础航班等派生数据；大文件不随 Git 克隆恢复 |
| `research/`、`experiments/` | 研究日志、实验索引、按 run_id 保存的运行证据 |
| `sources.json` | 原始数据来源和下载地址 |
| `requirements.txt`、`requirements.lock.txt` | 直接依赖与实际完整锁定环境 |

## 在新电脑克隆并核对

```bash
git clone --branch codex/apr-aug-research https://github.com/GeoRus63/Airline-delay.git airline_delay
cd airline_delay
python scripts/verify_downloads.py --raw-only
```

仓库直接保存原始数据文件。`.gitattributes` 保留文件原始字节，Windows 克隆时也不会因换行转换改变数据校验值。选模协议和逐轮源码审计需要完整 Git 历史。`processed/` 及各 run 的 `data/`、`predictions/`、`models/` 被 Git 忽略，迁移须另行同步并按收据核验；仅克隆仓库不能恢复完整实验。

命令只核对原始文件大小、哈希及压缩完整性，不联网、不清洗、不解析测试目标。检查通过后，按照[最终运行说明](docs/最终运行说明.md)恢复隔离环境与本地产物，再选择只读审计或使用新 run_id 重建。不要覆盖已完成运行。

机型只用 family/variant，不加入机场设施、座位数或机龄。开发比较使用 4–7 月 UTC 前向验证与成熟训练标签；唯一正式 8 月时间外测试正在按冻结方案运行。网络图、周转热力图、阈值敏感性和变量增量已有开发结果；最终结论须等待正式运行及核验完成。

<details><summary>English</summary>

This project covers 3,057,590 BTS flights from April–August 2026 across 355 airport IDs. Development experiments and registered model selection are complete. R0027 selected the R0025 random forest with 65 G4B inputs; development pooled MAE/RMSE were 12.1643/41.7654 minutes. Final review is complete and F0001 is frozen. The unique formal August assessment R0028 started at approximately 03:08:45 UTC on 24 October 2026 and is still running; no completed test result is claimed. Final RQ1, full refit reproduction and final deliverables remain pending. Do not reopen the formal test. Full Git history and separately copied ignored processed data, predictions, feature archives and models are required for migration and receipt verification.

</details>
