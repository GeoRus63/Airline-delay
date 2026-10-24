# Airline delay：项目与运行入口

研究范围：2026 年 4–8 月，3,057,590 条 BTS 航班，355 个机场。研究同机航班衔接中的延误吸收与放大，并预测下一班的出发延误。

当前维护分支为 [`codex/apr-aug-research`](https://github.com/GeoRus63/Airline-delay/tree/codex/apr-aug-research)；本轮保留 `main` 和默认分支设置，查看当前说明与恢复项目请使用研究分支。

**当前项目仍需修改：已封存的 P0–P7 使用 2024 年 B-43 作为飞机主映射，尚未采用本地已有、覆盖更好的 2025 年参考表重建并重跑。** 既有结果是 2024 参考版的历史结果，不能代表新版飞机映射下的最终结论。2025 年参考表的发布时间不作为排除新版的理由。本轮仅维护项目文件和发布已有模型、预测归档；暂不重训，视频处理暂停。

后续需要逐项核实航司与尾号对应关系、同尾号冲突和机型别名，再更新 family/variant 映射及覆盖审计；无法可靠确定机型的航班保留为 `unknown`，不为补全机型删除样本。新版须使用新的数据版本、配置和 run_id，重新检查特征、开发验证和结果解释。原 F0001、正式评估账本、run、报告、演示及视频继续封存；已见 8 月不能重新称作未见测试。

## 已封存的 2024 参考版结果

旧版已实际完成 P0–P7：五个月清洗、D0003 衔接、W0002 天气、G0–G4B 主线与 variant/NOAA 分支、开发 EDA/RQ1/RQ2、登记选模、正式评估和计算复现。实际模型比较包括 OLS、4 个决策树候选和 2 个随机森林候选，以及训练均值、中位数和 carry-forward 三个基线。**随机森林仅在已测试的候选和登记规则内胜出，不能据此声称优于所有监督学习模型。**

R0027 选中 R0025：G4B 65 个输入、200 树随机森林、depth16/leaf50、max_features=0.5、seed42；开发期 pooled MAE 12.1643 分钟、RMSE 41.7654 分钟。R0028 是 F0001 下唯一正式 8 月评估：560,077 条连接上 MAE **13.6551 分钟**、RMSE 42.9171 分钟，MAE 较训练中位数基线降低约 24.14%。

R0029 完成固定参考 RQ1：25 个机场均有支持，24 个标准化净延误变化估计为负，唯一正值 FLL 的区间包含 0；不能解释为机场因果责任。R0030 在同一 F0001 下重建特征并重拟合四种预测器，raw/clip 预测逐行最大差异均为 0，3,320 行指标通过既定容差。这是已见测试集上的计算复现，不增加独立测试证据。

R0031 及[交付后审计](research/audits/R0031_post_delivery_integrity.json)保留了全部 31 条旧版运行记录：27 completed、3 failed、1 invalidated；该审计通过不表示最新飞机映射的后续工作已完成。

## 档案与交付入口

| 内容 | 文件或下载入口 |
|---|---|
| 当前维护状态与待办 | [执行进度](docs/执行进度.md) |
| 全部已有模型与预测 | [GitHub Release](https://github.com/GeoRus63/Airline-delay/releases/tag/research-artifacts-v1) · [下载与校验说明](docs/模型与预测归档.md) · [逐文件清单](artifacts/model_prediction_manifest.json) |
| 旧版英文课程报告 | [PDF](reports/report_en.pdf) · [Markdown](reports/report_en.md)，总 22 页、正文 19 页 |
| 旧版中文结果解释 | [结果解读](reports/结果解读_中文.md) |
| 旧版演示及视频档案 | [12 页 PPTX](reports/presentation/deck_v01/airline_delay.pptx) · [英文讲稿](reports/presentation/deck_v01/speaker_script_en.md) · [600 秒字幕 MP4](reports/presentation/video_v01/airline_delay_captioned.mp4)，无旁白/无音轨，当前暂停处理 |
| 研究与证据 | [研究记录](research/研究记录.md) · [实验索引](experiments/index.csv) · [证据索引](reports/evidence_index.csv) |
| 旧版封存收据 | [271 文件交付收据](experiments/runs/R0031/delivery_receipts.json) · [交付后检查](research/audits/R0031_delivery_postcheck.json) · [S0001 来源快照](reports/source_snapshots/S0001/mapping.json) |
| 迁移和历史运行说明 | [迁移与运行](docs/迁移与运行.md) · [封存版最终运行说明](docs/最终运行说明.md) |

模型与预测清单覆盖 **17 个 run 下的 268 个文件（134 个模型、134 个预测文件），共 2,243,334,942 字节**，按 run 分为 17 个 ZIP；R0027 的两个解释样本预测文件另已保存在 Git 中。Release 包不包含全部 `processed/`、run `data/` 或所有其他本地生成文件，不能视作完整计算备份。下载、解压位置和 SHA-256 检查见归档说明。

封存报告、演示、视频和 S0001 保留原字节。它们反映当时的 2024 参考版研究，当前限制和后续修改状态由本页及执行进度说明；不把新结论回填旧 run。

## 新电脑恢复入口

使用 PowerShell 7，保留完整 Git 历史：

```powershell
git clone --branch codex/apr-aug-research https://github.com/GeoRus63/Airline-delay.git airline_delay
Set-Location airline_delay
python scripts/verify_downloads.py --raw-only
```

仓库包含原始数据；`.gitattributes` 保留原始字节。选模协议和逐轮源码审计依赖完整 Git 历史，不使用浅克隆。按[迁移与运行](docs/迁移与运行.md)使用实际 Python **3.13.15** 与 `requirements.lock.txt` 恢复隔离环境，再下载 Release 的模型/预测并校验。`processed/` 和各 run 的 `data/` 仍需从原电脑同步或在独立副本重建；仅克隆和下载模型预测不能恢复完整实验。

给 Codex 的恢复指令：

> 请先阅读 README、docs/执行进度.md、docs/迁移与运行.md 及 docs/最终运行说明.md。先只读核对旧版档案及本地大文件，明确 2024 主映射尚待升级到已有的 2025 参考表。不要覆盖封存 run、F0001、正式账本或成品，不重复启动 formal。未经后续明确指令暂不重训、不处理视频；新版研究使用新的数据版本和 run_id，保留 unknown 航班及失败、无增益证据，已见 8 月不称为未见测试。

[Codex 完整执行流程](docs/Codex完整执行流程.md)和[完整项目计划](docs/完整项目计划.md)保留研究定义与阶段规则；当前状态以执行进度、研究日志和实验索引为准。[下载完成说明](下载完成说明.md)记录原始输入清单与覆盖。

## 文件在哪里

| 目录/文件 | 内容 |
|---|---|
| `raw/flights/` | 五个月 BTS 原始航班 ZIP |
| `raw/aircraft/`、`references/aircraft_reference_*.csv` | 原始 B-43 ZIP 及 2024/2025 年参考表 |
| `raw/weather/` | METAR/SPECI、TAF 分段及原文、NOAA GHCNh 原始档案 |
| `references/`、`tables/`、`sources.json` | 参考表、字段文档、来源、覆盖、大小和 SHA-256 |
| `notebooks/`、`scripts/`、`configs/` | 中文 Notebook、共享处理/训练/评估逻辑和逐轮配置 |
| `processed/` | 已在本机生成的派生数据；大文件不随 Git 克隆恢复 |
| `research/`、`experiments/` | 研究记录、索引、按 run_id 保留的运行证据；大型 data/models/predictions 不在 Git 中 |
| `artifacts/model_prediction_manifest.json` | Release 内模型/预测的路径、大小及 SHA-256 清单 |
| `requirements.txt`、`requirements.lock.txt` | 直接依赖和实际完整锁定环境 |

固定 RQ1/RQ2；机型仅用 family/variant，不加入机场设施、座位数或机龄。开发比较遵守 4–7 月 UTC 前向验证、训练标签成熟、上游实际到达时预测及天气 as-of 可用性；不得用已见 8 月结果反向调参。

<details><summary>English</summary>

The project covers 3,057,590 BTS flights from April–August 2026 across 355 airport IDs. The archived P0–P7 study is complete for its **2024 B-43 primary mapping**, but the project still needs revision: the locally available 2025 reference has better coverage and has not yet replaced that mapping or been rerun. Aircraft completion must verify airline/tail identity and retain unresolved flights as unknown. Current maintenance preserves all archived outputs; retraining and video work are paused.

OLS, four decision-tree configurations and two random-forest configurations were compared alongside three simple baselines. Random forest was selected only among the tested candidates under the registered rules; no claim is made about all supervised models. Its archived August MAE is 13.6551 minutes. R0030 reproduced the frozen predictions exactly on the already-seen test set, which is not a new independent test.

The Release contains all 134 existing model files and 134 prediction files with a per-file checksum manifest. It is not a complete computational backup: ignored processed data and run data still require separate transfer or reconstruction. Clone full Git history and recreate the actual Python 3.13.15 environment using requirements.lock.txt. Preserve F0001, the formal-test ledger and all historical run/report files; future work must use new versions and run IDs.

</details>
