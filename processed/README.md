# 处理结果目录 / Processed outputs

本机已实际生成五个月清洗航班、D0003 航班衔接及 W0001/W0002 天气标准表，旧版 P0–P7 已运行。当前主分析使用 2024 年 B-43 飞机映射；本地已有、覆盖更好的 2025 参考表尚未替换主映射并重跑，因此这些派生结果仍属于 2024 参考版。

此目录中的大型文件被 Git 忽略，克隆不会恢复它们。模型与预测通过 [GitHub Release](https://github.com/GeoRus63/Airline-delay/releases/tag/research-artifacts-v1)另行提供，包内不包含全部 processed 或 run data；完整迁移需从原电脑同步并核验，或在独立副本使用新数据版本重建。具体见[迁移与运行](../docs/迁移与运行.md)及[模型与预测归档](../docs/模型与预测归档.md)。

保留现有输入和各版本，不在原档案旁直接重跑准备脚本覆盖文件。后续更新飞机映射时核实航司/尾号对应关系，无法可靠补全的机型保留 unknown，航班不删除。暂不重训、不处理视频。

实验配置、指标、图表、已执行 Notebook、模型和预测按 `experiments/runs/<run_id>/` 留档；不能用更新本目录缓存代替保存研究证据。原始下载始终保存在 `raw/`，保持原样。

Processed data already exist locally but are ignored by Git. The archived study uses the 2024 aircraft mapping; the 2025 update and rerun remain pending. The model/prediction Release is not a complete backup of this directory or run data. Transfer and verify those files separately, or rebuild in an independent copy with new version/run IDs.
