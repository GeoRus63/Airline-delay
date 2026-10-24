# 第八次 Pro：冻结前固定 RF 预测累加顺序

网页版 Pro 回答耗时 1m49s，原提问、全文及时间凭据与本文件同前缀；页面截图仅留本机。此时 R0025 的两个开发折完成，第三折仍运行；R0026未启动，没有 freeze 或 August 正式访问 ledger。

## 新证据

原复现契约对逐行 raw/clip 预测和完整 metrics.csv 数值列均使用绝对 1e-10、rtol=0。后者包含未归一化 absolute_error_sum 和 squared_error_sum，先前小型合成树级/逐行验证不足以证明这些汇总量通过。

主代理只读加载 R0025 已保存五月模型和 R0017 开发特征，seed42随机抽取五月50000行，未训练、未访问八月、未改磁盘模型。24线程重复预测最大逐行差2.2737367544323206e-13，MAE/RMSE差约6.13e-14以内，但平方误差总和最大差4.656612873077393e-10，两次重复分别有10/18行超过门槛。原模型重复1线程预测和整张生产指标表严格一致。见 `research/audits/rf_prediction_reduction_development.*`。

## 采用的执行修正

Pro接受为冻结前的数值复现执行修正，要求明确训练/预测预算、恢复调用前原n_jobs、模型实例不得并发使用、重新加载后仍走统一入口，以及实际完整复现必须另行通过。主代理代码核对后采用这些要求。

- 仅最终保留RF时显式 `final_prediction_threads=1`；训练仍按此前决定24。R25获选另加final_threads=24，R26获选沿用原24训练预算。
- create_freeze先验证原选择/来源科学配置，再深拷贝设置，记录execution_delta、原config SHA与最终配置canonical hash。参数只允许整数1且仅RF；默认缺省行为不变。
- 共享predict_pipeline在预测时临时设置RF n_jobs，finally恢复进入时原值，未硬编码恢复24。fold timing与manifest分别记录fit_threads和prediction_threads。每折模型私有，没有同实例并发。
- 保存joblib时模型恢复训练n_jobs，因此裸model.predict不履行预测单线程契约；正式与完整复现使用读取同freeze的共享入口。
- 不改学习参数、开发分数、R25/R26预测设置、选择规则或任何复现容差；不删去误差总和、不进行四舍五入。未创建真实freeze。

## 实际验收及范围

主代理再次从磁盘加载五月模型，经新统一入口重复完整“加载→预测→生产metrics”路径。spy记录两次实际预测n_jobs均为1，调用后恢复归档原值8；50000条raw/clip及完整818行、12列指标表精确一致，含AE/SE总和；所有差为0，模型磁盘SHA不变。见 `research/audits/rf_deterministic_prediction_path.*`，脚本原文另存py.txt供复盘。此证据不证明未执行的最终全量重建/重训复现已通过。

核心子代理23项定向测试通过40.334秒，末次运行参数类型检查后另2项通过。主代理独立验证13项冻结/选模/final测试通过；同条命令把复现测试模块名写错为tests.test_reproduce_final，保留原导入失败日志，没有研究执行失败。改用实际tests.test_final_reproduction，10项24.871秒通过。报告预算披露7项7.169秒通过，分别覆盖预测单项/训练预测双项delta及冲突拒绝。原始日志均保留在research/audits。

完整P5最终选择/冻结审查仍须待全部候选完成；本专项不预设RF获选、不替代后续实际最终测试和完整复现。
