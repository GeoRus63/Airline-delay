# 文献核读与本项目采用范围

核读日期：2026-10-24。本记录只依据本地全文和已核验的官方项目页；不将论文中的案例结果写成本项目结果。

## Li et al. (2024)

来源：`references/papers/1-s2.0-S1366554524001169-main.pdf`，43页，DOI [10.1016/j.tre.2024.103525](https://doi.org/10.1016/j.tre.2024.103525)。第1页核对题名、作者、卷185、文章号103525及发表年份。第4页数据源表列出BTS、NOAA等，第9–10页区分flight-chain和airport-network视角；第9页明确将最低周转时间与turnaround buffer分开。

本项目采用同尾号相邻航段的微观分析单位，再将衔接表现汇总到机场网络。由于没有最低地面作业时间，不把计划周转直接称为buffer，不估计论文所述正式buffer，也不将观测到的净变化分解成因果性新增与传播延误。网络颜色、衔接数量及航线流量分别定义，避免把三个量混用。

## Chauhan et al. (2023)

来源：`references/papers/2211.08140v2.pdf`，20页；本地首页标注2023年4月修订版，arXiv:2211.08140v2。第1–3页及摘要说明时刻表connection network与历史delay network的区别，以及airport–flight、flight–flight-connection和多层网络的建模选择。

本项目只借鉴结构表达和口径分离。机场节点大小表示有效同机衔接数、颜色表示共同支持标准化后的每次净变化，边宽表示BTS航班数。没有实现该文中心性/渗流识别，也不引用其专有航司案例的“四个航班”传播距离或旅客衔接结论作为本项目证据。

## UC Berkeley (2025)

来源：[官方项目页](https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches)，2026-10-24经官方域名搜索结果核验；直接抓取返回403，搜索索引返回页面正文，页面标注最后更新2025-12-20。标题为 *Air Travel Delay Prediction Feature Engineering and ML Approaches*。

该项目将预测时点设为计划出发前两小时，并讨论飞机运行历史、预测时点可得性、天气对齐、逐项特征实验及严重延误的欠预测。本项目借鉴明确时点、逐组比较和误差分层的组织方式；预测时点改为上游实际到达，不沿用其两小时前可得性假设，也不直接比较不同年份、范围和样本的MAE。其XGBoost和神经网络不属于本项目必做模型，报告不能写成已复现该项目。

## 课程要求核对

原始要求为 `references/ind5003-2610_projects.pdf`。最终英文报告需覆盖文献、数据和准备、EDA、研究问题、新学方法、代码、结果与局限；正文最多25页，封面/目录/参考文献/附录不计，Times New Roman至少10pt，正文连贯行文。技术视频约10分钟，先讲发现，再说明方法；任务书要求避免AI配音。最终数值和图以有效实验归档为准，引用文献只说明借鉴来源。
