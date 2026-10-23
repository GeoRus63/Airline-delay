# 天气代码独立审查（2026-10-24）

范围：`prepare_weather.py`、`weather_features.py` 及合成测试。未运行全量准备或训练，未读取 8 月航班目标。实际源核对仅涉及天气 CSV schema、站点映射和预定缺全文覆盖审计。

## 已修复问题

1. UTC datetime 精度：pandas 字符串可能生成微秒时间，而航班时间为纳秒，导致 as-of merge 报错或整数区间比较静默失配。天气入口统一到 UTC 纳秒。
2. TAF 批查询 untimed 分支：merge 后继续使用原查询索引 mask，在非默认索引下报错。现在按当前工作表重建 mask。
3. 同刻未知公告：单条 selector 原用严格大于，批处理用大于等于；现在同刻未知公告也阻断旧公告，两条路径一致。
4. 缺全文产品遗漏：bulletin 表原先仅来自 `references/taf_validity.csv`，原始段表里存在但未恢复全文的产品不会参与最新状态选择。现在按原始 `(product_id, station)` 对 validity 做 anti-join，加入 `status=unverified` 占位；原始 `valid` 仅作 archive 锚点，body issue 和 availability 均为 NaT，有效期也不赋值。若同产品有多个 archive 时间，取首次档案证据作为保守阻断边界。缺正文段保持不可用，允许 revision 的敏感性选项也不能绕过未知正文时间。下一份正常公告可以恢复。

## 实际缺全文来源证据

`tables/taf_text_missing_products.csv` 的三个产品均在原始 TAF CSV 中找到，下面是首次匹配的 Observation 段元数据；未使用航班结果：

| station | product_id | raw valid (UTC) |
|---|---|---|
| KGRR | 202608042106-KGRR-FTUS43-TAFGRR-AAA | 2026-08-04 21:06 |
| KOMA | 202608042106-KOAX-FTUS43-TAFOMA-AAA | 2026-08-04 21:06 |
| KAZO | 202608042107-KGRR-FTUS43-TAFAZO-AAA | 2026-08-04 21:07 |

这些时间与 product ID 前缀一致，但不能证明正文发布时间或实时接收时间。不能把段文本当作完整公告来赋予新预测天气。

## 保留的策略与限制

- METAR：没有接收时间的 COR 不赋值；允许继续使用未超过 120 分钟的旧合法原文。事后 COR 不追溯删除旧原文。这与 TAF 最新公告状态阻断是两种明确区分的政策。
- TAF：先按可用代理时间选择最新公告，再找适用于查询时点的段；CNL/NIL、默认禁用的 AMD/COR/delayed、未知正文时间均不回退旧公告。FM 与 TEMPO/PROB 分开处理。
- NOAA：取最近合格观测，允许跳过较新的不合格 QC 行；来源白名单之外不算通过。缺失、零、trace 分开；温度、风、能见度、云底转换单位，累计降水不冒充 METAR 即时天气。此次未逐项核验官方完整 QC 文档，不宣称全部来源规则得到外部认证。
- 当前站点及别名映射未发现跨机场冲突；特征查询未发现读取下游标签。合成测试覆盖输入顺序及索引。
- 观测时间加 5 分钟、TAF 正文/档案发布时间最大值都是回顾性可用性代理。档案和最终 QC 不能证明历史实时接收顺序；不得宣称已实现完全在线可部署系统。
- 若原始缺全文产品连 archive 时间也缺失，则无法确定阻断起点；占位保持未知，本次三个实际产品不存在此问题。

## 检查结果

执行 `.venv\\Scripts\\python.exe -m unittest tests.test_weather_contract tests.test_weather_features -q`：23 项通过。新增测试验证缺全文产品阻断旧预报（包括允许 revision 选项）、占位不可用，以及下一正常产品恢复。未运行全量；天气派生数据尚须主流程重新生成后才包含本修复。

额外只读确认：`complete_rq1.run` 将 final 模式传为 `development=False` 给 `analyze_rq1.run`，raw RQ1 使用 2026-09-01 UTC 作为右开边界，不会把最终 4–8 月误截为开发 4–7 月。
