# 数据字典 / Data dictionary

列名保持英文，便于 pandas 读取。下表提供中文和英文含义。CSV 中的时间字符串可用 `pd.to_datetime` 解析，UTC 列加 `utc=True`。ID 按字符串读取；布尔列表示 True/False。

Column names remain in English for pandas. The tables below give Chinese and English meanings. Parse timestamp strings with `pd.to_datetime`, adding `utc=True` for UTC columns. Read IDs as strings; boolean fields represent True/False.

## 1. 航班表 / Flight table

`data/flights_clean.csv.gz`：每行一班航班，保留全部 1,209,654 条记录。

`data/flights_clean.csv.gz`: one flight per row, retaining all 1,209,654 records.

| 字段 / Field | 中文说明 | English definition |
| --- | --- | --- |
| `DayOfWeek` | 星期，周一为 1、周日为 7 | Weekday, Monday=1 and Sunday=7 |
| `FlightDate` | 航班日期 | Flight date |
| `Reporting_Airline` | 报告承运人代码 | Reporting carrier code |
| `Tail_Number` | 飞机尾号 | Aircraft tail number |
| `Flight_Number_Reporting_Airline` | 航班号，字符串标识 | Flight number, stored as an identifier |
| `OriginAirportID` | 出发机场 DOT 标识 | DOT origin airport identifier |
| `Origin` | 出发机场 IATA 代码 | Origin airport IATA code |
| `OriginState` | 出发州代码 | Origin state code |
| `DestAirportID` | 到达机场 DOT 标识 | DOT destination airport identifier |
| `Dest` | 到达机场 IATA 代码 | Destination airport IATA code |
| `DestState` | 到达州代码 | Destination state code |
| `CRSDepTime` | 当地计划出发时间，HHMM | Scheduled local departure time, HHMM |
| `DepTime` | 当地实际出发时间，HHMM | Actual local departure time, HHMM |
| `DepDelay` | 有符号出发延误，分钟；提前为负 | Signed departure delay in minutes; early departure is negative |
| `DepDelayMinutes` | 非负出发延误，分钟；提前记零 | Nonnegative departure delay in minutes; early departure is zero |
| `DepDel15` | 出发延误至少 15 分钟的 0/1 指示 | Indicator of departure delay of at least 15 minutes |
| `CRSArrTime` | 当地计划到达时间，HHMM | Scheduled local arrival time, HHMM |
| `ArrTime` | 当地实际到达时间，HHMM | Actual local arrival time, HHMM |
| `ArrDelay` | 有符号到达延误，分钟；提前为负 | Signed arrival delay in minutes; early arrival is negative |
| `ArrDelayMinutes` | 非负到达延误，分钟；提前记零 | Nonnegative arrival delay in minutes; early arrival is zero |
| `ArrDel15` | 到达延误至少 15 分钟的 0/1 指示 | Indicator of arrival delay of at least 15 minutes |
| `Cancelled` | 取消指示，0/1 | Cancellation indicator, 0/1 |
| `CancellationCode` | 取消原因代码 | Cancellation reason code |
| `Diverted` | 备降指示，0/1 | Diversion indicator, 0/1 |
| `CRSElapsedTime` | 计划飞行历时，分钟 | Scheduled elapsed time in minutes |
| `ActualElapsedTime` | 实际飞行历时，分钟 | Actual elapsed time in minutes |
| `Distance` | 航段距离，英里 | Flight distance in miles |
| `CarrierDelay` | 承运人原因延误，分钟 | Carrier-related delay in minutes |
| `WeatherDelay` | 天气原因延误，分钟 | Weather-related delay in minutes |
| `NASDelay` | 国家航空系统原因延误，分钟 | National Airspace System delay in minutes |
| `SecurityDelay` | 安全原因延误，分钟 | Security-related delay in minutes |
| `LateAircraftDelay` | 前序飞机晚到原因延误，分钟 | Late-aircraft delay in minutes |
| `source_file` | 原始 ZIP 文件名 | Source ZIP filename |
| `source_row` | 原始 CSV 行号，表头为第 1 行 | Source CSV row number, counting the header as row 1 |
| `flight_id` | 月份和原始行号构成的唯一标识 | Unique identifier built from source month and row number |
| `duplicate_key` | 日期、航司、航班号、起终点及计划出发组成的键重复 | Duplicate key based on date, carrier, flight number, airports and scheduled departure |
| `normal_operation` | 未取消且未备降 | Neither cancelled nor diverted |
| `origin_tz` | 出发机场 IANA 时区 | Origin airport IANA time zone |
| `dest_tz` | 到达机场 IANA 时区 | Destination airport IANA time zone |
| `scheduled_dep_utc` | 计划出发 UTC 时刻 | Scheduled departure timestamp in UTC |
| `scheduled_arr_utc` | 计划到达 UTC 时刻 | Scheduled arrival timestamp in UTC |
| `actual_dep_utc` | 实际出发 UTC 时刻 | Actual departure timestamp in UTC |
| `actual_arr_utc` | 实际到达 UTC 时刻 | Actual arrival timestamp in UTC |
| `scheduled_dep_hour` | 当地计划出发小时，0—23 | Scheduled local departure hour, 0–23 |
| `scheduled_arr_clock_error` | 计划到达钟表差异，分钟 | Scheduled arrival clock discrepancy in minutes |
| `actual_dep_clock_error` | 实际出发钟表差异，分钟 | Actual departure clock discrepancy in minutes |
| `actual_arr_clock_error` | 实际到达钟表差异，分钟 | Actual arrival clock discrepancy in minutes |
| `actual_elapsed_error` | 实际历时差异，分钟 | Actual elapsed-time discrepancy in minutes |
| `delay_values_valid` | 延误分钟及 15 分钟指示一致 | Delay values and 15-minute indicators are consistent |
| `time_valid` | 时间和延误检查通过 | Time and delay checks passed |
| `tail_schedule_incomplete` | 该尾号有计划出发时间缺失 | An aircraft has at least one missing scheduled departure timestamp |
| `schedule_tie` | 同尾号存在相同计划出发时刻 | Tied scheduled departures for the same aircraft |
| `actual_order_conflict` | 实际出发顺序与计划顺序冲突 | Actual departure order conflicts with scheduled order |
| `has_accepted_incoming_connection` | 作为有效衔接的下游航班出现 | Appears as a downstream flight in an eligible connection |


## 2. 衔接表 / Connection tables

`connection_candidates.csv.gz` 包含所有候选，`flight_connections.csv.gz` 包含 1,131,645 对有效衔接；两表字段相同。上游指上一班，下游指下一班。

`connection_candidates.csv.gz` contains all candidates; `flight_connections.csv.gz` contains 1,131,645 eligible connections. Both tables share the same fields. Upstream means the previous flight; downstream means the next flight.

| 字段 / Field | 中文说明 | English definition |
| --- | --- | --- |
| `connection_id` | 上游与下游航班 ID 组合 | Combined upstream and downstream flight IDs |
| `upstream_flight_id` | 上游航班 ID | Upstream flight identifier |
| `upstream_airline` | 上游航司代码 | Upstream carrier code |
| `upstream_flight_date` | 上游航班日期 | Upstream flight date |
| `upstream_scheduled_dep_utc` | 上游计划出发 UTC | Upstream scheduled departure UTC |
| `upstream_scheduled_arr_utc` | 上游计划到达 UTC | Upstream scheduled arrival UTC |
| `upstream_actual_dep_utc` | 上游实际出发 UTC | Upstream actual departure UTC |
| `upstream_actual_arr_utc` | 上游实际到达 UTC | Upstream actual arrival UTC |
| `downstream_flight_id` | 下游航班 ID | Downstream flight identifier |
| `downstream_airline` | 下游航司代码 | Downstream carrier code |
| `downstream_flight_date` | 下游航班日期 | Downstream flight date |
| `downstream_scheduled_dep_utc` | 下游计划出发 UTC | Downstream scheduled departure UTC |
| `downstream_scheduled_arr_utc` | 下游计划到达 UTC | Downstream scheduled arrival UTC |
| `downstream_actual_dep_utc` | 下游实际出发 UTC | Downstream actual departure UTC |
| `downstream_actual_arr_utc` | 下游实际到达 UTC | Downstream actual arrival UTC |
| `tail_number` | 两段航班共同的飞机尾号 | Aircraft tail number shared by both legs |
| `airport` | 衔接机场，即上游终点与下游起点 | Connection airport: upstream destination and downstream origin |
| `upstream_origin` | 上游航班起点 | Upstream origin airport |
| `downstream_origin` | 下游航班起点 | Downstream origin airport |
| `downstream_dest` | 下游航班终点 | Downstream destination airport |
| `upstream_arr_delay` | 上游有符号到达延误，分钟 | Signed upstream arrival delay in minutes |
| `upstream_arr_delay_minutes` | 上游非负到达延误，分钟 | Nonnegative upstream arrival delay in minutes |
| `downstream_dep_delay` | 下游有符号出发延误，分钟 | Signed downstream departure delay in minutes |
| `downstream_dep_delay_minutes` | 下游非负出发延误，预测目标，分钟 | Nonnegative downstream departure delay, the prediction target, in minutes |
| `downstream_sched_hour` | 下游当地计划出发小时 | Downstream scheduled local departure hour |
| `downstream_day_of_week` | 下游星期，周一为 1 | Downstream weekday, Monday=1 |
| `downstream_distance` | 下游航段距离，英里 | Downstream flight distance in miles |
| `downstream_crs_elapsed_time` | 下游计划飞行历时，分钟 | Downstream scheduled elapsed time in minutes |
| `scheduled_turnaround_minutes` | 下游计划出发减上游计划到达，分钟 | Scheduled downstream departure minus scheduled upstream arrival, in minutes |
| `actual_ground_minutes` | 下游实际出发减上游实际到达，分钟 | Actual downstream departure minus actual upstream arrival, in minutes |
| `delta_delay_minutes` | 下游非负出发延误减上游非负到达延误，分钟 | Nonnegative downstream departure delay minus nonnegative upstream arrival delay, in minutes |
| `overnight_scheduled` | 计划衔接跨过机场当地午夜 | Scheduled connection crosses local midnight at the connection airport |
| `first_exclusion_reason` | 首个未通过条件，或 accepted | First failed condition, or accepted |
| `check_tail_schedule_complete` | 尾号的计划时间完整 | Aircraft scheduled times are complete |
| `check_unique_schedule` | 两端计划顺序唯一且业务键无重复 | Both legs have unique schedule positions and business keys |
| `check_normal_both` | 两端均未取消、未备降 | Both legs are non-cancelled and non-diverted |
| `check_time_valid_both` | 两端时间检查通过 | Both legs pass time checks |
| `check_actual_order_consistent` | 实际出发顺序与计划顺序一致 | Actual departure order agrees with scheduled order |
| `check_airport_continuity` | 上游终点等于下游起点 | Upstream destination equals downstream origin |
| `check_scheduled_gap_nonnegative` | 计划周转至少 0 分钟 | Scheduled turnaround is at least zero minutes |
| `check_scheduled_gap_in_scope` | 计划周转不超过 1440 分钟 | Scheduled turnaround is at most 1,440 minutes |
| `check_actual_ground_nonnegative` | 实际地面间隔非负 | Actual ground interval is nonnegative |
| `main_analysis_eligible` | 通过所有衔接筛选 | Passes all connection eligibility rules |


## 3. 指标与时间 / Metrics and time

- 有符号延误用于还原实际时间；非负延误用于比较延误分钟，提前值归零。
- 延误率为延误至少 15 分钟的比例，分母取标签非空的航班。
- 延误变化小于零表示恢复，大于零表示增加。
- 计划周转窗口为 0—24 小时；隔夜按衔接机场当地日期判断。
- 原因字段按 BTS 的报告条件填写，缺失值保持为空。

- Signed delays reconstruct actual timestamps; nonnegative delays measure delay minutes, with early values set to zero.
- A delay rate is the proportion delayed by at least 15 minutes, among flights with an available label.
- Negative delay change means recovery; positive change means an increase.
- Scheduled turnaround spans 0–24 hours; overnight status uses local dates at the connection airport.
- Cause fields follow BTS reporting conditions; missing values remain missing.

## 4. 汇总表 / Summary tables

| 文件 / File | 内容 / Contents |
| --- | --- |
| `quality_summary.csv`、`.json` | 航班与衔接数量 / Flight and connection counts |
| `missing_values.csv` | 各字段缺失数量与比例 / Missing counts and shares by field |
| `time_checks.csv` | 时间核对项目与数量 / Time-check descriptions and counts |
| `connection_exclusions.csv` | 互斥的首个排除原因 / Mutually exclusive first exclusion reasons |
| `connection_check_failures.csv` | 各检查失败数量 / Failed counts for each check |
| `coverage_by_month_airline.csv` | 月份、航司及衔接覆盖 / Coverage by month and airline |
| `connection_audit_samples.csv` | 分组抽取的衔接示例 / Connection examples selected by group |
| `flights_sample_200.csv` | 200 行航班浏览样本 / A 200-row flight preview |
| `time_anomalies_normal_flights.csv` | 完成航班中的时间差异记录 / Time discrepancies among completed flights |
| `schema.csv` | 表名、字段与运行时类型 / Table, field and runtime data type |
| `eda_monthly_overview.csv` | 月度航班量、取消及备降 / Monthly flights, cancellations and diversions |
| `eda_daily_departure_delay_rate.csv` | 每日出发延误率 / Daily departure delay rates |
| `eda_hour_weekday_groups.csv` | 小时与星期分组 / Hour and weekday groups |
| `eda_airline_normal_operation.csv` | 航司航班量与延误率 / Airline flight counts and delay rates |
| `eda_origin_airport_normal_operation.csv` | 出发机场航班量与延误率 / Origin airport counts and delay rates |
| `eda_turnaround_delta_summary.csv` | 周转组延误变化 / Delay changes by turnaround group |
| `eda_summary.json` | EDA 总体统计 / Overall EDA statistics |

汇总字段：`sample_n` 为样本数，`label_n` 为非空标签数，`delay_rate` 为 0—1 比例，`delay_rate_pct` 为百分比；`median` 为中位数，`q25` 与 `q75` 为第 25 和第 75 百分位。

Summary fields: `sample_n` is the sample count, `label_n` the nonmissing-label count, `delay_rate` a proportion from 0 to 1, and `delay_rate_pct` a percentage. `median` is the median; `q25` and `q75` are the 25th and 75th percentiles.

## 5. 模型字段 / Modelling fields

在上游实际到达时预测下游出发延误。上游延误、计划周转和航班计划信息属于输入；`downstream_dep_delay_minutes` 是目标。`delta_delay_minutes`、`actual_ground_minutes` 和下游实际时刻用于事后分析。

Predict downstream departure delay at upstream actual arrival. Upstream delay, scheduled turnaround and schedule information are inputs; `downstream_dep_delay_minutes` is the target. `delta_delay_minutes`, `actual_ground_minutes` and downstream actual timestamps support retrospective analysis.
