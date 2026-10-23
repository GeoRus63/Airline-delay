# 衔接样本复算 / Connection sample recalculation

运行：`python _build/verify_samples.py`。该脚本使用标准库从原始 ZIP 重算时间和衔接指标。

Run `python _build/verify_samples.py`. The script uses the standard library to recalculate timestamps and connection metrics from the original ZIP files.

2026-10-18 的复算涵盖普通、隔夜、跨月、提前到达和上游延误超过 180 分钟五类，每类 5 对，共 25 对衔接和 43 条不同原始航班。UTC 时刻、机场、尾号、周转时间、实际地面时间及延误变化与导出样本一致。

The 2026-10-18 check covered five examples each of ordinary, overnight, cross-month, early-arrival and upstream-delay-over-180-minute connections: 25 pairs and 43 distinct raw flights. UTC timestamps, airports, tail numbers, turnaround, actual ground time and delay changes matched the exported samples.

延误变化采用下游非负出发延误减上游非负到达延误；候选与有效衔接总数分别为 1,202,754 和 1,131,645。

Delay change uses nonnegative downstream departure delay minus nonnegative upstream arrival delay. Candidate and eligible connection totals are 1,202,754 and 1,131,645 respectively.
