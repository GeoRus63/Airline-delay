# 来源资料 / Source references

原始资料保留其发布时的英文内容，字段的中英解释见上一级 [数据字典 / Data dictionary](../data_dictionary.md)。

Source documents retain their published English text. Chinese and English field definitions are in the parent folder’s data dictionary.

| 文件 / File | 用途 / Purpose |
| --- | --- |
| `2026_04_readme.html`、`2026_05_readme.html` | BTS 原始字段定义 / Original BTS field definitions |
| `airportsdata_snapshot.csv` | 固定版本机场资料 / Version-pinned airport reference |
| `airport_timezones_used.csv` | 实际使用的机场与时区 / Airports and time zones used in the analysis |
| `airportsdata_README.rst` | airportsdata 项目说明 / airportsdata project documentation |
| `airportsdata_LICENSE.txt` | MIT 许可证原文 / Original MIT licence |
| `sources.json` | 来源、版本和文件哈希 / Sources, version and file hashes |

机场资料字段：`iata`、`icao` 是机场代码，`name` 是名称，`country` 是国家，`lat`、`lon` 是经纬度，`tz` 是 IANA 时区。

Airport fields: `iata` and `icao` are airport codes; `name` is the airport name; `country` is the country; `lat` and `lon` are coordinates; `tz` is the IANA time zone.

时区映射来自保存的 airportsdata 快照，并通过 BTS 当地钟表记录核对。

Time-zone mapping uses the saved airportsdata snapshot and is checked against BTS local clock records.
