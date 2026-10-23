# 演示材料构建

`scripts/build_presentation.mjs` 使用本机 Codex 随附的 `@oai/artifact-tool` 生成可编辑 PPTX、逐页 PNG、英文人声讲稿、英文字幕和逐页时长。它不生成 AI 配音。最终内容须来自有效运行的冻结结果，尚未生成的指标不得填入演示稿。

从仓库根目录使用 PowerShell 7 执行。输出目录必须是尚不存在的新目录，避免覆盖既有版本：

```powershell
& "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe" scripts/build_presentation.mjs reports/presentation/content.json reports/presentation/final_v01
```

顶层 JSON 字段为 `title`、`duration`、`slides`。约十分钟演示使用 `duration: 600`，所有页的 `duration` 必须精确加总为 600。每页字段：

- `title`：英文标题。研究发现页优先明确写出该页支持的发现。
- `claims`：最多四条英文陈述。图表页一般一至两条，避免挤占证据空间。
- `table`：二维矩形数组，首行为表头，最多八行。数字保留所需单位和精度。
- `chart`：与 `table` 二选一，含 `type`（`bar` 或 `line` 等原生图表类型）、`categories`、`series`（每项含 `name`、`values`），以及可选 `xLabel`、`yLabel`、`numberFormat`、`minimum`、`showValues`、`direction`。
- `image`：与 `table`、`chart` 互斥。支持仓库相对路径字符串，或 `{ "path": "experiments/runs/Rxxxx/figures/network.png", "alt": "English description" }`。用于网络、热图、PDP 等科学图，不接受整页截图替代幻灯片。原始 PNG/JPEG 完整嵌入并保纵横比，源文件路径和 SHA256 写入该页 speaker notes 及 `source_images.json`。科学图需提前使用适合投影的字号。
- `caption`：可选的英文图表释义或影响结论的样本限制。
- `notes`：逐页英文人声讲稿。不要在其中放构建、验证过程说明。
- `citations`：该页结论和数值对应的 `run_id`、原结果相对路径及行号或筛选条件。写入 PPTX speaker notes 和讲稿。
- `duration`：该页展示秒数。
- `layout`：可选 `cover`，适合简短封面；不添加自动封面。

建议最终 10–12 页，以实际发现开场，然后说明 RQ1/RQ2、预测时点与前向验证、RQ1 机场结果、特征增量、模型比较、8 月冻结测试、错误集中的场景和研究限制。结构应随最终证据调整，不预写结论。

每次构建生成：

- `airline_delay.pptx`：原生文本、表格和图表，图表含完整字面数值的嵌入工作簿。
- `slides/slide-NN.png`：从通过 finalizer 的最终 PPTX 重新导入后渲染。
- `speaker_script_en.md`：逐页英文讲稿、时间段和证据引用。
- `captions_en.srt`：按讲稿字数分配的初始字幕时间轴，供字幕演示使用。
- `timing.json`：逐页时长及 `human-recording-required` 状态。
- `content.json`：输入内容副本。
- `source_images.json`：外部科学图的页号、源路径和 SHA256 记录。

finalizer 的草稿和完整验证收据位于输出目录同级的隐藏构建目录 `.<输出目录名>-build/`。验证覆盖结构、16:9 尺寸、指定 Arial 字体、页数、原生图表/表格和重新导入。它不代替逐页视觉检查和研究证据核对。不得将合成测试数值当作研究结果。

最终需逐页检查全部 PNG，核对字幕和讲稿事实，再执行字幕视频合成：

```powershell
& "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" scripts/render_presentation_video.py reports/presentation/final_v01 reports/presentation/video_v01
```

`render_presentation_video.py` 默认要求 600 秒，拒绝覆盖已有目录。本机 FFmpeg/FFprobe 默认使用 `E:/JasnaDev/assets/ffmpeg/bin/`，可通过 `--ffmpeg`、`--ffprobe` 指定其他已验证路径。编码为 1920×1080、5 fps、H.264、CRF 22、yuv420p、faststart，无音轨。字幕从 SRT 转换后烧录，整段显示 “Captioned demonstration (no narration)”。脚本实际检查时长、尺寸、音轨和文件小于 100 MB，并提取首段字幕截图及最后一帧。`video_receipt.json` 保存工具确切路径、版本与 SHA256、源 PNG/时间轴/字幕 SHA256、完整编码参数和 FFprobe 输出。不得仅凭编码成功声称逐帧内容已检查，至少复核两张提取图及每页转场。

12 秒合成测试使用 `--expected-duration 12`，不要把该参数用于最终课程视频。提交课程技术分享时按同一份 `speaker_script_en.md` 录制真人英文讲解，合成有真人旁白的版本，并依据实际录音调整字幕。自动分配字幕并不代表已经与真人录音同步。

环境路径可通过 `CODEX_PRESENTATION_RUNTIME`、`RUNTIME_NODE_MODULES`、`RUNTIME_PYTHON` 和 `PRESENTATION_SKILL_DIR` 覆盖。应继续使用提供的 runtime，不安装替代 PPTX 库。PPTX authoring marker 在本次首次合成生成前已成功执行一次。
