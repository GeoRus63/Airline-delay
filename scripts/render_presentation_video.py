"""Render a captioned, silent presentation MP4 from build_presentation.mjs output."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def sha256(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


def run(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}\n{result.stderr[-6000:]}")
    return result


def srt_seconds(value):
    h, m, s, ms = map(int, re.split(r"[:,]", value))
    return h * 3600 + m * 60 + s + ms / 1000


def ass_time(seconds):
    centiseconds = round(seconds * 100)
    h, rem = divmod(centiseconds, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-duration", type=float, default=600)
    parser.add_argument("--ffmpeg", type=Path, default=Path("E:/JasnaDev/assets/ffmpeg/bin/ffmpeg.exe"))
    parser.add_argument("--ffprobe", type=Path, default=Path("E:/JasnaDev/assets/ffmpeg/bin/ffprobe.exe"))
    args = parser.parse_args()
    deck, output = args.deck_directory.resolve(), args.output_directory.resolve()
    ffmpeg, ffprobe = args.ffmpeg.resolve(), args.ffprobe.resolve()
    if output.exists():
        raise FileExistsError(f"Output exists. Use a new revision directory: {output}")
    timeline_path, captions_path = deck / "timing.json", deck / "captions_en.srt"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8-sig"))
    duration = float(timeline["duration"])
    if abs(duration - args.expected_duration) > .001 or duration <= 0:
        raise ValueError("Timeline duration does not equal --expected-duration")
    if not timeline["slides"] or any(float(s["duration"]) <= 0 for s in timeline["slides"]):
        raise ValueError("A nonempty timeline with positive slide durations is required")
    if abs(sum(float(s["duration"]) for s in timeline["slides"]) - duration) > .001:
        raise ValueError("Slide durations do not sum to timeline duration")
    source_pngs = [(deck / s["png"]).resolve() for s in timeline["slides"]]
    if any(not p.is_relative_to(deck) or not p.is_file() for p in source_pngs):
        raise ValueError("All source PNGs must exist inside deck_directory")
    srt = captions_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
    cues = []
    for block in re.split(r"\n\s*\n", srt):
        lines = block.splitlines()
        match = re.fullmatch(r"(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})", lines[1]) if len(lines) >= 3 else None
        if not match or len(lines) < 3:
            raise ValueError("Malformed SRT cue")
        start, end = map(srt_seconds, match.groups())
        if start < 0 or end <= start or end > duration + .001 or (cues and start < cues[-1][1] - .001):
            raise ValueError("SRT cues must be ordered, nonoverlapping and within timeline")
        cues.append((start, end, "\n".join(lines[2:])))
    if not cues or abs(cues[0][0]) > .001 or abs(cues[-1][1] - duration) > .001:
        raise ValueError("SRT must cover the timeline start and end")
    output.mkdir(parents=True)
    build = output / "_build"
    build.mkdir()
    versions = {"ffmpeg": run([str(ffmpeg), "-version"]).stdout, "ffprobe": run([str(ffprobe), "-version"]).stdout}
    # Keep concat paths relative and controlled. This also snapshots the exact source images.
    concat = ["ffconcat version 1.0"]
    for i, (source, slide) in enumerate(zip(source_pngs, timeline["slides"]), 1):
        name = f"slide-{i:02d}.png"
        (build / name).write_bytes(source.read_bytes())
        concat.extend([f"file '{name}'", f"duration {float(slide['duration']):.9f}"])
    concat.append(f"file 'slide-{len(source_pngs):02d}.png'")
    (build / "slides.ffconcat").write_text("\n".join(concat) + "\n", encoding="utf-8")
    ass = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial,36,&H00FFFFFF,&H00FFFFFF,&H001A1A1A,&H001A1A1A,0,0,0,0,100,100,0,0,3,8,0,2,100,100,48,1
Style: Label,Arial,22,&H003E4B5A,&H003E4B5A,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,3,4,0,8,80,80,12,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass += f"Dialogue: 0,0:00:00.00,{ass_time(duration)},Label,,0,0,0,,Captioned demonstration (no narration)\n"
    for start, end, text in cues:
        # Escape ASS control syntax without altering the visible source text.
        escaped = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", r"\N")
        ass += f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Caption,,0,0,0,,{escaped}\n"
    (build / "captions.ass").write_text(ass, encoding="utf-8-sig")
    target = output / "airline_delay_captioned.mp4"
    # Keep the whole 16:9 slide above a separate subtitle band. Full-frame slides
    # put their bottom captions underneath the two-line narration subtitles.
    # Expand stills before rendering subtitles so cues change throughout each slide.
    video_filter = "scale=1600:900:flags=lanczos,pad=1920:1080:160:0:color=white,setsar=1,fps=5,ass=captions.ass"
    command = [str(ffmpeg), "-hide_banner", "-n", "-f", "concat", "-safe", "1", "-i", "slides.ffconcat", "-vf", video_filter, "-r", "5", "-fps_mode", "cfr", "-t", str(duration), "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-metadata", "title=Airline delay captioned demonstration", "-metadata", "comment=Captioned demonstration, no narration. Human narration has not been recorded.", str(target)]
    result = run(command, cwd=build)
    (build / "encode.log").write_text(result.stderr, encoding="utf-8")
    probe = json.loads(run([str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(target)]).stdout)
    video = [s for s in probe["streams"] if s["codec_type"] == "video"]
    if len(video) != 1 or (video[0]["width"], video[0]["height"]) != (1920, 1080):
        raise ValueError("Encoded video dimensions or video stream count failed")
    if any(s["codec_type"] == "audio" for s in probe["streams"]):
        raise ValueError("Captioned demonstration must contain no audio")
    if abs(float(probe["format"]["duration"]) - duration) > .001:
        raise ValueError("Encoded video duration mismatch")
    if target.stat().st_size >= 100_000_000:
        raise ValueError("Video exceeds the GitHub 100 MB file limit")
    for name, time in [("first-caption.png", min(1, duration / 4)), ("last-frame.png", duration - .2)]:
        run([str(ffmpeg), "-hide_banner", "-n", "-ss", str(time), "-i", str(target), "-frames:v", "1", str(output / name)])
    receipt = {"video": str(target), "duration_seconds": duration, "width": 1920, "height": 1080, "fps": 5, "audio": False, "captions": "burned from source SRT via ASS", "captioned_label": "Captioned demonstration (no narration)", "size_bytes": target.stat().st_size, "sha256": sha256(target), "tools": {k: {"path": str(p), "sha256": sha256(p), "version": versions[k]} for k, p in [("ffmpeg", ffmpeg), ("ffprobe", ffprobe)]}, "source_files": [{"path": str(p), "sha256": sha256(p)} for p in [timeline_path, captions_path, *source_pngs]], "command": command, "working_directory": str(build), "ffprobe": probe, "visual_review_required": ["first-caption.png", "last-frame.png"]}
    receipt["display_layout"] = {"slide": {"x": 160, "y": 0, "width": 1600, "height": 900}, "subtitle_band": {"y": 900, "height": 180}, "subtitle_font": "Arial", "subtitle_font_size": 36, "subtitle_bottom_margin": 48, "canvas_background": "white"}
    (output / "video_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["video", "duration_seconds", "width", "height", "fps", "audio", "size_bytes", "sha256"]}, indent=2))


if __name__ == "__main__":
    main()
