"""Render the English Markdown report using embedded Times New Roman fonts.

Run with the Codex bundled Python (ReportLab/Pillow/pypdf), for example:
  <bundled-python> scripts/render_report_pdf.py --input reports/report_en.md
Existing project matplotlib is used only in an isolated formula-image subprocess.
No report calculations, model fitting or research archive changes occur here.
The caller must follow the PDF skill's authoring-marker contract before creating
the first PDF; this renderer never repeats that session-level marker.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from PIL import Image as PillowImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image, KeepTogether,
                               PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
FONT_FILES = {"TNR": "times.ttf", "TNR-Bold": "timesbd.ttf", "TNR-Italic": "timesi.ttf", "TNR-BoldItalic": "timesbi.ttf"}
SECTION_TYPES = {"cover", "contents", "body", "references", "appendix"}
BODY_LIMIT = 25


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _protect(path):
    for parent in [Path(path).resolve().parent, *Path(path).resolve().parents]:
        manifest = parent / "manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8-sig")).get("status") == "completed":
            raise ValueError("PDF rendering must not modify a completed research archive")


def _frontmatter(text):
    metadata = {}
    if text.startswith("---\n") or text.startswith("---\r\n"):
        lines = text.splitlines()
        end = next((i for i in range(1, len(lines)) if lines[i] == "---"), None)
        if end is None:
            raise ValueError("Unclosed report metadata")
        for line in lines[1:end]:
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip("\"'")
        text = "\n".join(lines[end + 1:])
    size = float(metadata.get("fontsize", "11pt").removesuffix("pt"))
    if not 10 <= size <= 14:
        raise ValueError("English report body font must be 10--14 points")
    if metadata.get("mainfont", "Times New Roman") != "Times New Roman" or metadata.get("lang", "en") != "en":
        raise ValueError("Renderer expects English and Times New Roman metadata")
    return metadata, text, size


def _fonts(directory):
    paths = {name: Path(directory) / filename for name, filename in FONT_FILES.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required Times New Roman font missing: {path}")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-Bold", italic="TNR-Italic", boldItalic="TNR-BoldItalic")
    return paths


def _styles(size):
    common = {"fontName": "TNR", "fontSize": size, "leading": size * 1.45, "spaceAfter": 8,
              "allowWidows": 0, "allowOrphans": 0, "splitLongWords": 1, "autoLeading": "max"}
    body = ParagraphStyle("Body", **common)
    return {"body": body,
            "title": ParagraphStyle("Title", parent=body, fontName="TNR-Bold", fontSize=24, leading=30, alignment=TA_CENTER, spaceAfter=24),
            "subtitle": ParagraphStyle("Subtitle", parent=body, fontSize=12, leading=18, alignment=TA_CENTER),
            "h2": ParagraphStyle("Heading2", parent=body, fontName="TNR-Bold", fontSize=15, leading=20, spaceBefore=12, spaceAfter=8, keepWithNext=True),
            "h3": ParagraphStyle("Heading3", parent=body, fontName="TNR-Bold", fontSize=12, leading=17, spaceBefore=8, spaceAfter=6, keepWithNext=True),
            "caption": ParagraphStyle("Caption", parent=body, fontSize=10, leading=13.5, spaceBefore=6, spaceAfter=12),
            "table": ParagraphStyle("TableCell", parent=body, fontSize=10, leading=13, spaceAfter=0),
            "table_head": ParagraphStyle("TableHead", parent=body, fontName="TNR-Bold", fontSize=10, leading=13, spaceAfter=0),
            "equation": ParagraphStyle("Equation", parent=body, alignment=TA_CENTER, spaceBefore=8, spaceAfter=12),
            "toc0": ParagraphStyle("TOC0", parent=body, fontSize=11, leading=16, spaceBefore=5),
            "toc1": ParagraphStyle("TOC1", parent=body, fontSize=10, leading=14, leftIndent=16)}


MATH_SCRIPT = r'''
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import mathtext, font_manager
from PIL import Image
request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
font_manager.fontManager.addfont(request['font'])
font_manager.fontManager.addfont(request['italic'])
font_manager.fontManager.addfont(request['bold'])
matplotlib.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'custom',
 'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
 'mathtext.bf':'Times New Roman:bold','mathtext.sf':'Times New Roman',
 'mathtext.tt':'Times New Roman','mathtext.cal':'Times New Roman:italic',
 'mathtext.fallback':None})
for item in request['items']:
    mathtext.math_to_image('$'+item['tex']+'$', item['path'],
        prop=font_manager.FontProperties(fname=request['font'],size=15),dpi=240,format='png',color='black')
'''


def _math_assets(text, directory, fonts, math_python):
    expressions = list(dict.fromkeys(match.group(1).strip() for match in re.finditer(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", text, re.S)
                                    if match.group(1) is not None))
    expressions += [match.group(1).strip() for match in re.finditer(r"\\\[(.*?)\\\]", text, re.S)]
    expressions = list(dict.fromkeys(expressions))
    if not expressions:
        return {}
    if not Path(math_python).is_file():
        raise FileNotFoundError("Formula rendering requires the existing matplotlib Python; set --math-python")
    directory.mkdir()
    items = [{"tex": expression, "path": str(directory / (hashlib.sha256(expression.encode()).hexdigest()[:20] + ".png"))} for expression in expressions]
    request = {"font": str(fonts["TNR"]), "italic": str(fonts["TNR-Italic"]), "bold": str(fonts["TNR-Bold"]), "items": items}
    request_path = directory / "requests.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    result = subprocess.run([str(math_python), "-c", MATH_SCRIPT, str(request_path)], capture_output=True, text=True)
    if result.returncode or "does not have a glyph" in result.stderr or "Substituting with a dummy symbol" in result.stderr:
        raise ValueError("Formula rendering failed; raw TeX is never published: " + result.stderr[-2000:])
    return {item["tex"]: Path(item["path"]) for item in items}


def _inline(text, math):
    # Normalize typographic dashes to the PDF skill's portable ASCII hyphen.
    text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    tokens = {}
    def equation(match):
        expression = match.group(1).strip()
        path = math[expression]
        with PillowImage.open(path) as image:
            width, height = image.size
        token = f"MATHPLACEHOLDER{len(tokens)}END"
        tokens[token] = f'<img src="{html.escape(str(path), quote=True)}" width="{width * 72 / 240:.3f}" height="{height * 72 / 240:.3f}" valign="middle"/>'
        return token
    text = re.sub(r"\\\((.*?)\\\)", equation, text, flags=re.S)
    result = html.escape(text, quote=False)
    result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<link href="{html.escape(m[2], quote=True)}" color="#1E4262">{m[1]}</link>', result)
    result = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", result)
    result = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", result)
    result = re.sub(r"`([^`]+)`", r"\1", result)
    for token, rendered in tokens.items():
        result = result.replace(token, rendered)
    if "\\(" in result or "\\[" in result:
        raise ValueError("Unparsed math delimiter")
    return result


class SectionMarker(Flowable):
    def __init__(self, section):
        super().__init__()
        self.section = section
        self.width = self.height = 0

    def draw(self):
        pass


class ReportDocument(BaseDocTemplate):
    def __init__(self, path, *, title, **kwargs):
        super().__init__(str(path), pagesize=A4, leftMargin=54, rightMargin=54, topMargin=58, bottomMargin=54, title=title, **kwargs)
        self.report_title = title
        self.addPageTemplates(PageTemplate(id="report", frames=[Frame(self.leftMargin, self.bottomMargin, self.width, self.height, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPageEnd=self._page_end))

    def beforeDocument(self):
        self.section = "cover"
        self.page_sections = {}

    def afterFlowable(self, flowable):
        if isinstance(flowable, SectionMarker):
            self.section = flowable.section
        if hasattr(flowable, "heading_level"):
            self.canv.bookmarkPage(flowable.bookmark)
            self.notify("TOCEntry", (flowable.heading_level, flowable.getPlainText(), self.page, flowable.bookmark))

    def _page_end(self, canvas, doc):
        self.page_sections[self.page] = self.section
        if self.section != "cover":
            canvas.saveState()
            canvas.setFont("TNR", 10)
            canvas.setFillColor(colors.HexColor("#4B4B4B"))
            canvas.drawRightString(A4[0] - 54, 32, str(self.page))
            if self.section == "body":
                canvas.setFont("TNR-Italic", 10)
                canvas.drawString(54, A4[1] - 35, self.report_title[:85])
            canvas.restoreState()


def _heading(text, level, styles, math, index):
    item = Paragraph(_inline(text, math), styles["h2" if level <= 2 else "h3"])
    item.heading_level = max(0, level - 2)
    item.bookmark = f"heading-{index}"
    return item


def _table(lines, width, styles, math):
    split = lambda line: [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
    rows = [split(lines[0])] + [split(line) for line in lines[2:]]
    if any(len(row) != len(rows[0]) for row in rows) or len(rows[0]) > 7:
        raise ValueError("Report tables require consistent rows and at most seven columns")
    # Wrap every cell; never reduce text below the ten-point minimum to fit.
    content = [[Paragraph(_inline(cell.replace("\\|", "|"), math), styles["table_head" if index == 0 else "table"]) for cell in row] for index, row in enumerate(rows)]
    weights = [max(8, min(40, max(len(row[column]) for row in rows))) for column in range(len(rows[0]))]
    widths = [width * weight / sum(weights) for weight in weights]
    table = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF2")),
        ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#5C6F7D")),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#A9B2B9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return table


def _story(text, metadata, styles, math, source, *, include_contents):
    lines = text.splitlines()
    title_line = next((i for i, line in enumerate(lines) if re.match(r"^#\s", line)), None)
    title = metadata.get("title", "Research report")
    if title_line is not None:
        title = lines.pop(title_line)[2:].strip()
    story = [SectionMarker("cover"), Spacer(1, 145), Paragraph(_inline(title, math), styles["title"])]
    for name in ["author", "course", "date"]:
        if metadata.get(name):
            story.append(Paragraph(_inline(metadata[name], math), styles["subtitle"]))
    if include_contents:
        toc = TableOfContents()
        toc.levelStyles = [styles["toc0"], styles["toc1"]]
        story += [PageBreak(), SectionMarker("contents"), Paragraph("Contents", styles["h2"]), toc]
    story += [PageBreak(), SectionMarker("body")]
    figures, counts, index = [], {"figures": 0, "tables": 0, "equations": 0}, 0
    section = "body"
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        marker = re.fullmatch(r"<!--\s*report-section:\s*([a-z]+)\s*-->", line)
        if marker:
            requested = marker[1]
            if requested not in SECTION_TYPES or requested in {"cover", "contents"}:
                raise ValueError("Inline section marker supports body/references/appendix only")
            if requested != section:
                story += [PageBreak(), SectionMarker(requested)]
                section = requested
            index += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            label = heading[2]
            requested = "references" if re.fullmatch(r"References|Bibliography", label, re.I) else "appendix" if label.lower().startswith("appendix") else section
            if requested != section:
                story += [PageBreak(), SectionMarker(requested)]
                section = requested
            story.append(_heading(label, len(heading[1]), styles, math, index))
            index += 1
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image:
            if not image[1].strip():
                raise ValueError("Report figures require a nonempty descriptive caption")
            image_path = (source.parent / image[2]).resolve()
            if not image_path.is_relative_to(source.parent) or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError("Report figures must be local PNG/JPEG files within the report directory")
            with PillowImage.open(image_path) as raw:
                width_px, height_px = raw.size
            width = min(A4[0] - 108, width_px * 72 / 110)
            height = width * height_px / width_px
            if height > 475:
                width *= 475 / height
                height = 475
            following = index + 1
            while following < len(lines) and not lines[following].strip():
                following += 1
            caption = image[1]
            if following < len(lines) and lines[following].strip().startswith(caption):
                caption = lines[following].strip()
                index = following
            counts["figures"] += 1
            caption = f"Figure {counts['figures']}. {caption}"
            story.append(KeepTogether([Image(str(image_path), width=width, height=height), Paragraph(_inline(caption, math), styles["caption"])]))
            figures.append({**_receipt(image_path), "width_points": width, "height_points": height,
                            "effective_dpi": width_px / (width / 72), "caption": caption})
            index += 1
            continue
        if line.startswith(r"\["):
            expression = line[2:]
            while r"\]" not in expression:
                index += 1
                if index >= len(lines):
                    raise ValueError("Unclosed display equation")
                expression += " " + lines[index].strip()
            if expression.split(r"\]", 1)[1].strip():
                raise ValueError("Display math closing delimiter must end its paragraph")
            expression = expression.split(r"\]", 1)[0].strip()
            story.append(Paragraph(_inline(r"\(" + expression + r"\)", math), styles["equation"]))
            counts["equations"] += 1
            index += 1
            continue
        if index + 1 < len(lines) and "|" in line and re.fullmatch(r"\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*", lines[index + 1]):
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            story += [_table(table_lines, A4[0] - 108, styles, math), Spacer(1, 10)]
            counts["tables"] += 1
            continue
        if line.startswith("```") or re.match(r"^[-*]\s|^\d+\.\s", line):
            raise ValueError("Course report body must use continuous prose; fenced code and Markdown lists are unsupported")
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip():
            if re.match(r"^(#|!\[|<!--|\\\[)", lines[index].strip()):
                break
            paragraph.append(lines[index].strip())
            index += 1
        style = styles["caption"] if re.match(r"^(Table|Figure)\s+\d+[.:]", line) else styles["body"]
        story.append(Paragraph(_inline(" ".join(paragraph), math), style))
    return story, title, figures, counts


def _embedded_fonts(reader, used_fonts):
    embeddings = {}
    for page in reader.pages:
        resource = page["/Resources"].get("/Font")
        for entry in resource.get_object().values() if resource else []:
            font = entry.get_object()
            name = str(font.get("/BaseFont", "unknown"))
            if name in used_fonts:
                descriptor = font.get("/FontDescriptor")
                embeddings[name] = bool(descriptor and "/FontFile2" in descriptor.get_object())
    return embeddings


def render_report(input_path, output_path=None, *, font_dir="C:/Windows/Fonts", math_python=None,
                  include_contents=True, pdftoppm_path=None):
    source = Path(input_path).resolve()
    output = Path(output_path).resolve() if output_path else source.with_suffix(".pdf")
    layout_path, qa_dir = output.with_suffix(".layout.json"), output.with_suffix(".qa")
    for target in [output, layout_path, qa_dir]:
        _protect(target)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing PDF/evidence: {target}")
    if source.suffix.lower() != ".md" or output.suffix.lower() != ".pdf":
        raise ValueError("Expected Markdown input and PDF output")
    metadata, text, size = _frontmatter(source.read_text(encoding="utf-8-sig"))
    fonts = _fonts(font_dir)
    poppler = pdftoppm_path or shutil.which("pdftoppm")
    if not poppler:
        raise FileNotFoundError("pdftoppm is required for every-page visual-QA images")
    output.parent.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir()
    math_python = math_python or (ROOT / ".venv/Scripts/python.exe")
    math = _math_assets(text, qa_dir / "math", fonts, math_python)
    story, title, figures, counts = _story(text, metadata, _styles(size), math, source, include_contents=include_contents)
    document = ReportDocument(output, title=title)
    document.multiBuild(story)
    reader = PdfReader(output)
    used_fonts, sizes = set(), []
    extracted = []
    for page in reader.pages:
        def visit(text, cm, tm, font, font_size):
            if text.strip():
                used_fonts.add(str((font or {}).get("/BaseFont", "unknown")))
                sizes.append(float(font_size))
        extracted.append(page.extract_text(visitor_text=visit))
    if len(document.page_sections) != len(reader.pages):
        raise ValueError("Page classification is incomplete")
    if re.search(r"\\(?:frac|sqrt|sum|\(|\[)", "\n".join(extracted)):
        raise ValueError("Rendered PDF still contains unparsed TeX")
    pages = [{"page": number, "section": document.page_sections[number]} for number in range(1, len(reader.pages) + 1)]
    body_pages = sum(item["section"] == "body" for item in pages)
    prefix = qa_dir / "page"
    subprocess.run([str(poppler), "-r", "130", "-png", str(output), str(prefix)], check=True, capture_output=True)
    previews = sorted(qa_dir.glob("page-*.png"))
    if len(previews) != len(reader.pages):
        raise ValueError("Not all PDF pages rendered to QA images")
    embeddings = _embedded_fonts(reader, used_fonts)
    font_passed = (bool(sizes) and min(sizes) >= 10 and all("TimesNewRoman" in name for name in used_fonts)
                   and set(embeddings) == used_fonts and all(embeddings.values()))
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "input": _receipt(source), "pdf": _receipt(output), "page_count": len(reader.pages),
              "body_page_count": body_pages, "body_page_limit": BODY_LIMIT, "body_page_limit_passed": body_pages <= BODY_LIMIT,
              "page_sections": pages, "font_family": "Times New Roman", "font_files": [_receipt(path) for path in fonts.values()],
              "pdf_text_fonts": sorted(used_fonts), "minimum_pdf_text_points": min(sizes) if sizes else None,
              "font_embedding_truetype": embeddings,
              "fonts_passed": font_passed, "body_points": size, "caption_table_points": 10,
              "math": {"expression_count": len(math), "base_points": 15, "font": "Times New Roman", "renderer": "existing matplotlib.mathtext to 240-dpi PNG; no raw TeX output",
                       "python": str(math_python) if math else None},
              "figures": figures, "elements": counts, "qa_images": [str(path) for path in previews],
              "visual_review_required": True, "visual_review_completed": False,
              "layout_checks_passed": font_passed and body_pages <= BODY_LIMIT,
              "limitations": ["Raster chart axis-label sizes require visual review; document font checks cannot certify image text",
                              "Page/font checks apply only to this supplied Markdown; they do not certify a future final report"],
              "fit_called": False, "research_data_read": False}
    result["runtime"] = {"python": sys.version, "executable": sys.executable,
                         "reportlab": importlib.metadata.version("reportlab"),
                         "pypdf": importlib.metadata.version("pypdf")}
    layout_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result["layout_checks_passed"]:
        raise ValueError(f"Report layout failed; keep PDF and evidence for review: {layout_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="reports/report_en.md")
    parser.add_argument("--output")
    parser.add_argument("--font-dir", default="C:/Windows/Fonts")
    parser.add_argument("--math-python")
    parser.add_argument("--no-contents", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render_report(args.input, args.output, font_dir=args.font_dir, math_python=args.math_python,
                                   include_contents=not args.no_contents), ensure_ascii=False, indent=2))
