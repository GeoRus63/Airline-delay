"""One retained synthetic PDF for layout review; no final report is generated.

Execute with the bundled ReportLab Python, after the session's PDF skill marker.
The project venv is used only for its existing matplotlib formula renderer.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import uuid

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ["reportlab", "pypdf", "PIL"])
if AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont
    from pypdf import PdfReader
    from scripts import render_report_pdf as renderer


@unittest.skipUnless(AVAILABLE, "Run with the bundled ReportLab/Pillow/pypdf Python")
class ReportPDF(unittest.TestCase):
    def test_one_synthetic_document_fonts_sections_math_tables_and_overwrite_guard(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "tmp/pdf_renderer_test" / ("synthetic_" + uuid.uuid4().hex[:10])
        directory.mkdir(parents=True)
        (directory / "figures").mkdir()
        image = Image.new("RGB", (1500, 650), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 44)
        draw.text((100, 45), "Synthetic monthly error comparison", fill="#203C53", font=font)
        draw.line((150, 515, 1370, 515), fill="black", width=4)
        for x, value, label in [(300, 290, "May"), (720, 220, "June"), (1140, 190, "July")]:
            draw.rectangle((x, 515-value, x+150, 515), fill="#557A95")
            draw.text((x+18, 540), label, fill="black", font=font)
        draw.text((90, 130), "MAE (minutes)", fill="black", font=font)
        image.save(directory / "figures/chart.png")
        prose = ("The synthetic development example illustrates a fixed temporal comparison. "
                 "All estimates in this fixture are invented for layout verification. "
                 "They are not results from the airline project. ")
        text = r'''---
title: Synthetic flight delay report
lang: en
mainfont: Times New Roman
fontsize: 11pt
course: Layout verification fixture
---
# Synthetic flight delay report

## Research questions and temporal design

This paragraph contains **bold emphasis**, *italic emphasis*, a [source link](https://example.com), and the inline target \(y_i=\max(D_i,0)\). The strict condition is \(t_{prediction}<t_{label}\).

\[
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

Table 1. Synthetic model errors and unequal monthly sample support.

| Model | May MAE (min) | June MAE (min) | July MAE (min) |
| --- | ---: | ---: | ---: |
| Training median | 12.4 | 12.1 | 12.0 |
| Fixed linear regression | 12.0 | 11.8 | 11.7 |
| A wrapped model label that must remain readable | 11.9 | 11.5 | 11.4 |

## Results and evidence

![Monthly error patterns](figures/chart.png)

Monthly error patterns [E0001]. This is a synthetic chart caption.

'''
        text += "\n\n".join(prose * 3 for _ in range(6))
        text += "\n\n## References\n\nExample Author (2026). Synthetic reference for renderer verification. https://example.com/reference.\n\n"
        text += "## Appendix A: Additional definitions\n\nThe appendix is excluded from the main-text page count. This fixture does not certify a future final report.\n"
        source = directory / "synthetic_report.md"
        source.write_text(text, encoding="utf-8")
        result = renderer.render_report(source)
        self.assertTrue(result["fonts_passed"])
        self.assertTrue(all(result["font_embedding_truetype"].values()))
        self.assertGreaterEqual(result["minimum_pdf_text_points"], 10)
        self.assertGreater(result["page_count"], result["body_page_count"])
        self.assertEqual({page["section"] for page in result["page_sections"]}, {"cover", "contents", "body", "references", "appendix"})
        self.assertEqual(len(result["qa_images"]), result["page_count"])
        self.assertEqual(result["elements"]["tables"], 1)
        self.assertEqual(result["elements"]["figures"], 1)
        self.assertEqual(result["math"]["expression_count"], 3)
        self.assertFalse(result["fit_called"])
        pdf_text = "\n".join(page.extract_text() for page in PdfReader(source.with_suffix(".pdf")).pages)
        self.assertIn("Contents", pdf_text)
        self.assertIn("Training median", pdf_text)
        self.assertNotIn(r"\frac", pdf_text)
        self.assertNotIn(r"\(", pdf_text)
        with self.assertRaises(FileExistsError):
            renderer.render_report(source)
        print("SYNTHETIC_PDF_EVIDENCE=" + json.dumps({"directory": str(directory), "page_count": result["page_count"],
              "body_page_count": result["body_page_count"], "fonts": result["pdf_text_fonts"], "qa_images": result["qa_images"]}))

    def test_preflight_rejects_small_font_and_completed_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "report.md"
            source.write_text("---\nfontsize: 9pt\n---\n# Too small\n\nBody.")
            with self.assertRaisesRegex(ValueError, "10--14"):
                renderer.render_report(source)
            (directory / "manifest.json").write_text(json.dumps({"status": "completed"}))
            with self.assertRaisesRegex(ValueError, "completed"):
                renderer.render_report(source)
            self.assertFalse(source.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
