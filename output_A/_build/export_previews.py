"""导出双语静态 HTML / Export bilingual static HTML previews."""
from pathlib import Path
import re

import nbformat
from nbconvert import HTMLExporter

out = Path(__file__).resolve().parents[1]
for path in sorted(out.glob('0*.ipynb')):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in code_cells), f'{path.name} 尚未全部执行 / Notebook execution incomplete'
    assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
    html, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
    # 保留静态表格和内嵌图 / Keep static tables and embedded images.
    html = re.sub(r'<script\b[^>]*>.*?</script>', '', html, flags=re.S | re.I)
    html = html.replace('<html lang="en">', '<html lang="zh-CN">')
    style = '''<style>
    body {background:#fff; color:#222;}
    main {max-width:1180px; margin:0 auto;}
    .jp-RenderedHTMLCommon {font-size:15px; line-height:1.75;}
    .jp-RenderedHTMLCommon h1 {font-size:28px;}
    .jp-RenderedHTMLCommon h2 {margin-top:1.5em;}
    .jp-OutputArea-output {overflow-x:auto;}
    .jp-RenderedImage img {max-width:100%; height:auto;}
    .jp-InputArea-editor {font-size:13px;}
    @media print {.jp-InputArea {display:none;} body {font-size:11pt;}}
    </style>'''
    html = html.replace('</head>', style + '\n</head>')
    target = path.with_suffix('.html')
    target.write_text(html, encoding='utf-8')
    print(target.name, len(code_cells), 'executed code cells', len(html), 'characters')
