"""
Convert Discrete_Mathematics_Project.md → Discrete_Mathematics_Project.pdf
Pipeline: Python-Markdown → styled HTML → Microsoft Edge headless → PDF
"""

import markdown
import subprocess
import sys
import tempfile
from pathlib import Path

MD_FILE   = Path(__file__).parent / "Discrete_Mathematics_Project.md"
PDF_FILE  = Path(__file__).parent / "Discrete_Mathematics_Project.pdf"
EDGE_EXE  = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
/* ── Page layout ── */
@page {
    size: A4;
    margin: 2.4cm 2.7cm 2.4cm 2.7cm;
}

/* ── Base typography ── */
body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11.5pt;
    line-height: 1.65;
    color: #1a1a1a;
    text-align: justify;
    max-width: 100%;
}

/* ── Title (h1) ── */
h1 {
    font-size: 16.5pt;
    font-weight: bold;
    text-align: center;
    margin-top: 0;
    margin-bottom: 6pt;
    line-height: 1.3;
    color: #0d0d0d;
    border-bottom: 2px solid #1a1a1a;
    padding-bottom: 8pt;
}

/* ── Cover metadata block (first <p> after h1) ── */
h1 + p {
    text-align: center;
    font-size: 10.5pt;
    color: #444;
    margin-top: 4pt;
    margin-bottom: 24pt;
    line-height: 1.8;
}

/* ── Section headings ── */
h2 {
    font-size: 13pt;
    font-weight: bold;
    margin-top: 22pt;
    margin-bottom: 5pt;
    color: #0d0d0d;
    border-bottom: 1px solid #ccc;
    padding-bottom: 3pt;
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 14pt;
    margin-bottom: 4pt;
    color: #1a1a1a;
    page-break-after: avoid;
}

h4 {
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 12pt;
    margin-bottom: 3pt;
    color: #1a1a1a;
    page-break-after: avoid;
}

/* ── Paragraphs ── */
p {
    margin: 0 0 7pt 0;
    orphans: 3;
    widows: 3;
}

/* ── Horizontal rules ── */
hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 12pt 0;
}

/* ── Blockquotes (displayed math / formal expressions) ── */
blockquote {
    margin: 8pt 0 8pt 1.8em;
    padding: 5pt 5pt 5pt 12pt;
    border-left: 3px solid #888;
    background: #f8f8f8;
    font-family: 'Courier New', Courier, monospace;
    font-size: 10pt;
    line-height: 1.5;
    color: #222;
}
blockquote p { margin: 2pt 0; }

/* ── Pseudocode blocks ── */
pre {
    font-family: 'Courier New', Courier, monospace;
    font-size: 9.3pt;
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-left: 4px solid #444;
    padding: 9pt 11pt;
    margin: 10pt 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.4;
    page-break-inside: avoid;
}

code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 9.8pt;
    background: #efefef;
    padding: 1pt 3pt;
    border-radius: 2pt;
}

pre code {
    background: none;
    padding: 0;
    font-size: 9.3pt;
}

/* ── Tables ── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 11pt 0;
    font-size: 10pt;
    page-break-inside: avoid;
}

thead tr {
    background-color: #2b2b2b;
    color: #fff;
}

thead th {
    padding: 6pt 8pt;
    text-align: left;
    font-weight: bold;
}

tbody tr:nth-child(even)  { background: #f2f2f2; }
tbody tr:nth-child(odd)   { background: #ffffff; }

td {
    padding: 5pt 8pt;
    border-bottom: 1px solid #ddd;
    vertical-align: top;
}

/* ── Lists ── */
ul, ol {
    margin: 4pt 0 7pt 1.5em;
    padding: 0;
}
li { margin-bottom: 3pt; line-height: 1.55; }

/* ── Inline emphasis ── */
strong { font-weight: bold; color: #0d0d0d; }
em     { font-style: italic; }

/* ── Page numbers via CSS counter ── */
@page { @bottom-center { content: counter(page); font-size: 10pt; color: #666; } }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Discrete Mathematics Project</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>"""

def build_html():
    raw = MD_FILE.read_text(encoding="utf-8")
    md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "toc"])
    body = md.convert(raw)
    return HTML_TEMPLATE.format(css=CSS, body=body)

def render_pdf(html_content):
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(html_content)
        tmp_html = Path(f.name)

    try:
        cmd = [
            EDGE_EXE,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=5000",
            f"--print-to-pdf={PDF_FILE}",
            "--print-to-pdf-no-header",
            tmp_html.as_uri(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print("Edge stderr:", result.stderr[:500])
            sys.exit(1)
    finally:
        tmp_html.unlink(missing_ok=True)

if __name__ == "__main__":
    print("Building HTML ...")
    html = build_html()
    print("Rendering PDF via Edge headless ...")
    render_pdf(html)
    size_kb = PDF_FILE.stat().st_size // 1024
    print(f"Done -> {PDF_FILE}  ({size_kb} KB)")
