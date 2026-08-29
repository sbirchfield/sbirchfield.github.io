#!/usr/bin/env python
"""Build lesson pages: execute each notebook in notebooks/ and convert it to
HTML in lessons/, wrapped so it matches the site's look and feel.

Usage:
    python build.py               # build every notebook
    python build.py lesson31      # build only notebooks matching this substring
"""
import pathlib
import re
import sys

from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from pygments.formatters import HtmlFormatter
import nbformat

from references_data import REFERENCES

ROOT = pathlib.Path(__file__).parent
NOTEBOOKS_DIR = ROOT / "notebooks"
LESSONS_DIR = ROOT / "lessons"
CSS_DIR = ROOT / "css"

GITHUB_REPO = "sbirchfield/sbirchfield.github.io"
GITHUB_BRANCH = "main"
NOTEBOOK_REPO_PATH = "cvintro/notebooks"

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Introduction to Computer Vision</title>
    <link rel="stylesheet" href="../../css/styles.css">
    <link rel="stylesheet" href="../css/pygments.css">
    <link rel="stylesheet" href="../css/notebook.css">
    <script>
    window.MathJax = {{
        tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }}
    }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <ul class="nav-links">
                <li><a href="../index.html">Intro to Computer Vision</a></li>
                <li style="display: flex; align-items: center; color: #2c3e50;">&bull;</li>
                <li><a href="https://sbirchfield.github.io/">Stan Birchfield</a></li>
            </ul>
        </div>
    </nav>
    <div class="container notebook-container">
        <div class="notebook-links">
            <a href="https://colab.research.google.com/github/{repo}/blob/{branch}/{nb_repo_path}/{nb_name}" target="_blank" rel="noopener">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
            </a>
            <a href="https://raw.githubusercontent.com/{repo}/{branch}/{nb_repo_path}/{nb_name}" download>
                <img src="https://img.shields.io/badge/Download-.ipynb-F37626?logo=jupyter&logoColor=white" alt="Download .ipynb">
            </a>
        </div>
{body}
    </div>
</body>
</html>
"""


def build_notebook(nb_path: pathlib.Path):
    print(f"Executing {nb_path.name} ...")
    nb = nbformat.read(nb_path, as_version=4)
    ExecutePreprocessor(timeout=120, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(NOTEBOOKS_DIR)}}
    )

    exporter = HTMLExporter(template_name="basic")
    body, _ = exporter.from_notebook_node(nb)

    title = nb_path.stem.replace("_", " ").title()
    page = PAGE_TEMPLATE.format(
        title=title,
        body=body,
        repo=GITHUB_REPO,
        branch=GITHUB_BRANCH,
        nb_repo_path=NOTEBOOK_REPO_PATH,
        nb_name=nb_path.name,
    )

    out_path = LESSONS_DIR / (nb_path.stem + ".html")
    out_path.write_text(page, encoding="utf-8")
    print(f"  -> {out_path.relative_to(ROOT)}")


REFERENCES_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>References - Introduction to Computer Vision</title>
    <link rel="stylesheet" href="../css/styles.css">
    <link rel="stylesheet" href="css/notebook.css">
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <ul class="nav-links">
                <li><a href="index.html">Intro to Computer Vision</a></li>
                <li style="display: flex; align-items: center; color: #2c3e50;">&bull;</li>
                <li><a href="https://sbirchfield.github.io/">Stan Birchfield</a></li>
            </ul>
        </div>
    </nav>
    <div class="container">
        <h1>References</h1>
        <p style="text-align: right;"><span class="landmark-paper">&#9733;</span> marks especially influential, highly-cited papers.</p>
        <ul class="references-list">
{items}
        </ul>
    </div>
</body>
</html>
"""


def find_citing_lessons():
    """Map each reference id to the sorted list of (lesson_number, filename) that cite it,
    by scanning the already-built lesson pages for links back to references.html#id."""
    citing = {}
    for html_path in sorted(LESSONS_DIR.glob("lesson*.html")):
        m = re.match(r"lesson(\d+)_", html_path.stem)
        if not m:
            continue
        lesson_num = int(m.group(1))
        text = html_path.read_text(encoding="utf-8")
        for ref_id in set(re.findall(r'references\.html#([a-zA-Z0-9\-]+)"', text)):
            citing.setdefault(ref_id, []).append((lesson_num, html_path.name))
    for ref_id in citing:
        citing[ref_id].sort()
    return citing


def build_references_page():
    entries = sorted(REFERENCES, key=lambda r: (r["sort_name"].lower(), r["year"]))
    citing_lessons = find_citing_lessons()
    items = []
    for r in entries:
        star = ' <span class="landmark-paper">&#9733;</span>' if r.get("landmark") else ""
        title = r["title"]
        if r.get("url"):
            title_html = f'<a href="{r["url"]}" target="_blank" rel="noopener">{title}</a>'
        else:
            title_html = title

        lessons_html = ""
        cited_in = citing_lessons.get(r["id"], [])
        if cited_in:
            links = ", ".join(f'<a href="lessons/{fname}">{num}</a>' for num, fname in cited_in)
            lessons_html = f' ({links})'

        items.append(f'            <li id="{r["id"]}">{r["authors"]} ({r["year"]}). {title_html}.{star}{lessons_html}</li>')

    page = REFERENCES_PAGE_TEMPLATE.format(items="\n".join(items))
    out_path = ROOT / "references.html"
    out_path.write_text(page, encoding="utf-8")
    print(f"  -> {out_path.relative_to(ROOT)}")


def build_pygments_css():
    css = HtmlFormatter(style="default").get_style_defs(".highlight")
    out_path = CSS_DIR / "pygments.css"
    out_path.write_text(css, encoding="utf-8")
    print(f"  -> {out_path.relative_to(ROOT)}")


def main():
    LESSONS_DIR.mkdir(exist_ok=True)
    build_pygments_css()
    build_references_page()
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if len(sys.argv) > 1:
        needle = sys.argv[1]
        notebooks = [p for p in notebooks if needle in p.stem]
    if not notebooks:
        print("No matching notebooks found in notebooks/")
        return
    for nb_path in notebooks:
        build_notebook(nb_path)


if __name__ == "__main__":
    main()
