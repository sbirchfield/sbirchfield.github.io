#!/usr/bin/env python
"""Build lesson pages: execute each notebook in notebooks/ and convert it to
HTML in lessons/, wrapped so it matches the site's look and feel.

Usage:
    python build.py
"""
import pathlib

from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from pygments.formatters import HtmlFormatter
import nbformat

ROOT = pathlib.Path(__file__).parent
NOTEBOOKS_DIR = ROOT / "notebooks"
LESSONS_DIR = ROOT / "lessons"
CSS_DIR = ROOT / "css"

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
                <li><a href="../index.html">All Lessons</a></li>
                <li><a href="../../index.html#teaching">Back to Site</a></li>
            </ul>
        </div>
    </nav>
    <div class="container notebook-container">
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
    page = PAGE_TEMPLATE.format(title=title, body=body)

    out_path = LESSONS_DIR / (nb_path.stem + ".html")
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
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found in notebooks/")
        return
    for nb_path in notebooks:
        build_notebook(nb_path)


if __name__ == "__main__":
    main()
