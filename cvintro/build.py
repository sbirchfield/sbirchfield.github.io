#!/usr/bin/env python
"""Build lesson pages: execute each notebook in notebooks/ and convert it to
HTML in lessons/, wrapped so it matches the site's look and feel.

Usage:
    python build.py               # build every notebook, then all quiz pages
    python build.py lesson31      # build only notebooks matching this substring, then all quiz pages
    python build.py --quizzes-only  # skip notebook execution; just regenerate quizzes/*.html
"""
import pathlib
import re
import sys

from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor, TagRemovePreprocessor
from pygments.formatters import HtmlFormatter
import nbformat

from references_data import REFERENCES
from quizzes_data import QUIZZES

ROOT = pathlib.Path(__file__).parent
NOTEBOOKS_DIR = ROOT / "notebooks"
LESSONS_DIR = ROOT / "lessons"
QUIZZES_DIR = ROOT / "quizzes"
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
{quiz_button}
        <div class="lesson-nav">
            <div class="lesson-nav-prev">{prev_link}</div>
            <div class="lesson-nav-next">{next_link}</div>
        </div>
    </div>
</body>
</html>
"""

QUIZ_BUTTON_TEMPLATE = """        <div class="quiz-cta">
            <a href="../quizzes/quiz{lesson_num:02d}.html" class="quiz-button">Take the Lesson {lesson_num} Quiz &rarr;</a>
        </div>"""


def lesson_title(nb_path: pathlib.Path) -> str:
    """The 'Lesson N: ...' heading from the notebook's title cell. Some lessons (Part openers)
    have an extra '# Part N: ...' heading line first, so search for the 'Lesson' line specifically
    rather than assuming it's the first line."""
    nb = nbformat.read(nb_path, as_version=4)
    lines = nb.cells[0].source.split("\n")
    for line in lines:
        if re.match(r"#\s*Lesson\s+\d+", line):
            return line.lstrip("#").strip()
    return lines[0].lstrip("#").strip()


def get_lesson_sequence():
    """All lesson notebooks in course order, as a list of pathlib.Path."""
    return sorted(NOTEBOOKS_DIR.glob("lesson*.ipynb"))


def build_notebook(nb_path: pathlib.Path, sequence=None):
    print(f"Executing {nb_path.name} ...")
    nb = nbformat.read(nb_path, as_version=4)
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(NOTEBOOKS_DIR)}}
    )

    exporter = HTMLExporter(template_name="basic")
    exporter.register_preprocessor(TagRemovePreprocessor(remove_cell_tags={"remove-cell"}), enabled=True)
    body, _ = exporter.from_notebook_node(nb)

    sequence = sequence if sequence is not None else get_lesson_sequence()
    idx = sequence.index(nb_path)
    prev_link = ""
    if idx > 0:
        prev_path = sequence[idx - 1]
        prev_link = f'<a href="{prev_path.stem}.html">&larr; {lesson_title(prev_path)}</a>'
    next_link = ""
    if idx < len(sequence) - 1:
        next_path = sequence[idx + 1]
        next_link = f'<a href="{next_path.stem}.html">{lesson_title(next_path)} &rarr;</a>'

    quiz_button = ""
    m = re.match(r"lesson(\d+)_", nb_path.stem)
    if m and int(m.group(1)) in QUIZZES:
        quiz_button = QUIZ_BUTTON_TEMPLATE.format(lesson_num=int(m.group(1)))

    title = nb_path.stem.replace("_", " ").title()
    page = PAGE_TEMPLATE.format(
        title=title,
        body=body,
        quiz_button=quiz_button,
        repo=GITHUB_REPO,
        branch=GITHUB_BRANCH,
        nb_repo_path=NOTEBOOK_REPO_PATH,
        nb_name=nb_path.name,
        prev_link=prev_link,
        next_link=next_link,
    )

    out_path = LESSONS_DIR / (nb_path.stem + ".html")
    out_path.write_text(page, encoding="utf-8")
    print(f"  -> {out_path.relative_to(ROOT)}")


QUIZ_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lesson {lesson_num} Quiz - Introduction to Computer Vision</title>
    <link rel="stylesheet" href="../../css/styles.css">
    <link rel="stylesheet" href="../css/notebook.css">
    <link rel="stylesheet" href="../css/quiz.css">
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
        <p class="quiz-breadcrumb"><a href="{lesson_href}">&larr; Back to Lesson {lesson_num}</a></p>
        <h1>{quiz_heading}</h1>
        <p class="quiz-score" id="quiz-score">Score: 0 / {num_questions}</p>
        <div id="quiz-questions">
{questions_html}
        </div>
        <div class="lesson-nav">
            <div class="lesson-nav-prev">{prev_quiz_link}</div>
            <div class="lesson-nav-next">{next_quiz_link}</div>
        </div>
    </div>
    <script>
    let score = 0;
    const answered = new Set();

    function shuffleChildren(container) {{
        const children = Array.from(container.children);
        for (let i = children.length - 1; i > 0; i--) {{
            const j = Math.floor(Math.random() * (i + 1));
            [children[i], children[j]] = [children[j], children[i]];
        }}
        children.forEach(child => container.appendChild(child));
        return children;
    }}

    function shuffleQuiz() {{
        const container = document.getElementById('quiz-questions');
        const questions = shuffleChildren(container);
        questions.forEach((q, i) => {{
            q.querySelector('.quiz-q-number').textContent = i + 1;
            shuffleChildren(q.querySelector('.quiz-choices'));
        }});
    }}
    shuffleQuiz();

    function checkAnswer(qIndex, choiceIndex, correctIndex) {{
        if (answered.has(qIndex)) return;
        answered.add(qIndex);

        const buttons = document.querySelectorAll(`[data-q="${{qIndex}}"] .quiz-choice`);
        buttons.forEach(btn => {{
            btn.disabled = true;
            const c = parseInt(btn.dataset.c, 10);
            if (c === correctIndex) btn.classList.add('quiz-correct');
            else if (c === choiceIndex) btn.classList.add('quiz-incorrect');
        }});

        document.querySelector(`[data-q="${{qIndex}}"] .quiz-explanation`).style.display = 'block';

        if (choiceIndex === correctIndex) score += 1;
        document.getElementById('quiz-score').textContent = `Score: ${{score}} / {num_questions}`;
    }}
    </script>
</body>
</html>
"""

QUIZ_QUESTION_TEMPLATE = """            <div class="quiz-question" data-q="{q_index}">
                <p class="quiz-question-text"><span class="quiz-q-number"></span>. {question}</p>
                <div class="quiz-choices">
{choices_html}
                </div>
                <p class="quiz-explanation"><strong>Explanation:</strong> {explanation}</p>
            </div>"""

QUIZ_CHOICE_TEMPLATE = """                    <button class="quiz-choice" data-c="{c_index}" onclick="checkAnswer({q_index}, {c_index}, {correct_index})">{choice}</button>"""


def build_quiz_page(lesson_num: int, questions: list, sequence, quiz_lesson_nums: list):
    """Generate quizzes/quizNN.html for one lesson's question bank."""
    nb_path = next(p for p in sequence if p.stem.startswith(f"lesson{lesson_num:02d}_"))
    title = lesson_title(nb_path)
    # "Lesson N: Subtitle" -> "Quiz N: Subtitle"
    subtitle = title.split(":", 1)[1].strip() if ":" in title else title
    quiz_heading = f"Quiz {lesson_num}: {subtitle}"

    questions_html = []
    for q_index, q in enumerate(questions):
        choices_html = "\n".join(
            QUIZ_CHOICE_TEMPLATE.format(q_index=q_index, c_index=c_index, correct_index=q["correct"], choice=choice)
            for c_index, choice in enumerate(q["choices"])
        )
        questions_html.append(QUIZ_QUESTION_TEMPLATE.format(
            q_index=q_index,
            question=q["question"],
            choices_html=choices_html,
            explanation=q["explanation"],
        ))

    idx = quiz_lesson_nums.index(lesson_num)
    prev_quiz_link = ""
    if idx > 0:
        prev_num = quiz_lesson_nums[idx - 1]
        prev_quiz_link = f'<a href="quiz{prev_num:02d}.html">&larr; Lesson {prev_num} Quiz</a>'
    next_quiz_link = ""
    if idx < len(quiz_lesson_nums) - 1:
        next_num = quiz_lesson_nums[idx + 1]
        next_quiz_link = f'<a href="quiz{next_num:02d}.html">Lesson {next_num} Quiz &rarr;</a>'

    page = QUIZ_PAGE_TEMPLATE.format(
        lesson_num=lesson_num,
        lesson_title=title,
        quiz_heading=quiz_heading,
        lesson_href=f"../lessons/{nb_path.stem}.html",
        num_questions=len(questions),
        questions_html="\n".join(questions_html),
        prev_quiz_link=prev_quiz_link,
        next_quiz_link=next_quiz_link,
    )
    out_path = QUIZZES_DIR / f"quiz{lesson_num:02d}.html"
    out_path.write_text(page, encoding="utf-8")
    print(f"  -> {out_path.relative_to(ROOT)}")


def build_quizzes():
    QUIZZES_DIR.mkdir(exist_ok=True)
    sequence = get_lesson_sequence()
    quiz_lesson_nums = sorted(QUIZZES)
    for lesson_num, questions in QUIZZES.items():
        build_quiz_page(lesson_num, questions, sequence, quiz_lesson_nums)


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
        <p style="text-align: right;"><span class="landmark-paper">&#9733;</span> marks especially influential, <a href="https://www.papelist.app/" target="_blank" rel="noopener">highly-cited papers</a>.</p>
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
    args = sys.argv[1:]
    if "--quizzes-only" in args:
        build_quizzes()
        return

    LESSONS_DIR.mkdir(exist_ok=True)
    build_pygments_css()
    build_references_page()
    sequence = get_lesson_sequence()
    notebooks = sequence
    if args:
        needle = args[0]
        notebooks = [p for p in notebooks if needle in p.stem]
    if not notebooks:
        print("No matching notebooks found in notebooks/")
        return
    for nb_path in notebooks:
        build_notebook(nb_path, sequence=sequence)
    build_quizzes()


if __name__ == "__main__":
    main()
