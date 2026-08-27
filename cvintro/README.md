# cvintro — Computer Vision lecture notes

Workflow for authoring lessons with working Python code, published as static
pages on GitHub Pages (no server required).

## Layout

- `notebooks/` — source Jupyter notebooks (`.ipynb`), one per lesson. Edit these.
- `lessons/` — generated HTML pages (output of `build.py`). Do not hand-edit.
- `css/notebook.css` — styling layered on top of `../css/styles.css` to match the main site.
- `index.html` — lesson listing page.
- `build.py` — executes each notebook and converts it to a standalone HTML page.

## Adding a lesson

1. Create a new notebook in `notebooks/`, e.g. `notebooks/lesson02_filtering.ipynb`.
2. Write markdown + code cells as usual (open with `jupyter lab notebooks/`).
3. Run the build:

   ```
   python build.py
   ```

   This executes every notebook fresh (so output is always reproducible) and
   writes a matching page to `lessons/`. Plots are embedded as base64 so the
   page is self-contained — no extra image files to commit.
4. Add a link to the new lesson in `index.html`.
5. Commit `notebooks/*.ipynb` and `lessons/*.html` together.

## Setup

```
pip install nbconvert nbformat numpy matplotlib ipykernel opencv-python-headless pywavelets
```

Add other CV libraries (scikit-image, etc.) as needed per lesson.

## Interactive lessons

For lessons where students should edit/run code live in the browser, consider
[JupyterLite](https://jupyterlite.readthedocs.io/) instead of the static
nbconvert pipeline above — it runs real Python via Pyodide entirely
client-side, so it also works on GitHub Pages with no backend.
