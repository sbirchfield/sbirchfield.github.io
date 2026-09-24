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
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Part 2 (deep learning) lessons
```

Add other CV libraries (scikit-image, etc.) as needed per lesson.

## Course structure

The lessons are organized into four parts, all listed on `index.html`:

- **Part 1: Classical 2D Image Processing** (Lessons 1-17) — pixels, filtering, convolution, frequency-domain methods, and compression, built from scratch and validated against OpenCV.
- **Part 2: Classical 3D Computer Vision** (Lessons 18-29) — color, clustering, features, optical flow, stereo, projective geometry, and structure from motion, continuing the same from-scratch/validate-then-use style.
- **Part 3: Deep Learning for Computer Vision** (Lessons 30-44) — neural networks, CNNs, and applications (classification, detection, segmentation), plus a closing lesson on the practical engineering of training at scale (numeric precision, mixed-precision training, data and model parallelism). Kept lightweight/CPU-friendly (synthetic or small datasets, e.g. CIFAR-10-scale) throughout.
- **Part 4: Transformers and Foundation Models** (Lessons 45-61) — attention, the Transformer architecture, Vision Transformers, self-supervised/contrastive learning, DINOv2, vision-language models (CLIP-style and generative VQA), Segment Anything, open-vocabulary detection, monocular/stereo depth estimation (including Depth Anything), diffusion/generative models, neural rendering (NeRF/Gaussian Splatting), and feed-forward camera pose estimation (VGGT-style). Real foundation models (SAM, DINOv2, CLIP, FoundationStereo, Depth Anything, VGGT) are far too large to train or run directly in this course, so these lessons implement each system's core *mechanism* on small synthetic tasks (same from-scratch/validate-then-use style as Parts 1-3) and discuss what changes when the same mechanism is scaled up to a real foundation model. Some of these lessons (e.g. Lesson 57) report genuine negative/null results from the toy-scale experiment where that's what the validated numbers actually show, rather than overclaiming success.

## Interactive lessons

For lessons where students should edit/run code live in the browser, consider
[JupyterLite](https://jupyterlite.readthedocs.io/) instead of the static
nbconvert pipeline above — it runs real Python via Pyodide entirely
client-side, so it also works on GitHub Pages with no backend.
