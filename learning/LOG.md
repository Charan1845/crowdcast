# Learning Log

Crowdcast is being built as a learning project: each entry is a concept learned, with a note on how it applies to this project. Applied code (scripts that actually touch project data) lands in a separate implementation session and its own commit.

---

### Lesson 0.1 — Images as arrays, NumPy basics (2026-10-07)

**Concept:** An image is a grid of numbers, nothing more.
- Grayscale image → 2D array, shape `(height, width)`, one number (0–255) per pixel.
- Color image → 3D array, shape `(height, width, 3)` for the R/G/B channels.
- Real images are stored as `uint8` (0–255, 1 byte per value) — that's *why* 255 is max brightness.
- NumPy indexing is `[row, col]` i.e. `[y, x]`, not `(x, y)`.
- Vectorized ops (`img + 5`) apply to every pixel at once — no Python loop. This is the same idea every CV/DL framework (PyTorch, TensorFlow) is built on.

**Why it matters for Crowdcast:** whichever counting approach gets picked (YOLO boxes vs CSRNet density maps), the model's output is itself just another array the same height/width as the input — "image = array" is the one idea underneath both.

**Applied step (2026-10-08):** downloaded one real ShanghaiTech Part A image (`IMG_1.jpg`, via Kaggle mirror) and inspected it — see [lesson_01_image_as_array.py](lesson_01_image_as_array.py).

Real results:
```
PIL size (width, height): (1024, 704)
NumPy shape (height, width, channels): (704, 1024, 3)
dtype: uint8
min/max pixel value: 0 / 255
mean brightness: 118.46
```
Confirms the concept exactly: PIL reports (width, height), NumPy flips it to (height, width, channels) — the gotcha called out above, now seen on real data, not just a toy example.

**Status:** done — concept covered and applied on real data.

---

### Lesson 0.2 — Ground-truth `.mat` files, head-point annotations (2026-10-08)

**Concept:** ShanghaiTech's labels are not boxes, just points.
- `.mat` is MATLAB's binary format; `scipy.io.loadmat` reads it into a Python dict.
- Each `GT_IMG_*.mat` holds a nested struct with two fields: `location` (an N×2 array of `(x, y)` head-pixel coordinates) and `number` (the stated count — a cross-check, not extra information).
- The ground-truth count *is* `len(location)`. There's no separate labeling of "how many people" — it falls out of counting the points.
- This is why density-map methods (CSRNet) fit this data natively, and why box-based detectors (YOLO) need an extra conversion step — the raw annotation has no box information to use directly. This is the concrete reasoning behind the open "Counting approach" decision in [DECISIONS.md](DECISIONS.md).

**Why it matters for Crowdcast:** before any model can be trained, these sparse points need to become either synthetic boxes or a density map (each point blurred into a small Gaussian blob). Understanding the raw format is the prerequisite for that conversion step, which is next.

**Applied step (2026-10-08):** loaded `GT_IMG_1.mat` (paired with yesterday's `IMG_1.jpg`), extracted the 172 head points, verified `len(points) == stated count` (172 == 172), and plotted the points over the image — see [lesson_02_head_point_annotations.py](lesson_02_head_point_annotations.py).

**Status:** done — concept covered and applied on real data (172 head points confirmed two ways, overlay generated).
