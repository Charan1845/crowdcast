# Decision Log

Every major decision for Crowdcast, with the reasoning. Newest decisions at the bottom.

---

### 1. No live camera collection pipeline (2026-10-04)
**Decision:** Train the counting model on existing public image and video datasets, then test it on real images and videos uploaded through the app.
**Why:** A live-camera pipeline would need weeks of collection before anything could be evaluated. Public datasets let training start immediately and come with ground-truth labels, which gives real, comparable error numbers.

### 2. Keep forecasting, using public pedestrian-count data (2026-10-04)
**Decision:** Forecast crowd levels using the Melbourne Pedestrian Counting System (hourly sensor counts).
**Why:** Forecasting needs counts over time, which uploaded images and videos cannot provide. This dataset has years of real history, including holidays and events, which is far more than a few weeks of self-collected data. Any claims about holiday or event effects will be made only if the data supports them.

### 3. Demo as a Streamlit or web app (2026-10-04)
**Decision:** Users interact through a web app: upload an image or video to get a count and heatmap, and view forecasts.
**Why:** A live link that anyone can open in seconds is the most convincing proof that the system works.

### 4. Evaluate against simple baselines (2026-10-04)
**Decision:** Report counting MAE on standard benchmark test sets, and forecast error compared to a "same time last week" baseline.
**Why:** Numbers mean nothing without a reference point. Beating a simple baseline is an honest, explainable result.

### 5. Transfer learning for counting, with a from-scratch comparison (2026-10-17)
**Decision:** The counting model starts from pretrained weights (VGG-16 on ImageNet as CSRNet's front end; COCO-pretrained weights for YOLO) and is fine-tuned on ShanghaiTech. A small CNN trained from scratch is built once as a learning exercise and comparison point. Forecasting does not use transfer learning (baseline → Prophet → LSTM trained on the Melbourne data).
**Why:** ShanghaiTech has only 300 (Part A) / 400 (Part B) training images, which is too few to train a deep network from zero without overfitting. Pretrained models already recognise edges, shapes and people, so fine-tuning is faster and more accurate. Comparing against a from-scratch model measures how much transfer learning actually helps, instead of just claiming it. Pretrained image models don't apply to tabular time series, and years of Melbourne data are enough to train forecasting models directly.

---

## Open decisions

- **Counting approach:** YOLO (detects each person; good for sparse scenes) vs CSRNet-style density estimation (handles dense, overlapping crowds), either way starting from pretrained weights (see #5)
- **Forecasting model:** Prophet vs LSTM (or similar)
- **App framework:** Streamlit vs a custom web app
