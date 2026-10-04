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

---

## Open decisions

- **Counting approach:** YOLO (detects each person; good for sparse scenes) vs CSRNet-style density estimation (handles dense, overlapping crowds)
- **Forecasting model:** Prophet vs LSTM (or similar)
- **App framework:** Streamlit vs a custom web app
