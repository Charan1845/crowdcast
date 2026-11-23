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

### 6. CSRNet (VGG-16 front end) over YOLO, built in TensorFlow/Keras (2026-10-21)
**Decision:** The counting model is CSRNet-style density estimation, with a VGG-16/ImageNet front end (per Decision #5) fine-tuned on ShanghaiTech, implemented in TensorFlow/Keras.
**Why:** Phase 1's real counts on Part A range 33-3,138 people per image, averaging 541 — dense enough that individual heads overlap and occlude heavily. YOLO would need a distinct box around each of ~500+ overlapping people, exactly the failure mode box-detection struggles with at that density. CSRNet's density map never needs to separate individuals, just get the local density right, which is what the point annotations (Lesson 0.2) and cached density maps (Phase 1) already produce directly. Keras ships VGG-16/ImageNet weights (`tf.keras.applications.VGG16`), so the pretrained front end is available the same way in TensorFlow as in PyTorch; the dilated-convolution back end is built directly from the CSRNet paper's spec rather than adapted from existing (mostly PyTorch) reference code.

### 7. Streamlit for the app, not a custom web app (2026-11-10)
**Decision:** The demo is a Streamlit app (`app/streamlit_app.py`): upload a photo, choose the crowd type (Dense crowd uses the Part A model, Street scene uses the Part B model), and get a count plus a density heatmap. Each result shows that model's measured test error.
**Why:** The project is about the model and the data, not front-end work. Streamlit turns the Python inference code into a usable page with very little extra code, and it deploys directly to Hugging Face Spaces or Streamlit Community Cloud. Two models are offered instead of one because each was trained on a different crowd density (Part A test MAE 79.6, Part B 13.4). Using the dense model on a sparse street scene, or the reverse, would give worse counts.

### 8. "Best time to travel" uses Bengaluru Metro data, not Melbourne or place-level estimates (2026-11-10)
**Decision:** The best-time feature is a **photo-driven metro crowd check**. The user uploads a photo taken at a **Namma Metro (Bengaluru) station**, the CSRNet model counts the people, and the app recommends the least crowded hour to travel from that station, using BMRCL's station-wise hourly ridership for Aug–Sep 2025 (obtained through RTI, published on OpenCity). It replaces the Melbourne pedestrian plan in Decision #2.
**Why:** The goal changed to India-only. No free hourly crowd data exists for Indian temples, malls or tourist spots, and published tourism statistics are yearly or monthly. Metro entries near a place would measure commuters, not visitors, so using them to advise "when to visit the temple" would be invalid. The data answers exactly one question honestly: how crowded a station is each hour. That is the feature. Hyderabad was preferred, but no public station-level hourly ridership could be found for it. If such data becomes available (for example through an RTI to HMRL), the same notebook and app page can use it.
**Models:** Same hour last week (baseline) vs average weekly pattern vs XGBoost, tested on an unseen final week. XGBoost won narrowly (MAE 54.0 vs 56.3 vs 62.7) and powers the app.
**How the two models connect:** the photo's time is read from its EXIF data (or the current time). The "people you'd see then" figure scales the photo's count by the ratio of usual entries in the suggested hour to the photo's hour. This assumes the visible crowd rises and falls with station entries, so the app labels it an estimate.

---

### 9. Remove the metro feature; focus on counting accuracy (2026-11-18)
**Decision:** The "best time to travel" feature (Decision #8) is removed from the app and the repo: the metro page, its data table and notebook 04. Git history keeps the work. The project now focuses on making the people count itself more accurate.
**Why:** Friends testing the live app found that counts on everyday photos were off. For example, 8 people in a group photo were counted as 12–18, and a team lined up for the national anthem was overcounted too. The counting model is the core of the project, so accuracy there matters more than a second feature built on top of it. Next steps: measure train vs validation vs test error and error on a real-world small-group test set; add a person detector for small groups (exact counts) next to CSRNet for dense crowds; and add a feedback loop so users can report the correct count.

---

### 10. Hybrid counting: a person detector for small groups, CSRNet for crowds (2026-11-19)
**Decision:** The app no longer asks the user for a crowd type. The Street CSRNet makes a first estimate. If it is **40 or fewer**, an **EfficientDet-D2 person detector** (TensorFlow, pretrained on COCO, confidence ≥ 0.30) counts each person and draws boxes. Otherwise the **Dense CSRNet** runs, and its estimate is used if it is **above 250**. If not, the Street CSRNet's estimate is used.
**Why:** The diagnosis (notebook 05) showed the CSRNet models are healthy on ShanghaiTech-style photos (Street: train 12.6 / validation 12.8 / test 13.4), but they overcount everyday photos of small groups. On 225 COCO photos with 1–13 people, the error was 7.2 people with +6.0 bias. That's domain shift, not overfitting: heatmaps showed heads "seen" in trees, windows and patterned objects. Training longer on ShanghaiTech can't fix that. A detector trained on everyday photos counts small groups exactly, but it misses people in dense crowds, so each method is used where it works.
**How the thresholds were chosen (validation photos only):**
- Detector confidence: 0.30 gave the lowest error on 75 COCO validation photos (MAE 1.48 vs 1.84 for EfficientDet-D0). D2 was chosen over D0 despite being slower (about 2.2 s vs 0.8 s per photo).
- Routing on detector count failed: the detector finds few people in dense crowds too.
- The Street CSRNet estimate separates the cases well. At ≤ 40, COCO validation error fell from 8.5 to 3.4 people, at a cost of about 5% on ShanghaiTech B.
- Choosing Dense on the Street estimate failed on very dense crowds, because Street undercounts them (997 counted as 153). Using the Dense model's own estimate (> 250) brought A validation error to 107, the same as the Dense model alone.
**Still to do:** report all of this once on the held-out test photos (150 COCO, ShanghaiTech A and B test).

### 11. Smaller detector (EfficientDet-Lite4) to stay on free hosting (2026-11-19)
**Decision:** Replace the full EfficientDet-D2 with **EfficientDet-Lite4** (TensorFlow Lite, confidence ≥ 0.25), and keep hosting on free Streamlit Community Cloud.
**Why:** The live app crashed with "503" on crowd photos. Measured locally, loading D2 alone took about 1.35 GB, and the app peaked at about 3.5 GB, far above the free tier's limit. Moving to Hugging Face Spaces was considered, but running apps (Docker or Gradio) there now needs a paid PRO plan, so the user chose to slim the app down instead. On the 75 COCO validation photos, Lite4 matches D2 (MAE 1.47 vs 1.48) while using about 64 MB and running about 7× faster (0.3 s vs 2.2 s per photo). With all three models loaded, the app now sits at about 630 MB (was 1.9 GB) and peaks at about 2.0 GB while counting (was 3.5 GB). Lite4 returns at most 25 objects, so a full 25 falls back to CSRNet rather than reporting a capped count. Routing on validation photos is essentially unchanged: COCO groups 3.3 people off (3.4 with D2), ShanghaiTech B 15.7 (15.0), A 107 (107).
**Second fix, same day:** Lite4 alone wasn't enough: the live app still crashed on a dense crowd. The old single-model app, which ran fine on the cloud, peaked at about 1.8 GB locally, so the hybrid had to stay below that. Running each CSRNet as a compiled graph (`tf.function`) instead of eagerly gives identical counts, with about 40% less peak memory per full-size photo (909 vs 1,524 MB) and about 7× the speed. The full hybrid now peaks at about 1.24 GB. A TensorFlow Lite version of CSRNet was also tried: it used little memory at rest, but it was slow (18–40 s per photo) and needed tiling that shifted counts slightly, so it was dropped.

### 12. Keep the hybrid after the final held-out test (2026-11-23)
**Decision:** Keep the hybrid as the app's counter, with its thresholds unchanged.
**Result (notebook 06, scored once):** everyday groups of 1–13 people (COCO, 150 photos) improved from MAE 6.52 to 3.53. Dense crowds (ShanghaiTech A) went from 79.6 to 87.0 and street crowds (B) from 13.4 to 17.0. The old figures assume the visitor always picks the right model.
**Why:** The reported failures were on everyday photos, and those improved by 46%. The crowd numbers got worse because the app now has to infer the crowd type. These test photos can't be used to re-tune the thresholds, so any later routing change needs fresh validation and a new test set.
**Correction:** earlier notes described the COCO subset as 1–20 people. The photos that passed its filters actually contain at most 13, so the "11-20" bin is really 11–13.

---

## Open decisions

- **Large groups:** how to count 40–150 people in everyday photos, which sit between what the detector and CSRNet each handle well.
