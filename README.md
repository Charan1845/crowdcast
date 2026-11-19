# Crowdcast

**Count the people in a crowd photo, from a small group to a packed stadium.**

**▶ Live demo: [crowdcast-charan.streamlit.app](https://crowdcast-charan.streamlit.app)**

> Every number below is a real measured result on held-out test data. The free host puts the app to sleep after a few days without visitors; if you see a "wake up" button, it takes about a minute to start.

| | Model | Result on unseen test data |
|---|---|---|
| 👥 **People counting**, street scenes | VGG-16 CSRNet (transfer learning) | **MAE 13.4** people (ShanghaiTech B; baseline 72.0, published 10.6) |
| 👥 **People counting**, dense crowds | VGG-16 CSRNet (transfer learning) | **MAE 79.6** people (ShanghaiTech A; baseline 301.8, published 68.2) |

## What it does

A Streamlit app: upload any photo, from a small group to a packed stadium. It picks the counting method by itself:

- **Small groups:** a person detector finds each person, draws a numbered box around them, and gives an **exact count**.
- **Crowds:** CSRNet estimates the count from a **density map** and shows a heatmap of where the crowd is.

> 🚧 **In progress:** a final check of the hybrid on held-out test photos, better handling of large groups, and a feedback loop. See [Progress](#progress).

## How it works

### Choosing the method automatically

```
 Photo ──► CSRNet (Street) estimate ──► ≤ 40 people? ──yes──► person detector: exact count, box per person
                                              │
                                              no ──► CSRNet (Dense) estimate > 250? ──yes──► Dense crowd density map
                                                                   │
                                                                   no ──► Street scene density map
```

- **Why a detector for small groups:** CSRNet was trained on crowds filmed from above. On everyday photos of a few people it "sees" heads in textures (trees, windows, patterned objects) and overcounts. A detector trained on everyday photos counts small groups exactly.
- **Why not the detector for crowds:** it misses most people when heads overlap. It found about 9 people in validation crowds averaging 561.
- **Detector:** EfficientDet-Lite4 (TensorFlow Lite, pretrained on COCO), keeping "person" detections with confidence ≥ 0.25. It's as accurate as the full EfficientDet-D2 on our validation photos (1.47 vs 1.48 people off) but uses about 64 MB of memory instead of 1.35 GB, which is what lets it run on free hosting. It returns at most 25 people, so a full 25 falls back to CSRNet.
- **Thresholds** (0.25, 40, 250) were tuned on **validation photos only**: COCO groups and ShanghaiTech A and B validation splits. See [DECISIONS.md](DECISIONS.md) #10 and #11.

### People counting: CSRNet with a VGG-16 front end

```
 Crowd photo ──► VGG-16 front end ──► 6 dilated conv layers ──► density map ──► sum = people count
                (pretrained on ImageNet,  (trained from scratch;
                 last block fine-tuned)    wider view, no extra shrinking)
```

- **Density map, not boxes.** Boxes around each person fail when hundreds of heads overlap. The model predicts a density map instead, and the count is its sum.
- **Labels:** each annotated head becomes a small Gaussian blob summing to 1. Dense crowds use blob widths adapted to head spacing; street scenes use a fixed width (sigma 15).
- **Transfer learning:** VGG-16's ImageNet weights are frozen first. Then only its last block is fine-tuned, at a low learning rate.
- **Overfitting controls:**
  - a held-out validation set, with only the best epoch kept
  - early stopping
  - random crop and flip augmentation
  - learning-rate halving when progress stalls
  - an L2 weight penalty
- Built in TensorFlow/Keras and trained on a laptop CPU.

Why each choice was made (CSRNet over YOLO, transfer learning, Streamlit): [DECISIONS.md](DECISIONS.md).

## Results

### People counting: ShanghaiTech Part B (street scenes, 316 test images, 9–539 people each)

| Model | Test MAE | Test RMSE |
|---|---|---|
| Baseline: always guess the average count | 72.0 | 95.1 |
| **Crowdcast (best of 13 epochs)** | **13.4** | **20.6** |
| Published CSRNet paper (for reference) | 10.6 | 16.0 |

- The test images average 124 people, so the model is off by about **11%** on average.
- Error grows with crowd size: about 3 people on the smallest crowds, about 25 on the largest.
- **Stopped by hand:** training ended after 13 epochs, with no new best for 3 epochs and an already tiny learning rate. The best epoch (10) was saved during training and is the one tested. [Notebook](notebooks/03_counting_model_partB.ipynb)

### People counting: ShanghaiTech Part A (dense crowds, 182 test images, 66–2,256 people each)

| Model | Test MAE | Test RMSE |
|---|---|---|
| Baseline: always guess the average count | 301.8 | 368.6 |
| Crowdcast v1 (best of 5 epochs) | 94.3 | 130.7 |
| **Crowdcast v2 (best of 20 epochs)** | **79.6** | **112.2** |
| Published CSRNet paper (for reference) | 68.2 | 115.0 |

- The test images average 433 people, so v2 is off by about **18%** on average.
- **v1 → v2 changes:** a lower learning rate, fine-tuning only VGG-16's last block, a larger validation set, and more epochs with learning-rate halving. v2 beats v1 at every crowd size.
- Notebooks: [v1](notebooks/01_counting_model_partA.ipynb), [v2 with the v1 vs v2 comparison](notebooks/02_counting_model_partA_v2.ipynb)

## Limitations

- **Counting models** were trained on ShanghaiTech photos. Photos with very different angles, lighting or places may count less accurately. Both models still overcount sparse scenes a little.
- **Small groups at eye level** used to be badly overcounted by CSRNet: on 225 everyday COCO photos with 1–20 people it was off by 7.2 people on average, and overcounted by 6.0 ([diagnosis](notebooks/05_accuracy_diagnosis.ipynb)). The hybrid fixes most of this (validation error 8.5 → 3.4 people). The final held-out test is still to come.
- **The detector counts people it can see.** Heavily hidden people, or groups larger than about 40, go to CSRNet instead.

## Project layout

```
notebooks/
  01_counting_model_partA.ipynb      CSRNet v1 on ShanghaiTech A
  02_counting_model_partA_v2.ipynb   improved v2 + v1 vs v2 comparison
  03_counting_model_partB.ipynb      CSRNet on ShanghaiTech B
  05_accuracy_diagnosis.ipynb        train/val/test check + everyday small-group photos (COCO)
app/
  streamlit_app.py                   entry point
  app_pages/count_crowd.py           the counting page
  crowd_model.py, shared.py          CSRNet + person detector, automatic method choice, heatmap/boxes
.streamlit/config.toml               app theme
DECISIONS.md, DATASETS.md            why each choice was made; data sources and licenses
```

## Data

| Used for | Dataset | License / credit |
|---|---|---|
| People counting | ShanghaiTech Part A/B (1,198 images with head annotations) | BSD-2-Clause (dataset repo) |

Details, file checks and quirks: [DATASETS.md](DATASETS.md). Raw datasets are not stored in this repo. The trained counting weights (65 MB each) are published as the [weights-v1 release](https://github.com/Charan1845/crowdcast/releases/tag/weights-v1).

## Run it yourself

Requires Python 3.12 (TensorFlow has no stable release for 3.14 yet).

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

1. Download ShanghaiTech into `data/shanghaitech/` (see [DATASETS.md](DATASETS.md)).
2. Run the notebooks in `notebooks/`. They save the trained counting weights to `weights/`. On CPU, TensorFlow's oneDNN option is turned off in the first cell, because it made full-image counting about 60× slower on our machine.
3. Start the app:

```bash
.venv\Scripts\streamlit run app/streamlit_app.py
```

On first start the app downloads the trained weights from the [weights-v1 release](https://github.com/Charan1845/crowdcast/releases/tag/weights-v1), so you can run it without training anything. Counting a photo takes about 2 seconds on a laptop CPU. `app/requirements.txt` lists only what the app needs (used by Streamlit Community Cloud).

## Progress

- [x] Datasets downloaded and documented
- [x] People counting on ShanghaiTech Part A (v1, then improved v2) and Part B
- [x] Streamlit app: upload a photo, get the count and a heatmap
- [x] Trained weights published as a GitHub Release; the app downloads them on first start
- [x] Live demo on Streamlit Community Cloud: [crowdcast-charan.streamlit.app](https://crowdcast-charan.streamlit.app)
- [x] Removed the metro "best time to travel" feature to focus on counting accuracy (see [DECISIONS.md](DECISIONS.md) #9)
- [x] Measured train vs validation vs test error, and error on a real-world small-group test set ([notebook 05](notebooks/05_accuracy_diagnosis.ipynb))
- [x] Hybrid counting: a person detector for small groups (exact counts with boxes), CSRNet for dense crowds, chosen automatically
- [ ] Feedback loop: users can report the correct count to help improve the model
