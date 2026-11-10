# Crowdcast

**Count the people in a crowd photo, and (next) recommend the best time to visit a place.**

> Status: 🚧 In progress. Results below are only real measured numbers, updated as the project is built.

## What it does

1. **Count** (working): give it a crowd photo and it estimates how many people are in it, with a density heatmap showing where the crowd is packed.
2. **Best time to visit** (planned): using years of real hourly pedestrian counts, show how busy a type of place usually is by hour and day, and suggest the quietest times.

## How the counting model works

A **CSRNet**-style density-estimation model, built in TensorFlow/Keras with **transfer learning**:

```
 Crowd photo ──► VGG-16 front end ──► 6 dilated conv layers ──► density map ──► sum = people count
                (pretrained on ImageNet,  (trained from scratch;
                 last block fine-tuned)    wider view, no extra shrinking)
```

- Instead of drawing a box around each person (which fails when hundreds of heads overlap), the model predicts a **density map**, and the people count is its sum.
- Training labels come from head-position annotations, each turned into a small Gaussian blob that sums to 1 (geometry-adaptive kernel).
- **Overfitting controls:**
  - a held-out validation set
  - keep only the best epoch
  - early stopping
  - random crop/flip augmentation
  - VGG-16 frozen at first, then partly fine-tuned
  - learning-rate halving when progress stalls
  - L2 weight penalty

Why CSRNet over YOLO, and why transfer learning: see [DECISIONS.md](DECISIONS.md) (#5, #6).

## Results

**ShanghaiTech Part A** (dense crowds, 182 test images, 66–2,256 people each). Lower is better.

| Model | Test MAE | Test RMSE |
|---|---|---|
| Baseline: always guess the average count | 301.8 | 368.6 |
| Crowdcast v1 (best of 5 epochs) | 94.3 | 130.7 |
| **Crowdcast v2 (best of 20 epochs)** | **79.6** | **112.2** |
| Published CSRNet paper (for reference) | 68.2 | 115.0 |

- **MAE** is the average number of people the count is off by. The test images average 433 people, so v2 is off by about 18% on average.
- **v1 → v2 changes:**
  - lower learning rate
  - fine-tuning only VGG-16's last block
  - larger validation set
  - more epochs with learning-rate halving

  v2 is better than v1 at every crowd size.
- **Known weakness:** v2 still overcounts the sparsest images (about +15 people) and undercounts the densest (about −40). The overall bias is near zero only because these cancel out.
- **Training setup:** trained on a laptop CPU (no GPU). The published model trained for about 400 epochs; ours trained for 20.

Notebooks with every cell's output:
- [01_counting_model_partA.ipynb](notebooks/01_counting_model_partA.ipynb) (v1)
- [02_counting_model_partA_v2.ipynb](notebooks/02_counting_model_partA_v2.ipynb) (v2, including the v1 vs v2 comparison)

**ShanghaiTech Part B** (sparse street scenes, 316 test images, 9–539 people each). Lower is better.

| Model | Test MAE | Test RMSE |
|---|---|---|
| Baseline: always guess the average count | 72.0 | 95.1 |
| **Crowdcast (best of 13 epochs)** | **13.4** | **20.6** |
| Published CSRNet paper (for reference) | 10.6 | 16.0 |

- **How it was trained:** the Part A v2 recipe, except the density maps use a fixed Gaussian width (sigma 15), as in the CSRNet paper, because the crowds are sparse.
- **Accuracy:** the test images average 124 people, so the model is off by about 11% on average. Error grows with crowd size, from about 3 people on the smallest crowds (9–55) to about 25 on the largest (167–539).
- **Known weakness:** like Part A, it tends to overcount sparse scenes (overall bias +4.6 people).
- **Training was stopped by hand after 13 epochs:** there was no new best for 3 epochs and the learning rate was already very small. The best epoch (10) was saved during training and is the model tested here. The notebook says so and shows the full per-epoch history.

Notebook: [03_counting_model_partB.ipynb](notebooks/03_counting_model_partB.ipynb)

| Still to measure | Baseline | Crowdcast |
|---|---|---|
| Best-time forecast error vs "same time last week" | – | – |

## Progress

- [x] Datasets downloaded and documented ([DATASETS.md](DATASETS.md))
- [x] Counting model on ShanghaiTech Part A (v1, then improved v2)
- [x] Counting model on ShanghaiTech Part B
- [ ] Best time to visit: hourly patterns by type of place, baseline vs forecasting model
- [ ] Web app: upload a photo → count + heatmap; best-time tab
- [ ] Live demo (Hugging Face Spaces / Streamlit Cloud)

## Data

| Part | Dataset | Why |
|---|---|---|
| Counting | ShanghaiTech Part A/B | Standard benchmark: dense (A) and sparse street (B) scenes |
| Counting (optional) | JHU-Crowd++ / UCF-QNRF | Very dense crowds, varied weather and lighting |
| Best time to visit | Melbourne Pedestrian Counting System | Years of real hourly sensor counts (CC BY 4.0) |

Full details (sizes, label formats, licenses, download links): [DATASETS.md](DATASETS.md). Datasets and trained weights are not stored in this repo.

## Run it yourself

Requires Python 3.12 (TensorFlow has no stable release for 3.14 yet).

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Download ShanghaiTech into `data/shanghaitech/` (see [DATASETS.md](DATASETS.md)), then run the notebooks in `notebooks/`. On CPU, TensorFlow's oneDNN option is turned off in the first cell, because it made full-image counting about 60× slower on our machine.

## Design decisions

See [DECISIONS.md](DECISIONS.md) for every major decision and the reasoning behind it.
