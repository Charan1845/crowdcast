# Crowdcast

**Count the people in a crowd photo, and find the best time to travel from a Namma Metro (Bengaluru) station.**

**▶ Live demo: [crowdcast-charan.streamlit.app](https://crowdcast-charan.streamlit.app)**

> Every number below is a real measured result on held-out test data. The free host puts the app to sleep after a few days without visitors; if you see a "wake up" button, it takes about a minute to start.

| | Model | Result on unseen test data |
|---|---|---|
| 👥 **People counting**, street scenes | VGG-16 CSRNet (transfer learning) | **MAE 13.4** people (ShanghaiTech B; baseline 72.0, published 10.6) |
| 👥 **People counting**, dense crowds | VGG-16 CSRNet (transfer learning) | **MAE 79.6** people (ShanghaiTech A; baseline 301.8, published 68.2) |
| 🚇 **Hourly metro crowds**, 83 stations | XGBoost | **MAE 54.0** entries/hour, **14% better** than "same hour last week" |

## What it does

A two-page Streamlit app:

1. **Count a crowd:** upload any crowd photo and get the number of people plus a heatmap of where they are. Choose *Street scene* or *Dense crowd* to pick the matching model.
2. **Metro crowd check:** upload a photo taken at a Namma Metro station:

```
 Photo at a metro station ──► CSRNet counts the people ─────────────────┐
                                                                          ├──► "91 people now · 6–7 pm is Very busy ·
 Station + time (read from the photo, ──► XGBoost hourly crowd model ───┘      go at 9–10 pm: ~74% fewer people,
 or the current time)                     (BMRCL ridership, 83 stations)         about 23 in a photo like this"
```

The "people you'd see then" figure scales your photo's count by how the suggested hour compares with the photo's hour at that station. It's labelled as an estimate (see [Limitations](#limitations)).

## How it works

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

### Hourly metro crowds: XGBoost

- **Data:** real BMRCL hourly **station entries** for 83 Namma Metro stations, 1–18 Aug and 1–30 Sep 2025. That's 92,280 station-hours.
- **Features:** station, day of week, hour, weekend, holiday. **Target:** people entering per hour.
- **Training:** gradient-boosted trees with absolute-error loss, with the number of trees (2,370) chosen by early stopping on the last training week.
- **Output:** the usual number of entries for every station × weekday × hour, saved as a small table ([app/data/metro_hourly_pattern.csv](app/data/metro_hourly_pattern.csv)) that the app looks up instantly.

Why each choice was made (CSRNet over YOLO, transfer learning, Streamlit, Bengaluru Metro data): [DECISIONS.md](DECISIONS.md).

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

### Hourly metro crowds: Namma Metro, test week 22–30 Sep 2025 (never seen in training)

Error is in people entering per station-hour, 6 am to 11 pm. An average station-hour has 516 people entering.

| Model | Test MAE | vs baseline | Picks the quietest hour (9 am–9 pm) | Picks one of the 3 quietest |
|---|---|---|---|---|
| Baseline: same hour last week | 62.7 | – | 74% | 93% |
| Average weekly pattern | 56.3 | −10% | 77% | 95% |
| **XGBoost** | **54.0** | **−14%** | **79%** | 94% |

- **Entries, checked:** August's file only says "ridership". Its hourly pattern matches September's *entries* with a correlation of 0.997, and doesn't match exits, so both months are treated as entries.
- **What didn't work first:** XGBoost trained on log counts scored 68, worse than the baseline, because the log scale under-weights busy hours. Raw counts with absolute-error loss fixed it.
- **XGBoost's edge over a plain weekly average is small** (54.0 vs 56.3). With 48 days of data, the average is already strong. [Notebook](notebooks/04_metro_best_time.ipynb)

## Limitations

- **Counting models** were trained on ShanghaiTech photos. Photos with very different angles, lighting or places may count less accurately. Both models still overcount sparse scenes a little.
- **Metro patterns** cover one city over two months of **normal days**. Festivals, events, strikes and heavy rain aren't in the data, and the 15 stations that opened in August 2025 have less history.
- **"People you'd see then"** assumes the crowd in your camera's view rises and falls with station entries. It's a sensible estimate, but it hasn't been tested against real platform photos yet.
- **Real metro photos** haven't been tested yet. The metro page has only been tried with street-scene test photos so far.

## Project layout

```
notebooks/
  01_counting_model_partA.ipynb      CSRNet v1 on ShanghaiTech A
  02_counting_model_partA_v2.ipynb   improved v2 + v1 vs v2 comparison
  03_counting_model_partB.ipynb      CSRNet on ShanghaiTech B
  04_metro_best_time.ipynb           Namma Metro hourly crowds: baseline vs weekly average vs XGBoost
app/
  streamlit_app.py                   entry point and page navigation
  app_pages/count_crowd.py           page 1: count a crowd
  app_pages/metro_best_time.py       page 2: metro crowd check
  crowd_model.py, shared.py          model loading, counting, heatmap, photo time
  data/metro_hourly_pattern.csv      XGBoost output used by page 2
.streamlit/config.toml               app theme
DECISIONS.md, DATASETS.md            why each choice was made; data sources and licenses
```

## Data

| Used for | Dataset | License / credit |
|---|---|---|
| People counting | ShanghaiTech Part A/B (1,198 images with head annotations) | BSD-2-Clause (dataset repo) |
| Hourly metro crowds | BMRCL station-wise hourly ridership, Aug–Sep 2025 | Obtained through RTI by Vivek Mathew, published on [OpenCity](https://data.opencity.in/dataset/bmrcl-station-wise-ridership-data); Creative Commons Non-Commercial |

Details, file checks and quirks: [DATASETS.md](DATASETS.md). Raw datasets are not stored in this repo. The trained counting weights (65 MB each) are published as the [weights-v1 release](https://github.com/Charan1845/crowdcast/releases/tag/weights-v1).

## Run it yourself

Requires Python 3.12 (TensorFlow has no stable release for 3.14 yet).

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

1. Download ShanghaiTech into `data/shanghaitech/` and the BMRCL files into `data/bmrcl/` (see [DATASETS.md](DATASETS.md)).
2. Run the notebooks in `notebooks/`. They save the trained counting weights to `weights/`. The metro table is already in the repo, but notebook 04 rebuilds it. On CPU, TensorFlow's oneDNN option is turned off in the first cell, because it made full-image counting about 60× slower on our machine.
3. Start the app:

```bash
.venv\Scripts\streamlit run app/streamlit_app.py
```

On first start the app downloads the trained weights from the [weights-v1 release](https://github.com/Charan1845/crowdcast/releases/tag/weights-v1), so you can run it without training anything. Counting a photo takes about 2 seconds on a laptop CPU. `app/requirements.txt` lists only what the app needs (used by Streamlit Community Cloud).

## Progress

- [x] Datasets downloaded and documented
- [x] People counting on ShanghaiTech Part A (v1, then improved v2) and Part B
- [x] Hourly metro crowd model: baseline vs weekly average vs XGBoost
- [x] Streamlit app: count a crowd; metro crowd check (photo → count → best time to travel)
- [x] Trained weights published as a GitHub Release; the app downloads them on first start
- [x] Live demo on Streamlit Community Cloud: [crowdcast-charan.streamlit.app](https://crowdcast-charan.streamlit.app)
- [ ] Test the metro page with real Namma Metro photos
- [ ] More cities when hourly data becomes available (e.g. Hyderabad Metro via RTI)
