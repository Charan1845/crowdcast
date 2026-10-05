# Datasets

Research notes on the datasets Crowdcast plans to use: where to get them, what's inside, how they're labeled, and what we're allowed to do with them.

> Checked on 2026-10-05. Anything marked **(to verify)** could not be confirmed from an official source yet. Check it before relying on it.

**Important:** Datasets are never committed to this repo (they're large and have their own licenses). They go in a local `data/` folder, which is git-ignored.

---

## Summary

| Dataset | Used for | Size | Labels | License / terms |
|---|---|---|---|---|
| ShanghaiTech Part A/B | Counting (main benchmark) | 1,198 images | Head points (`.mat`) | BSD-2-Clause (repo license) |
| JHU-Crowd++ | Counting (dense, varied conditions) | 4,372 images, ~2.9 GB | Head points, approx. boxes, blur, occlusion | CC BY 4.0, academic / non-commercial, cite paper |
| UCF-QNRF | Counting (very dense) | 1,535 images, ~4.3 GB | Head points | **(to verify)**, research use, cite paper |
| Mall | Counting on video frames | 2,000 frames, 640×480 | Head points | Research only, non-commercial, cite paper |
| Melbourne Pedestrian Counting System | Forecasting | Hourly counts since 2009 | Count per sensor per hour | City of Melbourne open data, **exact license to verify** |

---

## Counting datasets

### ShanghaiTech (Part A and Part B)
- **What:** The standard crowd-counting benchmark, from the CVPR 2016 MCNN paper (Zhang et al.).
- **Part A:** 482 images collected from the internet, **dense** crowds. 300 train / 182 test.
- **Part B:** 716 photos of busy Shanghai streets, **sparse** crowds. 400 train / 316 test.
- **Labels:** A `.mat` file per image (e.g. `GT_IMG_1.mat`) holding the (x, y) position of each person's head.
- **Get it:** Official repo [desenzhou/ShanghaiTechDataset](https://github.com/desenzhou/ShanghaiTechDataset) (Dropbox / Baidu links), or a [Kaggle mirror](https://www.kaggle.com/datasets/tthien/shanghaitech).
- **License:** The official repo carries a BSD-2-Clause license.
- **Known issue:** Users have reported some [ground-truth inconsistencies in Part B](https://github.com/desenzhou/ShanghaiTechDataset/issues/2).
- **Plan:** Our **first dataset**. It's small and well documented, and almost every paper reports results on it, so our MAE is directly comparable.

### JHU-Crowd++
- **What:** A large dataset with varied scenes, weather (rain, fog, snow) and lighting.
- **Size:** 4,372 images, ~1.51 million head annotations; download `jhu_crowd_v2.0.zip` is ~2.87 GB.
- **Labels:** Head points, plus approximate bounding boxes, blur level and occlusion type.
- **Get it:** [crowd-counting.com](http://www.crowd-counting.com) · paper: [arXiv 2004.03597](https://arxiv.org/abs/2004.03597)
- **License:** CC BY 4.0 for academic, non-commercial use; cite the JHU-Crowd++ papers.
- **Plan:** Optional, for testing robustness in harder conditions, after ShanghaiTech works.

### UCF-QNRF
- **What:** Very dense, high-resolution crowd images (from web search, Flickr and Hajj footage).
- **Size:** 1,535 images (1,201 train / 334 test), ~1.25 million annotations, ~4.33 GB, average ~2013×2902 px.
- **Labels:** Head points.
- **Get it:** [UCF CRCV page](https://www.crcv.ucf.edu/data/ucf-qnrf/) · paper: [arXiv 1808.01050](https://arxiv.org/pdf/1808.01050)
- **License:** **(to verify)** The official page couldn't be loaded during research. Treat it as research-only and cite the paper.
- **Plan:** Optional stretch goal. The images are very large and need more GPU memory.

### Mall
- **What:** 2,000 frames from a fixed public webcam in a shopping mall. It's the closest thing to real video in our list.
- **Size:** 2,000 frames, 640×480, over 60,000 labeled pedestrians. Commonly split as 800 train / 1,200 test.
- **Labels:** Head positions in every frame, plus a perspective map (how much people shrink with distance).
- **Get it:** [Official page](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html) (Chen Change Loy, CUHK).
- **License:** Research purposes only, no commercial use; cite the relevant papers.
- **Plan:** Use it to test how the counting model behaves on consecutive video frames.

---

## Forecasting dataset

### Melbourne Pedestrian Counting System
- **What:** Automated sensors across Melbourne's CBD counting people walking past, every hour, since 2009.
- **Labels/fields:** Hourly counts per sensor, linked by `sensor_id` to a separate sensor-locations dataset.
- **Updates:** Monthly.
- **Get it:** [City of Melbourne Open Data Portal](https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour/information/) (CSV / API) · sensor locations: [map](https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-sensor-locations/map/)
- **License:** Published as open data by the City of Melbourne. Related datasets use Creative Commons Attribution licenses. **Exact license to verify** on the portal before publishing results.
- **Why it fits:** Years of real data mean we can see daily, weekly and yearly patterns, and holidays and events. Any holiday or event claims will be made only if the data supports them.
- **Note:** Some sensors were removed or moved over the years, so we'll need to pick sensors with long, continuous history.

---

## Order of use

1. **ShanghaiTech:** learn the data format, density maps and the first counting model
2. **Melbourne:** time-series basics and forecasting
3. **Mall:** test counting on video frames
4. **JHU-Crowd++ / UCF-QNRF:** optional, if time allows
