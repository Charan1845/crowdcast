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
| BMRCL Namma Metro station-wise hourly ridership | Best time to travel (used) | 1–18 Aug + 1–30 Sep 2025, 83 stations | Hourly entries per station (Sep also exits) | Creative Commons Non-Commercial; via RTI (Vivek Mathew), published on OpenCity |
| Melbourne Pedestrian Counting System | Forecasting (not used, see DECISIONS #8) | Hourly counts, May 2009–Oct 2022 + Nov 2024–present (gap between) | Count per sensor per hour | CC BY 4.0 (verified 2026-11-07) |

---

## Counting datasets

### ShanghaiTech (Part A and Part B)
- **What:** The standard crowd-counting benchmark, from the CVPR 2016 MCNN paper (Zhang et al.).
- **Part A:** 482 images collected from the internet, **dense** crowds. 300 train / 182 test.
- **Part B:** 716 photos of busy Shanghai streets, **sparse** crowds. 400 train / 316 test.
- **Labels:** A `.mat` file per image (e.g. `GT_IMG_1.mat`) holding the (x, y) position of each person's head.
- **Get it:** Official repo [desenzhou/ShanghaiTechDataset](https://github.com/desenzhou/ShanghaiTechDataset) (Dropbox / Baidu links), or a [Kaggle mirror](https://www.kaggle.com/datasets/tthien/shanghaitech).
- **License:** The official repo carries a BSD-2-Clause license.
- **Known issue:** Users have reported some [ground-truth inconsistencies in Part B](https://github.com/desenzhou/ShanghaiTechDataset/issues/2). We also found, while batch-processing Part A in an earlier version of the pipeline, that 24/482 images (~5%) have 1-2 annotated head points landing just outside the image's pixel bounds — those points are dropped from the density map, so `stated_count` is used as the training label rather than the density map's own sum wherever the two differ.
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

## Best time to travel dataset

### BMRCL (Namma Metro) station-wise hourly ridership
- **What:** Hourly counts per Bengaluru Metro station, from BMRCL, obtained through an RTI request by Vivek Mathew and published on [OpenCity](https://data.opencity.in/dataset/bmrcl-station-wise-ridership-data).
- **Files used** (downloaded 2026-11-10 into `data/bmrcl/`, not committed):
  - `Station-wise Hourly Data - August 2025`: 1–18 Aug 2025, one row per station per day, 24 hourly columns plus a total. 239 KB; MD5 matches OpenCity's published hash.
  - `Station-wise Hourly Entry and Exit Data - September 2025`: 1–30 Sep 2025, separate Entry and Exit sheets, same layout.
  - `Station Codes Used in the Files`: code to name mapping. Not needed, since station names are in the data.
- **What "ridership" means:** August's file doesn't say. Its hourly pattern matches September's **entries** (correlation 0.997 at MG Road), not exits (0.27), and station daily totals line up with entries (median ratio 0.96). So both months are treated as **entries**.
- **Quirks:** station 111 is called "Bangalore City Station" in August and "KSR Railway Station" in September, so stations are matched by code. 15 stations opened during August 2025 (9 on 11 Aug); days before opening are dropped. 83 stations, 48 days, 92,280 station-hours, no missing values.
- **License:** the dataset page says Creative Commons Non-Commercial; the resource page says CC BY-NC-SA 4.0. Treat it as **non-commercial with attribution, share-alike**. The derived `app/data/metro_hourly_pattern.csv` carries the same terms.

## Forecasting dataset (planned earlier, not used)

### Melbourne Pedestrian Counting System
- **What:** Automated sensors across Melbourne's CBD counting people walking past, every hour.
- **Comes in two parts, with a gap** (checked 2026-11-07):
  - **Historical archive:** a zip attached to the dataset page (`Pedestrian_Counting_System_Monthly_counts_per_hour_may_2009_to_14_dec_2022.csv`). 4,562,230 rows, 82 sensors. Despite the filename, the data actually ends **2022-10-31**. Columns: `Date_Time, Year, Month, Mdate, Day, Time, Sensor_ID, Sensor_Name, Hourly_Counts`.
  - **Live dataset (API/CSV export):** a **rolling 2-year window only** (2024-11-07 to 2026-11-06 when downloaded), 1,629,587 rows, 102 sensors. Columns: `location_id, sensing_date, hourday, direction_1, direction_2, pedestriancount, sensor_name, location`.
  - **Gap:** Nov 2022 – Nov 2024 is no longer downloadable from either source. Older live data rolls off as new data comes in, so re-downloading later shifts the window.
- **Schemas differ:** column names differ between the two parts, and whether `Sensor_ID` (archive) and `location_id` (live) refer to the same physical sensors still needs checking before joining them.
- **Sensor locations:** separate dataset, `pedestrian-counting-system-sensor-locations`.
- **Get it:** [City of Melbourne Open Data Portal](https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour/information/) (archive is under the dataset's attachments) · sensor locations: [map](https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-sensor-locations/map/)
- **License:** CC BY 4.0 (verified via the portal API, 2026-11-07). Attribute the City of Melbourne.
- **Why it fits:** 13+ years of real hourly data show daily, weekly and yearly patterns, plus holidays and events. Any holiday or event claims will be made only if the data supports them.
- **Notes:** Some sensors were removed or moved over the years, so we'll need to pick sensors with long, continuous history. Melbourne had extended COVID lockdowns in 2020–2021, which will show up as a major disruption in the archive. Train/test splits need to account for that rather than ignore it.

---

## Order of use

1. **ShanghaiTech:** learn the data format, density maps and the first counting model
2. **BMRCL Namma Metro:** best time to travel (replaced the Melbourne plan; see DECISIONS #8)
3. **Mall:** test counting on video frames
4. **JHU-Crowd++ / UCF-QNRF:** optional, if time allows
