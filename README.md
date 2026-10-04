# Crowdcast

**Crowd counting from images and video, plus crowd forecasting from real pedestrian data.**

> Status: 🚧 In progress. This README is updated as the project is built. Results below are filled in only with real measured numbers.

## What it does

1. **Count**: Upload an image or video, and Crowdcast estimates how many people are in it, with a density heatmap showing where the crowd is packed. For videos, it also plots the count over time.
2. **Forecast**: Using years of real hourly pedestrian counts, Crowdcast predicts how busy a location will be at a given time, and compares its error against a simple "same time last week" baseline.

## Architecture (planned)

```
 Images / Video ──► Counting model ──┐
 (public datasets for training,      │
  real uploads for testing)          ▼
                                 Web app (Streamlit / web)
                                     ▲
 Hourly pedestrian counts ──► Forecasting model
 (Melbourne Pedestrian Counting System)
```

## Data

| Part | Dataset | Why |
|---|---|---|
| Counting | ShanghaiTech Part A/B | Standard benchmark, dense (A) and sparse street (B) scenes |
| Counting | JHU-Crowd++ / UCF-QNRF | Very dense crowds, varied weather and lighting |
| Counting | Mall dataset | Fixed-camera video frames for testing on video |
| Forecasting | Melbourne Pedestrian Counting System | Years of real hourly sensor counts, including holidays and events |

## Roadmap (about 6 weeks, from Oct 4, 2026)

- [ ] **Week 1**: Project setup, dataset download and exploration, choose counting approach
- [ ] **Week 2**: Train and evaluate the counting model (MAE on benchmark test sets)
- [ ] **Week 3**: Forecasting: baseline ("same time last week") vs model, evaluated on held-out data
- [ ] **Week 4**: App: image/video upload, count, density heatmap, forecasting tab
- [ ] **Week 5**: Deploy a live demo (Hugging Face Spaces / Streamlit Cloud)
- [ ] **Week 6**: Polish: results, architecture diagram, demo GIF, write-up

## Results

_To be added with real measured numbers._

| Metric | Baseline | Crowdcast |
|---|---|---|
| Counting MAE (ShanghaiTech) | – | – |
| Forecast error vs "same time last week" | – | – |

## Design decisions

See [DECISIONS.md](DECISIONS.md) for every major decision and the reasoning behind it.
