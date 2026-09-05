# OceanF 🌊

## Satellite Embedding-Based Deep Learning Framework for Subsurface Ocean Temperature Reconstruction

OceanF is a deep-learning-based framework designed to reconstruct **depth-wise subsurface ocean temperature** from daily satellite-derived surface observations.

The project focuses on the **North Indian Ocean (NIO)** and aims to generate daily subsurface temperature fields at a standardized **0.25° spatial resolution** using surface ocean and atmospheric observations.

---

## 🌍 Study Domain

| Parameter | Value |
|---|---|
| Region | North Indian Ocean |
| Latitude | 5°N – 30°N |
| Longitude | 45°E – 105°E |
| Spatial Resolution | 0.25° × 0.25° |
| Temporal Resolution | Daily |
| Study Period | 1 July 2025 – 31 December 2025 |
| Number of Days | 184 |

---

## 🎯 Objective

The primary objective is to develop a deep-learning framework that learns the relationship between observable surface conditions and the vertical structure of ocean temperature.

The model will use satellite/reanalysis-derived surface variables as inputs and reconstruct temperature at multiple subsurface depths.

### Input Features

The prepared ML input contains **7 surface features**:

1. SST — Sea Surface Temperature
2. SSS — Sea Surface Salinity
3. SLA — Sea Level Anomaly
4. UO — Eastward Ocean Current
5. VO — Northward Ocean Current
6. U Wind — 10 m Eastward Wind Component
7. V Wind — 10 m Northward Wind Component

### Target Variable

- `thetao` — Ocean Potential Temperature

Target depths:

```text
0 m
5 m
10 m
20 m
30 m
50 m
75 m
100 m
125 m
150 m
200 m
300 m
500 m
700 m
1000 m


OceanF/
│
├── data/
│   ├── raw/
│   │   ├── SST/
│   │   ├── SSS/
│   │   ├── SSA/
│   │   ├── Winds/
│   │   ├── Currents/
│   │   └── SubsurfaceTemp/
│   │
│   ├── processed/
│   │   ├── SST/
│   │   ├── SSS/
│   │   ├── SSA/
│   │   ├── Winds/
│   │   ├── Currents/
│   │   ├── SubsurfaceTemp/
│   │   └── ML/
│   │
│   └── eda/
│
├── scripts/
│   ├── sst_preprocessing.py
│   ├── sss_preprocessing.py
│   ├── ssa_preprocessing.py
│   ├── winds_preprocessing.py
│   └── subsurface_temp_preprocessing.py
│
├── notebooks/
│   ├── sst_eda.ipynb
│   ├── sss_eda.ipynb
│   ├── ssa_eda.ipynb
│   ├── winds_eda.ipynb
│   ├── subsurface_temp_eda.ipynb
│   └── ml_dataset_preparation.ipynb
│
├── .gitignore
├── DATA_HANDOFF.md
└── README.md
