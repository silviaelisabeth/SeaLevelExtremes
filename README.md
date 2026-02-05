# SeaLevelExtremes · Storm Surge GEV Analysis

A climate data analysis with focus on sea-level extremes shall be conducted. The work is carried out as part of the **CLIMEX project** (_CLIMate influences on EXtreme Sea-level events_) funded by the Spanish MICIU and the European Union with Prof. Francisco Calafat as Project PI.

The objectives for this project are:

1. Ingest and prepare data from multi-model ensembles of sea-level simulations for extreme-value analysis. Data will be provided by Prof. Francisco Calafat and is based on hindcast data from the Decadal Climate Prediction Project (_DCPP_).
2. Compute extreme-value statistics by fitting a **Generalized Extreme Value** (GEV) distribution to the simulated data at each coastal location, including trends in the location parameter of the GEV distribution with associated uncertainties.
3. Deliver outputs (i.e., extreme-value statistics) and a short technical report (~5 pages).

**Timeline:** Deliverables by May 2026.

**More about the data:** The data consist of sea-level annual maxima along the European coastlines (including the Mediterranean Sea but excluding the Baltic) for the period 1960-2026. The simulations come from a total of 8 models, each with 2 ensemble members for 11022 locations along the European coastline. The whole dataset is about 12 GB. Before executing the GEV analysis, the correct bias correction must be collected; only data flagged as valid must be considered in the analysis.

**Deliverables:** Extreme-value statistics and report.

![License: CC BY‑NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)

---
<img width="1440" height="900" alt="Screenshot 2026-02-05 at 16 11 39" src="https://github.com/user-attachments/assets/174b379f-049d-46c8-90e2-f72415fa0e4d" />

You can view the interactive [map](https://drive.google.com/file/d/1qlvSHVBruSPMDb5cr5vqGndFN1aEjc_T/view?usp=sharing) of all locations here.

---

## Features

- **Data ingestion and preparation**
  - Supports multiple NetCDF climate models or observational datasets.
  - Combines data across models, aligning by location and time.
  - Handles missing data and generates summaries of data availability.

- **GEV analysis per location**
  - Fits GEV distributions to site-level storm surge or extreme water level data.
  - Supports return period calculations and warnings for failed fits.
  - Parallelized workflow using `joblib` for efficient processing of thousands of locations.

- **Annual extreme value statistics**
  - Computes per-year GEV statistics for stationary and non-stationary fits.
  - Weighted least squares regression to assess temporal trends.

- **Visualization and reporting**
  - Plots μ-trends (location parameter) over time for each site.
  - Saves HTML summary files for regression analyses.
  - Optional report export per location to facilitate reproducibility.

- **Scalable and reproducible**
  - Configurable input/output directories and file patterns.
  - Command-line interface for batch processing.
  - Runs efficiently outside Jupyter notebooks in production environments.

---

## Installation

This project requires Python 3.8+. The required packages can be installed by executing:

```bash
pip install -r requirements.txt
```

---

## Folder Structure

The project has a simple structure

```
├── script/       # Data analysis scripts and helper functions
├── output/       # Generated results, plots, and logs
```

The `script/` folder contains all code relevant for data exploration and GEV analysis:

- All `.py` files starting with `func_` contain helper functions for data preparation, GEV analysis, and plotting.
- The Jupyter notebooks `data_exploration.ipynn` and `GEVanalysis.ipynb` are step-by-step notebooks for interactive exploration of the raw data and a guide to walk you through the GEV analysis for a subset of ~10 locations.
- Finally, `GEVanalysis.py` is the optimized version of the respective notebook for efficient analysis of all locations using multiprocessing.

---

## Running the Analysis

You can run the analysis from a terminal or an IDE like PyCharm:

```zsh
python analysis.py --input_dir "/path/to/netcdf/files" --pattern "_.nc"

Arguments:
--input_dir: Directory containing input NetCDF files.
--pattern (optional): Filename pattern to select files (default: _.nc).
```

The script will then

- Import and prepare data from all models
- Pool and combine data by location
- Fit GEV distributions for all locations in parallel
- Compute annual extreme value statistics
- Optionally generate plots and save per-location reports
- Store detailed analysis notes and logs in the output directory

**💡 Tip ·** Use `GEVanalysis.py` for large-scale analysis, as it leverages multiprocessing for faster execution.

---

## Data

I am not in a position to disclose the data. Anyone interested in running the script themselves and playing around with
the original data should please contact
<a href="mailto:&#102;&#114;&#97;&#110;&#99;&#105;&#115;&#99;&#111;&#46;&#109;&#99;&#97;&#108;&#97;&#102;&#97;&#116;&#64;&#117;&#105;&#98;&#46;&#99;&#97;&#116;">Prof. Francisco Calafat</a>.

---

# License
Copyright © 2023–2026<br>
SilviaE. Zieger<br><br>
This project/repo is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
