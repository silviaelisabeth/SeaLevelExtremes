# SeaLevelExtremes · Storm Surge GEV Analysis

Here, a climate data analysis with focus on sea-level extremes shall be conducted. The work is carried out as part of the **CLIMEX project** (_CLIMate influences on EXtreme Sea-level events_) funded by the Spanish MICIU and the European Union with Prof. Francisco XY as Project PI.

The objectives for this project are:

1. Ingest and prepare data from multi-model ensembles of sea-level simulations for extreme-value analysis. The ensembles will be provided by Prof. Francisco and are based on hindcast data from the Decadal Climate Prediction Project (_DCPP_).
2. Compute extreme-value statistics by fitting a **Generalized Extreme Value** (GEV) distribution to the simulated data at each coastal location, including trends in the location parameter of the GEV distribution with associated uncertainties. The preference would be to use a **Bayesian model for the fit of the GEV**, though that is not essential.
3. Deliver outputs (i.e., extreme-value statistics) and a short technical report (~5 pages).

**Timeline:** Deliverables by 31 May 2026.

**More about the data:** The data consist of sea-level annual maxima along the European coastlines (including the Mediterranean Sea but excluding the Baltic) for the period 1960-2019. The simulations come from a total of 8 models, each with 2 ensemble members except for one model which has 10 ensemble members. In total, there are approximately 13000 annual maxima at each coastal location. The whole dataset is about 12 GB.

**Deliverables:** Extreme-value statistics and report.

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

This project requires Python 3.8+ and the following packages:

```bash
pip install numpy pandas xarray joblib matplotlib
```

---

## Usage

Run the analysis from the terminal or PyCharm:
python analysis.py --input_dir "/path/to/netcdf/files" --pattern "_.nc"
Arguments:
--input_dir : Directory containing input NetCDF files.
--pattern (optional) : Filename pattern to select files (default: _.nc).
The script will:
Import and prepare data from all models.
Pool and combine data by location.
Fit GEV distributions for all locations in parallel.
Compute annual extreme value statistics.
Optionally generate plots and save per-location reports.
Store detailed analysis notes and logs in the output directory.

---

# Performance

Optimized for >10,000 locations and hundreds of temporal samples per site.
Parallelized per-location computations using all available CPU cores.
Deferred plotting/reporting minimizes runtime for large-scale datasets.

---

# License

MIT License
