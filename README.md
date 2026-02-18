# PCA Driving Behavior

This repository contains the main analysis workflow for PCA-based driving behavior analysis.

## Data Source (OSF)

Data are hosted on OSF:

- https://doi.org/10.17605/OSF.IO/B4XKN

Expected core data files:

- Raw data (folder: `raw/`)
  - `EEDA_autonomous_raw_unitvectors.csv`
  - `EEDA_manual_raw_unitvectors.csv`
- Cleaned data
  - `EEDA_cleaned.csv`
- Questionnaire data
  - `EEDA_Qdata.csv`

## Main Notebook Workflow

Run the main analysis notebooks in this order:

1. `0_preprocessing.ipynb`
2. `1_demographics.ipynb`
3. `2_pca.ipynb`
4. `3_variance_analysis.ipynb`
5. `4_discriminant_analysis.ipynb`

## Minimal Setup

1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run notebooks from the project root directory.

## Data Placement (Current Notebook Paths)

The current notebook code expects:

- Raw vector CSVs in `raw/`
- Cleaned data as `data/EEDA_cleaned.csv`

Place these files directly in `raw/`:

- `raw/EEDA_autonomous_raw_unitvectors.csv`
- `raw/EEDA_manual_raw_unitvectors.csv`
