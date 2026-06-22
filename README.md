# FCBGA with Lid Project

Thermo-mechanical analysis and machine-learning-based prediction for an FCBGA package with lid.

## Overview

This project builds a reproducible ML pipeline from raw simulation data through exploratory analysis, model training (Random Forest and XGBoost), hyperparameter tuning, and final reporting. Five target outputs are modeled across assembly and SJR datasets.

## Dataset

- **Source file:** `2D_assembly_lid_300_v4.xlsx` (sheet: `2D_assembly_lid_300`)
- **Design points:** 300
- **Processed data:** cleaned and merged CSV files in `data/`

Place the original Excel file in `data/` to re-run from step 1, or use the existing CSVs to start from later pipeline steps.

### Input design parameters

- Cu-pillar pitch–diameter
- Bulk silicon thickness
- Bump solder height
- Substrate core thickness
- Substrate core (E, CTE)
- UF (E, CTE)
- Lid foot width
- Lid thickness
- Bump solder material

### Target outputs

| Dataset  | Target                      |
|----------|-----------------------------|
| Assembly | ELK stress                  |
| Assembly | Warpage Post UF cure        |
| Assembly | Warpage post lid attach     |
| SJR      | DeltaW_BGA                  |
| SJR      | DeltaW_bump                 |

## Project structure

```
FCBGA_with_Lid_Project/
├── data/       # Raw and processed datasets
├── figures/    # Exploratory and model plots
├── results/    # Metrics, feature importance, summaries
├── scripts/    # Pipeline scripts (01–15)
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Pipeline

Run scripts in order from the project root:

| Step | Script | Purpose |
|------|--------|---------|
| 01 | `scripts/01_data_check.py` | Load Excel, clean columns, export CSVs |
| 02 | `scripts/02_exploratory_analysis.py` | Descriptive stats and distribution plots |
| 03 | `scripts/03_check_sjr_data.py` | Validate SJR dataset |
| 04 | `scripts/04_merge_datasets.py` | Merge assembly and SJR data |
| 05 | `scripts/05_validate_and_prepare_combined_data.py` | Validate combined dataset |
| 06 | `scripts/06_merge_by_design_variables.py` | Merge on design variables |
| 07 | `scripts/07_train_assembly_models.py` | Train assembly Random Forest models |
| 08 | `scripts/08_train_sjr_models.py` | Train SJR Random Forest models |
| 09 | `scripts/09_create_summary_table.py` | Summarize model performance |
| 10 | `scripts/10_feature_importance.py` | Feature importance analysis |
| 11 | `scripts/11_train_xgboost_models.py` | Train XGBoost models |
| 12 | `scripts/12_compare_rf_xgboost.py` | Compare RF vs XGBoost |
| 13 | `scripts/13_tune_weak_models.py` | Hyperparameter tuning for weak targets |
| 14 | `scripts/14_final_best_model_summary.py` | Select best model per target |
| 15 | `scripts/15_create_report_results.py` | Generate report-ready summary |

Example:

```bash
python scripts/01_data_check.py
```

## Key results

Best models by test R² (see `results/final_results_interpretation.txt`):

| Target | Best model | Test R² | Quality |
|--------|------------|---------|---------|
| ELK stress | XGBoost | 0.977 | Very good |
| Warpage Post UF cure | XGBoost | 0.977 | Very good |
| Warpage post lid attach | Tuned Extra Trees | 0.177 | Weak |
| DeltaW_BGA | Tuned Random Forest | 0.635 | Medium |
| DeltaW_bump | Tuned Random Forest | 0.345 | Weak |

## Author

[baharprh](https://github.com/baharprh)
