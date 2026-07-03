# FCBGA with Lid Project

Thermo-mechanical analysis and machine-learning-based co-design for an FCBGA package with lid.

## Overview

This project builds a reproducible workflow from raw simulation data through surrogate modeling, multi-objective optimization (NSGA-II), Net Flow Method (NFM) ranking, and champion design export — aligned with thermo-mechanical co-design manuscripts.

## Dataset

- **Source file:** `2D_assembly_lid_300_v4.xlsx` (sheet: `2D_assembly_lid_300`)
- **Design points:** 300
- **Processed data:** cleaned CSV files in `data/`

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
├── data/                  # Raw and processed datasets
├── figures/               # Plots (codesign_latest/ = stable co-design figures)
├── results/               # Metrics and optimization outputs
│   └── codesign_latest/   # Stable co-design results (updated each full run)
├── scripts/               # Pipeline scripts (01–16 + integrated workflows)
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Recommended workflows

### Full co-design (term paper — ML + optimization)

```bash
python scripts/fcbga_full_codesign_pipeline.py
```

Implements: data cleaning, fixed-depth RF/XGB surrogates, validation/learning curves, **5×4 parity and learning-curve grids**, correlation heatmaps, NSGA-II (pop=150, gen=100), NFM ranking (**histogram + pairwise plots**), **RadViz per objective**, champion export, and nearest-FEA validation proxy.

Stable outputs are copied to `results/codesign_latest/` and `figures/codesign_latest/`.

#### Key figures (`figures/codesign_latest/`)

| Figure | Description |
|--------|-------------|
| `combined_actual_vs_predicted_5x4.png` | RF/XGB × train/test for all 5 targets |
| `combined_learning_curves_5x4.png` | RF/XGB × R²/MSE learning curves |
| `combined_radviz_by_objectives_champion.png` | RadViz panel colored by each objective |
| `radviz_opt_*.png` | Individual RadViz per objective |
| `nfm_scores_histogram.png` | Net Flow score distribution |
| `combined_nfm_pairwise_scatter.png` | NFM rank-bucketed Pareto pairs |
| `combined_scatter_pareto.png` | Raw Pareto objective pairs |
| `objective_correlation_*.png` | Simulation + Pareto correlation heatmaps |

### Surrogate-only (validation plots, no optimization)

```bash
python scripts/fcbga_combined_surrogate_pipeline.py
```

### Highest hold-out R² (tuned models per target)

```bash
python scripts/16_unified_best_models.py
python scripts/15_create_report_results.py
```

Writes `results/unified_best_model_summary.csv` — do **not** confuse with co-design fixed-depth summaries.

### Step-by-step pipeline (01–16)

| Step | Script | Purpose |
|------|--------|---------|
| 01 | `scripts/01_data_check.py` | Load Excel, clean columns, export CSVs |
| 02 | `scripts/02_exploratory_analysis.py` | Descriptive stats and distribution plots |
| 03 | `scripts/03_check_sjr_data.py` | Validate SJR dataset |
| 04 | `scripts/04_merge_datasets.py` | Merge assembly and SJR data |
| 05 | `scripts/05_validate_and_prepare_combined_data.py` | Validate combined dataset |
| 06 | `scripts/06_merge_by_design_variables.py` | Merge on design variables |
| 07–15 | `scripts/07`–`15` | RF/XGB training, tuning, reporting |
| 16 | `scripts/16_unified_best_models.py` | Best tuned model per target |

## Key results

### Tuned surrogates (`results/unified_best_model_summary.csv`)

| Target | Best model | Test R² | Quality |
|--------|------------|---------|---------|
| ELK stress | XGBoost | 0.982 | Very good |
| Warpage Post UF cure | XGBoost | 0.972 | Very good |
| Warpage post lid attach | Tuned Extra Trees | 0.177 | Weak |
| DeltaW_BGA | Tuned Random Forest | 0.635 | Medium |
| DeltaW_bump | Tuned Random Forest | 0.345 | Weak |

### Co-design run (`results/codesign_latest/`)

Fixed-depth surrogates used for optimization (pop=150, gen=100):

| Target | Test R² | Quality |
|--------|---------|---------|
| ELK stress | 0.984 | Very good |
| Warpage Post UF cure | 0.966 | Very good |
| Warpage post lid attach | 0.166 | Weak |
| DeltaW_BGA | 0.579 | Medium |
| DeltaW_bump | 0.328 | Weak |

**Champion design (NFM rank 1):** see `results/codesign_latest/champion_design.csv`

| Parameter | Value |
|-----------|-------|
| Bulk silicon thickness | 0.75 mm |
| Bump solder height | 0.06 mm |
| Substrate core thickness | 0.8 mm |
| Lid foot width | 9 mm |
| Lid thickness | 2.0 mm |
| Cu-pillar / BGA solder | SAC405 / SAC305 |

Validation proxy vs nearest FEA simulation (Design_ID 145): `results/codesign_latest/champion_fea_validation_proxy.csv`

