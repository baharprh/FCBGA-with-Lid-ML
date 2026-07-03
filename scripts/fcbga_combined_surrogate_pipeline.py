#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCBGA with Lid — Combined ML Surrogate Pipeline
===============================================

Term-paper-ready script that merges:

  (A) Project pipeline strengths
      - Excel/CSV loading with explicit data cleaning
      - Harmonized column names across Assembly and SJR files
      - Dataset-specific input features (no cross-file zero-fill)
      - Removal of constant design variables
      - sklearn Pipeline + OneHotEncoder for categoricals

  (B) Professor / surrogate-template strengths
      - Fixed tree depths for reproducible surrogate models
      - 5-fold validation curves (R² and MSE)
      - Learning curves and 2×3 diagnostic figure panels
      - Combined performance summary plots
      - Exported validation metrics table for the report

Workflow (for Methods section):
  1. Load raw Excel or pre-cleaned CSV
  2. Standardize duplicate / inconsistent variable names
  3. Prepare features per simulation dataset (Assembly vs SJR)
  4. Train Random Forest and XGBoost surrogates (fixed depth)
  5. Validate with cross-validation curves and hold-out test set
  6. Select the better surrogate per target and export figures/tables

Run from project root:
    python scripts/fcbga_combined_surrogate_pipeline.py
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

# Timestamped run folder (professor-style) plus stable copies in results/
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_RESULTS_DIR = RESULTS_DIR / f"surrogate_run_{RUN_STAMP}"
RUN_FIGURES_DIR = FIGURES_DIR / f"surrogate_run_{RUN_STAMP}"
RUN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ESTIMATORS = 100

# Fixed depths (reproducible surrogates — professor / updatev2_depth convention)
RF_FIXED_MAX_DEPTH = 8
XGB_FIXED_MAX_DEPTH = 4

# Depth ranges used ONLY for validation-curve diagnostics (not for final model selection)
RF_DEPTH_RANGE = [3, 5, 8, 10, 12, 15]
XGB_DEPTH_RANGE = [2, 3, 4, 5, 6, 8]

# Canonical names for outputs used in this project
ALL_TARGETS = {
    "ELK stress",
    "Warpage Post UF cure",
    "Warpage post lid attach",
    "DeltaW_BGA",
    "DeltaW_bump",
}

# Map inconsistent column labels to one canonical name (handles duplicate semantics)
COLUMN_ALIASES = {
    "Lid thicknes": "Lid thickness",
    "lid thickness": "Lid thickness",
    "Warpage post lid atach": "Warpage post lid attach",
    "Bump solder material": "Cu-pillar bump solder material",
}

# Input files: prefer cleaned CSV; fall back to Excel in data/
ASSEMBLY_SOURCES = [
    DATA_DIR / "cleaned_fcbga_lid_data.csv",
    DATA_DIR / "2D_assembly_lid_300_v4.xlsx",
]
SJR_SOURCES = [
    DATA_DIR / "cleaned_sjr_lid_data.csv",
    DATA_DIR / "2D_SJR_lid_300_v4.xlsx",
]

ASSEMBLY_TARGETS = [
    "ELK stress",
    "Warpage Post UF cure",
    "Warpage post lid attach",
]
SJR_TARGETS = ["DeltaW_BGA", "DeltaW_bump"]

VALIDATION_METRICS: list[dict] = []

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.color": "0.85",
    }
)


# =============================================================================
# 2. DATA CLEANING AND COLUMN HARMONIZATION
# =============================================================================


def _read_first_available(paths: list[Path], sheet_name: str | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the first existing file from a list of candidate paths."""
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            if sheet_name:
                df = pd.read_excel(path, sheet_name=sheet_name)
            else:
                xl = pd.ExcelFile(path)
                df = pd.read_excel(path, sheet_name=xl.sheet_names[0])
        print(f"Loaded {path.name} ({len(df)} rows, {len(df.columns)} columns)")
        return df, path
    raise FileNotFoundError(f"No input file found among: {[str(p) for p in paths]}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and apply canonical aliases."""
    out = df.copy()
    out.columns = out.columns.str.strip()
    return out.rename(columns=COLUMN_ALIASES)


def remove_constant_inputs(df: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop input columns with a single unique value (zero predictive information)."""
    inputs = [c for c in df.columns if c not in targets and c not in ALL_TARGETS]
    constant = [c for c in inputs if df[c].nunique() <= 1]
    if constant:
        print("  Constant columns removed:", ", ".join(constant))
    keep = [c for c in df.columns if c not in constant]
    return df[keep].copy(), constant


def report_missing_values(df: pd.DataFrame, label: str) -> None:
    """Print missing-value counts (term-paper data-quality check)."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n{label} — missing values:")
    if missing.empty:
        print("  None")
    else:
        for col, count in missing.items():
            print(f"  {col}: {count}")


def load_and_clean_dataset(
    sources: list[Path],
    targets: list[str],
    dataset_name: str,
    excel_sheet: str | None = None,
) -> pd.DataFrame:
    """
    Load Assembly or SJR data and apply cleaning steps documented in the term paper:
      - strip / harmonize names
      - verify targets exist
      - remove constant inputs
    """
    print(f"\n{'=' * 70}\nDATA CLEANING: {dataset_name}\n{'=' * 70}")
    df, source = _read_first_available(sources, sheet_name=excel_sheet)
    df = clean_column_names(df)

    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        raise ValueError(f"{dataset_name}: missing target columns {missing_targets}")

    report_missing_values(df, dataset_name)
    df, _ = remove_constant_inputs(df, targets)

    clean_path = RUN_RESULTS_DIR / f"cleaned_{dataset_name.lower()}.csv"
    df.to_csv(clean_path, index=False)
    print(f"  Saved cleaned snapshot: {clean_path.name}")
    return df


# =============================================================================
# 3. FEATURE PREPARATION (dataset-specific — no cross-file feature pooling)
# =============================================================================


def build_preprocessor(df: pd.DataFrame, input_columns: list[str]) -> ColumnTransformer:
    """One-hot encode categoricals; pass numeric design variables through unchanged."""
    cat_cols = df[input_columns].select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = df[input_columns].select_dtypes(exclude=["object", "string"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("numerical", "passthrough", num_cols),
        ]
    )


def get_input_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Inputs = all columns except targets."""
    return [c for c in df.columns if c not in ALL_TARGETS]


def quality_label(r2: float) -> str:
    if r2 >= 0.90:
        return "Very good"
    if r2 >= 0.75:
        return "Good"
    if r2 >= 0.50:
        return "Medium"
    return "Weak"


# =============================================================================
# 4. VALIDATION AND DIAGNOSTIC PLOTS
# =============================================================================


def log_validation_curve(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_name: str,
    param_range: list,
    target: str,
    model_label: str,
) -> None:
    """Record mean CV train/test scores across a hyperparameter range."""
    param_key = f"regressor__{param_name}"
    train_scores, test_scores = validation_curve(
        pipeline,
        X_train,
        y_train,
        param_name=param_key,
        param_range=param_range,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        error_score=np.nan,
    )
    for value, tr, te in zip(param_range, train_scores.mean(axis=1), test_scores.mean(axis=1)):
        VALIDATION_METRICS.append(
            {
                "Target": target,
                "Model": model_label,
                "ParamName": param_name,
                "ParamValue": value,
                "CV_Train_R2_Mean": float(tr),
                "CV_Test_R2_Mean": float(te),
            }
        )


def _plot_validation_curve_ax(ax, param_range, train_mean, test_mean, param_name, title):
    ax.plot(param_range, train_mean, "o-", color="darkorange", label="Training (CV mean)")
    ax.plot(param_range, test_mean, "o-", color="navy", label="Validation (CV mean)")
    ax.set_xlabel(param_name)
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)


def _plot_learning_curve_ax(ax, train_sizes, train_scores, test_scores, title):
    tr = train_scores.mean(axis=1)
    te = test_scores.mean(axis=1)
    ax.plot(train_sizes, tr, "o-", color="darkorange", label="Training")
    ax.plot(train_sizes, te, "o-", color="navy", label="Cross-validation")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)


def _plot_parity_ax(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, edgecolors="k", alpha=0.7)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)


def save_diagnostic_panel(
    pipeline: Pipeline,
    sweep_pipeline: Pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    depth_range: list,
    fixed_depth: int,
    target: str,
    model_label: str,
    dataset_name: str,
) -> None:
    """
    2×3 diagnostic panel (professor-style):
      [Val curve R²] [Learning curve R²] [Parity — train]
      [Val curve — fixed depth note] [Learning MSE] [Parity — test]
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Validation curve (max_depth sweep for documentation)
    tr_list, te_list = [], []
    for d in depth_range:
        sweep_pipeline.set_params(regressor__max_depth=d)
        tr, te = validation_curve(
            sweep_pipeline,
            X_train,
            y_train,
            param_name="regressor__max_depth",
            param_range=[d],
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
        )
        tr_list.append(tr.mean())
        te_list.append(te.mean())
    _plot_validation_curve_ax(
        axes[0, 0],
        depth_range,
        tr_list,
        te_list,
        "max_depth",
        f"{model_label} — validation curve (R²)",
    )

    # Learning curve at fixed depth
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="r2",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
    )
    _plot_learning_curve_ax(
        axes[0, 1],
        train_sizes,
        train_scores,
        test_scores,
        f"{model_label} — learning curve (fixed depth={fixed_depth})",
    )

    y_train_pred = pipeline.predict(X_train)
    _plot_parity_ax(axes[0, 2], y_train, y_train_pred, f"{model_label} — train parity")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.05,
        0.5,
        f"Final surrogate settings\n"
        f"  Model: {model_label}\n"
        f"  max_depth: {fixed_depth}\n"
        f"  n_estimators: {N_ESTIMATORS}\n"
        f"  CV folds: {CV_FOLDS}\n"
        f"  Test fraction: {TEST_SIZE}",
        fontsize=12,
        va="center",
    )

    _, train_mse, test_mse = learning_curve(
        pipeline,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
    )
    axes[1, 1].plot(train_sizes, -train_mse.mean(axis=1), "o-", color="darkorange", label="Training")
    axes[1, 1].plot(train_sizes, -test_mse.mean(axis=1), "o-", color="navy", label="CV")
    axes[1, 1].set_xlabel("Training samples")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].set_title(f"{model_label} — learning curve (MSE)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    y_test_pred = pipeline.predict(X_test)
    _plot_parity_ax(axes[1, 2], y_test, y_test_pred, f"{model_label} — test parity")

    safe = target.replace(" ", "_")
    fig.suptitle(f"{dataset_name} — {target} — {model_label} diagnostics", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RUN_FIGURES_DIR / f"{dataset_name}_{safe}_{model_label.replace(' ', '_')}_diagnostics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved diagnostics: {out.name}")


# =============================================================================
# 5. SURROGATE TRAINING (fixed depth RF & XGB per target)
# =============================================================================


def make_rf_pipeline(preprocessor: ColumnTransformer, max_depth: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_xgb_pipeline(preprocessor: ColumnTransformer, max_depth: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_depth=max_depth,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    tree_method="hist",
                ),
            ),
        ]
    )


def evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    y_tr = pipeline.predict(X_train)
    y_te = pipeline.predict(X_test)
    return {
        "Train_R2": float(r2_score(y_train, y_tr)),
        "Test_R2": float(r2_score(y_test, y_te)),
        "Train_MSE": float(mean_squared_error(y_train, y_tr)),
        "Test_MSE": float(mean_squared_error(y_test, y_te)),
        "y_test_pred": y_te,
    }


def train_target_surrogates(df: pd.DataFrame, target: str, dataset_name: str) -> tuple[dict, list[dict]]:
    """
    Train fixed-depth RF and XGB surrogates for one output variable.
    Returns the best model row and all candidate metric rows.
    """
    print(f"\n{'=' * 70}\nSURROGATE TRAINING: {dataset_name} — {target}\n{'=' * 70}")

    input_cols = get_input_columns(df, target)
    print("  Input features:", ", ".join(input_cols))

    X = df[input_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(df, input_cols)

    candidates = [
        ("Random Forest", make_rf_pipeline(preprocessor, RF_FIXED_MAX_DEPTH), RF_DEPTH_RANGE, RF_FIXED_MAX_DEPTH),
        ("XGBoost", make_xgb_pipeline(preprocessor, XGB_FIXED_MAX_DEPTH), XGB_DEPTH_RANGE, XGB_FIXED_MAX_DEPTH),
    ]

    rows = []
    for model_label, pipeline, depth_range, fixed_depth in candidates:
        # Clone preprocessor for sweep pipeline (independent fit)
        pre = build_preprocessor(df, input_cols)
        sweep = (
            make_rf_pipeline(pre, fixed_depth)
            if model_label == "Random Forest"
            else make_xgb_pipeline(pre, fixed_depth)
        )

        log_validation_curve(
            sweep, X_train, y_train, "max_depth", depth_range, target, model_label
        )
        metrics = evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test)
        save_diagnostic_panel(
            pipeline,
            sweep,
            X_train,
            X_test,
            y_train,
            y_test,
            depth_range,
            fixed_depth,
            target,
            model_label,
            dataset_name,
        )

        row = {
            "Dataset": dataset_name,
            "Target": target,
            "Model": model_label,
            "Train_R2": metrics["Train_R2"],
            "Test_R2": metrics["Test_R2"],
            "Train_MSE": metrics["Train_MSE"],
            "Test_MSE": metrics["Test_MSE"],
        }
        rows.append(row)
        print(
            f"  {model_label}: Train R²={metrics['Train_R2']:.4f}, "
            f"Test R²={metrics['Test_R2']:.4f}"
        )

    best = max(rows, key=lambda r: r["Test_R2"])
    # Re-run predict on best only — pipeline already fitted in loop
    winning_pipeline = next(p for lbl, p, _, _ in candidates if lbl == best["Model"])
    best_y_test_pred = winning_pipeline.predict(X_test)

    best_row = {
        "Dataset": dataset_name,
        "Target": target,
        "Model": best["Model"],
        "Train_R2": best["Train_R2"],
        "Test_R2": best["Test_R2"],
        "Train_MSE": best["Train_MSE"],
        "Test_MSE": best["Test_MSE"],
        "Model_Quality": quality_label(best["Test_R2"]),
        "RF_Max_Depth": RF_FIXED_MAX_DEPTH,
        "XGB_Max_Depth": XGB_FIXED_MAX_DEPTH,
    }
    print(f"  >> Selected surrogate: {best_row['Model']} (Test R²={best_row['Test_R2']:.4f})")

    # Best-model parity figure (term-paper figure)
    safe = target.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(6, 6))
    _plot_parity_ax(ax, y_test, best_y_test_pred, f"Best surrogate — {target}")
    fig.savefig(
        RUN_FIGURES_DIR / f"best_parity_{dataset_name}_{safe}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return best_row, rows


# =============================================================================
# 6. SUMMARY PLOTS AND TABLES
# =============================================================================


def save_combined_test_r2_bar(summary_df: pd.DataFrame) -> None:
    """Bar chart of hold-out test R² for all targets (report figure)."""
    labels = summary_df["Dataset"] + "\n" + summary_df["Target"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, summary_df["Test_R2"], color="steelblue", edgecolor="black")
    ax.axhline(0.90, ls="--", color="green", lw=1, label="Very good (0.90)")
    ax.axhline(0.50, ls="--", color="orange", lw=1, label="Medium (0.50)")
    ax.set_ylabel("Hold-out test R²")
    ax.set_xlabel("Target")
    ax.set_title("FCBGA with Lid — surrogate model performance (fixed depth)")
    ax.set_ylim(0, 1.05)
    for bar, r2 in zip(bars, summary_df["Test_R2"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{r2:.3f}", ha="center", fontsize=9)
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    for folder in (RUN_FIGURES_DIR, FIGURES_DIR):
        path = folder / "surrogate_test_r2_summary.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved summary bar chart: {path}")
    plt.close(fig)


def save_interpretation(summary_df: pd.DataFrame) -> None:
    text = f"""
FCBGA with Lid — Combined Surrogate ML Summary
Run timestamp: {RUN_STAMP}

Method:
  - Cleaned Assembly and SJR datasets separately
  - Harmonized inconsistent column names (e.g. lid thickness variants)
  - Removed constant inputs; one-hot encoded categoricals
  - Fixed-depth surrogates: Random Forest (depth={RF_FIXED_MAX_DEPTH}),
    XGBoost (depth={XGB_FIXED_MAX_DEPTH}), n_estimators={N_ESTIMATORS}
  - 5-fold validation curves + 80/20 hold-out test split (seed={RANDOM_STATE})
  - Best RF vs XGB selected by hold-out test R² per target

Results:
{summary_df.to_string(index=False)}

Output folders:
  - {RUN_RESULTS_DIR}
  - {RUN_FIGURES_DIR}
"""
    for folder in (RUN_RESULTS_DIR, RESULTS_DIR):
        path = folder / "surrogate_run_interpretation.txt"
        path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Saved interpretation: {RUN_RESULTS_DIR / 'surrogate_run_interpretation.txt'}")


# =============================================================================
# 7. MAIN
# =============================================================================


def main() -> None:
    print("\nFCBGA with Lid — Combined ML Surrogate Pipeline")
    print(f"Run output: {RUN_RESULTS_DIR}")

    assembly_df = load_and_clean_dataset(
        ASSEMBLY_SOURCES,
        ASSEMBLY_TARGETS,
        "Assembly",
        excel_sheet="2D_assembly_lid_300",
    )
    sjr_df = load_and_clean_dataset(
        SJR_SOURCES,
        SJR_TARGETS,
        "SJR",
    )

    best_rows = []
    all_candidate_rows = []

    for target in ASSEMBLY_TARGETS:
        best, candidates = train_target_surrogates(assembly_df, target, "Assembly")
        best_rows.append(best)
        all_candidate_rows.extend(candidates)

    for target in SJR_TARGETS:
        best, candidates = train_target_surrogates(sjr_df, target, "SJR")
        best_rows.append(best)
        all_candidate_rows.extend(candidates)

    summary_df = pd.DataFrame(best_rows)
    candidates_df = pd.DataFrame(all_candidate_rows)
    val_df = pd.DataFrame(VALIDATION_METRICS)

    # Save to timestamped run folder and stable project results
    for name, frame in [
        ("surrogate_best_model_summary.csv", summary_df),
        ("surrogate_all_candidates.csv", candidates_df),
        ("validation_cv_metrics.csv", val_df),
    ]:
        frame.to_csv(RUN_RESULTS_DIR / name, index=False)
        if name in {
            "surrogate_best_model_summary.csv",
            "validation_cv_metrics.csv",
        }:
            frame.to_csv(RESULTS_DIR / name, index=False)

    val_df.to_excel(RUN_RESULTS_DIR / "validation_cv_metrics.xlsx", index=False)
    val_df.to_excel(RESULTS_DIR / "validation_cv_metrics.xlsx", index=False)

    save_combined_test_r2_bar(summary_df)
    save_interpretation(summary_df)

    print("\n" + "=" * 70)
    print("FINAL SURROGATE SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nAll run artifacts: {RUN_RESULTS_DIR}")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
