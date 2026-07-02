#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combined_surrogate_optimization_with_nfm_nsga2.py
=================================================

- SURROGATE toggle at top ("xgb" or "rf")
- NSGA-II (population-based) with user-tunable POP_SIZE and N_GEN
- All outputs saved into a timestamped subfolder under the running folder (RESULTS_DIR)
- Separate Excel export for Pareto objectives: pareto_objectives_only.xlsx
- Integrated Net Flow Method (NFM) ranking over Pareto objectives (utilities provided)
- Plots and file formats:
  • Validation/Learning/Parity diagnostic panels per target (RF & XGB)
  • 4×4 Actual-vs-Predicted grid (RF/XGB × Train/Test for the 4 targets)
  • 4×4 Learning Curves grid (RF/XGB × R²/MSE for the 4 targets)
  • RadViz per objective
  • NEW: Combined RadViz panel (single figure; each subplot colored by one objective)
  • Pairwise 2D Pareto scatter (raw) + combined 2×3 panel
  • (Optional) NFM histogram and NFM pairwise (rank-bucketed) plots with champion marker
    - individual pairwise: NO titles, axis labels kept
    - combined 2×3 panel with Champion in legend
  • Correlation heatmap

CHANGES:
- Optimization strictly discrete/categorical (no in-betweens).
- Fix 'PCB k_z' to remain a single numeric discrete variable (NOT split & categorical).
- Save NFM results merged with decoded designs/outputs in separate Excel: nfm_with_designs.xlsx (+ CSV).

Returns from run_combined_workflow:
    dict(res, pareto_F, pareto_X, objectives, results_dir)
"""

import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pandas.api import types as ptypes
import itertools
import os
import re
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from matplotlib.ticker import FuncFormatter

# =========================
# --- Global Configuration
# =========================
SURROGATE = "xgb"  # choose: "xgb" or "rf"

# NSGA-II hyperparameters
POP_SIZE = 500
N_GEN = 500
SEED = 42

# Input Excel files.
# Keep the uploaded SJR file first. Add/rename thermal files here when available.
FILE_CONFIGS = [
    {'name': '2D_SJR_lid_300_v4 (1).xlsx'},
    {'name': '2D_assembly_lid_300_v4.xlsx'},
]

TARGET_KEYWORDS = ["Theta", "DeltaW", "stress", "Warpage"]
RF_MAX_DEPTH_RANGE = [3, 5, 8, 10, 12, 15]
XGB_MAX_DEPTH_RANGE = [2, 3, 4, 5, 6, 8]
N_ESTIMATORS_DEFAULT = 100
TOP_N_FEATURES = 10

RUN_DIR = os.path.abspath(os.getcwd())
# Search both the current folder and /mnt/data (useful inside ChatGPT sandbox).
# When you run locally, put the Excel files in the same folder as this script.
INPUT_SEARCH_DIRS = [Path(RUN_DIR), Path('/mnt/data')]
RESULTS_DIR = os.path.join(RUN_DIR, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"All outputs will be saved to: {RESULTS_DIR}")

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.color": "gray"
})

# ===========================================
# --- Objective Display Names (consistent) ---
# ===========================================
DISPLAY_OBJ_LABELS = {
    "bga":   r"ΔW$_{BGA}$",
    "bump":  r"ΔW$_{Bump}$",
    "c4":    r"ΔW$_{C4}$",
    "theta": r"θ$_{JA}$",
    "ja":    r"θ$_{JA}$",
}

def objective_display_name(raw_name: str) -> str:
    s = str(raw_name).lower()
    if "delta" in s or "Δ" in s or "deltaw" in s or "delta w" in s or "dw" in s:
        if "bga"  in s: return DISPLAY_OBJ_LABELS["bga"]
        if "bump" in s: return DISPLAY_OBJ_LABELS["bump"]
        if "c4"   in s: return DISPLAY_OBJ_LABELS["c4"]
    if "theta" in s or "θ" in s or "ja" in s:
        return DISPLAY_OBJ_LABELS["theta"]
    return raw_name

# Utility to detect ΔW-family objectives (for non-negativity enforcement)
def is_deltaw_objective(raw_name: str) -> bool:
    s = str(raw_name).lower()
    return ("delta" in s) or ("deltaw" in s) or ("delta w" in s) or ("dw" in s) or ("Δ" in s)

# ===========================================
# --- Net Flow Method (NFM) Configuration ---
# ===========================================
SENSE = {}   # default minimize
WEIGHTS = None
Q_EXPL   = None
P_EXPL   = None
V_EXPL   = None
Q_FRACTION = 0.05
P_FRACTION = 0.15
V_FRACTION = 0.30

# Collect validation-curve metrics across all targets / models
VALIDATION_METRICS = []

# =====================
# --- Helper Functions
# =====================

def sanitize_filename_component(name):
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*]+', '_', name)
    name = name.replace(' ', '_')
    return name

def resolve_input_file(file_name: str) -> Path | None:
    """Return the first existing path for an input file.

    The original script used hard-coded Excel names. This helper makes the
    workflow safer when the file was downloaded/uploaded with suffixes such as
    "(1)" or when it is stored in /mnt/data inside the ChatGPT environment.
    """
    candidate = Path(file_name)
    if candidate.exists():
        return candidate
    for folder in INPUT_SEARCH_DIRS:
        direct = folder / file_name
        if direct.exists():
            return direct
    # Loose fallback: same extension and similar stem, ignoring spaces and punctuation.
    wanted = re.sub(r'[^a-z0-9]+', '', Path(file_name).stem.lower())
    for folder in INPUT_SEARCH_DIRS:
        if not folder.exists():
            continue
        for path in folder.glob('*.xls*'):
            stem = re.sub(r'[^a-z0-9]+', '', path.stem.lower())
            if wanted and (wanted in stem or stem in wanted):
                return path
    return None

def build_column_defaults(dfs, cols_for_defaults):
    defaults = {}
    for c in cols_for_defaults:
        vals = []
        for df_item in dfs:
            if c in df_item.columns:
                numeric_col = pd.to_numeric(df_item[c], errors='coerce')
                valid_numeric_vals = numeric_col.dropna()
                if not valid_numeric_vals.empty:
                    vals.extend(valid_numeric_vals.tolist())
        defaults[c] = np.median(vals) if vals else 0.0
    return defaults

def get_best_param_from_r2_vc(estimator_class, base_params, X_train, y_train,
                              param_name, param_range, cv=5, n_jobs=-1, current_target_name_for_log=""):
    global VALIDATION_METRICS

    log_prefix = f"Target '{current_target_name_for_log}': " if current_target_name_for_log else ""
    model_name_for_log = estimator_class.__name__
    print(f"{log_prefix}Determining best '{param_name}' for {model_name_for_log} using R² validation curve (range: {param_range})...")
    current_tuning_params = base_params.copy()
    if param_name in current_tuning_params:
        del current_tuning_params[param_name]
    estimator = estimator_class(**current_tuning_params)
    try:
        X_train_np = X_train if isinstance(X_train, np.ndarray) else np.asarray(X_train)
        y_train_np = y_train if isinstance(y_train, np.ndarray) else np.asarray(y_train)

        # Get train & validation (CV) scores
        train_scores, test_scores = validation_curve(
            estimator,
            X_train_np,
            y_train_np,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring="r2",
            n_jobs=n_jobs,
            error_score=np.nan
        )

        raw_train_scores_mean = np.nanmean(train_scores, axis=1)
        raw_test_scores_mean  = np.nanmean(test_scores,  axis=1)

        # === NEW: log validation metrics for export ===
        param_range_array = np.array(param_range)
        for val, tr_mean, te_mean in zip(param_range_array, raw_train_scores_mean, raw_test_scores_mean):
            VALIDATION_METRICS.append({
                "Target":     current_target_name_for_log,
                "Model":      estimator_class.__name__,
                "ParamName":  param_name,
                "ParamValue": float(val),
                "CV_Train_R2_Mean": float(tr_mean) if not np.isnan(tr_mean) else np.nan,
                "CV_Test_R2_Mean":  float(te_mean) if not np.isnan(te_mean) else np.nan,
            })
        # ==============================================

        if np.all(np.isnan(raw_test_scores_mean)):
            print(f"{log_prefix}Warning: All CV R² scores are NaN; defaulting to first param.")
            return param_range[0] if param_range else None

        best_param_idx = np.nanargmax(raw_test_scores_mean)
        param_range_array = np.array(param_range)
        best_param_value = param_range_array[best_param_idx]
        best_score = raw_test_scores_mean[best_param_idx]
        print(f"{log_prefix}Best {param_name}: {best_param_value} (CV R²: {best_score:.4f})")
        return best_param_value
    except Exception as e:
        print(f"{log_prefix}ERROR during validation_curve: {e}")
        return param_range[0] if param_range else None


# --- Plotting helpers (VC/LC/Pred) ---

def _plot_single_validation_curve(ax, estimator_instance, X, y, param_name, param_range, scoring, title_suffix, cv=5,
                                  n_jobs=-1):
    try:
        X_np = X if isinstance(X, np.ndarray) else np.asarray(X)
        y_np = y if isinstance(y, np.ndarray) else np.asarray(y)
        train_scores, test_scores = validation_curve(
            estimator_instance, X_np, y_np, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=n_jobs, error_score=np.nan
        )
    except Exception as e:
        ax.text(0.5, 0.5, f'VC Error:\n{e}', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(f"Validation Curve {title_suffix}\nError")
        return

    raw_train_scores_mean = np.nanmean(train_scores, axis=1)
    raw_test_scores_mean  = np.nanmean(test_scores,  axis=1)
    train_scores_std = np.nanstd(train_scores, axis=1)
    test_scores_std  = np.nanstd(test_scores,  axis=1)

    plot_train_scores_mean = raw_train_scores_mean.copy()
    plot_test_scores_mean  = raw_test_scores_mean.copy()
    ylabel = f"Score ({scoring})"

    if scoring == "neg_mean_squared_error":
        plot_train_scores_mean = -raw_train_scores_mean
        plot_test_scores_mean  = -raw_test_scores_mean
        ylabel = "Mean Squared Error (MSE)"
        max_val = 0
        if not np.all(np.isnan(plot_train_scores_mean)):
            max_val = np.nanmax(plot_train_scores_mean[~np.isnan(plot_train_scores_mean)])
        if not np.all(np.isnan(plot_test_scores_mean)):
            max_val = max(max_val, np.nanmax(plot_test_scores_mean[~np.isnan(plot_test_scores_mean)]))
        ax.set_ylim(bottom=0, top=max_val * 1.1 if max_val > 0 else 1.0)
    elif scoring == "r2":
        ylabel = "R² Score"
        min_y = min(np.nanmin(raw_train_scores_mean - train_scores_std),
                    np.nanmin(raw_test_scores_mean  - test_scores_std))
        max_y = max(np.nanmax(raw_train_scores_mean + train_scores_std),
                    np.nanmax(raw_test_scores_mean  + test_scores_std))
        ax.set_ylim(min_y - 0.05, max_y + 0.05)

    param_range_array = np.array(param_range)
    plot_param_range = np.arange(len(param_range_array)) if not np.all([isinstance(pr, (int, float)) for pr in param_range_array]) else param_range_array.astype(float)
    tick_labels = [str(pr) for pr in param_range_array]

    ax.plot(plot_param_range, plot_train_scores_mean, label="Training score", color="darkorange", marker='o', lw=2)
    ax.fill_between(plot_param_range, plot_train_scores_mean - train_scores_std,
                    plot_train_scores_mean + train_scores_std, alpha=0.2, color="darkorange")
    ax.plot(plot_param_range, plot_test_scores_mean, label="CV score", color="navy", marker='o', lw=2)
    ax.fill_between(plot_param_range, plot_test_scores_mean - test_scores_std,
                    plot_test_scores_mean + test_scores_std, alpha=0.2, color="navy", label="±1 std")

    ax.set_xticks(plot_param_range); ax.set_xticklabels(tick_labels, rotation=30, ha='right')
    ax.set_title(f"Validation Curve {title_suffix} ({param_name})", fontweight='bold', fontsize=20)
    ax.set_xlabel(str(param_name), fontsize=20); ax.set_ylabel(ylabel, fontsize=20)
    ax.legend(loc="best", fontsize=16)
    ax.tick_params(axis='x', labelsize=18); ax.tick_params(axis='y', labelsize=18)
    ax.grid(True)

def _plot_single_learning_curve(ax, estimator_instance, X, y, scoring, title_suffix, cv=5, n_jobs=-1,
                                train_sizes=np.linspace(.1, 1.0, 5)):
    try:
        X_np = X if isinstance(X, np.ndarray) else np.asarray(X)
        y_np = y if isinstance(y, np.ndarray) else np.asarray(y)
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator_instance, X_np, y_np, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes,
            error_score=np.nan
        )
    except Exception as e:
        ax.text(0.5, 0.5, f'LC Error:\n{e}', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(f"Learning Curve {title_suffix}\nError")
        return

    raw_train_scores_mean = np.nanmean(train_scores, axis=1)
    raw_test_scores_mean  = np.nanmean(test_scores,  axis=1)
    train_scores_std = np.nanstd(train_scores, axis=1)
    test_scores_std  = np.nanstd(test_scores,  axis=1)

    plot_train_scores_mean = raw_train_scores_mean.copy()
    plot_test_scores_mean  = raw_test_scores_mean.copy()
    ylabel = f"Score ({scoring})"

    if scoring == "neg_mean_squared_error":
        plot_train_scores_mean = -raw_train_scores_mean
        plot_test_scores_mean  = -raw_test_scores_mean
        ylabel = "Mean Squared Error (MSE)"
        max_val = 0
        if not np.all(np.isnan(plot_train_scores_mean)):
            max_val = np.nanmax(plot_train_scores_mean[~np.isnan(plot_train_scores_mean)])
        if not np.all(np.isnan(plot_test_scores_mean)):
            max_val = max(max_val, np.nanmax(plot_test_scores_mean[~np.isnan(plot_test_scores_mean)]))
        ax.set_ylim(bottom=0, top=max_val * 1.1 if max_val > 0 else 1.0)
    elif scoring == "r2":
        ylabel = "R² Score"
        min_y = min(np.nanmin(raw_train_scores_mean - train_scores_std),
                    np.nanmin(raw_test_scores_mean  - test_scores_std))
        max_y = max(np.nanmax(raw_train_scores_mean + train_scores_std),
                    np.nanmax(raw_test_scores_mean  + test_scores_std))
        ax.set_ylim(min_y - 0.05, max_y + 0.05)

    ax.plot(train_sizes_abs, plot_train_scores_mean, 'o-', color="darkorange", label="Training score", lw=2)
    ax.fill_between(train_sizes_abs,
                    plot_train_scores_mean - train_scores_std,
                    plot_train_scores_mean + train_scores_std,
                    alpha=0.1, color="darkorange")

    ax.plot(train_sizes_abs, plot_test_scores_mean, 'o-', color="navy", label="CV score", lw=2)
    ax.fill_between(train_sizes_abs,
                    plot_test_scores_mean - test_scores_std,
                    plot_test_scores_mean + test_scores_std,
                    alpha=0.1, color="navy", label="±1 std")

    ax.set_title(f"Learning Curve {title_suffix}", fontweight='bold', fontsize=20)
    ax.set_xlabel("Training examples", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.legend(loc="best", fontsize=16)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True)
    ax.set_xlim(left=0, right=train_sizes_abs.max() * 1.05 if train_sizes_abs.size > 0 else 1)

def _plot_single_actual_vs_predicted(ax, y_true, y_pred, dataset_name_str):
    if y_true is None or y_pred is None:
        ax.text(0.5, 0.5, 'Data N/A', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Actual vs. Predicted - {dataset_name_str}\nData N/A")
        return
    y_true_clean = y_true[~np.isnan(y_true) & ~np.isnan(y_pred)]
    y_pred_clean = y_pred[~np.isnan(y_true) & ~np.isnan(y_pred)]
    if len(y_true_clean) == 0:
        ax.text(0.5, 0.5, 'No valid data points', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Actual vs. Predicted - {dataset_name_str}\nNo Valid Data", fontweight='bold')
        return
    ax.scatter(y_true_clean, y_pred_clean, alpha=0.6, edgecolors='k', s=50, label="Data points")
    all_vals = np.concatenate([y_true_clean, y_pred_clean])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    margin = (max_val - min_val) * 0.05 if (max_val - min_val) > 0 else 0.1
    plot_min, plot_max = min_val - margin, max_val + margin
    if plot_min == plot_max:
        plot_min -= 0.5; plot_max += 0.5
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label="Ideal (y=x)")
    ax.set_xlabel("Actual Values", fontsize=20); ax.set_ylabel("Predicted Values", fontsize=20)
    ax.set_title(f"Actual vs. Predicted - {dataset_name_str}", fontweight='bold', fontsize=20)
    ax.legend(loc="best", fontsize=16)
    ax.tick_params(axis='x', labelsize=18); ax.tick_params(axis='y', labelsize=18)
    ax.grid(True)
    ax.set_xlim(plot_min, plot_max); ax.set_ylim(plot_min, plot_max)
    def smart_format(x, _): return f"{x:.3f}".rstrip('0').rstrip('.') if isinstance(x, float) else str(x)
    ax.xaxis.set_major_formatter(FuncFormatter(smart_format))
    ax.yaxis.set_major_formatter(FuncFormatter(smart_format))
    ax.set_aspect('equal', adjustable='box')

def generate_diagnostic_subplots(estimator_class, base_params,
                                 X_train, y_train, X_test, y_test,
                                 vc_param_name, vc_param_range,
                                 best_param_value_for_lc,
                                 main_plot_title, output_plot_filename_base,
                                 all_learning_curve_data=None, all_actual_vs_pred_data=None, target_name=""):
    print(f"  Generating diagnostic plot: {main_plot_title}")
    fig, axs = plt.subplots(2, 3, figsize=(24, 14))
    vc_estimator_params_for_plot = base_params.copy()
    if vc_param_name in vc_estimator_params_for_plot: del vc_estimator_params_for_plot[vc_param_name]
    vc_estimator = estimator_class(**vc_estimator_params_for_plot)
    _plot_single_validation_curve(axs[0, 0], vc_estimator, X_train, y_train, vc_param_name, vc_param_range, "r2", "(R²)")
    _plot_single_validation_curve(axs[0, 1], vc_estimator, X_train, y_train, vc_param_name, vc_param_range, "neg_mean_squared_error", "(MSE)")
    model_params_for_lc_pred = base_params.copy()
    initial_model_fit_error = False
    if best_param_value_for_lc is not None:
        model_params_for_lc_pred[vc_param_name] = best_param_value_for_lc
    else:
        print(f"    Warning: best_param_value_for_lc for {vc_param_name} is None. Using first from range.")
        if vc_param_range and len(vc_param_range) > 0:
            model_params_for_lc_pred[vc_param_name] = vc_param_range[0]
        else:
            initial_model_fit_error = True
    if 'n_estimators' not in model_params_for_lc_pred:
        model_params_for_lc_pred['n_estimators'] = base_params.get('n_estimators', N_ESTIMATORS_DEFAULT)
    model_for_lc_pred = None
    if not initial_model_fit_error:
        try:
            model_for_lc_pred = estimator_class(**model_params_for_lc_pred)
        except Exception as e_inst:
            print(f"    Error instantiating model for LC/Pred plots: {e_inst}")
            initial_model_fit_error = True
    y_train_pred, y_test_pred = None, None
    predictions_available = False
    actual_fitting_error = False
    if model_for_lc_pred is not None:
        try:
            model_for_lc_pred.fit(X_train, y_train)
            y_train_pred = model_for_lc_pred.predict(X_train)
            y_test_pred = model_for_lc_pred.predict(X_test)
            predictions_available = True
        except Exception as e_fit:
            print(f"    Error fitting model for LC/Pred plots: {e_fit}"); actual_fitting_error = True
    else:
        actual_fitting_error = True
    if predictions_available:
        _plot_single_actual_vs_predicted(axs[0, 2], y_train, y_train_pred, "Training Data")
        if all_actual_vs_pred_data is not None and target_name:
            all_actual_vs_pred_data.append({
                'target_name': target_name,
                'model_type': estimator_class.__name__,
                'y_train_true': np.asarray(y_train),
                'y_train_pred': np.asarray(y_train_pred),
                'y_test_true':  np.asarray(y_test),
                'y_test_pred':  np.asarray(y_test_pred)
            })
    else:
        err_msg = 'Model fit/instantiation failed' if (initial_model_fit_error or actual_fitting_error) else 'Pred. N/A'
        axs[0, 2].text(0.5, 0.5, err_msg, ha='center', va='center', transform=axs[0, 2].transAxes)
        axs[0, 2].set_title(f"Actual vs. Predicted - Training\n{err_msg}", fontweight='bold')
    if model_for_lc_pred is not None and not actual_fitting_error:
        lc_title_suffix_r2 = f"(R², {vc_param_name}={model_params_for_lc_pred.get(vc_param_name, 'N/A')}, N_est={model_params_for_lc_pred.get('n_estimators', 'N/A')})"
        lc_title_suffix_mse = f"(MSE, {vc_param_name}={model_params_for_lc_pred.get(vc_param_name, 'N/A')}, N_est={model_params_for_lc_pred.get('n_estimators', 'N/A')})"
        _plot_single_learning_curve(axs[1, 0], model_for_lc_pred, X_train, y_train, "r2", lc_title_suffix_r2)
        _plot_single_learning_curve(axs[1, 1], model_for_lc_pred, X_train, y_train, "neg_mean_squared_error", lc_title_suffix_mse)
        if all_learning_curve_data is not None and target_name:
            try:
                sizes_r2, tr_r2, cv_r2 = learning_curve(model_for_lc_pred, X_train, y_train, cv=5, scoring="r2", n_jobs=-1,
                                                        train_sizes=np.linspace(.1, 1.0, 5), error_score=np.nan)
                all_learning_curve_data.append({
                    'target_name': target_name,
                    'model_type': estimator_class.__name__,
                    'metric_type': 'r2',
                    'train_sizes': sizes_r2,
                    'train_scores': tr_r2,
                    'test_scores':  cv_r2
                })
                sizes_mse, tr_mse, cv_mse = learning_curve(model_for_lc_pred, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1,
                                                           train_sizes=np.linspace(.1, 1.0, 5), error_score=np.nan)
                all_learning_curve_data.append({
                    'target_name': target_name,
                    'model_type': estimator_class.__name__,
                    'metric_type': 'mse',
                    'train_sizes': sizes_mse,
                    'train_scores': tr_mse,
                    'test_scores':  cv_mse
                })
            except Exception as e:
                print(f"    Error collecting learning curve data: {e}")
    else:
        err_msg = 'LC not plotted (model error)'
        axs[1, 0].text(0.5, 0.5, err_msg, ha='center', va='center'); axs[1, 0].set_title(f"Learning Curve (R²)\n{err_msg}")
        axs[1, 1].text(0.5, 0.5, err_msg, ha='center', va='center'); axs[1, 1].set_title(f"Learning Curve (MSE)\n{err_msg}")
    if predictions_available:
        _plot_single_actual_vs_predicted(axs[1, 2], y_test, y_test_pred, "Test Data")
    else:
        err_msg = 'Model fit/instantiation failed' if (initial_model_fit_error or actual_fitting_error) else 'Pred. N/A'
        axs[1, 2].text(0.5, 0.5, err_msg, ha='center', va='center')
        axs[1, 2].set_title(f"Actual vs. Predicted - Test\n{err_msg}")
    fig.suptitle(main_plot_title, fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0)
    output_filename = os.path.join(RESULTS_DIR, f"{output_plot_filename_base}_diagnostic_plots.png")
    plt.savefig(output_filename); print(f"  Saved diagnostic plots to {output_filename}")
    plt.close(fig)

# -------- 4×4 combined plots (parity & learning curves) --------

def generate_combined_actual_vs_predicted_4x4(all_pred_data, output_path):
    target_order = [r"ΔW$_{BGA}$", r"ΔW$_{Bump}$", r"ΔW$_{C4}$", r"θ$_{JA}$"]
    target_map = {}
    for data in all_pred_data:
        target_map[data['target_name']] = objective_display_name(data['target_name'])
    fig, axs = plt.subplots(4, 4, figsize=(24, 20))
    column_config = [
        ('RandomForestRegressor', 'train'),
        ('XGBRegressor',         'train'),
        ('RandomForestRegressor', 'test'),
        ('XGBRegressor',          'test')
    ]
    for row, target_disp in enumerate(target_order):
        actual_t = None
        for raw, disp in target_map.items():
            if disp == target_disp:
                actual_t = raw; break
        if actual_t is None:
            for col in range(4): axs[row, col].axis('off')
            continue
        for col, (model_type, data_type) in enumerate(column_config):
            ax = axs[row, col]
            plot_data = next((d for d in all_pred_data if d['target_name']==actual_t and d['model_type']==model_type), None)
            if plot_data is None:
                ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes); continue
            y_true = plot_data['y_train_true'] if data_type=='train' else plot_data['y_test_true']
            y_pred = plot_data['y_train_pred'] if data_type=='train' else plot_data['y_test_pred']
            _plot_single_actual_vs_predicted(ax, y_true, y_pred, "temp")
            model_short = "RF" if model_type=="RandomForestRegressor" else "XGB"
            data_short  = "Train" if data_type=="train" else "Test"
            ax.set_title(f"{target_disp} | {model_short} | {data_short}", fontweight='bold', fontsize=20)
            if ax.legend_: ax.legend_.remove()
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined 4x4 Actual vs Predicted plot to {output_path}")

def generate_combined_learning_curves_4x4(all_learning_curve_data, output_path):
    target_order = [r"ΔW$_{BGA}$", r"ΔW$_{Bump}$", r"ΔW$_{C4}$", r"θ$_{JA}$"]
    target_map = {}
    for data in all_learning_curve_data:
        target_map[data['target_name']] = objective_display_name(data['target_name'])
    fig, axs = plt.subplots(4, 4, figsize=(24, 20))
    fig.suptitle("Learning Curves - All Targets and Models", fontsize=24, fontweight='bold', y=0.98)
    column_config = [
        ('RandomForestRegressor', 'r2'),
        ('RandomForestRegressor', 'mse'),
        ('XGBRegressor',          'r2'),
        ('XGBRegressor',          'mse')
    ]
    for row, target_disp in enumerate(target_order):
        actual_t = None
        for raw, disp in target_map.items():
            if disp == target_disp:
                actual_t = raw; break
        if actual_t is None:
            for col in range(4): axs[row, col].axis('off')
            continue
        for col, (model_type, metric_type) in enumerate(column_config):
            ax = axs[row, col]
            plot_data = next((d for d in all_learning_curve_data
                              if d['target_name']==actual_t and d['model_type']==model_type and d['metric_type']==metric_type), None)
            if plot_data is None:
                ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{target_disp} | {'RF' if model_type=='RandomForestRegressor' else 'XGB'} | {metric_type.upper()}",
                             fontweight='bold', fontsize=16); continue
            train_sizes = plot_data['train_sizes']
            train_scores = plot_data['train_scores']; test_scores  = plot_data['test_scores']
            if metric_type == 'mse':
                train_scores = -train_scores; test_scores  = -test_scores
            tr_mean = np.mean(train_scores, axis=1); tr_std = np.std(train_scores, axis=1)
            cv_mean = np.mean(test_scores,  axis=1); cv_std = np.std(test_scores,  axis=1)
            ax.plot(train_sizes, tr_mean, 'o-', color="darkorange", label="Train", lw=2)
            ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.1, color="darkorange")
            ax.plot(train_sizes, cv_mean, 'o-', color="navy", label="CV", lw=2)
            ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.1, color="navy")
            ylabel = "R²" if metric_type == 'r2' else "MSE"
            model_short = "RF" if model_type == "RandomForestRegressor" else "XGB"
            ax.set_xlabel("Training examples", fontsize=14); ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(f"{target_disp} | {model_short} | {metric_type.UPPER()}", fontweight='bold', fontsize=16) if False else ax.set_title(f"{target_disp} | {model_short} | {metric_type.upper()}", fontweight='bold', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.5)
            handles, labels = ax.get_legend_handles_labels()
            sd_patch = mpatches.Patch(alpha=0.1, facecolor='grey', edgecolor='none', label='±1 SD')
            ax.legend(handles + [sd_patch], labels + ['±1 SD'], loc="best", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined 4x4 Learning Curves plot to {output_path}")

# -------- Feature-importance aggregation over categorical OHE --------

def aggregate_fi_over_categories(importances: np.ndarray, feature_names: list[str],
                                 cat_col_details: dict) -> tuple[list[str], np.ndarray]:
    ohe_to_base = {}
    for base_col, details in cat_col_details.items():
        for ohe in details.get('one_hot_names', []):
            ohe_to_base[ohe] = base_col
    agg = {}
    for fi, fname in zip(importances, feature_names):
        base = ohe_to_base.get(fname, fname)
        agg[base] = agg.get(base, 0.0) + float(fi)
    names = list(agg.keys())
    vals  = np.array([agg[n] for n in names], dtype=float)
    return names, vals

# -------- NFM utilities --------

def build_netflow_arrays(F: np.ndarray,
                         obj_names: list[str],
                         sense_map: dict[str, str],
                         weights_map: dict[str, float] | None,
                         q_map: dict[str, float] | None,
                         p_map: dict[str, float] | None,
                         v_map: dict[str, float] | None,
                         q_frac: float,
                         p_frac: float,
                         v_frac: float):
    n = len(obj_names)
    if weights_map is None:
        W = np.ones(n) / n
    else:
        W = np.array([weights_map.get(name, 0.0) for name in obj_names], dtype=float)
        W = np.ones(n)/n if W.sum() <= 0 else W / W.sum()
    ranges = F.max(axis=0) - F.min(axis=0)
    def resolve_threshold(vec_map, frac_default):
        if vec_map is not None:
            return np.array([vec_map.get(name, frac_default * ranges[i]) for i, name in enumerate(obj_names)], dtype=float)
        else:
            return np.array([frac_default * ranges[i] for i in range(n)], dtype=float)
    Q = np.maximum(resolve_threshold(q_map, q_frac), 0.0)
    P = np.maximum(resolve_threshold(p_map, p_frac), Q)
    V = np.maximum(resolve_threshold(v_map, v_frac), P + 1e-12)
    S = np.array([1.0 if sense_map.get(name, "min").strip().lower()=="max" else -1.0 for name in obj_names], dtype=float)
    return F, W, Q, P, V, S

def net_flow(F: np.ndarray, W: np.ndarray, Q: np.ndarray, P: np.ndarray, V: np.ndarray, S: np.ndarray):
    M, n = F.shape
    c = np.zeros((M, M, n))
    for i in range(M):
        for j in range(M):
            d = S * (F[i] - F[j])
            c[i, j] = np.where(d <= Q, 1.0, np.where(d <= P, 0.5, np.where(d <= V, 0.25, 0.0)))
    pref = (c * W.reshape((1, 1, n))).sum(axis=2)
    phi_plus  = pref.sum(axis=1)
    phi_minus = pref.sum(axis=0)
    return phi_plus - phi_minus

# -------- NFM plotting helpers --------

def plot_scores_hist(scores: np.ndarray, out_path: str):
    plt.figure(figsize=(7, 5))
    plt.hist(scores, bins="auto", edgecolor='k', alpha=0.75)
    plt.title("Net Flow Scores Histogram", fontweight='bold')
    plt.xlabel("Score"); plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

def pairwise_scatter(df: pd.DataFrame, obj_cols: list[str], ranks: np.ndarray, out_dir: str,
                     label_map: dict[str, str] | None = None):
    order = np.argsort(ranks)  # best->worst
    n = len(order)
    cut10 = max(n // 10, 1); cut25 = max(n // 4, 1); cut50 = max(n // 2, 1)
    buckets = {
        "Top 10%": set(order[:cut10]),
        "Top 10-25%": set(order[cut10:cut25]),
        "Top 25-50%": set(order[cut25:cut50]),
        "Bottom 50%": set(order[cut50:])
    }
    colors = {
        "Bottom 50%": "black",
        "Top 25-50%": "blue",
        "Top 10-25%": "green",
        "Top 10%": "cyan",
    }
    champion_idx = int(order[0])
    # Figure out the champion's bucket color to redraw underlying point
    champ_color = None
    for label, idxset in buckets.items():
        if champion_idx in idxset:
            champ_color = colors[label]
            break

    for i in range(len(obj_cols)):
        for j in range(i+1, len(obj_cols)):
            xcol, ycol = obj_cols[i], obj_cols[j]
            xlab = label_map.get(xcol, xcol) if label_map else xcol
            ylab = label_map.get(ycol, ycol) if label_map else ycol
            plt.figure(figsize=(8, 6))
            for label in ["Bottom 50%", "Top 25-50%", "Top 10-25%", "Top 10%"]:
                idxs = list(buckets[label])
                if idxs:
                    plt.scatter(df.loc[idxs, xcol], df.loc[idxs, ycol], s=30, label=label, color=colors[label], alpha=0.85)
            # Hollow champion marker (so the point beneath remains visible)
            plt.scatter(df.loc[champion_idx, xcol], df.loc[champion_idx, ycol],
                        s=160, marker='^', facecolors='none', edgecolors='red', linewidths=1.8, label='Champion', zorder=5)
            # Redraw the actual champion point on top in its bucket color
            if champ_color is not None:
                plt.scatter(df.loc[champion_idx, xcol], df.loc[champion_idx, ycol],
                            s=40, color=champ_color, edgecolors='k', linewidths=0.6, zorder=6)
            plt.xlabel(xlab, fontsize=13, fontweight='bold'); plt.ylabel(ylab, fontsize=13, fontweight='bold')
            plt.legend(loc='best', fontsize=10, framealpha=1, edgecolor='black')
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            fname = f"netflow_pairwise_scatter_{sanitize_filename_component(xcol)}_VS_{sanitize_filename_component(ycol)}.png"
            plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=300); plt.close()

def pairwise_scatter_combined_nfm(df, obj_cols, ranks, out_path, label_map=None):
    order = np.argsort(ranks)
    n = len(order)
    cut10 = max(n // 10, 1); cut25 = max(n // 4,  1); cut50 = max(n // 2,  1)
    buckets = {
        "Top 10%":     set(order[:cut10]),
        "Top 10-25%":  set(order[cut10:cut25]),
        "Top 25-50%":  set(order[cut25:cut50]),
        "Bottom 50%":  set(order[cut50:])
    }
    draw_order = ["Bottom 50%", "Top 25-50%", "Top 10-25%", "Top 10%"]
    colors = {
        "Bottom 50%": "black",
        "Top 25-50%": "blue",
        "Top 10-25%": "green",
        "Top 10%":    "cyan"
    }
    champion_idx = int(order[0])
    champ_color = None
    for label, idxset in buckets.items():
        if champion_idx in idxset:
            champ_color = colors[label]
            break

    pairs = list(itertools.combinations(range(len(obj_cols)), 2))[:6]
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.ravel()
    for ax_i, (ax, (i, j)) in enumerate(zip(axs, pairs)):
        xcol, ycol = obj_cols[i], obj_cols[j]
        xlab = label_map.get(xcol, xcol) if label_map else xcol
        ylab = label_map.get(ycol, ycol) if label_map else ycol
        for label in draw_order:
            idxs = list(buckets[label])
            if idxs:
                ax.scatter(df.loc[idxs, xcol], df.loc[idxs, ycol],
                           s=30, label=label if ax_i == 0 else None, color=colors[label], alpha=0.85)
        # Hollow champion triangle + redraw of the actual point
        ax.scatter(df.loc[champion_idx, xcol], df.loc[champion_idx, ycol],
                   s=160, marker='^', facecolors='none', edgecolors='red', linewidths=1.8,
                   label='Champion' if ax_i == 0 else None, zorder=5)
        if champ_color is not None:
            ax.scatter(df.loc[champion_idx, xcol], df.loc[champion_idx, ycol],
                       s=40, color=champ_color, edgecolors='k', linewidths=0.6, zorder=6)
        ax.set_xlabel(xlab, fontsize=13, fontweight='bold'); ax.set_ylabel(ylab, fontsize=13, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.5)
    handles, labels = axs[0].get_legend_handles_labels()
    label_order = draw_order + ["Champion"]
    ordered = [(h, l) for l in label_order for (h, lbl) in zip(handles, labels) if lbl == l]
    if ordered:
        handles_o, labels_o = zip(*ordered)
        leg = fig.legend(handles_o, labels_o, loc='upper center', ncol=5, fontsize=16, frameon=True)
        leg.get_frame().set_edgecolor('black'); leg.get_frame().set_linewidth(1.0); leg.get_frame().set_alpha(1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[NFM] Saved combined pairwise Pareto (NO TITLES, labels kept) -> {out_path}")

# === NEW: Objective correlation heatmap ===
def plot_objective_corr_heatmap(
    F_df: pd.DataFrame,
    raw_obj_names: list[str],
    out_path: str,
    method: str = "pearson",
    title: str | None = None,
    annotate: bool = True,
    triangular: bool = False,
):
    """
    Fancy, high-quality correlation heatmap for objectives (Pareto-only by default).
    - F_df: DataFrame whose columns are 'Obj_<sanitized objective name>'
    - raw_obj_names: the original objective names (used to build F_df column order & pretty labels)
    - method: 'pearson' or 'spearman'
    - triangular: if True, hide upper triangle for a clean look
    """
    obj_cols = [f"Obj_{sanitize_filename_component(n)}" for n in raw_obj_names if f"Obj_{sanitize_filename_component(n)}" in F_df.columns]
    if not obj_cols or len(obj_cols) < 2:
        print("[Heatmap] Not enough objective columns to plot.")
        return

    data = F_df[obj_cols].astype(float).copy()
    if method.lower() == "spearman":
        corr = data.rank(axis=0).corr(method="pearson").values
        method_used = "Spearman"
    else:
        corr = data.corr(method="pearson").values
        method_used = "Pearson"

    avg_abs = np.mean(np.abs(corr), axis=0)
    order = np.argsort(-avg_abs)
    corr = corr[order][:, order]
    obj_cols = [obj_cols[i] for i in order]
    disp_labels = [objective_display_name(name.replace("Obj_", "")) for name in obj_cols]


    if triangular:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_masked = np.ma.array(corr, mask=mask)
    else:
        corr_masked = corr

    n = len(obj_cols)
    fig_size = (max(6, 1.8 * n), max(5.5, 1.6 * n))
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)
    im = ax.imshow(corr_masked, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_aspect('equal')
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(disp_labels, rotation=35, ha='right', fontsize=12)
    ax.set_yticklabels(disp_labels, fontsize=12)


    if annotate:
        for i in range(n):
            for j in range(n):
                if triangular and j > i:
                    continue
                val = corr[i, j]
                if np.isnan(val):
                    continue
                txt = f"{val:.2f}"
                kw = dict(color="black", fontsize=11, fontweight="bold") if abs(val) >= 0.70 else dict(color="black", fontsize=10)
                ax.text(j, i, txt, ha="center", va="center", **kw)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.040, pad=0.08)
    cbar.set_label(f"{method_used} correlation", fontsize=12)

    ax.set_title("Objective Correlation Heatmap", fontsize=14, fontweight="bold", pad=14)

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[Heatmap] Saved -> {out_path}")


# -------- Raw combined 2×3 panel --------

def pairwise_scatter_combined_raw(pareto_F: np.ndarray, objective_names: list[str], out_path: str):
    if pareto_F is None or pareto_F.shape[1] < 2: return
    pairs = list(itertools.combinations(range(pareto_F.shape[1]), 2))[:6]
    fig, axs = plt.subplots(2, 3, figsize=(18, 10)); axs = axs.ravel()
    for ax, (i, j) in zip(axs, pairs):
        obj1, obj2 = objective_names[i], objective_names[j]
        ax.scatter(pareto_F[:, i], pareto_F[:, j], color='tab:blue', s=60, edgecolor='black', label='Pareto Points', zorder=5)
        ax.set_xlabel(objective_display_name(obj1), fontsize=13, fontweight='bold')
        ax.set_ylabel(objective_display_name(obj2), fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(out_path, dpi=300); plt.close(fig)
    print(f"Saved combined raw pairwise panel -> {out_path}")

# -------- Combined RadViz panels --------

def generate_combined_radviz_by_objectives(XY_radviz: np.ndarray,
                                           pareto_F: np.ndarray,
                                           objective_names: list[str],
                                           out_path: str,
                                           max_subplots: int = 6):
    if XY_radviz is None or XY_radviz.shape[0] == 0 or pareto_F is None or pareto_F.shape[1] < 2:
        return
    n_obj = min(len(objective_names), pareto_F.shape[1], max_subplots)
    if n_obj == 4: nrows, ncols = 2, 2
    elif n_obj <= 3: nrows, ncols = 1, n_obj
    elif n_obj <= 6: nrows, ncols = 2, 3
    else:
        nrows = int(np.ceil(np.sqrt(n_obj))); ncols = int(np.ceil(n_obj / nrows))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6.5 * nrows)); axs = np.atleast_1d(axs).ravel()
    angles = 2 * np.pi * np.arange(pareto_F.shape[1]) / pareto_F.shape[1]
    anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    poly_x = np.append(anchors[:, 0], anchors[0, 0]); poly_y = np.append(anchors[:, 1], anchors[0, 1])
    for i in range(n_obj):
        ax = axs[i]; name = objective_names[i]; disp_name = objective_display_name(name)
        sc = ax.scatter(XY_radviz[:, 0], XY_radviz[:, 1], c=pareto_F[:, i], cmap="viridis", s=60, ec="k", alpha=0.7, label="Pareto Designs")
        ax.plot(poly_x, poly_y, "--", c="r", lw=1.5, label="Objective Anchors")
        ax.scatter(anchors[:, 0], anchors[:, 1], marker="^", c="r", s=120)
        for il, lab in enumerate(objective_names):
            x_anchor, y_anchor = anchors[il, 0], anchors[il, 1]
            rotation_angle, offset = 0, 0.9
            if abs(x_anchor) > 0.7 and abs(y_anchor) < 0.3: rotation_angle, offset = 90, 0.80
            elif abs(y_anchor) > 0.7 and abs(x_anchor) < 0.3: rotation_angle, offset = 0, 0.80
            ax.text(x_anchor * offset, y_anchor * offset, objective_display_name(lab), c="m",
                    ha="center", va="center", rotation=rotation_angle, fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round'))
        ax.set_title(f"RadViz (Color by {disp_name})"); ax.set_xlabel("RadViz X"); ax.set_ylabel("RadViz Y")
        ax.axhline(0, c='k', lw=0.5, ls='--'); ax.axvline(0, c='k', lw=0.5, ls='--'); ax.set_aspect('equal')
        cbar = plt.colorbar(sc, ax=ax, location='bottom', shrink=0.85, pad=0.12)
        ax.set_xlabel("RadViz X", labelpad=10)
        cbar.set_label(disp_name)
    for j in range(n_obj, nrows * ncols): axs[j].axis('off')
    plt.tight_layout(pad=2.0); plt.savefig(out_path, dpi=300); plt.close(fig)
    print(f"Saved combined RadViz panel -> {out_path}")

def generate_combined_radviz_by_objectives_champion(XY_radviz: np.ndarray,
                                                    pareto_F: np.ndarray,
                                                    objective_names: list[str],
                                                    out_path: str,
                                                    champion_index: int,
                                                    max_subplots: int = 6):
    if XY_radviz is None or XY_radviz.shape[0] == 0 or pareto_F is None or pareto_F.shape[1] < 2:
        return
    n_obj = min(len(objective_names), max_subplots)
    nrows = 2 if n_obj > 2 else 1; ncols = int(np.ceil(n_obj / nrows))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6.5 * nrows)); axs = np.atleast_1d(axs).ravel()
    angles = 2 * np.pi * np.arange(pareto_F.shape[1]) / pareto_F.shape[1]
    anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    poly_x = np.append(anchors[:, 0], anchors[0, 0]); poly_y = np.append(anchors[:, 1], anchors[0, 1])
    M = XY_radviz.shape[0]; champ_ok = 0 <= int(champion_index) < M
    for i in range(n_obj):
        ax = axs[i]; name = objective_names[i]; disp_name = objective_display_name(name)
        sc = ax.scatter(XY_radviz[:, 0], XY_radviz[:, 1], c=pareto_F[:, i], cmap="viridis",
                        s=60, ec="k", alpha=0.7, label="Pareto Designs")
        ax.plot(poly_x, poly_y, "--", c="r", lw=1.5, label="Objective Anchors")
        ax.scatter(anchors[:, 0], anchors[:, 1], marker="^", c="r", s=120)
        for il, lab in enumerate(objective_names):
            x_anchor, y_anchor = anchors[il, 0], anchors[il, 1]
            rotation_angle, offset = 0, 0.75
            if abs(x_anchor) > 0.7 and abs(y_anchor) < 0.3: rotation_angle, offset = 90, 0.80
            elif abs(y_anchor) > 0.7 and abs(x_anchor) < 0.3: rotation_angle, offset = 0, 0.80
            ax.text(x_anchor * offset, y_anchor * offset, objective_display_name(lab), c="m",
                    ha="center", va="center", rotation=rotation_angle, fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round'))
        if champ_ok:
            # Hollow star so the underlying colored point is visible
            ax.scatter(XY_radviz[int(champion_index), 0], XY_radviz[int(champion_index), 1],
                       s=220, marker="*", facecolors='none', edgecolors="black", linewidths=1.6,
                       label="NFM Champion (★)", zorder=6)
            # Re-draw a small dot on top to ensure visibility
            ax.scatter(XY_radviz[int(champion_index), 0], XY_radviz[int(champion_index), 1],
                       s=45, marker="o", edgecolors="k", linewidths=0.6, zorder=7)
        ax.set_title(f"RadViz (Color by {disp_name})"); ax.set_xlabel("RadViz X"); ax.set_ylabel("RadViz Y")
        ax.axhline(0, c='k', lw=0.5, ls='--'); ax.axvline(0, c='k', lw=0.5, ls='--'); ax.set_aspect('equal')
        cbar = plt.colorbar(sc, ax=ax, location='bottom', shrink=0.85, pad=0.12); cbar.set_label(disp_name)
        ax.set_xlabel("RadViz X", labelpad=10)

    for j in range(n_obj, len(axs)):
        axs[j].axis('off')

    # Shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=16,
               frameon=True, framealpha=1.0, edgecolor='black')

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined RadViz (with Champion) -> {out_path}")

def generate_combined_radviz_with_champion(XY_radviz: np.ndarray,
                                           champion_index: int,
                                           out_path: str,
                                           title: str,
                                           mask: np.ndarray | None = None):
    if XY_radviz is None or len(XY_radviz) == 0: return
    M = XY_radviz.shape[0]
    if mask is None:
        mask = np.ones(M, dtype=bool)
    else:
        mask = np.array(mask, dtype=bool)
        if mask.shape[0] != M:
            mask = np.ones(M, dtype=bool)
    fig, ax = plt.subplots(figsize=(7, 7)); pts = XY_radviz[mask]
    ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.75, edgecolors="none")
    if 0 <= int(champion_index) < M and mask[int(champion_index)]:
        ax.scatter(XY_radviz[int(champion_index), 0],
                   XY_radviz[int(champion_index), 1],
                   s=220, marker="*", facecolors='none', edgecolors="black", linewidths=1.6, zorder=5)
        ax.scatter(XY_radviz[int(champion_index), 0],
                   XY_radviz[int(champion_index), 1],
                   s=45, marker="o", edgecolors="k", linewidths=0.6, zorder=6)
    ax.set_xlabel("RadViz X"); ax.set_ylabel("RadViz Y"); ax.set_title(title, pad=12)
    ax.grid(True, linestyle="--", alpha=0.4); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close(fig)

# ==========================
# --- NEW: Discrete helpers
# ==========================

def collect_discrete_domains(original_dfs: list[pd.DataFrame],
                             feature_names: list[str],
                             objective_names: list[str],
                             max_unique: int = 10000) -> dict[str, list]:
    domains = {}
    for f in feature_names:
        if f in objective_names:
            continue
        vals = []
        for df in original_dfs:
            if f in df.columns:
                s = pd.to_numeric(df[f], errors='coerce').dropna()
                if not s.empty:
                    vals.append(s.values)
        if vals:
            u = np.unique(np.concatenate(vals).astype(float))
            if len(u) == 1:
                domains[f] = [float(u[0])]
            elif 1 < len(u) <= max_unique:
                domains[f] = [float(x) for x in np.sort(u)]
            else:
                u6 = np.unique(np.round(u.astype(float), 6))
                domains[f] = [float(x) for x in np.sort(u6)]
    return domains

def index_bounds_from_domains(discrete_domains: dict[str, list]) -> dict[str, tuple[int, int]]:
    return {k: (0, max(0, len(v) - 1)) for k, v in discrete_domains.items()}

def get_value_from_domain(discrete_domains: dict[str, list], name: str, idx_array: np.ndarray) -> np.ndarray:
    values = discrete_domains[name]
    idx = np.clip(idx_array.astype(int), 0, len(values) - 1)
    return np.array([values[i] for i in idx], dtype=float)

# ==========================
# --- Main Script Workflow
# ==========================

def run_combined_workflow():
    print("--- Starting Combined RF/XGB Training, NSGA-II Optimization, and NFM Ranking ---")
    all_feature_importance_data = []
    all_actual_vs_pred_data = []
    all_learning_curve_data = []

    # 1) Load Excel data
    print("\n1. Loading data...")
    raw_dfs = []
    for config in FILE_CONFIGS:
        fname = config['name']
        resolved_path = resolve_input_file(fname)
        if resolved_path is None:
            print(f"Warning: input file not found: {fname}. Proceeding without it.")
            raw_dfs.append(pd.DataFrame())
            continue
        try:
            df = pd.read_excel(resolved_path)
            raw_dfs.append(df)
            print(f"Loaded {resolved_path.name} ({len(df)} rows, {len(df.columns)} cols)")
        except Exception as e:
            print(f"Error loading {resolved_path}: {e}. Proceeding without it.")
            raw_dfs.append(pd.DataFrame())
    original_dfs = [df.copy() for df in raw_dfs]

    # Identify objectives by keyword
    initial_objective_names = []
    target_info_map = {}
    for idx, df_orig in enumerate(original_dfs):
        if df_orig.empty: continue
        file_name_prefix = sanitize_filename_component(os.path.splitext(FILE_CONFIGS[idx]['name'])[0]) if idx < len(FILE_CONFIGS) else f"DF_{idx}"
        for col_name in df_orig.columns:
            if any(keyword.lower() in col_name.lower() for keyword in TARGET_KEYWORDS):
                if col_name not in initial_objective_names:
                    initial_objective_names.append(col_name)
                    target_info_map[col_name] = {'df_index': idx, 'file_name_prefix': file_name_prefix}
    initial_objective_names = sorted(set(initial_objective_names))
    if not initial_objective_names:
        raise SystemExit("CRITICAL: No target columns identified based on keywords. Exiting.")
    print(f"Objectives detected ({len(initial_objective_names)}): {initial_objective_names}")

    # Detect categorical features (text)
    print("\nDetecting categorical features.")
    globally_identified_cat_cols_set = set()
    for df_idx, df_orig in enumerate(original_dfs):
        if df_orig.empty: continue
        for col_name in df_orig.columns:
            if col_name in initial_objective_names: continue
            if df_orig[col_name].notna().any() and df_orig[col_name].dtype == 'object':
                try:
                    if df_orig[col_name].dropna().apply(lambda x: isinstance(x, str)).any():
                        globally_identified_cat_cols_set.add(col_name)
                except Exception:
                    pass
    final_cat_cols_to_process = sorted(globally_identified_cat_cols_set)
    if final_cat_cols_to_process:
        print(f"Categorical features: {final_cat_cols_to_process}")
    else:
        print("No categorical features detected by text.")

    # Prepare OHE across all dataframes
    CAT_COL_DETAILS = {}
    globally_generated_ohe_col_names = []
    for col_name in final_cat_cols_to_process:
        all_vals = []
        for df_orig in original_dfs:
            if col_name in df_orig.columns:
                all_vals.extend(df_orig[col_name].dropna().astype(str).tolist())
        unique_cats = sorted(list(set(all_vals)))
        if not unique_cats:
            print(f"Warn: No unique categories for '{col_name}'. Skipping OHE."); continue
        ohe_names = [f"{sanitize_filename_component(col_name)}_{sanitize_filename_component(s_val)}" for s_val in unique_cats]
        CAT_COL_DETAILS[col_name] = {'categories': unique_cats, 'one_hot_names': ohe_names}
        globally_generated_ohe_col_names.extend(ohe_names)
    globally_generated_ohe_col_names = sorted(list(set(globally_generated_ohe_col_names)))
    print(f"Total global OHE columns: {len(globally_generated_ohe_col_names)}")

    processed_dfs_after_ohe = []
    target_to_ohe_df_map = {}
    for df_idx, df_orig in enumerate(original_dfs):
        if df_orig.empty:
            processed_dfs_after_ohe.append(df_orig.copy()); continue
        df_updated = df_orig.copy()
        cats_in_df = [c for c in final_cat_cols_to_process if c in df_updated.columns]
        non_cat = df_updated.drop(columns=cats_in_df, errors='ignore') if cats_in_df else df_updated.copy()
        ohe_gen = pd.DataFrame(index=non_cat.index)
        if cats_in_df:
            cat_part = df_updated[cats_in_df].astype(str)
            s_prefixes = {col: sanitize_filename_component(col) for col in cats_in_df}
            ohe_gen = pd.get_dummies(cat_part, columns=cats_in_df, prefix=s_prefixes, prefix_sep='_', dummy_na=False)
        ohe_reindexed = ohe_gen.reindex(columns=globally_generated_ohe_col_names, fill_value=0)
        df_full = pd.concat([non_cat, ohe_reindexed], axis=1)
        processed_dfs_after_ohe.append(df_full)

    for target_name, info in target_info_map.items():
        target_to_ohe_df_map[target_name] = processed_dfs_after_ohe[info['df_index']]
    print("OHE alignment complete.")

    # 2) Train models per objective (RF + XGB), collect diagnostics
    rf_models = {}
    xgb_models = {}
    rf_input_features = None

    # Pool all features present across targets (post-OHE)
    all_feature_columns = set()
    for tname, df_ohe in target_to_ohe_df_map.items():
        cols = [c for c in df_ohe.columns if c not in initial_objective_names]
        all_feature_columns.update(cols)
    all_feature_columns = sorted(list(all_feature_columns))

    # Build defaults for any missing columns
    defaults_for_rf = build_column_defaults(list(target_to_ohe_df_map.values()), all_feature_columns)

    target_metrics_summary = []

    for tname in initial_objective_names:
        df = target_to_ohe_df_map[tname].copy()
        # Ensure all features exist
        for c in all_feature_columns:
            if c not in df.columns:
                df[c] = defaults_for_rf.get(c, 0.0)

        feature_cols = [c for c in all_feature_columns if c not in initial_objective_names]
        X = df[feature_cols].astype(float).values
        y = pd.to_numeric(df[tname], errors='coerce').values

        mask = ~np.isnan(y)
        X = X[mask]; y = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Tune and fit both RF and XGB.
        # The previous version plotted validation curves but still trained the final
        # optimization models with fixed depths. Now the tuned depths are used.
        rf_base_params = dict(n_estimators=N_ESTIMATORS_DEFAULT, random_state=SEED, n_jobs=-1)
        xgb_base_params = dict(n_estimators=N_ESTIMATORS_DEFAULT, random_state=SEED,
                               objective='reg:squarederror', tree_method='hist')

        rf_best_depth = int(get_best_param_from_r2_vc(
            RandomForestRegressor, rf_base_params, X_train_s, y_train,
            "max_depth", RF_MAX_DEPTH_RANGE, current_target_name_for_log=tname
        ))
        xgb_best_depth = int(get_best_param_from_r2_vc(
            XGBRegressor, xgb_base_params, X_train_s, y_train,
            "max_depth", XGB_MAX_DEPTH_RANGE, current_target_name_for_log=tname
        ))

        rf_model = RandomForestRegressor(**rf_base_params, max_depth=rf_best_depth)
        xg_model = XGBRegressor(**xgb_base_params, max_depth=xgb_best_depth)
        rf_model.fit(X_train_s, y_train)
        xg_model.fit(X_train_s, y_train)

        rf_models[tname] = (rf_model, feature_cols, scaler)
        xgb_models[tname] = (xg_model, feature_cols, scaler)
        rf_input_features = feature_cols

        # Diagnostics plot (2x3) for both model classes, using the tuned depths.
        for model_name, model_class, p_range, base_p, best_val in [
            ("RF", RandomForestRegressor, RF_MAX_DEPTH_RANGE, rf_base_params, rf_best_depth),
            ("XGB", XGBRegressor, XGB_MAX_DEPTH_RANGE, xgb_base_params, xgb_best_depth),
        ]:
            main_title = f"{objective_display_name(tname)} — {model_name} Diagnostics"
            fname_base = f"{sanitize_filename_component(tname)}_{model_name}"
            generate_diagnostic_subplots(model_class, base_p, X_train_s, y_train, X_test_s, y_test,
                                         "max_depth", p_range, best_val, main_title, fname_base,
                                         all_learning_curve_data=all_learning_curve_data,
                                         all_actual_vs_pred_data=all_actual_vs_pred_data,
                                         target_name=tname)

        for tag, (mdl, fcols, sc) in [("RF", rf_models[tname]), ("XGB", xgb_models[tname])]:
            ytr = mdl.predict(X_train_s); yte = mdl.predict(X_test_s)
            target_metrics_summary.append({
                "Target": tname, "Model": tag,
                "R2_train": float(r2_score(y_train, ytr)), "R2_test": float(r2_score(y_test, yte)),
                "MSE_train": float(mean_squared_error(y_train, ytr)), "MSE_test": float(mean_squared_error(y_test, yte))
            })

        all_feature_importance_data.append((rf_model, xg_model, feature_cols, tname))

    # === ML EXPORTS ===
    try:
        if target_metrics_summary:
            metrics_df = pd.DataFrame(target_metrics_summary)
            metrics_csv = os.path.join(RESULTS_DIR, "ml_metrics_summary.csv")
            # === NEW: save validation-curve CV metrics to Excel ===
            if VALIDATION_METRICS:
                val_df = pd.DataFrame(VALIDATION_METRICS)
                val_xlsx = os.path.join(RESULTS_DIR, "validation_cv_metrics.xlsx")
                val_df.to_excel(val_xlsx, index=False)
                print(f"Saved: {val_xlsx}")
            metrics_df.to_csv(metrics_csv, index=False); print(f"Saved: {metrics_csv}")
        if all_actual_vs_pred_data:
            combined_4x4_path = os.path.join(RESULTS_DIR, "combined_4x4_actual_vs_predicted.png")
            generate_combined_actual_vs_predicted_4x4(all_actual_vs_pred_data, combined_4x4_path)
        if all_learning_curve_data:
            combined_lc_path = os.path.join(RESULTS_DIR, "combined_4x4_learning_curves.png")
            generate_combined_learning_curves_4x4(all_learning_curve_data, combined_lc_path)
        for (rf_model, xgb_model, f_names, tname_raw) in all_feature_importance_data:
            disp = objective_display_name(tname_raw)
            rf_imp = getattr(rf_model, 'feature_importances_', None) if rf_model is not None else None
            xg_imp = getattr(xgb_model, 'feature_importances_', None) if xgb_model is not None else None
            if rf_imp is None and xg_imp is None: continue
            agg_names = None
            if rf_imp is not None:
                agg_names, rf_vals = aggregate_fi_over_categories(rf_imp, f_names, CAT_COL_DETAILS)
            if xg_imp is not None:
                names_x,  xg_vals = aggregate_fi_over_categories(xg_imp, f_names, CAT_COL_DETAILS)
                if agg_names is None:
                    agg_names, rf_vals = names_x, np.zeros_like(xg_vals)
            if rf_imp is None: rf_vals = np.zeros_like(xg_vals)
            if xg_imp is None: xg_vals = np.zeros_like(rf_vals)
            fig, ax = plt.subplots(figsize=(12, 7)); x = np.arange(len(agg_names)); width = 0.38
            ax.bar(x - width/2, rf_vals, width, label='RF', alpha=0.8)
            ax.bar(x + width/2, xg_vals, width, label='XGB', alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels(agg_names, rotation=45, ha='right')
            ax.set_ylabel('Aggregated Importance'); ax.set_title(f'Feature Importance — {disp}')
            ax.grid(True, linestyle='--', alpha=0.4); ax.legend()
            plt.tight_layout()
            sfname = sanitize_filename_component(disp)
            fi_path = os.path.join(RESULTS_DIR, f"feature_importance_{sfname}.png")
            plt.savefig(fi_path, dpi=300); plt.close(fig); print(f"Saved: {fi_path}")
        try:
            fi2x2_path = os.path.join(RESULTS_DIR, "combined_feature_importance_2x2.png")
            generate_combined_feature_importance_2x2(all_feature_importance_data, CAT_COL_DETAILS, fi2x2_path, top_n=8)
        except Exception as e:
            print(f"Error generating combined 2x2 FI panel: {e}")
    except Exception as e:
        print(f"Error during ML export block: {e}")

    # 3) Set up Pymoo problem (ALL decision variables are discrete)
    # IMPORTANT: Reuse the authoritative CAT_COL_DETAILS from true categorical columns.
    CAT_OHE = CAT_COL_DETAILS.copy()  # <-- prevents mis-parsing like 'PCB k_z'
    ohe_name_set = set(sum([v["one_hot_names"] for v in CAT_OHE.values()], []))
    numeric_non_ohe_features = [c for c in all_feature_columns if c not in ohe_name_set and c not in initial_objective_names]
    DISCRETE_DOMAINS = collect_discrete_domains(original_dfs, numeric_non_ohe_features, initial_objective_names)
    pymoo_opt_vars_numeric_idx = list(DISCRETE_DOMAINS.keys())
    pymoo_opt_vars_categorical = list(CAT_OHE.keys())
    pymoo_opt_vars_all = pymoo_opt_vars_numeric_idx + pymoo_opt_vars_categorical
    bounds_list = []
    num_idx_bounds = index_bounds_from_domains(DISCRETE_DOMAINS)
    for name in pymoo_opt_vars_numeric_idx:
        lo, hi = num_idx_bounds[name]; bounds_list.append([lo, hi])
    for cat in pymoo_opt_vars_categorical:
        bounds_list.append([0, max(0, len(CAT_OHE[cat]['categories']) - 1)])
    bounds_np_array = np.array(bounds_list, dtype=float)

    active_objective_names_for_pymoo = initial_objective_names.copy()
    active_models_for_pymoo, active_feature_cols_for_pymoo, scalers_for_pymoo = [], [], []
    for tname in active_objective_names_for_pymoo:
        if SURROGATE.lower() == "rf":
            mdl, fcols, sc = rf_models[tname]
        else:
            mdl, fcols, sc = xgb_models[tname]
        active_models_for_pymoo.append(mdl)
        active_feature_cols_for_pymoo.append(fcols)
        scalers_for_pymoo.append(sc)

    # Precompute which objectives must be >= 0 (ΔW family)
    nonneg_obj_idx = [i for i, nm in enumerate(active_objective_names_for_pymoo) if is_deltaw_objective(nm)]

    class SurrogateProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(pymoo_opt_vars_all),
                             n_obj=len(active_objective_names_for_pymoo),
                             n_constr=0,
                             xl=bounds_np_array[:, 0],
                             xu=bounds_np_array[:, 1])
        def _evaluate(self, X_batch, out, *args, **kwargs):
            # Indices are integers; keep search discrete
            X_idx = np.rint(X_batch).astype(int)
            n_sols = X_idx.shape[0]
            X_rf = pd.DataFrame(columns=rf_input_features, index=range(n_sols))
            # numeric (index -> true value)
            for name in pymoo_opt_vars_numeric_idx:
                col_i = pymoo_opt_vars_all.index(name)
                X_rf[name] = get_value_from_domain(DISCRETE_DOMAINS, name, X_idx[:, col_i])
            # categorical (index -> one-hot)
            for cat_name, details in CAT_OHE.items():
                col_i = pymoo_opt_vars_all.index(cat_name)
                idx_choices = np.clip(X_idx[:, col_i], 0, len(details['categories']) - 1)
                for ohe_n in details['one_hot_names']:
                    if ohe_n in X_rf.columns: X_rf[ohe_n] = 0.0
                for row_i, choice in enumerate(idx_choices):
                    chosen_ohe = details['one_hot_names'][choice]
                    if chosen_ohe in X_rf.columns:
                        X_rf.loc[row_i, chosen_ohe] = 1.0
            # fill any missing
            for col_fill in rf_input_features:
                if col_fill not in X_rf.columns or X_rf[col_fill].isnull().all():
                    X_rf[col_fill] = defaults_for_rf.get(col_fill, 0.0)
                elif X_rf[col_fill].isnull().any():
                    X_rf[col_fill].fillna(defaults_for_rf.get(col_fill, 0.0), inplace=True)
            # Predict each objective with the scaler and feature order used during its own training.
            # This fixes the major bug in the original code: it used scalers_for_pymoo[0]
            # for every objective, which can distort Pareto predictions.
            predictions = []
            for obj_name, model, fcols, scaler in zip(
                active_objective_names_for_pymoo,
                active_models_for_pymoo,
                active_feature_cols_for_pymoo,
                scalers_for_pymoo,
            ):
                try:
                    X_obj = X_rf.reindex(columns=fcols, fill_value=0.0).copy()
                    for col_fill in fcols:
                        if X_obj[col_fill].isnull().any():
                            X_obj[col_fill] = X_obj[col_fill].fillna(defaults_for_rf.get(col_fill, 0.0))
                    X_scaled_i = scaler.transform(X_obj.values)
                    predictions.append(model.predict(X_scaled_i))
                except Exception as e:
                    raise RuntimeError(f"Failed surrogate prediction for objective '{obj_name}': {e}")
            F_pred = np.column_stack(predictions)
            # --- Enforce physical non-negativity on ΔW-family objectives (fix negative DeltaW_* display) ---
            if nonneg_obj_idx:
                for idx in nonneg_obj_idx:
                    F_pred[:, idx] = np.maximum(F_pred[:, idx], 0.0)
            out["F"] = F_pred

    problem = SurrogateProblem()
    print("MOO problem with discrete domains defined (no interpolation; exact observed values & categories).")

    # 4) Run NSGA-II
    n_obj = len(active_objective_names_for_pymoo)
    print(f"\n4. Running NSGA-II (pop={POP_SIZE}, gens={N_GEN}) over integer index domains...")
    res = None
    if n_obj < 2:
        print("Skipping NSGA-II: Need at least 2 objectives.")
    else:
        try:
            from pymoo.factory import get_sampling, get_crossover, get_mutation
            algo = NSGA2(
                pop_size=POP_SIZE,
                sampling=get_sampling("int_random"),
                crossover=get_crossover("int_sbx", prob=0.9, eta=15),
                mutation=get_mutation("int_pm", eta=20),
                eliminate_duplicates=True
            )
        except Exception:
            algo = NSGA2(pop_size=POP_SIZE, eliminate_duplicates=True)
        res = minimize(problem, algo, termination=get_termination("n_gen", N_GEN), seed=SEED, verbose=True)
        print("NSGA-II optimization finished.")

    pareto_F = res.F if res and hasattr(res, 'F') else None
    pareto_X = res.X if res and hasattr(res, 'X') else None
    if pareto_F is None or len(pareto_F) == 0:
        print("Pareto front is empty. Returning early.")
        return {"res": res, "pareto_F": pareto_F, "pareto_X": pareto_X,
                "objectives": active_objective_names_for_pymoo, "results_dir": RESULTS_DIR}

    # 5) Plotting & Output
    XY_radviz = None
    try:
        def radviz_projection(objs):
            if objs is None or objs.shape[0] == 0: return np.empty((0, 2))
            m = objs.shape[1]
            if m == 0: return np.empty((objs.shape[0], 2))
            obj_min = objs.min(axis=0); obj_range = np.ptp(objs, axis=0)
            obj_range_safe = np.where(obj_range == 0, 1e-9, obj_range)
            norm = (objs - obj_min) / obj_range_safe
            norm = 1.0 - norm
            angles = 2 * np.pi * np.arange(m) / m
            S_matrix = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            numerator = norm @ S_matrix
            denominator = norm.sum(axis=1, keepdims=True)
            denominator_safe = np.where(denominator == 0, 1e-12, denominator)
            return numerator / denominator_safe
        XY_radviz = radviz_projection(pareto_F)
        if XY_radviz is not None and XY_radviz.shape[0] > 0 and pareto_F.shape[1] >= 2:
            num_obj_plot = pareto_F.shape[1]
            angles = 2 * np.pi * np.arange(num_obj_plot) / num_obj_plot
            anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            for i, name in enumerate(active_objective_names_for_pymoo):
                disp_name = objective_display_name(name)
                fig, ax = plt.subplots(figsize=(9, 9))
                sc = ax.scatter(XY_radviz[:, 0], XY_radviz[:, 1], c=pareto_F[:, i], cmap="viridis", s=60, ec="k",
                                alpha=0.7, label="Pareto Designs")
                poly_x = np.append(anchors[:, 0], anchors[0, 0]); poly_y = np.append(anchors[:, 1], anchors[0, 1])
                ax.plot(poly_x, poly_y, "--", c="r", lw=1.5, label="Objective Anchors")
                ax.scatter(anchors[:, 0], anchors[:, 1], marker="^", c="r", s=120)
                for il, lab in enumerate(active_objective_names_for_pymoo):
                    x_anchor, y_anchor = anchors[il, 0], anchors[il, 1]
                    rotation_angle, offset = 0, 0.9
                    if abs(x_anchor) > 0.7 and abs(y_anchor) < 0.3: rotation_angle, offset = 90, 0.8
                    elif abs(y_anchor) > 0.7 and abs(x_anchor) < 0.3: rotation_angle, offset = 0, 0.8
                    ax.text(x_anchor * offset, y_anchor * offset, objective_display_name(lab), c="m", ha="center", va="center",
                            rotation=rotation_angle, fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round'))
                ax.axhline(0, c='k', lw=0.5, ls='--'); ax.axvline(0, c='k', lw=0.5, ls='--'); ax.set_aspect('equal')
                ax.set_title(f"RadViz (Color by {disp_name})"); ax.set_xlabel("RadViz X"); ax.set_ylabel("RadViz Y")
                cbar = plt.colorbar(sc, ax=ax, label=disp_name, location='bottom', shrink=0.85, pad=0.12)
                ax.set_xlabel("RadViz X", labelpad=10)
                ax.legend(loc="upper right")
                plt.tight_layout(pad=2.0)
                sfname = sanitize_filename_component(name)
                plt.savefig(os.path.join(RESULTS_DIR, f"radviz_opt_{sfname}.png")); plt.close(fig)
            combined_radviz_path = os.path.join(RESULTS_DIR, "combined_radviz_by_objectives.png")
            generate_combined_radviz_by_objectives(XY_radviz, pareto_F, active_objective_names_for_pymoo, combined_radviz_path)
    except Exception as e:
        print(f"RadViz error: {e}")

    if pareto_F.shape[1] >= 2:
        for i1, i2 in itertools.combinations(range(pareto_F.shape[1]), 2):
            obj1_name, obj2_name = active_objective_names_for_pymoo[i1], active_objective_names_for_pymoo[i2]
            obj1_disp, obj2_disp = objective_display_name(obj1_name), objective_display_name(obj2_name)
            fig_scatter, ax_scatter = plt.subplots(figsize=(9, 8))
            ax_scatter.scatter(pareto_F[:, i1], pareto_F[:, i2], color='tab:blue',
                               s=60, edgecolor='black', label='Pareto Points', zorder=5)
            ax_scatter.set_xlabel(obj1_disp, fontsize=24); ax_scatter.set_ylabel(obj2_disp, fontsize=24)
            ax_scatter.set_title(f"{obj1_disp} vs {obj2_disp}", fontweight='bold', fontsize=24)
            ax_scatter.legend(fontsize=20, loc='best')
            ax_scatter.tick_params(axis='x', labelsize=20); ax_scatter.tick_params(axis='y', labelsize=20)
            ax_scatter.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(pad=2.0)
            stitle = sanitize_filename_component(f"{obj1_name}_VS_{obj2_name}")
            plt.savefig(os.path.join(RESULTS_DIR, f"scatter_opt_{stitle}.png")); plt.close(fig_scatter)
        pairwise_scatter_combined_raw(pareto_F, active_objective_names_for_pymoo,
                                      os.path.join(RESULTS_DIR, "combined_scatter_opt_plots.png"))

    # Decode Pareto decisions & save
    print("\nSaving Pareto solutions and objectives.")
    X_idx_df = pd.DataFrame(np.rint(pareto_X).astype(int), columns=pymoo_opt_vars_all).copy()
    X_df_out = pd.DataFrame(index=X_idx_df.index)
    for name in pymoo_opt_vars_numeric_idx:
        if name in X_idx_df.columns:
            X_df_out[name] = get_value_from_domain(DISCRETE_DOMAINS, name, X_idx_df[name].values)
    for cat_name, details in CAT_OHE.items():
        if cat_name in X_idx_df.columns:
            idx = np.clip(X_idx_df[cat_name].values.astype(int), 0, len(details['categories']) - 1)
            X_df_out[cat_name] = [details['categories'][i] for i in idx]
    obj_cols = [f"Obj_{sanitize_filename_component(n)}" for n in active_objective_names_for_pymoo]
    F_df = pd.DataFrame(pareto_F, columns=obj_cols)
    radviz_df = pd.DataFrame(XY_radviz, columns=["RadViz_X", "RadViz_Y"]) if XY_radviz is not None and len(XY_radviz) == len(F_df) \
                else pd.DataFrame(index=F_df.index)
    results_df = pd.concat([radviz_df, F_df, X_df_out], axis=1)
    csv_path = os.path.join(RESULTS_DIR, "pareto_solutions_designs_opt.csv")
    results_df.to_csv(csv_path, index=False); print(f"Saved: {csv_path}")
    excel_obj_path = os.path.join(RESULTS_DIR, "pareto_objectives_only.xlsx")
    F_df.to_excel(excel_obj_path, index=False); print(f"Saved: {excel_obj_path}")


    # === NEW: Objective correlation heatmaps (Pareto-only) ===
    try:
        hm_path_pearson = os.path.join(RESULTS_DIR, "objective_correlation_heatmap_pearson.png")
        plot_objective_corr_heatmap(
            F_df=F_df,
            raw_obj_names=active_objective_names_for_pymoo,
            out_path=hm_path_pearson,
            method="pearson",
            title="Objective Correlation Heatmap (Pearson)",
            annotate=True,
            triangular=False
        )

    except Exception as e:
        print(f"[Heatmap] Failed to generate: {e}")



    # 6) Net Flow Method over Pareto F_df (export)
    try:
        print("\n6. Running Net Flow Method (NFM) ranking on Pareto objectives...")
        df_obj = F_df.reset_index().rename(columns={"index": "Sol_Index"})
        obj_cols = [c for c in df_obj.columns if c.startswith("Obj_")]
        if obj_cols:
            F = df_obj[obj_cols].to_numpy(dtype=float)
            weights_map = WEIGHTS if isinstance(WEIGHTS, dict) else None
            q_map = Q_EXPL if isinstance(Q_EXPL, dict) else None
            p_map = P_EXPL if isinstance(P_EXPL, dict) else None
            v_map = V_EXPL if isinstance(V_EXPL, dict) else None
            F_aligned, W, Q, P, V, S = build_netflow_arrays(
                F=F, obj_names=obj_cols, sense_map=SENSE,
                weights_map=weights_map, q_map=q_map, p_map=p_map, v_map=v_map,
                q_frac=Q_FRACTION, p_frac=P_FRACTION, v_frac=V_FRACTION
            )
            scores = net_flow(F_aligned, W, Q, P, V, S)
            order = np.argsort(scores)[::-1]
            ranks = np.empty_like(order); ranks[order] = np.arange(1, len(order) + 1)
            df_nf = df_obj.copy()
            df_nf["NetFlow_Score"] = scores; df_nf["NetFlow_Rank"]  = ranks
            df_nf_sorted = df_nf.sort_values(by="NetFlow_Rank", ascending=True).reset_index(drop=True)
            nf_xlsx = os.path.join(RESULTS_DIR, "netflow_results.xlsx")
            with pd.ExcelWriter(nf_xlsx, engine="xlsxwriter") as writer:
                df_nf.to_excel(writer,        sheet_name="Unsorted", index=False)
                df_nf_sorted.to_excel(writer, sheet_name="Sorted",   index=False)
            print(f"NFM: saved {nf_xlsx}")
            config_used = {
                "objective_columns": obj_cols,
                "sense":   {name: SENSE.get(name, "min") for name in obj_cols},
                "weights": {name: float(w) for name, w in zip(obj_cols, W)},
                "Q":       {name: float(q) for name, q in zip(obj_cols, Q)},
                "P":       {name: float(p) for name, p in zip(obj_cols, P)},
                "V":       {name: float(v) for name, v in zip(obj_cols, V)},
                "notes": "If WEIGHTS/Q/P/V not provided, auto-scaled from objective ranges."
            }
            with open(os.path.join(RESULTS_DIR, "netflow_config.json"), "w", encoding="utf-8") as f:
                json.dump(config_used, f, indent=2)
            print("NFM: saved netflow_config.json")
            # NEW: Merge NFM with decoded designs/objectives
            merged_unsorted = df_nf.merge(results_df.reset_index().rename(columns={"index": "Sol_Index"}),
                                          on="Sol_Index", how="left")
            merged_sorted = df_nf_sorted.merge(results_df.reset_index().rename(columns={"index": "Sol_Index"}),
                                               on="Sol_Index", how="left")
            nfm_designs_xlsx = os.path.join(RESULTS_DIR, "nfm_with_designs.xlsx")
            with pd.ExcelWriter(nfm_designs_xlsx, engine="xlsxwriter") as writer:
                merged_unsorted.to_excel(writer, sheet_name="Unsorted", index=False)
                merged_sorted.to_excel(writer,  sheet_name="Sorted",   index=False)
            print(f"NFM+Designs: saved {nfm_designs_xlsx}")
            merged_sorted.to_csv(os.path.join(RESULTS_DIR, "nfm_with_designs_sorted.csv"), index=False)
            # NFM plots
            nfm_label_map = {}
            for raw_name in active_objective_names_for_pymoo:
                base = sanitize_filename_component(raw_name)
                obj_col = f"Obj_{base}"
                pretty = objective_display_name(raw_name)
                overrides = {
                    "ΔW_BGA": r"$\Delta W_{\mathrm{BGA}}$",
                    "ΔW_Bump": r"$\Delta W_{\mathrm{bump}}$",
                    "ΔW_bump": r"$\Delta W_{\mathrm{bump}}$",
                    "ΔW_C4": r"$\Delta W_{\mathrm{C4}}$",
                    "θ_JA": r"$\Theta_{JA}$",
                    "Theta_JA": r"$\Theta_{JA}$",
                    "θJA": r"$\Theta_{JA}$",
                }
                label = overrides.get(pretty, pretty)
                nfm_label_map[obj_col] = label
            plot_scores_hist(scores, os.path.join(RESULTS_DIR, "netflow_scores_hist.png"))
            pairwise_scatter(df_nf, obj_cols, ranks, RESULTS_DIR, label_map=nfm_label_map)
            pairwise_scatter_combined_nfm(df_nf, obj_cols, ranks,
                                          os.path.join(RESULTS_DIR, "netflow_pairwise_scatter_combined.png"),
                                          label_map=nfm_label_map)
            print("NFM: plots saved.")
            try:
                if XY_radviz is not None and len(XY_radviz) == len(F_df):
                    champion_idx = int(df_nf_sorted.iloc[0]["Sol_Index"]) if "Sol_Index" in df_nf_sorted.columns else int(df_nf_sorted.index[0])
                    out_panel = os.path.join(RESULTS_DIR, "combined_radviz_by_objectives_champion.png")
                    generate_combined_radviz_by_objectives_champion(
                        XY_radviz, F_df.values, active_objective_names_for_pymoo,
                        out_panel, champion_index=champion_idx, max_subplots=6
                    )
                else:
                    print("XY_radviz unavailable or length mismatch; skipping combined RadViz (with Champion).")
            except Exception as e:
                print(f"Combined RadViz (with Champion) failed: {e}")
        else:
            print("No objective columns found for NFM; skipping.")
    except Exception as e:
        print(f"NFM export failed: {e}")

    return {
        "res": res,
        "pareto_F": pareto_F,
        "pareto_X": pareto_X,
        "objectives": active_objective_names_for_pymoo,
        "results_dir": RESULTS_DIR,
    }

def generate_combined_feature_importance_2x2(all_feature_importance_data, cat_col_details, output_path, top_n=8):
    targets = []
    for (rf_model, xgb_model, f_names, tname_raw) in all_feature_importance_data:
        if tname_raw not in targets: targets.append(tname_raw)
    targets = targets[:4]
    if not targets:
        print("No feature-importance data available for combined 2x2 plot."); return
    fig, axs = plt.subplots(2, 2, figsize=(18, 12)); axs = axs.ravel()
    for ax, tname_raw in zip(axs, targets):
        tup = None
        for item in all_feature_importance_data:
            if item[3] == tname_raw: tup = item
        if tup is None: ax.axis('off'); continue
        rf_model, xgb_model, f_names, _ = tup
        rf_imp = getattr(rf_model, 'feature_importances_', None) if rf_model is not None else None
        xg_imp = getattr(xgb_model, 'feature_importances_', None) if xgb_model is not None else None
        if rf_imp is None and xg_imp is None: ax.axis('off'); continue
        agg_names = None
        if rf_imp is not None:
            agg_names, rf_vals = aggregate_fi_over_categories(rf_imp, f_names, cat_col_details)
        if xg_imp is not None:
            names_x,  xg_vals = aggregate_fi_over_categories(xg_imp, f_names, cat_col_details)
            if agg_names is None:
                agg_names, rf_vals = names_x, np.zeros_like(xg_vals)
        if rf_imp is None: rf_vals = np.zeros_like(xg_vals)
        if xg_imp is None: xg_vals = np.zeros_like(rf_vals)
        comb = rf_vals + xg_vals; order = np.argsort(comb)[::-1][:top_n]
        names_top = [agg_names[i] for i in order]; rf_top = rf_vals[order]; xg_top = xg_vals[order]
        idx = np.arange(len(names_top)); bw = 0.38
        ax.bar(idx - bw/2, rf_top, bw, label='RF', alpha=0.85)
        ax.bar(idx + bw/2, xg_top, bw, label='XGB', alpha=0.85)
        ax.set_xticks(idx); ax.set_xticklabels(names_top, rotation=35, ha='right')
        ax.set_ylabel('Aggregated Importance')
        ax.set_title(f"Feature Importance — {objective_display_name(tname_raw)}", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4); ax.legend()
    for j in range(len(targets), 4): axs[j].axis('off')
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    try:
        print(f"Saved combined 2x2 Feature Importances plot to {output_path}")
    except Exception: pass

if __name__ == "__main__":
    run_combined_workflow()
