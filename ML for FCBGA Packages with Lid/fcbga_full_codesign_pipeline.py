#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCBGA with Lid — Full Co-Design Pipeline
=========================================

End-to-end workflow aligned with the example thermo-mechanical co-design manuscript:

  1. Data cleaning and harmonized column names (Assembly + SJR)
  2. Merge paired simulation rows (Design_ID) for correlation analysis
  3. Train ANN / CatBoost surrogates per target (dataset-specific features)
  4. Validation curves, learning curves, parity plots, feature importance
  5. Combined 5×4 actual-vs-predicted and learning-curve grids (ANN/CatBoost)
  6. Objective correlation heatmap (Pearson)
  7. NSGA-II multi-objective optimization on discrete design variables
  8. Net Flow Method (NFM) ranking + histogram and pairwise NFM plots
  9. RadViz (per objective + combined with champion marker)
 10. Champion design export + nearest-FEA validation proxy table

Run:
    python scripts/fcbga_full_codesign_pipeline.py

Term-paper note:
  Replace the validation-proxy step with a full ANSYS re-simulation of the
  champion design when high-fidelity confirmation is required.
"""

from __future__ import annotations

import itertools
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split, validation_curve
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"codesign_run_{RUN_STAMP}"
RUN_FIG = FIGURES_DIR / f"codesign_run_{RUN_STAMP}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
RUN_FIG.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ESTIMATORS = 100  # CatBoost iterations

# ANN — improved defaults (regularized, target-scaled, CV-tuned)
ANN_HIDDEN_LAYERS = (64, 32)
ANN_MAX_ITER = 3000
ANN_ALPHA_DEFAULT = 1e-3
ANN_LR_DEFAULT = 1e-3
ANN_LAYER_SWEEP = [(16,), (32,), (32, 16), (64, 32)]
ANN_ALPHA_SWEEP = [1e-4, 1e-3, 1e-2]
ANN_LR_SWEEP = [1e-4, 1e-3]
BASELINE_RUN_DIR = RESULTS_DIR / "codesign_run_20260706_201315"

# CatBoost
CATBOOST_DEPTH = 4
# CatBoost validation-curve sweep over tree depth
CATBOOST_DEPTH_SWEEP = [2, 3, 4, 5, 6]

# NSGA-II (reduce for quick runs; increase toward 500/500 for publication)
POP_SIZE = 150
N_GEN = 100

NFM_Q_FRAC, NFM_P_FRAC, NFM_V_FRAC = 0.05, 0.15, 0.30

ALL_TARGETS = [
    "ELK stress",
    "Warpage Post UF cure",
    "Warpage post lid attach",
    "DeltaW_BGA",
    "DeltaW_bump",
]
ASSEMBLY_TARGETS = ALL_TARGETS[:3]
SJR_TARGETS = ALL_TARGETS[3:]

COLUMN_ALIASES = {
    "Lid thicknes": "Lid thickness",
    "lid thickness": "Lid thickness",
    "Warpage post lid atach": "Warpage post lid attach",
    "Bump solder material": "Cu-pillar bump solder material",
}

MERGE_DESIGN_COLS = [
    "Cu-pillar pitch–diameter",
    "Bulk silicon thickness",
    "Bump solder height",
    "Substrate core thickness",
    "Substrate core (E, CTE)",
    "UF (E, CTE)",
    "Lid foot width",
    "Lid thickness",
    "Cu-pillar bump solder material",
]

VALIDATION_CV_ROWS: list[dict] = []
ALL_PRED_DATA: list[dict] = []
ALL_LC_DATA: list[dict] = []


# =============================================================================
# DATA CLEANING
# =============================================================================


def load_sources(sources: list[Path], sheet: str | None = None) -> pd.DataFrame:
    for path in sources:
        if path.exists():
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            return pd.read_excel(path, sheet_name=sheet or 0)
    raise FileNotFoundError(f"No file found: {sources}")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()
    return out.rename(columns=COLUMN_ALIASES)


def drop_constant_inputs(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    inputs = [c for c in df.columns if c not in ALL_TARGETS and c != "Design_ID"]
    drop = [c for c in inputs if df[c].nunique() <= 1]
    if drop:
        print(f"  Removed constant columns: {', '.join(drop)}")
    return df.drop(columns=drop, errors="ignore")


def load_assembly() -> pd.DataFrame:
    df = clean_df(
        load_sources(
            [DATA_DIR / "cleaned_fcbga_lid_data.csv", DATA_DIR / "2D_assembly_lid_300_v4.xlsx"],
            sheet="2D_assembly_lid_300",
        )
    )
    return drop_constant_inputs(df, ASSEMBLY_TARGETS)


def load_sjr() -> pd.DataFrame:
    df = clean_df(
        load_sources(
            [DATA_DIR / "cleaned_sjr_lid_data.csv", DATA_DIR / "2D_SJR_lid_300_v4.xlsx"],
        )
    )
    return drop_constant_inputs(df, SJR_TARGETS)


def build_master_table(assembly: pd.DataFrame, sjr: pd.DataFrame) -> pd.DataFrame:
    """
    Pair Assembly and SJR rows by Design_ID (same DOE index across simulations).
    This preserves all 300 design points for correlation / validation lookup.
    """
    a = assembly.copy()
    s = sjr.copy()
    a["Design_ID"] = np.arange(1, len(a) + 1)
    s["Design_ID"] = np.arange(1, len(s) + 1)

    a_cols = [c for c in MERGE_DESIGN_COLS if c in a.columns] + ASSEMBLY_TARGETS + ["Design_ID"]
    s_extra = ["Design_ID", "BGA solder material"] + [c for c in SJR_TARGETS if c in s.columns]
    s_extra = [c for c in s_extra if c in s.columns]

    master = pd.merge(a[a_cols], s[s_extra], on="Design_ID", how="inner")
    master.to_csv(RUN_DIR / "master_merged_300_designs.csv", index=False)
    print(f"  Master table: {len(master)} paired design points")
    return master


# =============================================================================
# ML HELPERS
# =============================================================================


def make_preprocessor(df: pd.DataFrame, inputs: list[str]) -> ColumnTransformer:
    cat = df[inputs].select_dtypes(include=["object", "string"]).columns.tolist()
    num = df[inputs].select_dtypes(exclude=["object", "string"]).columns.tolist()
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num),
        ]
    )


def ann_pipe(
    pre,
    hidden_layer_sizes=ANN_HIDDEN_LAYERS,
    alpha=ANN_ALPHA_DEFAULT,
    learning_rate_init=ANN_LR_DEFAULT,
) -> Pipeline:
    """ANN surrogate: scaled numeric inputs + scaled target + regularized MLP."""
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=ANN_MAX_ITER,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=SEED,
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=mlp,
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )


def tune_ann_hyperparams(pre, X, y, target: str) -> dict:
    """Select ANN architecture / regularization by 5-fold CV R²."""
    best_score = -np.inf
    best = {
        "hidden_layer_sizes": ANN_HIDDEN_LAYERS,
        "alpha": ANN_ALPHA_DEFAULT,
        "learning_rate_init": ANN_LR_DEFAULT,
        "CV_R2": np.nan,
    }
    for layers in ANN_LAYER_SWEEP:
        for alpha in ANN_ALPHA_SWEEP:
            for lr in ANN_LR_SWEEP:
                pipe = ann_pipe(pre, hidden_layer_sizes=layers, alpha=alpha, learning_rate_init=lr)
                scores = cross_val_score(pipe, X, y, cv=CV_FOLDS, scoring="r2", n_jobs=1)
                mean_sc = float(scores.mean())
                VALIDATION_CV_ROWS.append(
                    {
                        "Target": target,
                        "Model": "ANN",
                        "ParamName": "hidden_layer_sizes|alpha|lr",
                        "ParamValue": f"{layers}|{alpha}|{lr}",
                        "CV_Train_R2_Mean": np.nan,
                        "CV_Test_R2_Mean": mean_sc,
                    }
                )
                if mean_sc > best_score:
                    best_score = mean_sc
                    best = {
                        "hidden_layer_sizes": layers,
                        "alpha": alpha,
                        "learning_rate_init": lr,
                        "CV_R2": mean_sc,
                    }
    print(
        f"  ANN tuned for {target}: layers={best['hidden_layer_sizes']}, "
        f"alpha={best['alpha']}, lr={best['learning_rate_init']}, CV R²={best['CV_R2']:.4f}"
    )
    return best


def catboost_pipe(pre) -> Pipeline:
    """CatBoost gradient-boosting surrogate."""
    return Pipeline(
        [
            ("pre", pre),
            (
                "regressor",
                CatBoostRegressor(
                    iterations=N_ESTIMATORS,
                    depth=CATBOOST_DEPTH,
                    loss_function="RMSE",
                    random_seed=SEED,
                    verbose=0,
                    allow_writing_files=False,
                ),
            ),
        ]
    )


# Model registry: name -> (pipeline factory, sweep param name, sweep values, sklearn class name)
MODEL_SPECS = {
    "ANN": {
        "factory": ann_pipe,
        "param_name": "regressor__regressor__hidden_layer_sizes",
        "param_label": "hidden_layer_sizes",
        "param_range": ANN_LAYER_SWEEP,
        "model_type": "MLPRegressor",
    },
    "CatBoost": {
        "factory": catboost_pipe,
        "param_name": "regressor__depth",
        "param_label": "depth",
        "param_range": CATBOOST_DEPTH_SWEEP,
        "model_type": "CatBoostRegressor",
    },
}


def quality_label(r2: float) -> str:
    if r2 >= 0.90:
        return "Very good"
    if r2 >= 0.75:
        return "Good"
    if r2 >= 0.50:
        return "Medium"
    return "Weak"


def log_val_curve(pipe, X, y, target, model_name):
    spec = MODEL_SPECS[model_name]
    for v in spec["param_range"]:
        pipe.set_params(**{spec["param_name"]: v})
        tr, te = validation_curve(
            pipe, X, y, param_name=spec["param_name"], param_range=[v],
            cv=CV_FOLDS, scoring="r2", n_jobs=1,
        )
        VALIDATION_CV_ROWS.append(
            {
                "Target": target,
                "Model": model_name,
                "ParamName": spec["param_label"],
                "ParamValue": str(v),
                "CV_Train_R2_Mean": float(tr.mean()),
                "CV_Test_R2_Mean": float(te.mean()),
            }
        )


def save_diagnostics(pipe, X_train, X_test, y_train, y_test, target, model_name, dataset, ann_params=None):
    spec = MODEL_SPECS[model_name]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pre = pipe.named_steps["pre"]
    values = spec["param_range"]
    tr_m, te_m = [], []
    if model_name == "ANN" and ann_params:
        sweep = ann_pipe(
            pre,
            alpha=ann_params["alpha"],
            learning_rate_init=ann_params["learning_rate_init"],
        )
        param_name = spec["param_name"]
    else:
        sweep = spec["factory"](pre)
        param_name = spec["param_name"]
    for v in values:
        sweep.set_params(**{param_name: v})
        tr, te = validation_curve(
            sweep, X_train, y_train, param_name=param_name,
            param_range=[v], cv=CV_FOLDS, scoring="r2", n_jobs=1,
        )
        tr_m.append(tr.mean())
        te_m.append(te.mean())
    x_pos = range(len(values))
    axes[0].plot(x_pos, tr_m, "o-", label="Train CV")
    axes[0].plot(x_pos, te_m, "o-", label="Val CV")
    axes[0].set_xticks(list(x_pos))
    axes[0].set_xticklabels([str(v) for v in values], rotation=20, fontsize=8)
    axes[0].set_xlabel(spec["param_label"])
    axes[0].set_title("Validation curve (R²)")
    axes[0].legend()

    sizes, tr_lc, te_lc = learning_curve(
        pipe, X_train, y_train, cv=CV_FOLDS, scoring="r2",
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1,
    )
    axes[1].plot(sizes, tr_lc.mean(1), "o-", label="Train")
    axes[1].plot(sizes, te_lc.mean(1), "o-", label="CV")
    axes[1].set_title("Learning curve")
    axes[1].legend()

    y_pred = pipe.predict(X_test)
    axes[2].scatter(y_test, y_pred, edgecolors="k", alpha=0.7)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[2].plot([lo, hi], [lo, hi], "r--")
    axes[2].set_title("Test parity")

    safe = target.replace(" ", "_")
    fig.suptitle(f"{dataset} — {target} — {model_name}")
    fig.tight_layout()
    fig.savefig(RUN_FIG / f"diag_{dataset}_{safe}_{model_name.replace(' ', '_')}.png", dpi=200)
    plt.close(fig)


def train_target(df: pd.DataFrame, target: str, dataset: str) -> dict:
    inputs = [c for c in df.columns if c not in ALL_TARGETS and c != "Design_ID"]
    X, y = df[inputs], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    pre = make_preprocessor(df, inputs)

    best = None
    rows = []
    for name in ("ANN", "CatBoost"):
        spec = MODEL_SPECS[name]
        ann_params = None
        if name == "ANN":
            ann_params = tune_ann_hyperparams(pre, X_tr, y_tr, target)
            pipe = ann_pipe(
                pre,
                hidden_layer_sizes=ann_params["hidden_layer_sizes"],
                alpha=ann_params["alpha"],
                learning_rate_init=ann_params["learning_rate_init"],
            )
        else:
            pipe = spec["factory"](pre)
            log_val_curve(spec["factory"](pre), X_tr, y_tr, target, name)
        pipe.fit(X_tr, y_tr)
        y_tr_pred = pipe.predict(X_tr)
        y_pr = pipe.predict(X_te)
        model_type = spec["model_type"]
        ALL_PRED_DATA.append(
            {
                "target_name": target,
                "dataset": dataset,
                "model_type": model_type,
                "y_train_true": np.asarray(y_tr),
                "y_train_pred": np.asarray(y_tr_pred),
                "y_test_true": np.asarray(y_te),
                "y_test_pred": np.asarray(y_pr),
            }
        )
        train_sizes = np.linspace(0.2, 1.0, 5)
        for scoring, metric_type in [("r2", "r2"), ("neg_mean_squared_error", "mse")]:
            sizes, tr_sc, te_sc = learning_curve(
                pipe, X_tr, y_tr, cv=CV_FOLDS, scoring=scoring,
                train_sizes=train_sizes, n_jobs=1,
            )
            ALL_LC_DATA.append(
                {
                    "target_name": target,
                    "dataset": dataset,
                    "model_type": model_type,
                    "metric_type": metric_type,
                    "train_sizes": sizes,
                    "train_scores": tr_sc,
                    "test_scores": te_sc,
                }
            )
        row = {
            "Dataset": dataset,
            "Target": target,
            "Model": name,
            "Train_R2": r2_score(y_tr, pipe.predict(X_tr)),
            "Test_R2": r2_score(y_te, y_pr),
            "Train_MSE": mean_squared_error(y_tr, pipe.predict(X_tr)),
            "Test_MSE": mean_squared_error(y_te, y_pr),
            "pipeline": pipe,
            "inputs": inputs,
            "X_test": X_te,
            "y_test": y_te,
        }
        if ann_params:
            row["ANN_Layers"] = str(ann_params["hidden_layer_sizes"])
            row["ANN_Alpha"] = ann_params["alpha"]
            row["ANN_LR"] = ann_params["learning_rate_init"]
            row["ANN_CV_R2"] = ann_params["CV_R2"]
        rows.append(row)
        save_diagnostics(pipe, X_tr, X_te, y_tr, y_te, target, name, dataset, ann_params=ann_params)
        if best is None or row["Test_R2"] > best["Test_R2"]:
            best = row

    best_row = {
        k: v for k, v in best.items() if k not in ("pipeline", "inputs", "X_test", "y_test")
    }
    best_row["Model_Quality"] = quality_label(best_row["Test_R2"])
    return {"best": best, "candidates": rows, "summary": best_row}


def _clean_feature_name(name: str) -> str:
    """Strip ColumnTransformer prefixes for readable publication labels."""
    s = str(name)
    for prefix in ("cat__", "num__", "remainder__", "pre__"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def plot_feature_importance(best: dict, target: str, dataset: str) -> dict | None:
    """
    Feature importance for the best surrogate.
    CatBoost exposes feature_importances_ directly; for the ANN (MLP) we fall
    back to permutation importance on the hold-out test set.
    Returns a dict usable by the combined 5-target grid, or None on failure.
    """
    pipe: Pipeline = best["pipeline"]
    reg = pipe.named_steps["regressor"]
    if hasattr(reg, "regressor"):
        reg = reg.regressor
    pre = pipe.named_steps["pre"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        return None

    if hasattr(reg, "feature_importances_"):
        imp = np.asarray(reg.feature_importances_, dtype=float)
        method = "CatBoost importance"
    else:
        # Permutation importance on raw input columns for the ANN
        try:
            result = permutation_importance(
                pipe, best["X_test"], best["y_test"],
                n_repeats=10, random_state=SEED, scoring="r2",
            )
        except Exception:
            return None
        imp = np.asarray(result.importances_mean, dtype=float)
        names = np.asarray(best["inputs"])
        method = "Permutation importance (ANN)"

    names = np.asarray(names)
    idx = np.argsort(imp)[::-1][: min(10, len(imp))]
    top_names = [_clean_feature_name(names[i]) for i in idx]
    top_imp = imp[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(idx)), top_imp[::-1], color="steelblue", edgecolor="k", linewidth=0.4)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance — {target} ({method})")
    _bold_axis_text(ax, title_size=11, label_size=10, tick_size=8)
    fig.tight_layout()
    safe = target.replace(" ", "_")
    fig.savefig(RUN_FIG / f"fi_{dataset}_{safe}.png", dpi=200)
    plt.close(fig)

    return {
        "target": target,
        "dataset": dataset,
        "method": method,
        "feature_names": top_names,
        "importances": top_imp,
    }


def generate_combined_feature_importance_grid(
    importance_data: list[dict], output_path: Path, top_n: int = 8
):
    """Combined feature-importance panels for all five targets (publication grid)."""
    by_target = {d["target"]: d for d in importance_data}
    n_targets = len(ALL_TARGETS)
    nrows, ncols = (3, 2) if n_targets > 3 else (1, n_targets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.cm.viridis
    for i, target in enumerate(ALL_TARGETS):
        ax = axes[i]
        entry = by_target.get(target)
        if entry is None:
            ax.axis("off")
            continue
        names = list(entry["feature_names"][:top_n])
        vals = np.asarray(entry["importances"][:top_n], dtype=float)
        # plot most important at top
        names_r, vals_r = names[::-1], vals[::-1]
        colors = cmap(np.linspace(0.35, 0.9, len(vals_r)))
        ax.barh(range(len(vals_r)), vals_r, color=colors, edgecolor="k", linewidth=0.35)
        ax.set_yticks(range(len(vals_r)))
        ax.set_yticklabels(names_r, fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"{target}", fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        for label in ax.get_yticklabels() + ax.get_xticklabels():
            label.set_fontweight("bold")
        ax.xaxis.label.set_fontweight("bold")
    for j in range(n_targets, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        "Feature importance — all five targets (CatBoost)",
        fontweight="bold",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined feature-importance grid -> {output_path}")


def plot_pareto_pairwise_projections(
    pareto_f: np.ndarray,
    obj_names: list[str],
    scores: np.ndarray | None = None,
    output_path: Path | None = None,
):
    """
    Pairwise objective projections of the Pareto set.
    Points are colored by NFM net-flow score when provided (viridis).
    """
    pairs = list(itertools.combinations(range(len(obj_names)), 2))
    n_pairs = len(pairs)
    ncols = 5 if n_pairs > 6 else 3
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    color_vals = scores if scores is not None else np.zeros(len(pareto_f))
    sc_ref = None
    for ax, (i, j) in zip(axes, pairs):
        sc_ref = ax.scatter(
            pareto_f[:, i], pareto_f[:, j],
            c=color_vals, cmap="viridis", s=28, alpha=0.85, edgecolors="k", linewidths=0.25,
        )
        ax.set_xlabel(obj_names[i])
        ax.set_ylabel(obj_names[j])
        ax.grid(True, linestyle="--", alpha=0.4)
        _bold_axis_text(ax, title_size=10, label_size=9, tick_size=8)
    for k in range(n_pairs, len(axes)):
        axes[k].axis("off")
    if scores is not None and sc_ref is not None:
        cbar = fig.colorbar(sc_ref, ax=list(axes[:n_pairs]), fraction=0.02, pad=0.02)
        cbar.set_label("NFM net-flow score", fontweight="bold")
        cbar.ax.yaxis.label.set_fontweight("bold")
        for t in cbar.ax.get_yticklabels():
            t.set_fontweight("bold")
    fig.suptitle(
        "Pareto-optimal solutions — pairwise objective projections (NSGA-II)",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])
    out = output_path or (RUN_FIG / "combined_scatter_pareto.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pareto pairwise projections -> {out}")


def plot_corr_heatmap(master: pd.DataFrame):
    corr = master[ALL_TARGETS].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(ALL_TARGETS)))
    ax.set_yticks(range(len(ALL_TARGETS)))
    ax.set_xticklabels(ALL_TARGETS, rotation=35, ha="right")
    ax.set_yticklabels(ALL_TARGETS)
    for i in range(len(ALL_TARGETS)):
        for j in range(len(ALL_TARGETS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Objective correlation — simulation dataset (n=300)")
    fig.tight_layout()
    for folder in (RUN_FIG, FIGURES_DIR):
        fig.savefig(folder / "objective_correlation_simulation_data.png", dpi=300)
    plt.close(fig)
    corr.to_csv(RUN_DIR / "objective_correlation_matrix.csv")


# =============================================================================
# PROFESSOR-STYLE COMBINED FIGURES (5×4 grids, NFM, RadViz)
# =============================================================================


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


# Distinct colors for train vs test parity panels
TRAIN_COLOR = "#1f77b4"  # steel blue
TEST_COLOR = "#ff7f0e"   # orange


def _bold_axis_text(ax, title_size: float = 10, label_size: float = 9, tick_size: float = 8):
    """Bold titles, axis labels, and tick values for publication figures."""
    ax.title.set_fontweight("bold")
    ax.title.set_fontsize(title_size)
    ax.xaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def _plot_single_actual_vs_predicted(
    ax, y_true, y_pred, dataset_name_str: str, color: str = TRAIN_COLOR
):
    if y_true is None or y_pred is None:
        ax.text(0.5, 0.5, "Data N/A", ha="center", va="center", transform=ax.transAxes)
        return
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean, y_pred_clean = y_true[mask], y_pred[mask]
    if len(y_true_clean) == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.scatter(
        y_true_clean, y_pred_clean,
        alpha=0.65, edgecolors="k", s=40, c=color, linewidths=0.4,
    )
    lo = min(y_true_clean.min(), y_pred_clean.min())
    hi = max(y_true_clean.max(), y_pred_clean.max())
    margin = (hi - lo) * 0.05 if hi > lo else 0.1
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(dataset_name_str)
    ax.grid(True, linestyle="--", alpha=0.4)
    _bold_axis_text(ax)


def generate_combined_actual_vs_predicted_grid(all_pred_data: list[dict], output_path: Path):
    """5×4 grid: ANN/CatBoost × train/test for all five targets."""
    n_targets = len(ALL_TARGETS)
    fig, axs = plt.subplots(n_targets, 4, figsize=(20, 4 * n_targets))
    if n_targets == 1:
        axs = axs.reshape(1, -1)
    columns = [
        ("MLPRegressor", "train", "ANN | Train", TRAIN_COLOR),
        ("CatBoostRegressor", "train", "CatBoost | Train", TRAIN_COLOR),
        ("MLPRegressor", "test", "ANN | Test", TEST_COLOR),
        ("CatBoostRegressor", "test", "CatBoost | Test", TEST_COLOR),
    ]
    for row, target in enumerate(ALL_TARGETS):
        for col, (model_type, split, col_title, color) in enumerate(columns):
            ax = axs[row, col]
            entry = next(
                (d for d in all_pred_data if d["target_name"] == target and d["model_type"] == model_type),
                None,
            )
            if entry is None:
                ax.axis("off")
                continue
            y_true = entry[f"y_{split}_true"]
            y_pred = entry[f"y_{split}_pred"]
            _plot_single_actual_vs_predicted(ax, y_true, y_pred, col_title, color=color)
            if col == 0:
                ax.set_ylabel(f"{target}\nPredicted")
                ax.yaxis.label.set_fontweight("bold")
                ax.yaxis.label.set_fontsize(9)
    train_patch = mpatches.Patch(color=TRAIN_COLOR, label="Train")
    test_patch = mpatches.Patch(color=TEST_COLOR, label="Test")
    fig.legend(
        handles=[train_patch, test_patch],
        loc="upper right",
        fontsize=10,
        frameon=True,
        prop={"weight": "bold"},
    )
    fig.suptitle(
        "Actual vs Predicted — All Targets (ANN / CatBoost, Train / Test)",
        fontweight="bold",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined parity grid -> {output_path}")


def generate_combined_learning_curves_grid(all_lc_data: list[dict], output_path: Path):
    """5×4 grid: ANN/CatBoost × R²/MSE learning curves for all five targets."""
    n_targets = len(ALL_TARGETS)
    fig, axs = plt.subplots(n_targets, 4, figsize=(20, 4 * n_targets))
    if n_targets == 1:
        axs = axs.reshape(1, -1)
    columns = [
        ("MLPRegressor", "r2", "ANN | R²"),
        ("MLPRegressor", "mse", "ANN | MSE"),
        ("CatBoostRegressor", "r2", "CatBoost | R²"),
        ("CatBoostRegressor", "mse", "CatBoost | MSE"),
    ]
    for row, target in enumerate(ALL_TARGETS):
        for col, (model_type, metric_type, col_title) in enumerate(columns):
            ax = axs[row, col]
            entry = next(
                (
                    d
                    for d in all_lc_data
                    if d["target_name"] == target
                    and d["model_type"] == model_type
                    and d["metric_type"] == metric_type
                ),
                None,
            )
            if entry is None:
                ax.axis("off")
                continue
            train_sizes = entry["train_sizes"]
            tr = entry["train_scores"]
            te = entry["test_scores"]
            if metric_type == "mse":
                tr, te = -tr, -te
            tr_m, tr_s = tr.mean(1), tr.std(1)
            te_m, te_s = te.mean(1), te.std(1)
            ax.plot(train_sizes, tr_m, "o-", color="darkorange", label="Train", lw=1.5)
            ax.fill_between(train_sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color="darkorange")
            ax.plot(train_sizes, te_m, "o-", color="navy", label="CV", lw=1.5)
            ax.fill_between(train_sizes, te_m - te_s, te_m + te_s, alpha=0.15, color="navy")
            ax.set_title(col_title)
            ax.set_xlabel("Training examples")
            ax.set_ylabel("R²" if metric_type == "r2" else "MSE")
            ax.grid(True, linestyle="--", alpha=0.4)
            if row == 0 and col == 0:
                leg = ax.legend(fontsize=8, prop={"weight": "bold"})
                for text in leg.get_texts():
                    text.set_fontweight("bold")
            if col == 0:
                ax.set_ylabel(f"{target}\n" + ("R²" if metric_type == "r2" else "MSE"))
            _bold_axis_text(ax)
    fig.suptitle(
        "Learning Curves — All Targets (ANN / CatBoost, R² / MSE)",
        fontweight="bold",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined learning-curve grid -> {output_path}")


def plot_nfm_histogram(scores: np.ndarray, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(scores, bins="auto", edgecolor="k", alpha=0.75)
    ax.set_title("Net Flow Scores Histogram", fontweight="bold")
    ax.set_xlabel("Net Flow Score")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved NFM histogram -> {output_path}")


def plot_nfm_pairwise_combined(
    obj_df: pd.DataFrame,
    obj_cols: list[str],
    ranks: np.ndarray,
    output_path: Path,
    champion_idx: int = 0,
):
    """Combined 2×3 NFM pairwise scatter with rank buckets and champion marker."""
    order = np.argsort(ranks)
    n = len(order)
    cut10 = max(n // 10, 1)
    cut25 = max(n // 4, 1)
    cut50 = max(n // 2, 1)
    buckets = {
        "Bottom 50%": set(order[cut50:]),
        "Top 25-50%": set(order[cut25:cut50]),
        "Top 10-25%": set(order[cut10:cut25]),
        "Top 10%": set(order[:cut10]),
    }
    colors = {"Bottom 50%": "black", "Top 25-50%": "blue", "Top 10-25%": "green", "Top 10%": "cyan"}
    draw_order = ["Bottom 50%", "Top 25-50%", "Top 10-25%", "Top 10%"]
    champ_color = next((colors[l] for l, s in buckets.items() if champion_idx in s), "red")

    pairs = list(itertools.combinations(range(len(obj_cols)), 2))[:6]
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (i, j) in zip(axs.ravel(), pairs):
        xcol, ycol = obj_cols[i], obj_cols[j]
        for label in draw_order:
            idxs = list(buckets[label])
            if idxs:
                ax.scatter(
                    obj_df.iloc[idxs][xcol], obj_df.iloc[idxs][ycol],
                    s=30, color=colors[label], alpha=0.85,
                    label=label if ax is axs.ravel()[0] else None,
                )
        ax.scatter(
            obj_df.iloc[champion_idx][xcol], obj_df.iloc[champion_idx][ycol],
            s=160, marker="^", facecolors="none", edgecolors="red", linewidths=1.8,
            label="Champion" if ax is axs.ravel()[0] else None, zorder=5,
        )
        ax.scatter(
            obj_df.iloc[champion_idx][xcol], obj_df.iloc[champion_idx][ycol],
            s=40, color=champ_color, edgecolors="k", linewidths=0.6, zorder=6,
        )
        ax.set_xlabel(xcol.replace("Obj_", ""), fontsize=10)
        ax.set_ylabel(ycol.replace("Obj_", ""), fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
    handles, labels = axs.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved NFM pairwise scatter -> {output_path}")


def _draw_radviz_objective_labels(ax, anchors: np.ndarray, obj_names: list[str], fontsize: float = 7):
    """Place bold objective names near each RadViz anchor."""
    for j, lab in enumerate(obj_names):
        ax.text(
            anchors[j, 0] * 1.12, anchors[j, 1] * 1.12, lab,
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )


def plot_individual_radviz(
    xy: np.ndarray,
    pareto_f: np.ndarray,
    obj_names: list[str],
    champion_idx: int | None = None,
):
    """One RadViz figure per objective (professor-style)."""
    if xy.shape[0] == 0 or pareto_f.shape[1] < 2:
        return
    m = pareto_f.shape[1]
    angles = 2 * np.pi * np.arange(m) / m
    anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    poly_x = np.append(anchors[:, 0], anchors[0, 0])
    poly_y = np.append(anchors[:, 1], anchors[0, 1])
    for i, name in enumerate(obj_names):
        fig, ax = plt.subplots(figsize=(9, 9))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=pareto_f[:, i], cmap="viridis", s=60, ec="k", alpha=0.7)
        ax.plot(poly_x, poly_y, "--", c="r", lw=1.5)
        ax.scatter(anchors[:, 0], anchors[:, 1], marker="^", c="r", s=100)
        _draw_radviz_objective_labels(ax, anchors, obj_names, fontsize=9)
        if champion_idx is not None and 0 <= champion_idx < len(xy):
            ax.scatter(
                xy[champion_idx, 0], xy[champion_idx, 1],
                s=200, facecolors="none", edgecolors="gold", lw=2, label="NFM champion",
            )
            ax.legend(loc="upper right", fontsize=9, prop={"weight": "bold"})
        ax.set_title(f"RadViz (color by {name})", fontweight="bold")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, location="bottom", shrink=0.85, pad=0.12)
        fig.tight_layout()
        fig.savefig(RUN_FIG / f"radviz_opt_{_safe_filename(name)}.png", dpi=300)
        plt.close(fig)
    print(f"Saved {len(obj_names)} individual RadViz plots -> {RUN_FIG}")


# =============================================================================
# OPTIMIZATION (discrete NSGA-II + NFM)
# =============================================================================


def _is_categorical_column(series: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    if pd.api.types.is_categorical_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.isna().sum() > 0


def collect_domains(master: pd.DataFrame) -> tuple[dict, dict]:
    """Numeric discrete levels + categorical vocabularies from observed DOE."""
    num_domains: dict[str, list[float]] = {}
    cat_domains: dict[str, list[str]] = {}
    skip = set(ALL_TARGETS) | {"Design_ID"}
    for col in master.columns:
        if col in skip:
            continue
        s = master[col].dropna()
        if s.empty:
            continue
        if _is_categorical_column(s):
            cats = sorted(s.astype(str).unique().tolist())
            if len(cats) > 1:
                cat_domains[col] = cats
            continue
        vals = sorted(pd.to_numeric(s, errors="coerce").dropna().unique())
        if len(vals) > 1:
            num_domains[col] = [float(v) for v in vals]
    return num_domains, cat_domains


def decode_x(x_int: np.ndarray, var_names: list[str], num_dom: dict, cat_dom: dict) -> dict:
    row = {}
    for i, name in enumerate(var_names):
        raw = int(np.round(float(x_int[i])))
        if name in cat_dom:
            cats = cat_dom[name]
            idx = int(np.clip(raw, 0, len(cats) - 1))
            row[name] = cats[idx]
        else:
            vals = num_dom[name]
            idx = int(np.clip(raw, 0, len(vals) - 1))
            row[name] = vals[idx]
    return row


class FCBGAProblem(Problem):
    def __init__(self, var_names, num_dom, cat_dom, models: list[dict]):
        xl, xu = [], []
        for name in var_names:
            n = len(cat_dom[name]) if name in cat_dom else len(num_dom[name])
            xl.append(0)
            xu.append(max(0, n - 1))
        super().__init__(
            n_var=len(var_names),
            n_obj=len(ALL_TARGETS),
            xl=np.array(xl),
            xu=np.array(xu),
            vtype=int,
        )
        self.var_names = var_names
        self.num_dom = num_dom
        self.cat_dom = cat_dom
        self.models = models  # list aligned with ALL_TARGETS

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.rint(X).astype(int)
        F = []
        for x in X:
            design = decode_x(x, self.var_names, self.num_dom, self.cat_dom)
            preds = []
            for target, info in zip(ALL_TARGETS, self.models):
                df_row = pd.DataFrame([{c: design.get(c, np.nan) for c in info["inputs"]}])
                for c in info["inputs"]:
                    if c not in df_row.columns:
                        df_row[c] = np.nan
                df_row = df_row[info["inputs"]]
                val = float(info["pipeline"].predict(df_row)[0])
                if "DeltaW" in target:
                    val = max(val, 0.0)
                preds.append(val)
            F.append(preds)
        out["F"] = np.array(F)


def net_flow_rank(F: np.ndarray, obj_names: list[str]) -> np.ndarray:
    """PROMETHEE-style net flow (all objectives minimized)."""
    n_obj = F.shape[1]
    ranges = np.ptp(F, axis=0)
    ranges = np.where(ranges == 0, 1e-9, ranges)
    Q = NFM_Q_FRAC * ranges
    P = NFM_P_FRAC * ranges
    V = NFM_V_FRAC * ranges
    W = np.ones(n_obj) / n_obj
    M = F.shape[0]
    pref = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            d = F[j] - F[i]
            c = np.where(d <= Q, 1.0, np.where(d <= P, 0.5, np.where(d <= V, 0.25, 0.0)))
            pref[i, j] = (c * W).sum()
    return pref.sum(axis=1) - pref.sum(axis=0)


def radviz_projection(objs: np.ndarray) -> np.ndarray:
    if objs is None or objs.shape[0] == 0:
        return np.empty((0, 2))
    m = objs.shape[1]
    if m == 0:
        return np.empty((objs.shape[0], 2))
    obj_min = objs.min(axis=0)
    obj_range = np.ptp(objs, axis=0)
    obj_range_safe = np.where(obj_range == 0, 1e-9, obj_range)
    norm = 1.0 - (objs - obj_min) / obj_range_safe
    angles = 2 * np.pi * np.arange(m) / m
    anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    numerator = norm @ anchors
    denominator = np.where(norm.sum(axis=1, keepdims=True) == 0, 1e-12, norm.sum(axis=1, keepdims=True))
    return numerator / denominator


def plot_radviz_panels(
    xy: np.ndarray,
    pareto_f: np.ndarray,
    obj_names: list[str],
    champion_idx: int | None = None,
):
    if xy.shape[0] == 0 or pareto_f.shape[1] < 2:
        return
    n_obj = pareto_f.shape[1]
    angles = 2 * np.pi * np.arange(n_obj) / n_obj
    anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    poly_x = np.append(anchors[:, 0], anchors[0, 0])
    poly_y = np.append(anchors[:, 1], anchors[0, 1])

    nrows, ncols = (2, 3) if n_obj > 3 else (1, n_obj)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(obj_names):
        ax = axes[i]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=pareto_f[:, i], cmap="viridis", s=40, ec="k", alpha=0.7)
        ax.plot(poly_x, poly_y, "--", c="r", lw=1)
        ax.scatter(anchors[:, 0], anchors[:, 1], marker="^", c="r", s=80)
        _draw_radviz_objective_labels(ax, anchors, obj_names, fontsize=6.5)
        if champion_idx is not None and 0 <= champion_idx < len(xy):
            ax.scatter(
                xy[champion_idx, 0], xy[champion_idx, 1],
                s=220, facecolors="none", edgecolors="gold", lw=2.5, label="NFM champion",
                zorder=5,
            )
            if i == 0:
                ax.legend(loc="upper right", fontsize=7, prop={"weight": "bold"})
        ax.set_title(name, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        for t in cbar.ax.get_yticklabels():
            t.set_fontweight("bold")
    for j in range(n_obj, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        "RadViz of Pareto-optimal designs (NFM champion highlighted)",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    suffix = "_champion" if champion_idx is not None else ""
    fig.savefig(RUN_FIG / f"combined_radviz_by_objectives{suffix}.png", dpi=300)
    plt.close(fig)


def run_optimization(models: list[dict], master: pd.DataFrame):
    num_dom, cat_dom = collect_domains(master)
    var_names = list(num_dom.keys()) + list(cat_dom.keys())
    print(f"\nOptimization variables ({len(var_names)}): {var_names}")

    problem = FCBGAProblem(var_names, num_dom, cat_dom, models)
    try:
        from pymoo.factory import get_crossover, get_mutation, get_sampling
        algo = NSGA2(
            pop_size=POP_SIZE,
            sampling=get_sampling("int_random"),
            crossover=get_crossover("int_sbx", prob=0.9, eta=15),
            mutation=get_mutation("int_pm", eta=20),
            eliminate_duplicates=True,
        )
    except Exception:
        algo = NSGA2(pop_size=POP_SIZE, eliminate_duplicates=True)

    print(f"Running NSGA-II (pop={POP_SIZE}, gen={N_GEN})...")
    res = minimize(problem, algo, get_termination("n_gen", N_GEN), seed=SEED, verbose=False)
    if res.F is None or len(res.F) == 0:
        print("NSGA-II returned empty Pareto set.")
        return None

    pareto_F, pareto_X = res.F, res.X
    designs = [decode_x(x, var_names, num_dom, cat_dom) for x in pareto_X]
    design_df = pd.DataFrame(designs)
    obj_df = pd.DataFrame(pareto_F, columns=[f"Obj_{t}" for t in ALL_TARGETS])
    xy_radviz = radviz_projection(pareto_F)
    pareto_df = pd.concat(
        [
            pd.DataFrame({"RadViz_X": xy_radviz[:, 0], "RadViz_Y": xy_radviz[:, 1]}),
            design_df,
            obj_df,
        ],
        axis=1,
    )
    pareto_df.to_csv(RUN_DIR / "pareto_solutions_designs_opt.csv", index=False)
    obj_df.to_excel(RUN_DIR / "pareto_objectives_only.xlsx", index=False)

    # Correlation on Pareto objectives
    corr = obj_df.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(ALL_TARGETS)))
    ax.set_yticks(range(len(ALL_TARGETS)))
    ax.set_xticklabels(ALL_TARGETS, rotation=35, ha="right")
    ax.set_yticklabels(ALL_TARGETS)
    for i in range(len(ALL_TARGETS)):
        for j in range(len(ALL_TARGETS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Objective correlation — Pareto set")
    fig.tight_layout()
    fig.savefig(RUN_FIG / "objective_correlation_pareto.png", dpi=300)
    plt.close(fig)

    # NFM first so pairwise projections can be colored by score
    scores = net_flow_rank(pareto_F, ALL_TARGETS)
    order = np.argsort(scores)[::-1]
    nfm_df = pareto_df.copy()
    nfm_df["NetFlow_Score"] = scores
    nfm_df["NetFlow_Rank"] = np.argsort(np.argsort(-scores)) + 1
    nfm_sorted = nfm_df.iloc[order].reset_index(drop=True)
    nfm_sorted.to_excel(RUN_DIR / "nfm_with_designs.xlsx", index=False)
    nfm_sorted.to_csv(RUN_DIR / "nfm_with_designs_sorted.csv", index=False)

    plot_pareto_pairwise_projections(
        pareto_F, ALL_TARGETS, scores=scores,
        output_path=RUN_FIG / "combined_scatter_pareto.png",
    )

    champion = nfm_sorted.iloc[0]
    champion.to_frame().T.to_csv(RUN_DIR / "champion_design.csv", index=False)
    champ_idx = int(order[0])
    plot_radviz_panels(xy_radviz, pareto_F, ALL_TARGETS, champion_idx=None)
    plot_radviz_panels(xy_radviz, pareto_F, ALL_TARGETS, champion_idx=champ_idx)
    plot_individual_radviz(xy_radviz, pareto_F, ALL_TARGETS, champion_idx=champ_idx)

    obj_cols = [f"Obj_{t}" for t in ALL_TARGETS]
    plot_nfm_histogram(scores, RUN_FIG / "nfm_scores_histogram.png")
    plot_nfm_pairwise_combined(
        obj_df, obj_cols, -scores, RUN_FIG / "combined_nfm_pairwise_scatter.png",
        champion_idx=champ_idx,
    )

    # FEA validation proxy: nearest simulated design in master table
    champ_design = champion[var_names].to_dict()
    dists = []
    for _, row in master.iterrows():
        d = 0.0
        for k in var_names:
            if k in cat_dom:
                d += 0 if str(row.get(k, "")) == str(champ_design.get(k, "")) else 1
            else:
                d += (float(row.get(k, 0)) - float(champ_design.get(k, 0))) ** 2
        dists.append(d)
    nearest_idx = int(np.argmin(dists))
    nearest = master.iloc[nearest_idx]

    val_rows = []
    for t, info in zip(ALL_TARGETS, models):
        ml_pred = float(champion[f"Obj_{t}"])
        fea_actual = float(nearest[t])
        err = abs(ml_pred - fea_actual)
        pct = 100 * err / (abs(fea_actual) + 1e-12)
        val_rows.append(
            {
                "Target": t,
                "ML_Prediction": ml_pred,
                "Nearest_FEA_Simulation": fea_actual,
                "Abs_Error": err,
                "Pct_Error": pct,
                "Nearest_Design_ID": int(nearest["Design_ID"]),
            }
        )
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(RUN_DIR / "champion_fea_validation_proxy.csv", index=False)
    val_df.to_excel(RUN_DIR / "champion_fea_validation_proxy.xlsx", index=False)

    print("\nChampion design (NFM rank 1) saved.")
    print(val_df.to_string(index=False))
    return nfm_sorted


# =============================================================================
# MAIN
# =============================================================================


def publish_latest_artifacts():
    """Copy key co-design outputs to stable paths for papers and README links."""
    latest_res = RESULTS_DIR / "codesign_latest"
    latest_fig = FIGURES_DIR / "codesign_latest"
    latest_res.mkdir(parents=True, exist_ok=True)
    latest_fig.mkdir(parents=True, exist_ok=True)

    result_files = [
        "surrogate_best_model_summary.csv",
        "validation_cv_metrics.csv",
        "objective_correlation_matrix.csv",
        "pareto_solutions_designs_opt.csv",
        "nfm_with_designs_sorted.csv",
        "champion_design.csv",
        "champion_fea_validation_proxy.csv",
        "codesign_interpretation.txt",
    ]
    figure_files = [
        "surrogate_test_r2_summary.png",
        "objective_correlation_simulation_data.png",
        "objective_correlation_pareto.png",
        "combined_scatter_pareto.png",
        "combined_feature_importance_5targets.png",
        "combined_radviz_by_objectives.png",
        "combined_radviz_by_objectives_champion.png",
        "combined_actual_vs_predicted_5x4.png",
        "combined_learning_curves_5x4.png",
        "nfm_scores_histogram.png",
        "combined_nfm_pairwise_scatter.png",
    ]
    for name in result_files:
        src = RUN_DIR / name
        dst = latest_res / name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    for name in figure_files:
        src = RUN_FIG / name
        dst = latest_fig / name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    for src in RUN_FIG.glob("radviz_opt_*.png"):
        dst = latest_fig / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    (latest_res / "run_stamp.txt").write_text(RUN_STAMP + "\n", encoding="utf-8")


def write_ann_improvement_comparison(all_candidates: list[dict], run_dir: Path) -> pd.DataFrame:
    """Compare new ANN metrics against the baseline co-design run."""
    new_df = pd.DataFrame(
        [{k: v for k, v in row.items() if k not in ("pipeline", "inputs", "X_test", "y_test")}
         for row in all_candidates]
    )
    new_df.to_csv(run_dir / "surrogate_full_ann_catboost_comparison.csv", index=False)

    baseline_path = BASELINE_RUN_DIR / "surrogate_full_ann_catboost_comparison.csv"
    if not baseline_path.exists():
        print(f"Warning: baseline not found at {baseline_path}; skipping comparison file.")
        return new_df

    old_df = pd.read_csv(baseline_path)
    ann_old = old_df[old_df["Model"] == "ANN"].copy()
    ann_new = new_df[new_df["Model"] == "ANN"].copy()
    merge_keys = ["Dataset", "Target", "Model"]
    comparison = ann_old.merge(
        ann_new,
        on=merge_keys,
        how="outer",
        suffixes=("_Baseline", "_Improved"),
    )
    if "Test_R2_Baseline" in comparison.columns and "Test_R2_Improved" in comparison.columns:
        comparison["Test_R2_Delta"] = (
            comparison["Test_R2_Improved"] - comparison["Test_R2_Baseline"]
        )
    if "Test_MSE_Baseline" in comparison.columns and "Test_MSE_Improved" in comparison.columns:
        comparison["Test_MSE_Delta"] = (
            comparison["Test_MSE_Improved"] - comparison["Test_MSE_Baseline"]
        )

    out_csv = RESULTS_DIR / "ann_improvement_comparison.csv"
    comparison.to_csv(out_csv, index=False)
    print(f"Saved ANN improvement comparison -> {out_csv}")

    # Side-by-side bar chart: baseline vs improved ANN test R²
    if "Test_R2_Baseline" in comparison.columns and "Test_R2_Improved" in comparison.columns:
        fig, ax = plt.subplots(figsize=(11, 6))
        labels = comparison["Dataset"] + " | " + comparison["Target"]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, comparison["Test_R2_Baseline"], width, label="ANN Baseline", color="salmon")
        ax.bar(x + width / 2, comparison["Test_R2_Improved"], width, label="ANN Improved", color="steelblue")
        ax.axhline(0.5, ls="--", color="orange", label="Medium (0.5)")
        ax.axhline(0.0, ls="-", color="gray", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Test R²")
        ax.set_title("ANN Test R² — Baseline vs Improved")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "ann_improvement_test_r2_comparison.png", dpi=300)
        fig.savefig(FIGURES_DIR / "ann_improvement_test_r2_comparison.png", dpi=300)
        plt.close(fig)
    return comparison


def main():
    print("=" * 70)
    print("FCBGA with Lid — Full Co-Design Pipeline")
    print(f"Output: {RUN_DIR}")
    print("=" * 70)

    assembly = load_assembly()
    sjr = load_sjr()
    assembly.to_csv(RUN_DIR / "cleaned_assembly.csv", index=False)
    sjr.to_csv(RUN_DIR / "cleaned_sjr.csv", index=False)

    master = build_master_table(assembly, sjr)
    plot_corr_heatmap(master)

    trained = []
    summaries = []
    all_candidates = []
    importance_data: list[dict] = []
    for target in ASSEMBLY_TARGETS:
        out = train_target(assembly, target, "Assembly")
        trained.append(out["best"])
        summaries.append(out["summary"])
        all_candidates.extend(out["candidates"])
        fi = plot_feature_importance(out["best"], target, "Assembly")
        if fi:
            importance_data.append(fi)
    for target in SJR_TARGETS:
        out = train_target(sjr, target, "SJR")
        trained.append(out["best"])
        summaries.append(out["summary"])
        all_candidates.extend(out["candidates"])
        fi = plot_feature_importance(out["best"], target, "SJR")
        if fi:
            importance_data.append(fi)

    if importance_data:
        generate_combined_feature_importance_grid(
            importance_data, RUN_FIG / "combined_feature_importance_5targets.png"
        )

    comparison_df = write_ann_improvement_comparison(all_candidates, RUN_DIR)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RUN_DIR / "surrogate_best_model_summary.csv", index=False)
    pd.DataFrame(VALIDATION_CV_ROWS).to_csv(RUN_DIR / "validation_cv_metrics.csv", index=False)
    pd.DataFrame(VALIDATION_CV_ROWS).to_excel(RUN_DIR / "validation_cv_metrics.xlsx", index=False)

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = summary_df["Dataset"] + "\n" + summary_df["Target"]
    ax.bar(labels, summary_df["Test_R2"], color="steelblue", edgecolor="k")
    ax.axhline(0.9, ls="--", color="green", label="Very good")
    ax.axhline(0.5, ls="--", color="orange", label="Medium")
    ax.set_ylabel("Test R²")
    ax.set_title("Surrogate performance (ANN / CatBoost, best selected)")
    plt.xticks(rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN_FIG / "surrogate_test_r2_summary.png", dpi=300)
    plt.close(fig)

    generate_combined_actual_vs_predicted_grid(
        ALL_PRED_DATA, RUN_FIG / "combined_actual_vs_predicted_5x4.png"
    )
    generate_combined_learning_curves_grid(
        ALL_LC_DATA, RUN_FIG / "combined_learning_curves_5x4.png"
    )

    models_for_opt = trained  # order matches ALL_TARGETS
    run_optimization(models_for_opt, master)

    # Interpretation for term paper
    weak = summary_df[summary_df["Model_Quality"] == "Weak"]["Target"].tolist()
    text = f"""
FCBGA with Lid — Full Co-Design Pipeline Summary
Run: {RUN_STAMP}

Methods implemented:
- Separate Assembly / SJR cleaning with harmonized column names
- Improved ANN: scaled numeric inputs, target scaling, early stopping,
  CV-tuned hidden layers / alpha / learning rate
- CatBoost (depth={CATBOOST_DEPTH}) surrogates
- 5-fold validation curves and hold-out test metrics
- Combined 5×4 parity and learning-curve grids (ANN/CatBoost)
- Pearson correlation heatmap on 300 paired simulation rows
- NSGA-II on discrete DOE levels (pop={POP_SIZE}, gen={N_GEN})
- Net Flow Method ranking, histogram, and pairwise NFM plots
- RadViz per objective and combined panels with champion marker
- Champion export with nearest-FEA validation proxy

Surrogate summary:
{summary_df.to_string(index=False)}

Weak targets requiring careful discussion: {', '.join(weak) if weak else 'none'}

Note: Replace validation-proxy with ANSYS re-simulation of the champion for final confirmation.
"""
    (RUN_DIR / "codesign_interpretation.txt").write_text(text.strip() + "\n", encoding="utf-8")
    publish_latest_artifacts()

    print("\n" + summary_df.to_string(index=False))
    if not comparison_df.empty and "Test_R2_Delta" in comparison_df.columns:
        print("\nANN improvement (Test R² delta = Improved - Baseline):")
        for _, r in comparison_df.iterrows():
            print(
                f"  {r['Dataset']} | {r['Target']}: "
                f"{r.get('Test_R2_Baseline', np.nan):.4f} -> {r.get('Test_R2_Improved', np.nan):.4f} "
                f"(delta {r.get('Test_R2_Delta', np.nan):+.4f})"
            )
    print(f"\nDone. All artifacts in:\n  {RUN_DIR}\n  {RUN_FIG}")
    print(f"Stable copies: {RESULTS_DIR / 'codesign_latest'}")


if __name__ == "__main__":
    main()
