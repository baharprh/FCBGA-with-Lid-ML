#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCBGA with Lid — Full Co-Design Pipeline
=========================================

End-to-end workflow aligned with the example thermo-mechanical co-design manuscript:

  1. Data cleaning and harmonized column names (Assembly + SJR)
  2. Merge paired simulation rows (Design_ID) for correlation analysis
  3. Train fixed-depth RF / XGB surrogates per target (dataset-specific features)
  4. Validation curves, learning curves, parity plots, feature importance
  5. Objective correlation heatmap (Pearson)
  6. NSGA-II multi-objective optimization on discrete design variables
  7. Net Flow Method (NFM) ranking of Pareto designs
  8. Champion design export + nearest-FEA validation proxy table

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
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
N_ESTIMATORS = 100
RF_DEPTH = 8
XGB_DEPTH = 4

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
            ("num", "passthrough", num),
        ]
    )


def rf_pipe(pre) -> Pipeline:
    return Pipeline(
        [
            ("pre", pre),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_depth=RF_DEPTH,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def xgb_pipe(pre) -> Pipeline:
    return Pipeline(
        [
            ("pre", pre),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_depth=XGB_DEPTH,
                    objective="reg:squarederror",
                    random_state=SEED,
                    tree_method="hist",
                ),
            ),
        ]
    )


def quality_label(r2: float) -> str:
    if r2 >= 0.90:
        return "Very good"
    if r2 >= 0.75:
        return "Good"
    if r2 >= 0.50:
        return "Medium"
    return "Weak"


def log_val_curve(pipe, X, y, depths, target, model_name):
    for d in depths:
        pipe.set_params(regressor__max_depth=d)
        tr, te = validation_curve(
            pipe, X, y, param_name="regressor__max_depth", param_range=[d],
            cv=CV_FOLDS, scoring="r2", n_jobs=1,
        )
        VALIDATION_CV_ROWS.append(
            {
                "Target": target,
                "Model": model_name,
                "ParamName": "max_depth",
                "ParamValue": d,
                "CV_Train_R2_Mean": float(tr.mean()),
                "CV_Test_R2_Mean": float(te.mean()),
            }
        )


def save_diagnostics(pipe, X_train, X_test, y_train, y_test, target, model_name, dataset):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    depths = [3, 5, 8, 10, 12] if "Forest" in model_name else [2, 3, 4, 5, 6]
    pre = pipe.named_steps["pre"]
    sweep = (
        rf_pipe(pre) if "Forest" in model_name else xgb_pipe(pre)
    )
    tr_m, te_m = [], []
    for d in depths:
        sweep.set_params(regressor__max_depth=d)
        tr, te = validation_curve(
            sweep, X_train, y_train, param_name="regressor__max_depth",
            param_range=[d], cv=CV_FOLDS, scoring="r2", n_jobs=-1,
        )
        tr_m.append(tr.mean())
        te_m.append(te.mean())
    axes[0].plot(depths, tr_m, "o-", label="Train CV")
    axes[0].plot(depths, te_m, "o-", label="Val CV")
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
    for name, factory in [("Random Forest", rf_pipe), ("XGBoost", xgb_pipe)]:
        pipe = factory(pre)
        log_val_curve(
            rf_pipe(pre) if name == "Random Forest" else xgb_pipe(pre),
            X_tr, y_tr,
            [3, 5, 8, 10, 12] if name == "Random Forest" else [2, 3, 4, 5, 6],
            target, name,
        )
        pipe.fit(X_tr, y_tr)
        y_pr = pipe.predict(X_te)
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
        }
        rows.append(row)
        save_diagnostics(pipe, X_tr, X_te, y_tr, y_te, target, name, dataset)
        if best is None or row["Test_R2"] > best["Test_R2"]:
            best = row

    best_row = {k: v for k, v in best.items() if k not in ("pipeline", "inputs")}
    best_row["Model_Quality"] = quality_label(best_row["Test_R2"])
    return {"best": best, "candidates": rows, "summary": best_row}


def plot_feature_importance(pipe: Pipeline, inputs: list[str], target: str, dataset: str):
    reg = pipe.named_steps["regressor"]
    if not hasattr(reg, "feature_importances_"):
        return
    pre = pipe.named_steps["pre"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        return
    imp = reg.feature_importances_
    idx = np.argsort(imp)[::-1][:10]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(idx)), imp[idx][::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx][::-1])
    ax.set_title(f"Feature importance — {target}")
    fig.tight_layout()
    safe = target.replace(" ", "_")
    fig.savefig(RUN_FIG / f"fi_{dataset}_{safe}.png", dpi=200)
    plt.close(fig)


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
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(obj_names):
        ax = axes[i]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=pareto_f[:, i], cmap="viridis", s=40, ec="k", alpha=0.7)
        ax.plot(poly_x, poly_y, "--", c="r", lw=1)
        ax.scatter(anchors[:, 0], anchors[:, 1], marker="^", c="r", s=80)
        if champion_idx is not None and 0 <= champion_idx < len(xy):
            ax.scatter(xy[champion_idx, 0], xy[champion_idx, 1], s=200, facecolors="none", edgecolors="gold", lw=2)
        ax.set_title(name)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, shrink=0.8)
    for j in range(n_obj, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
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
    plt.colorbar(im, ax=ax)
    ax.set_title("Objective correlation — Pareto set")
    fig.tight_layout()
    fig.savefig(RUN_FIG / "objective_correlation_pareto.png", dpi=300)
    plt.close(fig)

    # Pairwise Pareto scatter (sample)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    pairs = list(itertools.combinations(range(len(ALL_TARGETS)), 2))[:6]
    for ax, (i, j) in zip(axes.ravel(), pairs):
        ax.scatter(pareto_F[:, i], pareto_F[:, j], s=20, alpha=0.7)
        ax.set_xlabel(ALL_TARGETS[i])
        ax.set_ylabel(ALL_TARGETS[j])
    fig.tight_layout()
    fig.savefig(RUN_FIG / "combined_scatter_pareto.png", dpi=300)
    plt.close(fig)

    # NFM
    scores = net_flow_rank(pareto_F, ALL_TARGETS)
    order = np.argsort(scores)[::-1]
    nfm_df = pareto_df.copy()
    nfm_df["NetFlow_Score"] = scores
    nfm_df["NetFlow_Rank"] = np.argsort(np.argsort(-scores)) + 1
    nfm_sorted = nfm_df.iloc[order].reset_index(drop=True)
    nfm_sorted.to_excel(RUN_DIR / "nfm_with_designs.xlsx", index=False)
    nfm_sorted.to_csv(RUN_DIR / "nfm_with_designs_sorted.csv", index=False)

    champion = nfm_sorted.iloc[0]
    champion.to_frame().T.to_csv(RUN_DIR / "champion_design.csv", index=False)
    plot_radviz_panels(xy_radviz, pareto_F, ALL_TARGETS, champion_idx=int(order[0]))

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
        "combined_radviz_by_objectives_champion.png",
    ]
    for name in result_files:
        src = RUN_DIR / name
        if src.exists():
            shutil.copy2(src, latest_res / name)
    for name in figure_files:
        src = RUN_FIG / name
        if src.exists():
            shutil.copy2(src, latest_fig / name)
    (latest_res / "run_stamp.txt").write_text(RUN_STAMP + "\n", encoding="utf-8")


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
    for target in ASSEMBLY_TARGETS:
        out = train_target(assembly, target, "Assembly")
        trained.append(out["best"])
        summaries.append(out["summary"])
        plot_feature_importance(out["best"]["pipeline"], out["best"]["inputs"], target, "Assembly")
    for target in SJR_TARGETS:
        out = train_target(sjr, target, "SJR")
        trained.append(out["best"])
        summaries.append(out["summary"])
        plot_feature_importance(out["best"]["pipeline"], out["best"]["inputs"], target, "SJR")

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
    ax.set_title("Surrogate performance (fixed depth RF/XGB, best selected)")
    plt.xticks(rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN_FIG / "surrogate_test_r2_summary.png", dpi=300)
    plt.close(fig)

    models_for_opt = trained  # order matches ALL_TARGETS
    run_optimization(models_for_opt, master)

    # Interpretation for term paper
    weak = summary_df[summary_df["Model_Quality"] == "Weak"]["Target"].tolist()
    text = f"""
FCBGA with Lid — Full Co-Design Pipeline Summary
Run: {RUN_STAMP}

Methods implemented:
- Separate Assembly / SJR cleaning with harmonized column names
- Fixed-depth surrogates (RF depth={RF_DEPTH}, XGB depth={XGB_DEPTH})
- 5-fold validation curves and hold-out test metrics
- Pearson correlation heatmap on 300 paired simulation rows
- NSGA-II on discrete DOE levels (pop={POP_SIZE}, gen={N_GEN})
- Net Flow Method ranking of Pareto designs
- Champion export with nearest-FEA validation proxy

Surrogate summary:
{summary_df.to_string(index=False)}

Weak targets requiring careful discussion: {', '.join(weak) if weak else 'none'}

Note: Replace validation-proxy with ANSYS re-simulation of the champion for final confirmation.
"""
    (RUN_DIR / "codesign_interpretation.txt").write_text(text.strip() + "\n", encoding="utf-8")
    publish_latest_artifacts()

    print("\n" + summary_df.to_string(index=False))
    print(f"\nDone. All artifacts in:\n  {RUN_DIR}\n  {RUN_FIG}")
    print(f"Stable copies: {RESULTS_DIR / 'codesign_latest'}")


if __name__ == "__main__":
    main()
