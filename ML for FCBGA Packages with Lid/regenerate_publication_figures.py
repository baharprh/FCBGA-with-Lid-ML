#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regenerate publication figures (RadViz, actual-vs-predicted, learning curves)
with updated styling, without re-running NSGA-II or ANN hyperparameter search.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split

import fcbga_full_codesign_pipeline as pipe

PROJECT_DIR = Path(__file__).resolve().parent
SRC_RUN = PROJECT_DIR / "results" / "codesign_run_20260706_204436"
SRC_FIG = PROJECT_DIR / "figures" / "codesign_run_20260706_204436"
OUT_FIG = PROJECT_DIR / "figures" / "codesign_latest"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Point pipeline figure writers at the stable publication folder
pipe.RUN_FIG = OUT_FIG
pipe.RUN_DIR = PROJECT_DIR / "results" / "codesign_latest"
pipe.RUN_DIR.mkdir(parents=True, exist_ok=True)

ANN_PARAMS = {
    "ELK stress": {"hidden_layer_sizes": (64, 32), "alpha": 0.001, "learning_rate_init": 0.001},
    "Warpage Post UF cure": {"hidden_layer_sizes": (64, 32), "alpha": 0.001, "learning_rate_init": 0.001},
    "Warpage post lid attach": {"hidden_layer_sizes": (32, 16), "alpha": 0.0001, "learning_rate_init": 0.001},
    "DeltaW_BGA": {"hidden_layer_sizes": (64, 32), "alpha": 0.01, "learning_rate_init": 0.001},
    "DeltaW_bump": {"hidden_layer_sizes": (64, 32), "alpha": 0.01, "learning_rate_init": 0.001},
}


def regenerate_radviz() -> None:
    pareto = pd.read_csv(SRC_RUN / "pareto_solutions_designs_opt.csv")
    nfm = pd.read_csv(SRC_RUN / "nfm_with_designs_sorted.csv")
    obj_cols = [f"Obj_{t}" for t in pipe.ALL_TARGETS]
    pareto_f = pareto[obj_cols].to_numpy(dtype=float)
    xy = pareto[["RadViz_X", "RadViz_Y"]].to_numpy(dtype=float)

    # Champion = NetFlow rank 1; match back to Pareto row index
    champ = nfm.iloc[0]
    champ_xy = np.array([champ["RadViz_X"], champ["RadViz_Y"]], dtype=float)
    champ_idx = int(np.argmin(np.sum((xy - champ_xy) ** 2, axis=1)))

    pipe.plot_radviz_panels(xy, pareto_f, pipe.ALL_TARGETS, champion_idx=None)
    pipe.plot_radviz_panels(xy, pareto_f, pipe.ALL_TARGETS, champion_idx=champ_idx)
    pipe.plot_individual_radviz(xy, pareto_f, pipe.ALL_TARGETS, champion_idx=champ_idx)
    print(f"RadViz regenerated (champion idx={champ_idx}) -> {OUT_FIG}")


def _collect_pred_and_lc(df: pd.DataFrame, target: str, dataset: str) -> None:
    inputs = [c for c in df.columns if c not in pipe.ALL_TARGETS and c != "Design_ID"]
    X, y = df[inputs], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=pipe.TEST_SIZE, random_state=pipe.SEED
    )
    pre = pipe.make_preprocessor(df, inputs)

    for name in ("ANN", "CatBoost"):
        spec = pipe.MODEL_SPECS[name]
        if name == "ANN":
            params = ANN_PARAMS[target]
            model = pipe.ann_pipe(
                pre,
                hidden_layer_sizes=params["hidden_layer_sizes"],
                alpha=params["alpha"],
                learning_rate_init=params["learning_rate_init"],
            )
        else:
            model = spec["factory"](pre)
        model.fit(X_tr, y_tr)
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)
        pipe.ALL_PRED_DATA.append(
            {
                "target_name": target,
                "dataset": dataset,
                "model_type": spec["model_type"],
                "y_train_true": np.asarray(y_tr),
                "y_train_pred": np.asarray(y_tr_pred),
                "y_test_true": np.asarray(y_te),
                "y_test_pred": np.asarray(y_te_pred),
            }
        )
        train_sizes = np.linspace(0.2, 1.0, 5)
        for scoring, metric_type in [("r2", "r2"), ("neg_mean_squared_error", "mse")]:
            sizes, tr_sc, te_sc = learning_curve(
                model, X_tr, y_tr, cv=pipe.CV_FOLDS, scoring=scoring,
                train_sizes=train_sizes, n_jobs=1,
            )
            pipe.ALL_LC_DATA.append(
                {
                    "target_name": target,
                    "dataset": dataset,
                    "model_type": spec["model_type"],
                    "metric_type": metric_type,
                    "train_sizes": sizes,
                    "train_scores": tr_sc,
                    "test_scores": te_sc,
                }
            )
        print(f"  Trained {name} for {dataset} / {target}")


def regenerate_parity_and_learning_curves() -> None:
    pipe.ALL_PRED_DATA.clear()
    pipe.ALL_LC_DATA.clear()

    assembly = pd.read_csv(SRC_RUN / "cleaned_assembly.csv")
    sjr = pd.read_csv(SRC_RUN / "cleaned_sjr.csv")

    for target in pipe.ASSEMBLY_TARGETS:
        _collect_pred_and_lc(assembly, target, "Assembly")
    for target in pipe.SJR_TARGETS:
        _collect_pred_and_lc(sjr, target, "SJR")

    parity_path = OUT_FIG / "combined_actual_vs_predicted_5x4.png"
    lc_path = OUT_FIG / "combined_learning_curves_5x4.png"
    pipe.generate_combined_actual_vs_predicted_grid(pipe.ALL_PRED_DATA, parity_path)
    pipe.generate_combined_learning_curves_grid(pipe.ALL_LC_DATA, lc_path)

    # Also refresh the stamped run folder used by the paper draft assets
    for name in (
        "combined_actual_vs_predicted_5x4.png",
        "combined_learning_curves_5x4.png",
        "combined_radviz_by_objectives.png",
        "combined_radviz_by_objectives_champion.png",
    ):
        src = OUT_FIG / name
        if src.exists():
            shutil.copy2(src, SRC_FIG / name)
    for src in OUT_FIG.glob("radviz_opt_*.png"):
        shutil.copy2(src, SRC_FIG / src.name)
    print(f"Parity + learning curves regenerated -> {OUT_FIG}")


def update_docx_caption() -> None:
    """Refresh Figure 13 caption to mention objective names on each RadViz panel."""
    try:
        from docx import Document
    except ImportError:
        print("python-docx not available; skipping Word caption update.")
        return

    docx_path = PROJECT_DIR / "ML for FCBGA Packages with Lid.docx"
    if not docx_path.exists():
        return
    doc = Document(str(docx_path))
    old = (
        "Figure 13: RadViz visualization of Pareto-optimal designs with the "
        "NFM-selected champion design highlighted."
    )
    new = (
        "Figure 13: RadViz visualization of Pareto-optimal designs with the "
        "objective names on each plot and the NFM-selected champion design highlighted."
    )
    updated = False
    for p in doc.paragraphs:
        if p.text.strip() == old or (
            p.text.strip().startswith("Figure 13:") and "RadViz" in p.text
        ):
            # Preserve runs style by rewriting full paragraph text when possible
            if p.runs:
                p.runs[0].text = new
                for r in p.runs[1:]:
                    r.text = ""
            else:
                p.text = new
            updated = True
            break
    if updated:
        doc.save(str(docx_path))
        print(f"Updated Figure 13 caption in {docx_path.name}")
    else:
        print("Figure 13 caption not found or already updated.")


def main() -> None:
    print("=" * 70)
    print("Regenerating publication figures")
    print("=" * 70)
    regenerate_radviz()
    regenerate_parity_and_learning_curves()
    update_docx_caption()
    print("Done.")


if __name__ == "__main__":
    main()
