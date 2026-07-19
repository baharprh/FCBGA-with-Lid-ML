#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate Pareto pairwise projections + combined feature-importance grid."""

from __future__ import annotations

import shutil
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import fcbga_full_codesign_pipeline as pipe

PROJECT = Path(__file__).resolve().parent
SRC_RUN = PROJECT / "results" / "codesign_run_20260706_204436"
SRC_FIG = PROJECT / "figures" / "codesign_run_20260706_204436"
OUT_FIG = PROJECT / "figures" / "codesign_latest"
OUT_FIG.mkdir(parents=True, exist_ok=True)

pipe.RUN_FIG = OUT_FIG
pipe.RUN_DIR = PROJECT / "results" / "codesign_latest"
pipe.RUN_DIR.mkdir(parents=True, exist_ok=True)


def regenerate_pareto_scatter() -> Path:
    nfm = pd.read_csv(SRC_RUN / "nfm_with_designs_sorted.csv")
    # Prefer original Pareto row order for scores alignment when available
    pareto = pd.read_csv(SRC_RUN / "pareto_solutions_designs_opt.csv")
    obj_cols = [f"Obj_{t}" for t in pipe.ALL_TARGETS]
    pareto_f = pareto[obj_cols].to_numpy(dtype=float)

    # Map NFM scores onto Pareto rows via RadViz coordinates
    scores = np.zeros(len(pareto))
    xy = pareto[["RadViz_X", "RadViz_Y"]].to_numpy(dtype=float)
    for _, row in nfm.iterrows():
        pt = np.array([row["RadViz_X"], row["RadViz_Y"]], dtype=float)
        idx = int(np.argmin(np.sum((xy - pt) ** 2, axis=1)))
        scores[idx] = float(row["NetFlow_Score"])

    out = OUT_FIG / "combined_scatter_pareto.png"
    pipe.plot_pareto_pairwise_projections(
        pareto_f, pipe.ALL_TARGETS, scores=scores, output_path=out
    )
    shutil.copy2(out, SRC_FIG / out.name)
    return out


def regenerate_feature_importance() -> Path:
    importance_data: list[dict] = []
    assembly = pd.read_csv(SRC_RUN / "cleaned_assembly.csv")
    sjr = pd.read_csv(SRC_RUN / "cleaned_sjr.csv")

    for target, df, dataset in (
        *[(t, assembly, "Assembly") for t in pipe.ASSEMBLY_TARGETS],
        *[(t, sjr, "SJR") for t in pipe.SJR_TARGETS],
    ):
        inputs = [c for c in df.columns if c not in pipe.ALL_TARGETS and c != "Design_ID"]
        X, y = df[inputs], df[target]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=pipe.TEST_SIZE, random_state=pipe.SEED
        )
        pre = pipe.make_preprocessor(df, inputs)
        # CatBoost was selected for every target in the published run
        model = pipe.catboost_pipe(pre)
        model.fit(X_tr, y_tr)
        best = {
            "pipeline": model,
            "inputs": inputs,
            "X_test": X_te,
            "y_test": y_te,
        }
        fi = pipe.plot_feature_importance(best, target, dataset)
        if fi:
            importance_data.append(fi)
            print(f"  FI ready for {dataset} / {target}")
        # copy per-target figure
        safe = target.replace(" ", "_")
        src = OUT_FIG / f"fi_{dataset}_{safe}.png"
        if src.exists():
            shutil.copy2(src, SRC_FIG / src.name)

    out = OUT_FIG / "combined_feature_importance_5targets.png"
    pipe.generate_combined_feature_importance_grid(importance_data, out)
    shutil.copy2(out, SRC_FIG / out.name)
    return out


def update_docx(pareto_png: Path, fi_png: Path) -> None:
    """Replace Fig. 10 (image11.jpg). FI figure is new — stored only as PNG asset."""
    docx = PROJECT / "ML for FCBGA Packages with Lid.docx"
    if not docx.exists():
        return
    im = Image.open(pareto_png).convert("RGB")
    max_side = 2800
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    tmpdir = Path(tempfile.mkdtemp())
    jpg = tmpdir / "image11.jpg"
    im.save(jpg, format="JPEG", quality=92, optimize=True)
    jpg_bytes = jpg.read_bytes()
    tmp_out = tmpdir / "out.docx"
    with zipfile.ZipFile(docx, "r") as zin, zipfile.ZipFile(
        tmp_out, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = jpg_bytes if item.filename == "word/media/image11.jpg" else zin.read(item.filename)
            zout.writestr(item, data)
    shutil.copy2(tmp_out, docx)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"Updated Figure 10 in {docx.name}")
    print(f"Combined FI figure saved (not yet in Word): {fi_png}")


def main() -> None:
    print("=" * 70)
    print("Updating Pareto pairwise + feature-importance figures")
    print("=" * 70)
    pareto_png = regenerate_pareto_scatter()
    fi_png = regenerate_feature_importance()
    update_docx(pareto_png, fi_png)
    # cleanup empty import-time run dirs
    for base in (PROJECT / "figures", PROJECT / "results"):
        for d in base.glob("codesign_run_2026*"):
            if d.name != SRC_RUN.name and d.is_dir():
                # only remove if empty-ish / today's temp
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
    print("Done.")


if __name__ == "__main__":
    main()
