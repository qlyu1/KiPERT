"""Redraw the numerical panels of Figure 4 from their published data axes."""
from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = Path(__file__).resolve().parents[2] / "source_data/fig4"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
summary = pd.read_csv(data / "Figure4_abcde_summary.csv")
panels = [
    "DE direction match",
    "L1 PDS",
    "Pearson Δ",
    "DE log-fold Spearman",
    "DE overlap",
]
lines = ["k562", "hepg2", "rpe1", "jurkat"]
names = ["K562", "HepG2", "RPE1", "Jurkat"]
palette = ["#287c95", "#808a99", "#bf7557", "#a89bb8", "#b6b8b1"]
fig = plt.figure(figsize=(11, 8), layout="constrained")
rows = fig.subfigures(3, 1)
for (letter, frame), name, ax in zip(
    summary.groupby("panel"), panels, rows[0].subplots(1, 5)
):
    ax.bar(
        np.arange(len(frame)), frame["mean"], yerr=frame.sd, color=palette, capsize=2
    )
    ax.set(
        xticks=np.arange(len(frame)),
        xticklabels=frame.model,
        title=f"{letter}   {name}",
    )
    ax.tick_params(axis="x", labelrotation=60)
for ctx, name, ax in zip(lines, names, rows[1].subplots(1, 4)):
    with np.load(data / f"SEH1L_{ctx}.npz") as z:
        x, y = z["true_mean"], z["pred_mean"]
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        ax.hexbin(x, y, gridsize=38, bins="log", mincnt=1, cmap="Blues")
        ax.set(
            title=f"{name}   R² = {r2:.3f}",
            xlabel="Observed mean",
            ylabel="Predicted mean",
        )
rows[1].suptitle("f   SEH1L")
for ctx, name, ax in zip(lines, names, rows[2].subplots(1, 4)):
    pair = [
        pd.read_csv(data / f"{m}_{ctx}_per_condition.csv").set_index("perturbation")
        for m in ("PMean", "UC")
    ]
    x, y = (
        pair[0].pds_l1_exclude_target,
        pair[1].loc[pair[0].index, "pds_l1_exclude_target"],
    )
    ax.hexbin(x, y, gridsize=30, bins="log", mincnt=1, cmap="BuGn")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    ax.set(
        title=f"{name}   n = {len(x)}", xlabel="PMean L1 PDS", ylabel="KiPERT L1 PDS"
    )
rows[2].suptitle("g   Matched perturbations")
fig.savefig(out / "Figure4.pdf")
plt.close(fig)
