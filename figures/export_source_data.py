"""Export the numeric source tables into one workbook per figure."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[1]
src = root / "source_data"
ed = root / "experiments/extended_data/reference"


def write(name, tables):
    with pd.ExcelWriter(a.output / f"{name}.xlsx", engine="openpyxl") as x:
        for sheet, frame in tables:
            frame.to_excel(x, sheet_name=sheet, index=False)


def csvs(directory, mapping):
    return [(s, pd.read_csv(directory / f)) for s, f in mapping]


fig2 = csvs(
    src / "fig2",
    [
        ("a_ranks", "ranks.csv"),
        ("b_conditions", "condition_metrics.csv"),
        ("d_targets", "target_metrics.csv"),
    ],
)
with np.load(src / "fig2/landscapes.npz") as z:
    for panel, ds in [("e", "S4"), ("f", "S6")]:
        for m in [
            "truth",
            "UC",
            "biolord",
            "scouter",
            "txpert",
            "gears",
            "stamp",
            "regformer",
        ]:
            wide = pd.DataFrame(z[f"{ds}_{m}"], columns=z[f"{ds}_target"])
            wide.insert(0, "target", z[f"{ds}_target"])
            fig2.append((panel + "_" + m, wide))
write("Figure2", fig2)
f3 = src / "fig3"
with np.load(f3 / "GI_D2_fold0_seen2.npz") as z:
    gene_ids = z["gene_id"]
    pairs = z["target"]
    axes = pd.DataFrame(
        {
            "index": np.arange(len(pairs) * len(gene_ids)),
            "target": np.repeat(pairs, len(gene_ids)),
            "gene_id": np.tile(gene_ids, len(pairs)),
        }
    )
fig3 = csvs(
    f3,
    [
        ("a_to_f", "Figure3_metric_source.csv"),
        ("g", "Figure3_case_source.csv"),
        ("h", "Figure3_scatter_source.csv.gz"),
    ],
)
write("Figure3", fig3 + [("h_axes", axes)])
f4 = src / "fig4"
f4tables = csvs(f4, [("a_to_e", "Figure4_abcde_summary.csv")])
for line in ("k562", "hepg2", "rpe1", "jurkat"):
    with np.load(f4 / f"SEH1L_{line}.npz") as z:
        f4tables.append(("f_" + line, pd.DataFrame({k: z[k] for k in z})))
    frames = [
        pd.read_csv(f4 / f"{m}_{line}_per_condition.csv") for m in ("UC", "PMean")
    ]
    f4tables.append(
        (
            "g_" + line,
            frames[0].merge(
                frames[1],
                on=["context", "perturbation"],
                validate="one_to_one",
                suffixes=("_KiPERT", "_PMean"),
            ),
        )
    )
write("Figure4", f4tables)
write(
    "Figure5",
    [
        (f.stem.split("_")[1], pd.read_csv(f))
        for f in sorted((src / "fig5").glob("panel*.csv"))
    ],
)
for n in range(1, 7):
    write(
        f"Extended_Data_Figure{n}",
        [(f.stem, pd.read_csv(f)) for f in sorted(ed.glob(f"{n}*.csv"))],
    )
