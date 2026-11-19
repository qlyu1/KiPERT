"""Fit a prepared double-perturbation fold using the recorded structure and penalty."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kipert.double.model import (
    Bank,
    context_data,
    fit,
    predict,
    prepare,
    score_validation,
)


def load(base, role):
    with np.load(base / (role + ".npz")) as z:
        return {k: z[k] for k in z}


def fit_arm(root, arm, ds, fold, bank, config, validate=False):
    base = root / ds / arm / f"fold{fold}"
    genes = pd.read_csv(base / "genes.tsv", sep="\t", dtype=str).gene_id.to_numpy(str)
    panel = {role: load(base, role) for role in ("train", "controls_fit")}
    queries = load(base, "validation" if validate else "query")
    target, context = queries["target"].astype(str), queries["context"].astype(str)
    candidates = [0.01, 0.1, 1.0] if validate else [config["chosen"]["alpha"]]
    predictions = {a: np.full((len(target), len(genes)), np.nan) for a in candidates}
    controls = {}
    for c in np.unique(context):
        train, y, controls[c], _ = context_data(panel, c)
        state = prepare(bank, train, y, genes, config["structure"])
        rows = np.flatnonzero(context == c)
        for a in candidates:
            predictions[a][rows] = predict(
                bank, fit(state, a), target[rows].tolist(), controls[c]
            )
    assert all(np.isfinite(p).all() for p in predictions.values())
    if validate:
        scores = [
            dict(
                alpha=a,
                **score_validation(
                    p,
                    queries["expression"],
                    target,
                    context,
                    controls,
                    ~np.isin(genes, bank.targets),
                ),
            )
            for a, p in predictions.items()
        ]
        best = min(
            scores,
            key=lambda s: (
                -(s["pds"] if s["pds"] is not None else -1),
                s["mse"],
                s["alpha"],
            ),
        )
        return best, scores
    return predictions[candidates[0]].astype(np.float32), target, context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Parent of D*/seen*/fold*"
    )
    parser.add_argument(
        "--priors", type=Path, required=True, help="Parent of prepared D* prior banks"
    )
    parser.add_argument("--dataset", choices=["D1", "D2", "D3", "D4"], required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument(
        "--reselect", action="store_true", help="Reselect alpha on seen0 validation"
    )
    a = parser.parse_args()
    if a.output.exists():
        parser.error("Output directory already exists")
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/double"
        / a.dataset
        / f"fold{a.fold}"
        / "selection.json"
    )
    cfg = json.loads(path.read_text())
    bank = Bank(a.priors / a.dataset)
    if a.reselect:
        cfg["chosen"], cfg["candidates"] = fit_arm(
            a.data, "seen0", a.dataset, a.fold, bank, cfg, True
        )
    a.output.mkdir(parents=True)
    (a.output / "selection.json").write_text(json.dumps(cfg, indent=2))
    for arm in ("seen0", "seen1", "seen2"):
        pred, target, context = fit_arm(a.data, arm, a.dataset, a.fold, bank, cfg)
        np.savez_compressed(
            a.output / (arm + ".npz"), prediction=pred, target=target, context=context
        )


if __name__ == "__main__":
    main()
