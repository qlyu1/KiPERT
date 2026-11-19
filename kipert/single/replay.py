"""Refit the recorded aligned_raw candidate without reading held-out responses."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .core import build, predict_candidate, configurations
from .inputs import Bank, make_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-data",
        type=Path,
        required=True,
        help="Directory containing S3/... through S7/...",
    )
    parser.add_argument(
        "--priors",
        type=Path,
        required=True,
        help="Directory containing the prepared 12-view prior banks",
    )
    parser.add_argument(
        "--dataset", choices=["S3", "S4", "S5", "S6", "S7"], required=True
    )
    parser.add_argument("--split", type=int, choices=[101, 202, 303], required=True)
    parser.add_argument(
        "--selections",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/single",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; an existing directory is rejected",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional sealed aligned_raw NPZ for numerical comparison",
    )
    args = parser.parse_args()
    base = args.formal_data / args.dataset / f"split{args.split}"
    selected = (
        args.selections / args.dataset / f"split{args.split}/SELECTIONS_FROZEN.json"
    )
    key = json.loads(selected.read_text())["aligned_raw"]
    cfg = next(c for c in configurations() if c["id"] == key)
    assert cfg["family"] == "aligned_raw"
    if args.output.exists():
        parser.error("Output directory already exists. Choose a new directory.")
    bank = Bank(args.priors / args.dataset)
    matrices, metadata = {}, {}
    for role in ["fit", "validation", "controls_fit"]:
        matrices[role] = sparse.load_npz(base / role / "expression.npz").tocsr()
        metadata[role] = pd.read_csv(base / role / "metadata.tsv", sep="\t", dtype=str)
    query = pd.read_csv(base / "query/metadata.tsv", sep="\t", dtype=str)
    genes = pd.read_csv(base / "genes.tsv", sep="\t").gene_id.astype(str).to_numpy()
    keep = ~np.isin(genes, bank.targets.astype(str))
    assert not set(metadata["fit"].target) & set(metadata["validation"].target)
    assert not (set(metadata["fit"].target) | set(metadata["validation"].target)) & set(
        query.target
    )
    prediction = np.full((len(query), len(genes)), np.nan, dtype=np.float32)
    weights = []
    for context in sorted(set(query.context)):
        sub, meta = {}, {}
        for role in matrices:
            ix = np.flatnonzero(metadata[role].context.to_numpy() == context)
            sub[role] = matrices[role][ix].astype(float)
            meta[role] = metadata[role].iloc[ix].reset_index(drop=True)
        controls = sub["controls_fit"]
        control = np.asarray(controls.mean(0)).ravel()
        variance = np.maximum(
            np.asarray(controls.power(2).mean(0)).ravel() - control**2, 0
        )
        rows, _ = make_rows(
            sparse.vstack([sub["fit"], sub["validation"]], format="csr"),
            pd.concat([meta["fit"], meta["validation"]], ignore_index=True),
            control,
            variance,
            len(meta["controls_fit"]),
        )
        ix = np.flatnonzero(query.context.to_numpy() == context)
        state = build(bank, rows, query.target.iloc[ix].tolist(), keep)
        prediction[ix] = predict_candidate(state, control, cfg)
        weights.extend(
            dict(context=context, view=v, weight=float(w), alignment=float(a))
            for v, w, a in zip(
                state["view_names"], state["weights"], state["alignment"]
            )
        )
    assert np.isfinite(prediction).all()
    report = dict(
        dataset=args.dataset,
        split=args.split,
        arm="aligned_raw",
        candidate=cfg,
        query_count=len(query),
        gene_count=len(genes),
        held_out_responses_read=False,
        selection_mode="replay existing frozen selection; no retuning",
    )
    if args.reference:
        with np.load(args.reference) as ref:
            np.testing.assert_array_equal(
                query.target.to_numpy(str), ref["target"].astype(str)
            )
            np.testing.assert_array_equal(
                query.context.to_numpy(str), ref["context"].astype(str)
            )
            delta = np.abs(prediction.astype(float) - ref["prediction"].astype(float))
            report["max_abs_error_vs_sealed"] = float(delta.max())
            report["prediction_array_equal"] = bool(
                np.array_equal(prediction, ref["prediction"])
            )
            np.testing.assert_allclose(prediction, ref["prediction"], atol=2e-6, rtol=0)
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "predictions.npz",
        prediction=prediction,
        target=query.target.to_numpy(str),
        context=query.context.to_numpy(str),
    )
    pd.DataFrame(weights).to_csv(args.output / "prior_weights.csv", index=False)
    (args.output / "replay.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
