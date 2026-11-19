"""Refit KiPERT-double for Replogle exp6 fold 0 from prepared training data."""
import json
from pathlib import Path
import numpy as np

from kipert.double.model import prepare, fit, predict

ROOT = Path(__file__).resolve().parent


class FoldKernels:
    """Supply kernels already normalized using the training components of this fold."""

    def __init__(self, targets, data):
        self.targets = targets
        self.training = data["training"].tolist()
        self.inputs = (
            data["views"],
            data["coverage"],
            data["training_components"].tolist(),
        )

    def kernels(self, conditions):
        if conditions != self.training:
            raise ValueError("Training conditions differ from the normalization fold")
        return self.inputs


def refit(arm):
    base = ROOT.parents[1] / "examples/double_small" / arm
    config = json.loads(
        (ROOT.parents[1] / "configs/double/D2/fold0/selection.json").read_text()
    )
    axes = np.load(base / "axes.npz")
    pred = np.full((len(axes["query"]), len(axes["genes"])), np.nan, np.float32)
    for i, ctx in enumerate(axes["contexts"]):
        with np.load(base / f"context_{i}.npz") as data:
            bank = FoldKernels(axes["targets"], data)
            idx = np.flatnonzero(axes["context"] == ctx)
            queries = axes["query"][idx].tolist()
            assert not set(queries).intersection(bank.training)
            state = prepare(
                bank, bank.training, data["delta"], axes["genes"], config["structure"]
            )
            fitted = fit(state, config["chosen"]["alpha"])
            pred[idx] = predict(bank, fitted, queries, data["control"])
    if not np.isfinite(pred).all():
        raise ValueError("Incomplete prediction matrix")
    return pred


def verify():
    result = {}
    for arm in ("seen0", "seen1", "seen2"):
        pred = refit(arm)
        base = ROOT.parents[1] / "examples/double_small" / arm
        with np.load(base / "expected.npz") as ref, np.load(base / "axes.npz") as axes:
            np.testing.assert_array_equal(axes["query"], ref["target"])
            np.testing.assert_array_equal(axes["context"], ref["context"])
            np.testing.assert_allclose(pred, ref["prediction"], atol=2e-6, rtol=0)
            result[arm] = dict(
                shape=pred.shape,
                max_abs_error=float(np.abs(pred - ref["prediction"]).max()),
            )
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))


def refit_figure3h():
    """Recreate panel h from a fresh seen-2 fit and matched observed references."""
    demo = ROOT.parents[1] / "examples/double_small/seen2"
    predicted = refit("seen2")
    source = ROOT.parents[1] / "source_data/fig3/GI_D2_fold0_seen2.npz"
    with np.load(demo / "axes.npz") as axes, np.load(source) as ref:
        gene_index = {g: i for i, g in enumerate(axes["genes"])}
        columns = [gene_index[g] for g in ref["gene_id"]]
        pair_means = np.stack(
            [
                predicted[axes["query"] == t][:, columns].mean(axis=0, dtype=float)
                for t in ref["target"]
            ]
        ).astype(np.float32)
        archived = ref["prediction"][ref["models"] == "kipert"].mean(axis=0)
        np.testing.assert_allclose(pair_means, archived, atol=2e-6, rtol=0)
        offset = ref["control"] - ref["single_a"] - ref["single_b"]
        observed = ref["truth"] + offset
        residual = pair_means + offset
        additive = (
            ref["prediction"][ref["models"] == "available_evidence_additive"].mean(
                axis=0
            )
            + offset
        )
        mse = np.square(residual - observed).mean()
        other = np.square(additive - observed).mean()
        assert round(float(100 * (1 - mse / other)), 1) == 51.0
        return dict(
            held_out_pairs=len(ref["target"]),
            genes=len(columns),
            pair_mean_max_abs_error=float(np.abs(pair_means - archived).max()),
            KiPERT_MSE=float(mse),
            additive_MSE=float(other),
            reduction_percent=float(100 * (1 - mse / other)),
        )
