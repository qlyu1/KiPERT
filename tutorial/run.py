"""Refit KiPERT on the Tian CRISPRi example from prepared input summaries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kipert.single.core import build_from_kernels, configurations, predict_candidate

DATA = Path(__file__).resolve().parents[1] / "examples/single_small"
CONFIG = DATA.parents[1] / "configs/single/S3/split101/SELECTIONS_FROZEN.json"


def refit(data=DATA):
    axes = np.load(data / "axes.npz", allow_pickle=False)
    choice = json.loads(CONFIG.read_text())["aligned_raw"]
    cfg = next(candidate for candidate in configurations() if candidate["id"] == choice)
    prediction = np.full(
        (len(axes["target"]), len(axes["genes"])), np.nan, dtype=np.float32
    )
    weights = []
    for i, context in enumerate(axes["contexts"]):
        rows = np.flatnonzero(axes["context"] == context)
        with np.load(data / f"context_{i}.npz", allow_pickle=False) as inputs:
            train, query = inputs["train"], inputs["queries"]
            np.testing.assert_array_equal(query, axes["target"][rows])
            n = len(train)
            state = build_from_kernels(
                train.tolist(),
                query.tolist(),
                inputs["train_delta"],
                axes["keep"],
                inputs["views"],
                np.arange(n),
                np.arange(n, n + len(query)),
                inputs["coverage"],
                inputs["names"],
            )
            prediction[rows] = predict_candidate(state, inputs["control"], cfg)
            for name, w in zip(state["view_names"], state["weights"]):
                weights.append((str(context), str(name), float(w)))
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete prediction matrix")
    return prediction, pd.DataFrame(weights, columns=["context", "prior", "weight"])


def verify(prediction, data=DATA):
    with np.load(data / "expected_predictions.npz") as ref, np.load(
        data / "axes.npz"
    ) as axes:
        for key in ("target", "context"):
            np.testing.assert_array_equal(axes[key], ref[key])
        np.testing.assert_allclose(prediction, ref["prediction"], atol=2e-6, rtol=0)
        return float(np.abs(prediction.astype(float) - ref["prediction"]).max())


if __name__ == "__main__":
    predicted, weights = refit()
    print(
        json.dumps(
            dict(
                dataset="Tian CRISPRi",
                split=101,
                shape=predicted.shape,
                contexts=weights.context.nunique(),
                max_abs_error=verify(predicted),
            )
        )
    )
