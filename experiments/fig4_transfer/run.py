"""Replay the four held-out SEH1L predictions using fitted source weights and maps."""
from pathlib import Path
import json
import numpy as np
from kipert.transfer.predict import transfer

DATA = Path(__file__).resolve().parents[2] / "examples/transfer_small"


def infer(line):
    with np.load(DATA / f"{line}.npz") as inputs:
        assert str(inputs["query"]) == "SEH1L" and line not in inputs["source"]
        bases = []
        for key in inputs["basis_keys"]:
            with np.load(DATA / f"basis_{key}.npz") as basis:
                bases.append(basis["U"])
        return transfer(
            inputs["delta"],
            inputs["control"],
            inputs["weights"],
            inputs["known"],
            bases,
            inputs["maps"],
        )


def verify():
    result = {}
    for line in ("k562", "hepg2", "rpe1", "jurkat"):
        pred = infer(line)
        with np.load(DATA / f"expected_{line}.npz") as ref:
            np.testing.assert_allclose(pred, ref["mean"], rtol=0, atol=2e-6)
            result[line] = dict(
                genes=len(pred), max_abs_error=float(np.abs(pred - ref["mean"]).max())
            )
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
