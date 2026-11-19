"""Apply the fitted Gaussian transport maps to measured source responses."""
import numpy as np


def transfer(delta, control, weights, known, bases, maps):
    if (
        len(delta) != len(bases)
        or len(delta) != len(maps)
        or delta.shape[1:] != control.shape
    ):
        raise ValueError("Inconsistent source or expression axes")
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)
    if (weights < 0).any() or np.any(weights[~known]):
        raise ValueError("Invalid source weights")
    mapped = []
    for response, basis, matrix in zip(delta, bases, maps):
        correction = (response @ basis @ (matrix - np.eye(matrix.shape[0]))) @ basis.T
        mapped.append(response + correction)
    return control + weights @ np.stack(mapped)
