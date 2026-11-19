"""Original aligned_raw single-KiPERT equations for fitted-response estimation.

The historical build() entry point forbids train/query identity overlap. This
module adds an explicit fitted-response task interface while retaining its
12-view transforms, response alignment, centered ridge, missing-prior fallback,
nonnegative projection, and float32 output rounding. It does not read files or
accept controls/reference cells. Hyperparameter selection is the caller's
input-only procedure, not the historical held-target PDS selection procedure.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse


ALPHAS = (0.01, 0.1, 1.0)
SHRINKAGES = (0., 0.25, 0.5, 0.75, 1.)  # coefficient on observed target residual
SELECTION_TIE_TOLERANCE = 1e-12  # Absolute MSE tolerance frozen in root protocol.
CANONICAL_METHOD = "aligned_raw"


def center(kernel):
    return kernel - kernel.mean(0)[None, :] - kernel.mean(1)[:, None] + kernel.mean()


def view_kernels(bank, train):
    """Exact Figure2 view preprocessing, fitted on supplied target identities."""
    ix = np.array([bank.index[g] for g in train])
    kernels, coverage = [], np.zeros(len(bank.targets), int)
    for view, raw in bank.raw.items():
        if not sparse.issparse(raw):
            present = bank.presence[view]
            use = ix[present[ix]]
            z = np.zeros(raw.shape)
            if len(use):
                z[present] = (raw[present] - raw[use].mean(0)) / np.maximum(raw[use].std(0), 1e-6)
            norm = np.linalg.norm(z, axis=1)
            z = np.divide(z, norm[:, None], out=np.zeros_like(z), where=norm[:, None] > 1e-12)
            kernel = z @ z.T
        else:
            frequency = (np.asarray(raw[ix].sum(0)).ravel()
                         if view in ["Reactome", "GO_BP", "CORUM", "CollecTRI"]
                         else bank.sizes[view])
            z = raw.multiply(1 / np.sqrt(np.maximum(frequency, 1))).tocsr()
            norm = np.sqrt(np.asarray(z.multiply(z).sum(1)).ravel())
            z = sparse.diags(np.divide(1, norm, out=np.zeros_like(norm), where=norm > 1e-12)) @ z
            kernel = (z @ z.T).toarray()
        kernels.append(kernel)
        coverage += norm > 1e-12
    return np.stack(kernels), ix, coverage, list(bank.raw)


def _columns(cols, n_genes, genes=None):
    if cols is None:
        return slice(None)
    cols = np.asarray(cols)
    if cols.ndim != 1:
        raise ValueError("Output columns must be one-dimensional")
    if cols.dtype == bool:
        if cols.shape != (n_genes,):
            raise ValueError("Boolean output selector must cover the full gene axis")
        cols = np.flatnonzero(cols)
    elif cols.dtype.kind in "USO":
        if genes is None:
            raise ValueError("Gene names require ModelContext(..., genes=...)")
        lookup = {g: i for i, g in enumerate(genes)}
        try:
            cols = np.array([lookup[str(g)] for g in cols], int)
        except KeyError as exc:
            raise ValueError("Unknown output gene: " + str(exc)) from exc
    elif cols.dtype.kind not in "iu":
        raise ValueError("Output selector must contain integer indices, names, or booleans")
    cols = cols.astype(int, copy=False)
    if not len(cols) or len(set(cols.tolist())) != len(cols):
        raise ValueError("Output gene selector must be nonempty and unique")
    if (cols < 0).any() or (cols >= n_genes).any():
        raise ValueError("Output gene index outside the fitted gene axis")
    return cols


class ModelContext:
    """Cache identity-only prior transforms for one batch's fixed training IDs.

    ``alignment_mask`` must be frozen before fitting. It excludes all assayed
    target genes and the prespecified alias safeguards. Passing eight output
    genes to predict_fitted never changes this full alignment mask.
    """

    def __init__(self, bank, train_targets, alignment_mask, genes=None):
        self.train_targets = tuple(str(t) for t in train_targets)
        if len(self.train_targets) < 2 or len(set(self.train_targets)) != len(self.train_targets):
            raise ValueError("At least two distinct training target identities are required")
        self.alignment_mask = np.array(alignment_mask, dtype=bool, copy=True)
        if self.alignment_mask.ndim != 1 or not self.alignment_mask.any():
            raise ValueError("A nonempty frozen alignment gene mask is required")
        self.n_genes = len(self.alignment_mask)
        self.genes = None if genes is None else tuple(str(g) for g in genes)
        if self.genes is not None:
            if len(self.genes) != self.n_genes or len(set(self.genes)) != self.n_genes:
                raise ValueError("Gene names must uniquely match the full alignment-mask axis")
        views, indices, coverage, names = view_kernels(bank, self.train_targets)
        if len(views) != 12:
            raise ValueError("Original aligned_raw requires exactly twelve prior views")
        self.view_names = tuple(names)
        self.train_bank_indices = indices
        self.coverage = coverage[indices].copy()
        self.missing = self.coverage == 0
        self.train_views = views[:, indices][:, :, indices].copy()
        if not np.isfinite(self.train_views).all():
            raise ValueError("Nonfinite prior kernel")
        self.off_diagonal = ~np.eye(len(self.train_targets), dtype=bool)
        # These centered prior vectors depend only on target IDs, never cells.
        self.centered_view_vectors = np.stack([center(k)[self.off_diagonal] for k in self.train_views])
        self.centered_view_norms = np.linalg.norm(self.centered_view_vectors, axis=1)

    def fit(self, means_absolute):
        means = np.asarray(means_absolute, dtype=np.float64)
        expected = (len(self.train_targets), self.n_genes)
        if means.shape != expected or not np.isfinite(means).all():
            raise ValueError("Input-only means must be a finite matrix of shape " + str(expected))
        intercept = means.mean(0)
        residual = means - intercept
        response = residual[:, self.alignment_mask]
        norm = np.linalg.norm(response, axis=1, keepdims=True)
        unit = np.divide(response, norm, out=np.zeros_like(response), where=norm > 1e-12)
        target_kernel = center(unit @ unit.T)
        target_vector = target_kernel[self.off_diagonal]
        target_norm = np.linalg.norm(target_vector)
        # Keep the canonical per-view dot product order for numerical parity.
        alignment = []
        for vec, vec_norm in zip(self.centered_view_vectors, self.centered_view_norms):
            denominator = vec_norm * target_norm
            alignment.append(max(0., float(vec @ target_vector / denominator))
                             if denominator > 1e-12 else 0.)
        alignment = np.array(alignment)
        normalized = (alignment / alignment.sum()
                      if alignment.sum() > 1e-12 else np.ones(12) / 12)
        weights = .5 * np.ones(12) / 12 + .5 * normalized
        kernel = np.einsum("v,vij->ij", weights, self.train_views)
        centered_kernel = center(kernel)
        eigenvalues, eigenvectors = np.linalg.eigh(centered_kernel)
        eigenvalues = np.maximum(eigenvalues, 0.)
        return FittedState(context=self, intercept=intercept, residual=residual,
                           alignment=alignment, weights=weights,
                           centered_kernel=centered_kernel,
                           eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    def metadata(self):
        return dict(method=CANONICAL_METHOD, task="Fitted response estimation for already measured targets",
                    train_targets=list(self.train_targets), view_names=list(self.view_names),
                    prior_coverage=self.coverage.tolist(), missing_prior_targets=[
                        t for t, missing in zip(self.train_targets, self.missing) if missing],
                    n_genes=self.n_genes, n_alignment_genes=int(self.alignment_mask.sum()),
                    alpha_grid=list(ALPHAS), shrinkage_grid=list(SHRINKAGES),
                    selection_tie_tolerance=SELECTION_TIE_TOLERANCE,
                    prior_normalization="Fixed training-target IDs only; unchanged across cell splits using those same IDs.",
                    output="Absolute mean expression; nonnegative projection and canonical float32-to-float64 conversion.",
                    control_or_reference_inputs=False,
                    model_scope="Original aligned_raw estimator; new in-sample task interface and input-only cardiac readout hyperparameter selection.")


@dataclass
class FittedState:
    context: ModelContext
    intercept: np.ndarray
    residual: np.ndarray
    alignment: np.ndarray
    weights: np.ndarray
    centered_kernel: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    def predict_fitted(self, alpha, cols=None):
        alpha = float(alpha)
        if alpha not in ALPHAS:
            raise ValueError("Use the frozen original alpha grid " + str(ALPHAS))
        selected = _columns(cols, self.context.n_genes, self.context.genes)
        # For query=train, original query-centered cross-kernel equals K_centered.
        u, ev = self.eigenvectors, self.eigenvalues
        smoother = (self.centered_kernel @ u / (ev + alpha)) @ u.T
        fitted = self.intercept[selected] + smoother @ self.residual[:, selected]
        fitted[self.context.missing] = self.intercept[selected]
        return np.maximum(fitted, 0.).astype(np.float32).astype(float)

    def audit(self):
        return dict(alignment=self.alignment.tolist(), weights=self.weights.tolist(),
                    n_positive_alignment=int((self.alignment > 0).sum()),
                    prior_views=list(self.context.view_names),
                    missing_prior_targets=[t for t, missing in zip(self.context.train_targets, self.context.missing) if missing],
                    n_training_targets=len(self.context.train_targets),
                    n_alignment_genes=int(self.context.alignment_mask.sum()),
                    centered_kernel_rank=int((self.eigenvalues > max(float(self.eigenvalues.max()) * 1e-12, 1e-12)).sum()))


def fit(train_targets, means_absolute, *, context):
    """Function interface with an explicit row-identity check."""
    if tuple(str(t) for t in train_targets) != context.train_targets:
        raise ValueError("Mean rows do not match the cached context target order")
    return context.fit(means_absolute)


def predict_fitted(state, alpha, cols=None):
    return state.predict_fitted(alpha, cols=cols)


def shrinkage_prediction(means_absolute, coefficient, cols=None):
    """Same-data target-mean shrinkage, with the target included in the intercept.

    coefficient=1 preserves the observed target mean; coefficient=0 predicts
    the common unweighted training-target mean. No reference outcomes enter.
    """
    coefficient = float(coefficient)
    if coefficient not in SHRINKAGES:
        raise ValueError("Use the frozen shrinkage grid " + str(SHRINKAGES))
    means = np.asarray(means_absolute, dtype=np.float64)
    if means.ndim != 2 or not np.isfinite(means).all():
        raise ValueError("Finite input-only mean matrix required")
    selected = _columns(cols, means.shape[1])
    means = means[:, selected]
    common = means.mean(0)
    return np.maximum(common + coefficient * (means - common), 0.)


def select_minimum_mse(scores, values, stronger):
    """Deterministic task-specific tie rule; it is not the old PDS tie rule.

    Caller supplies input-only, target-equal and fold-equal validation MSEs.
    MSE within the protocol-frozen absolute tolerance1e-12 of the minimum
    selects larger alpha, or smaller shrinkage residual coefficient. The
    tolerance is fixed before inference, not adjusted after observing outcomes.
    """
    score = np.asarray(scores, float)
    values = np.asarray(values, float)
    if score.shape != values.shape or score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("One finite validation MSE for each candidate is required")
    if stronger not in ("larger", "smaller"):
        raise ValueError("stronger must be 'larger' (alpha) or 'smaller' (residual coefficient)")
    tied = np.flatnonzero(score <= score.min() + SELECTION_TIE_TOLERANCE)
    index = tied[np.argmax(values[tied]) if stronger == "larger" else np.argmin(values[tied])]
    return float(values[index]), int(index)


def _load_original(path):
    spec = importlib.util.spec_from_file_location("canonical_uc_specific_methods", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def self_test(original_path):
    """Test fitted predictions against the unchanged canonical query code.

    Duplicate feature-identical query IDs make query/train names disjoint for
    the original API while evaluating exactly the fitted training locations.
    The duplicate IDs are confined to this synthetic test.
    """
    original = _load_original(original_path)
    rng = np.random.default_rng(2026111301)
    n, g = 8, 13
    train = ["t" + str(i) for i in range(n)]
    query = ["q" + str(i) for i in range(n)]
    names = ["DepMap", "GenePT", "STRING", "Reactome", "GO_BP", "CORUM",
             "CollecTRI", "HPA", "GO_MF", "GO_CC", "HPO", "WikiPathways"]
    bank = SimpleNamespace(targets=np.array(train + query), index={t: i for i, t in enumerate(train + query)},
                           raw={}, presence={}, sizes={})
    for view in names:
        if view in ["DepMap", "GenePT", "STRING", "HPA"]:
            raw = rng.normal(size=(n, 7))
            present = np.ones(n, bool)
            present[-1] = False
            raw[-1] = 0.
            bank.raw[view] = np.vstack((raw, raw))
            bank.presence[view] = np.r_[present, present]
        else:
            raw = (rng.random((n, 10)) < .35).astype(float)
            raw[-1] = 0.
            bank.raw[view] = sparse.csr_matrix(np.vstack((raw, raw)))
            if view in ["GO_MF", "GO_CC", "HPO", "WikiPathways"]:
                bank.sizes[view] = rng.uniform(1., 20., size=10)
    means = rng.gamma(2., 1., size=(n, g))
    control = rng.uniform(.2, 1., size=g)
    mask = np.ones(g, bool)
    mask[[0, 4]] = False
    genes = ["g" + str(i) for i in range(g)]
    ctx = ModelContext(bank, train, mask, genes=genes)
    state = fit(train, means, context=ctx)
    canonical = original.build(bank, [(t, y - control, []) for t, y in zip(train, means)], query, mask)
    np.testing.assert_allclose(state.weights, canonical["weights"], atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(state.alignment, canonical["alignment"], atol=1e-14, rtol=1e-13)
    maximum = 0.
    cols = np.array([1, 3, 5, 7, 8, 9, 11, 12])
    for alpha in ALPHAS:
        actual = predict_fitted(state, alpha)
        expected = original.predict_candidate(canonical, control,
                                              dict(kernel="aligned", head="raw", alpha=alpha, gain=1.))
        np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=1e-7)
        maximum = max(maximum, float(np.max(abs(actual - expected))))
        selected = predict_fitted(state, alpha, cols)
        np.testing.assert_array_equal(selected, actual[:, cols])
        by_name = predict_fitted(state, alpha, [genes[i] for i in cols])
        np.testing.assert_array_equal(by_name, selected)
        direct = state.intercept + state.centered_kernel @ np.linalg.solve(
            state.centered_kernel + alpha * np.eye(n), state.residual)
        direct[ctx.missing] = state.intercept
        direct = np.maximum(direct, 0.).astype(np.float32).astype(float)
        np.testing.assert_allclose(actual, direct, atol=5e-7, rtol=1e-7)
        np.testing.assert_array_equal(actual[-1], state.intercept.astype(np.float32).astype(float))
    np.testing.assert_allclose(shrinkage_prediction(means, 1.), means)
    np.testing.assert_allclose(shrinkage_prediction(means, 0.), np.broadcast_to(means.mean(0), means.shape))
    assert select_minimum_mse([1, 1, 1], ALPHAS, "larger")[0] == 1.
    assert select_minimum_mse([1] * 5, SHRINKAGES, "smaller")[0] == 0.
    assert select_minimum_mse([1., 1. + 5e-13, 1. + 2e-12], ALPHAS, "larger")[0] == .1
    assert select_minimum_mse([1. + 5e-13, 1., 1. + 2e-12, 2., 2.], SHRINKAGES, "smaller")[0] == 0.
    return dict(status="PASS", canonical_source_sha256=hashlib.sha256(Path(original_path).read_bytes()).hexdigest(),
                feature_identical_query_parity=True, control_intercept_cancellation=True,
                canonical_prediction_max_difference=maximum, independent_direct_ridge=True,
                selected_columns_match_full_prediction=True, original_missing_prior_fallback=True,
                shrinkage_endpoints=True, stronger_regularization_ties=True,
                selection_tie_tolerance=SELECTION_TIE_TOLERANCE,
                no_reference_or_control_arguments=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--original-module", type=Path, default=Path(
        "@STUDY@/uc_cross_context_transfer/experiments/cai_fig2ef_20261112/src/uc_specific_methods.py"))
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Import ModelContext/fit/predict_fitted for application runs; only --self-test is executable here.")
    print(json.dumps(self_test(args.original_module), indent=2), flush=True)
