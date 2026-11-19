"""Frozen D14 leave-one-sample-batch-out observed-source KiPERT transfer.

Predictions only: this script never loads the held-out batch's delta/cell_var
files, computes scores, chooses model settings from outcomes, or pools batches.
Original Gaussian-OT and GP-confidence equations use explicit Cai adaptations:
equal 12 Cai priors, count-limited OT rank, and batches in one biological context.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.linalg import cho_factor, cho_solve, eigh


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("KiPERT_OT_Cai_prior_adaptation", "equal_source_delta_mean",
           "same_weight_no_OT", "zero_control")
THRESHOLDS = (5, 10)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2) + "\n")


def load_core(path):
    spec = importlib.util.spec_from_file_location("cai_original_ot_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit_gaussian_bures


def default_core():
    candidates = [
        ROOT.parent / "stage_transfer/src/vendor_original/ot_core.py",
        Path("@STUDY@/uc_cross_context_transfer/experiments/"
             "pie_uc_state_benchmark_20261106/runs/uc/k562/src/ot_core.py"),
    ]
    return next((p for p in candidates if p.exists()), candidates[-1])


def load_fold(prepared, heldout):
    """Read controls/counts for all batches; expression responses for sources only."""
    prepared = Path(prepared)
    reads = []

    def read(path, mmap=False):
        reads.append(str(path.resolve()))
        return np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)

    with read(prepared / "axis.npz") as axis:
        targets = axis["targets"].astype(str)
        genes = axis["genes"].astype(str)
        batches = axis["batches"].astype(str)
        mask = axis["evaluation_gene_mask"].astype(bool)
    n_targets, n_genes, n_batches = len(targets), len(genes), len(batches)
    if not (0 <= heldout < n_batches):
        raise ValueError("Held-out batch index out of range")
    assert len(set(targets)) == n_targets and len(set(genes)) == n_genes
    assert len(set(batches)) == n_batches and mask.shape == (n_genes,)
    controls, counts, sources = [], [], {}
    for b in range(n_batches):
        folder = prepared / ("batch_" + str(b))
        ctrl = np.asarray(read(folder / "controls.npy"), dtype=np.float64)
        count = np.asarray(read(folder / "n.npy"))
        assert ctrl.ndim == 2 and ctrl.shape[1] == n_genes and len(ctrl) >= 2
        assert np.isfinite(ctrl).all()
        assert count.shape == (n_targets,) and np.isfinite(count).all()
        assert (count >= 0).all() and np.equal(count, np.floor(count)).all()
        controls.append(ctrl)
        counts.append(count.astype(np.int64))
        if b == heldout:
            continue
        use = np.flatnonzero(count >= THRESHOLDS[0])
        if not len(use):
            continue
        # mmap plus indexing avoids reading excluded low-count response rows.
        delta = read(folder / "delta.npy", mmap=True)
        cell_var = read(folder / "cell_var.npy", mmap=True)
        assert delta.shape == cell_var.shape == (n_targets, n_genes)
        y = np.array(delta[use], dtype=np.float64, copy=True)
        var = np.array(cell_var[use], dtype=np.float64, copy=True)
        del delta, cell_var
        assert np.isfinite(y).all() and np.isfinite(var).all()
        assert (var >= 0).all()
        sources[b] = dict(indices=use, delta=y, cell_var=var,
                          n=count[use].astype(np.int64))
    forbidden = {str((prepared / ("batch_" + str(heldout)) / name).resolve())
                 for name in ("delta.npy", "cell_var.npy")}
    assert not (forbidden & set(reads))
    return dict(targets=targets, genes=genes, batches=batches, mask=mask,
                controls=controls, counts=np.stack(counts, axis=1),
                sources=sources, heldout=heldout, paths_read=reads,
                forbidden_paths=sorted(forbidden))


def confidence(kernel, y, cell_var, counts, controls):
    """Exact full-kernel form of the original source GP relative variance.

    Query effects are the measured source effects. GP mean predictions are not
    used; GP uncertainty only determines source weights. NT uncertainty enters
    the original confidence noise rule, not a biological replication claim.
    """
    noise = (cell_var / counts[:, None]
             + controls.var(axis=0, ddof=1) / len(controls)).mean(axis=1)
    scale = max(float(np.var(y, axis=0).mean()), 1e-12)
    diagonal = np.maximum(noise / scale, 1e-6)
    factor = cho_factor(kernel + np.diag(diagonal), lower=True, check_finite=False)
    solved = cho_solve(factor, kernel, check_finite=False)
    posterior = np.diag(kernel) - np.einsum("ij,ji->i", kernel, solved)
    assert posterior.min() > -1e-8
    relative = np.maximum(posterior, 0) / np.diag(kernel)
    # Independent primal calculation checks parity with original Q/chol form.
    eigenvalues, vectors = eigh(kernel, check_finite=False)
    keep = eigenvalues > max(eigenvalues.max() * 1e-12, 1e-12)
    features = vectors[:, keep] * np.sqrt(eigenvalues[keep])
    primal = cho_factor(np.eye(features.shape[1])
                        + features.T @ (features / diagonal[:, None]),
                        lower=True, check_finite=False)
    solved_primal = cho_solve(primal, features.T, check_finite=False).T
    relative_primal = np.einsum("ij,ij->i", features, solved_primal) / np.diag(kernel)
    np.testing.assert_allclose(relative, relative_primal, atol=1e-9, rtol=1e-8)
    assert np.isfinite(relative).all() and (relative <= 1 + 1e-8).all()
    audit = dict(n_source_targets=len(y), source_scale=scale,
                 kernel_rank=int(keep.sum()), noise_min=float(noise.min()),
                 noise_max=float(noise.max()), noise_floor_count=int((noise / scale < 1e-6).sum()),
                 relative_variance_min=float(relative.min()),
                 relative_variance_max=float(relative.max()),
                 dual_primal_max_error=float(abs(relative - relative_primal).max()))
    return relative, audit


def transport(delta, source, target, fit_gaussian_bures):
    """Original response OT in a control-only, count-limited joint PCA basis."""
    n_genes = delta.shape[1]
    a = source - source.mean(axis=0)
    b = target - target.mean(axis=0)
    rank = min(128, len(a) - 1, len(b) - 1, n_genes)
    combined = np.vstack((a / np.sqrt(2 * len(a)), b / np.sqrt(2 * len(b))))
    eigenvalues, vectors = eigh(combined @ combined.T,
                               subset_by_index=[len(combined) - rank, len(combined) - 1],
                               check_finite=False, driver="evr")
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    if eigenvalues[-1] <= max(eigenvalues[0] * 1e-12, 1e-12):
        raise ValueError("Control-only requested PCA basis is numerically rank deficient; no automatic model fallback")
    basis = (combined.T @ vectors) / np.sqrt(eigenvalues)
    np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=1e-9)
    fit = fit_gaussian_bures(a @ basis, b @ basis, shrinkage=0., eigen_floor=1e-8)
    assert np.isfinite(fit["A"]).all()
    if fit["alignment_error_relative"] >= 1e-6:
        raise ValueError("Original Gaussian OT covariance alignment failed tolerance")
    projected = delta @ basis
    correction = (projected @ (fit["A"] - np.eye(rank))) @ basis.T
    mapped = delta + correction
    # Response transport equals the difference of original affine maps.
    probe = np.vstack((np.zeros(n_genes), delta[0])) @ basis
    affine = (probe - fit["mean_X"]) @ fit["A"] + fit["mean_Y"]
    np.testing.assert_allclose(affine[1] - affine[0], projected[0] @ fit["A"],
                               atol=1e-8, rtol=1e-8)
    norm = np.einsum("ij,ij->i", delta, delta)
    mapped_norm = np.einsum("ij,ij->i", mapped, mapped)
    correction_norm = np.einsum("ij,ij->i", correction, correction)
    denominator = np.maximum(norm, 1e-30)
    raw_source = np.linalg.eigvalsh(fit["raw_cov_X"])
    raw_target = np.linalg.eigvalsh(fit["raw_cov_Y"])
    audit = dict(source_control_n=len(a), target_control_n=len(b), rank=rank,
                 source_covariance_rank=int((raw_source > max(raw_source.max() * 1e-12, 1e-12)).sum()),
                 target_covariance_rank=int((raw_target > max(raw_target.max() * 1e-12, 1e-12)).sum()),
                 source_floor_count=int((raw_source < fit["source_covariance_floor"]).sum()),
                 target_floor_count=int((raw_target < fit["target_covariance_floor"]).sum()),
                 source_covariance_floor=float(fit["source_covariance_floor"]),
                 target_covariance_floor=float(fit["target_covariance_floor"]),
                 gain_min=float(fit["gain_eigenvalues"].min()),
                 gain_max=float(fit["gain_eigenvalues"].max()),
                 alignment_error=float(fit["alignment_error_relative"]),
                 retained_control_variance=float(eigenvalues.sum() / np.einsum("ij,ij->", combined, combined)),
                 shrinkage=0., eigen_floor=1e-8, pooled_target_perturbations_used=False)
    per_target = dict(raw_effect_energy=norm, mapped_effect_energy=mapped_norm,
                      correction_energy=correction_norm,
                      mapped_to_raw_norm=np.sqrt(mapped_norm / denominator),
                      correction_to_raw_norm=np.sqrt(correction_norm / denominator),
                      basis_energy_fraction=np.einsum("ij,ij->i", projected, projected) / denominator)
    return mapped, audit, per_target, dict(U=basis, A=fit["A"], eigenvalues=eigenvalues)


def predict_fold(data, prior_builder, fit_gaussian_bures, artifact_dir=None):
    targets, genes, batches = data["targets"], data["genes"], data["batches"]
    n_targets, n_genes, n_batches = len(targets), len(genes), len(batches)
    heldout = data["heldout"]
    sources = data["sources"]
    assert heldout not in sources
    if not sources:
        raise ValueError("No source batch with eligible target measurements")
    fit_indices = np.unique(np.concatenate([v["indices"] for v in sources.values()]))
    views_kernel, views, coverage = prior_builder(targets[fit_indices].tolist(), targets.tolist())
    assert views_kernel.shape == (12, n_targets, n_targets)
    kernel = views_kernel.mean(axis=0) + np.ones((n_targets, n_targets))
    kernel = (kernel + kernel.T) / 2
    assert np.isfinite(kernel).all() and np.linalg.eigvalsh(kernel).min() > -1e-8
    relative = np.full((n_targets, n_batches), np.nan)
    source_mapped, gp_audit, map_audit = {}, [], []
    diagnostics = []
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
    for source, item in sources.items():
        indices = item["indices"]
        rel, gp = confidence(kernel[np.ix_(indices, indices)], item["delta"],
                             item["cell_var"], item["n"], data["controls"][source])
        relative[indices, source] = rel
        gp.update(source_index=source, source_batch=str(batches[source]),
                  source_target_indices=indices)
        gp_audit.append(gp)
        mapped, audit, target_diagnostic, map_arrays = transport(
            item["delta"], data["controls"][source], data["controls"][heldout], fit_gaussian_bures)
        source_mapped[source] = mapped
        audit.update(source_index=source, source_batch=str(batches[source]),
                     target_index=heldout, target_batch=str(batches[heldout]))
        map_audit.append(audit)
        diagnostics.append(dict(source_index=source, target_indices=indices, **target_diagnostic))
        if artifact_dir is not None:
            np.savez_compressed(artifact_dir / ("source_" + str(source) + "_map.npz"), **map_arrays)
            np.savez_compressed(artifact_dir / ("source_" + str(source) + "_effect_diagnostics.npz"),
                                target_indices=indices, **target_diagnostic)
        del map_arrays
    result = {}
    for threshold in THRESHOLDS:
        eligible = data["counts"] >= threshold
        eligible[:, heldout] = False
        # Excludes a hypothetical source without fitted GP state as well.
        eligible &= np.isfinite(relative)
        valid = ((data["counts"][:, heldout] >= threshold)
                 & (eligible.sum(axis=1) >= 2))
        raw_weight = np.zeros((n_targets, n_batches))
        raw_weight[eligible] = 1. / (relative[eligible] + .05)
        denominator = raw_weight.sum(axis=1, keepdims=True)
        weight = np.divide(raw_weight, denominator, out=np.zeros_like(raw_weight), where=denominator > 0)
        weight[~valid] = 0.
        equal = np.divide(eligible.astype(float), eligible.sum(axis=1, keepdims=True),
                          out=np.zeros_like(raw_weight), where=eligible.sum(axis=1, keepdims=True) > 0)
        equal[~valid] = 0.
        assert not weight[:, heldout].any() and not weight[~eligible].any()
        np.testing.assert_allclose(weight[valid].sum(axis=1), 1., atol=1e-12)
        prediction = {name: np.zeros((n_targets, n_genes)) for name in METHODS}
        for source, item in sources.items():
            ix = item["indices"]
            prediction[METHODS[0]][ix] += weight[ix, source, None] * source_mapped[source]
            prediction[METHODS[1]][ix] += equal[ix, source, None] * item["delta"]
            prediction[METHODS[2]][ix] += weight[ix, source, None] * item["delta"]
        for value in prediction.values():
            assert np.isfinite(value[valid]).all()
            value[~valid] = np.nan
        result[threshold] = dict(prediction=prediction, valid=valid, weights=weight,
                                 equal_weights=equal, source_count=eligible.sum(axis=1))
    audit = dict(heldout_index=heldout, heldout_batch=str(batches[heldout]),
                 paths_read=data["paths_read"], forbidden_response_paths=data["forbidden_paths"],
                 target_perturbation_responses_read=False,
                 prior_normalization_fit_target_indices=fit_indices,
                 prior_views=views, prior_coverage=coverage,
                 prior_normalization_source_counts_only=True, temperature=None,
                 biological_context_count=1, geometry_ell=1.,
                 confidence_fit_count_threshold=5,
                 n10_rule="Restrict sources/query by n>=10, then renormalize unchanged n5-fitted source confidence; no GP or OT refit.",
                 confidence=gp_audit, maps=map_audit)
    if artifact_dir is not None:
        np.savez_compressed(artifact_dir / "confidence.npz", relative_variance=relative,
                            kernel=kernel, prior_coverage=coverage,
                            prior_views=np.asarray(views), fit_target_indices=fit_indices)
        dump(artifact_dir / "audit.json", audit)
    return result, audit


def self_test(fit_gaussian_bures):
    """Small structural tests, including absent held-out response files."""
    rng = np.random.default_rng(2026111207)
    n_targets, n_genes, n_batches, heldout = 6, 4, 4, 2
    targets = np.array(["t" + str(i) for i in range(n_targets)])
    features = rng.normal(size=(n_targets, 5))
    node = features @ features.T / 5

    def prior_builder(train, all_targets):
        assert set(train) <= set(all_targets)
        return np.repeat(node[None, :, :], 12, axis=0), ["v" + str(i) for i in range(12)], np.full(n_targets, 12)

    with tempfile.TemporaryDirectory(prefix="kipert_d14_smoke_") as folder:
        prepared = Path(folder)
        np.savez(prepared / "axis.npz", targets=targets,
                 genes=np.array(["g" + str(i) for i in range(n_genes)]),
                 batches=np.array(["b" + str(i) for i in range(n_batches)]),
                 evaluation_gene_mask=np.ones(n_genes, bool))
        for b in range(n_batches):
            directory = prepared / ("batch_" + str(b))
            directory.mkdir()
            np.save(directory / "controls.npy", rng.normal(size=(10 + b, n_genes)))
            count = np.array([5, 10, 12, 20, 4, 11])
            if b == 0:
                count[2] = 7
            if b == 1:
                count[1] = 8
            if b == 3:
                count[3] = 8
            np.save(directory / "n.npy", count)
            np.save(directory / "delta.npy", rng.normal(size=(n_targets, n_genes)))
            np.save(directory / "cell_var.npy", rng.uniform(.2, 1., size=(n_targets, n_genes)))
        data = load_fold(prepared, heldout)
        first, audit = predict_fold(data, prior_builder, fit_gaussian_bures)
        # Deleting held-out expression/variance files must leave inference intact.
        for name in ("delta.npy", "cell_var.npy"):
            (prepared / ("batch_" + str(heldout)) / name).unlink()
        second, _ = predict_fold(load_fold(prepared, heldout), prior_builder, fit_gaussian_bures)
        for threshold in THRESHOLDS:
            expected = np.array([threshold == 5, True, True, True, False, True])
            np.testing.assert_array_equal(first[threshold]["valid"], expected)
            for method in METHODS:
                np.testing.assert_array_equal(first[threshold]["prediction"][method],
                                              second[threshold]["prediction"][method])
            for i in np.flatnonzero(expected):
                direct_mean = np.mean([v["delta"][np.flatnonzero(v["indices"] == i)[0]]
                                       for b, v in data["sources"].items()
                                       if data["counts"][i, b] >= threshold], axis=0)
                np.testing.assert_allclose(first[threshold]["prediction"][METHODS[1]][i], direct_mean)
                direct_weighted = sum(first[threshold]["weights"][i, b]
                                      * v["delta"][np.flatnonzero(v["indices"] == i)[0]]
                                      for b, v in data["sources"].items())
                np.testing.assert_allclose(first[threshold]["prediction"][METHODS[2]][i], direct_weighted)
        assert first[5]["weights"][2, 0] > 0 and first[10]["weights"][2, 0] == 0
        ctrl = rng.normal(size=(20, n_genes))
        delta = rng.normal(size=(3, n_genes))
        mapped, _, _, _ = transport(delta, ctrl, ctrl, fit_gaussian_bures)
        np.testing.assert_allclose(mapped, delta, atol=1e-10, rtol=1e-10)
        return dict(status="PASS", absent_heldout_response_files=True,
                    count_rules=True, equal_source_mean=True, identity_OT=True,
                    GP_dual_primal_parity=True, fixed_threshold_no_refit=True,
                    max_GP_parity_error=max(q["dual_primal_max_error"] for q in audit["confidence"]))


def run(args, fit_gaussian_bures):
    prepared, output = args.prepared.resolve(), args.output.resolve()
    application_protocol = prepared.parent / "protocol.json"
    if not application_protocol.exists():
        raise FileNotFoundError("Freeze the root application protocol before inference: " + str(application_protocol))
    application_protocol_hash = sha(application_protocol)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite nonempty prediction output: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.prior_src.resolve()))
    from prior_adapter import kernels
    with np.load(prepared / "axis.npz", allow_pickle=False) as axis:
        targets, genes, batches = [axis[k].astype(str) for k in ("targets", "genes", "batches")]
        evaluation_mask = axis["evaluation_gene_mask"].astype(bool)
    n_targets, n_genes, n_batches = len(targets), len(genes), len(batches)
    protocol = dict(status="FROZEN_BEFORE_INFERENCE", task="D14 same-target leave-one-sample-batch-out response completion",
                    task_scope="Observed source target; neither unseen-target nor time transfer prediction.",
                    batch_replication="sample_batch identity is not assumed to represent independent animals or biological replicates.",
                    heldout_inputs="Only NT control cells, target identities and counts. No held-out delta or cell variance files.",
                    response_source="Observed batch-matched target mean delta; no posterior-mean imputation.",
                    temperature=None, temperature_selection="Fixed in advance; no target or source outcome selection.",
                    geometry_ell=1., biological_context_count=1,
                    confidence="Equal 12 source-normalized Cai prior kernels plus intercept; original full-kernel relative GP posterior variance and 1/(variance+.05) weighting.",
                    prior_adaptation="Cai priors replace the original fixed human 512D basis; unchanged transfer equations do not imply an unchanged frozen original model.",
                    OT="delta+(delta@U@(A-I))@U.T; control-only joint PCA; rank min(128,sourceNT-1,targetNT-1,G); shrinkage0; eigen_floor1e-8; no gain/output clipping.",
                    confidence_noise="Mean over genes of target cell_var/n plus source NT sample variance/nNT, divided by source across-target response variance; original diagonal floor1e-6.",
                    thresholds=THRESHOLDS, minimum_eligible_sources=2,
                    strict_n10="Restrict source/query counts only, renormalize source weights; GP remains n5 fitted and OT unchanged.",
                    methods=METHODS, output_shape=[n_targets, n_batches, n_genes],
                    output_type="float64 mean-expression delta relative to target-batch NT; invalid rows are NaN.",
                    evaluation="External evaluator only; no aggregation across batches or scoring in prediction script.",
                    original_core=str(args.ot_core.resolve()), original_core_sha256=sha(args.ot_core),
                    application_protocol_path=str(application_protocol),
                    application_protocol_sha256=application_protocol_hash,
                    script_sha256=sha(__file__), axis_sha256=sha(prepared / "axis.npz"),
                    manifest_sha256=sha(prepared / "manifest.json") if (prepared / "manifest.json").exists() else None,
                    prior_adapter_sha256=sha(args.prior_src / "prior_adapter.py"),
                    prior_inputs_sha256=sha(args.prior_src / "prior_inputs.py"))
    dump(output / "protocol.json", protocol)
    np.savez_compressed(output / "axis.npz", targets=targets, genes=genes, batches=batches,
                        evaluation_gene_mask=evaluation_mask)
    arrays = {}
    auxiliary = {}
    for threshold in THRESHOLDS:
        directory = output / ("n" + str(threshold))
        directory.mkdir()
        arrays[threshold] = {method: open_memmap(directory / (method + ".npy"), mode="w+",
                                               dtype=np.float64, shape=(n_targets, n_batches, n_genes))
                             for method in METHODS}
        # Each fold below writes every target row, including NaN invalid rows.
        auxiliary[threshold] = dict(valid=np.zeros((n_targets, n_batches), bool),
                                    source_weights=np.zeros((n_targets, n_batches, n_batches)),
                                    equal_source_weights=np.zeros((n_targets, n_batches, n_batches)),
                                    source_count=np.zeros((n_targets, n_batches), np.int64))
    audits = []
    for heldout in range(n_batches):
        data = load_fold(prepared, heldout)
        np.testing.assert_array_equal(data["targets"], targets)
        np.testing.assert_array_equal(data["genes"], genes)
        np.testing.assert_array_equal(data["batches"], batches)
        result, audit = predict_fold(data, kernels, fit_gaussian_bures,
                                     output / ("fold_" + str(heldout)))
        for threshold in THRESHOLDS:
            for method in METHODS:
                arrays[threshold][method][:, heldout, :] = result[threshold]["prediction"][method]
                arrays[threshold][method].flush()
            for saved, returned in (("valid", "valid"), ("source_weights", "weights"),
                                    ("equal_source_weights", "equal_weights"), ("source_count", "source_count")):
                auxiliary[threshold][saved][:, heldout] = result[threshold][returned]
        audit["input_sha256"] = {p: sha(p) for p in audit["paths_read"]}
        dump(output / ("fold_" + str(heldout)) / "audit.json", audit)
        audits.append(audit)
        print("FOLD_SEALED", heldout, str(batches[heldout]),
              {"n" + str(t): int(result[t]["valid"].sum()) for t in THRESHOLDS}, flush=True)
        del data, result
    del arrays
    for threshold, values in auxiliary.items():
        directory = output / ("n" + str(threshold))
        for key, value in values.items():
            np.save(directory / (key + ".npy"), value)
    dump(output / "fold_audit.json", audits)
    assert sha(application_protocol) == application_protocol_hash, "Application protocol changed during inference"
    predictions = {str(path.relative_to(output)): sha(path)
                   for threshold in THRESHOLDS
                   for path in sorted((output / ("n" + str(threshold))).glob("*.npy"))}
    seal = dict(status="SEALED", time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                target_response_access_in_respective_outer_fold=False,
                protocol_sha256=sha(output / "protocol.json"),
                application_protocol_sha256=application_protocol_hash,
                script_sha256=sha(__file__), prediction_sha256=predictions,
                valid_queries={"n" + str(t): int(auxiliary[t]["valid"].sum()) for t in THRESHOLDS},
                fold_audit_sha256=sha(output / "fold_audit.json"), scoring_performed=False)
    dump(output / "PREDICTIONS_SEALED.json", seal)
    print("PREDICTIONS_SEALED", seal["valid_queries"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "prepared")
    parser.add_argument("--output", type=Path, default=ROOT / "results_transfer")
    parser.add_argument("--prior-src", type=Path, default=ROOT / "src")
    parser.add_argument("--ot-core", type=Path, default=default_core())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    fit_gaussian_bures = load_core(args.ot_core)
    if args.self_test:
        print(json.dumps(self_test(fit_gaussian_bures), indent=2), flush=True)
    else:
        run(args, fit_gaussian_bures)


if __name__ == "__main__":
    main()
