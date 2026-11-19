"""Sealed-prediction evaluation of the frozen Cai low-coverage application.

No data are read at import.  --self-test uses synthetic arrays only.  The actual
evaluation opens reference files only after verifying the complete six-array
prediction seal.  This module never fits, selects, or modifies a predictor.
"""
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("KiPERT", "raw", "shrinkage")
BUDGETS = (5, 10)
BOOTSTRAPS = 2000
TOP_FRACTION = 0.20
SCRIPT_VERSION = "1.0-frozen-before-reference"
LIMITATIONS = [
    "The sampling unit for uncertainty is the perturbation target, not a cell or split.",
    "Twenty overlapping cell splits are repeated measurements, not independent n.",
    "sample_batch identities have not been mapped to independent animals or cultures.",
    "Intervals are conditional on the observed batches/cell pools and sealed predictions; "
    "shared controls and overlapping fitting targets are not separately resampled or refitted.",
    "Reference means and upper-quintile labels remain noisy measurements, not functional ground truth.",
    "This is retrospective low-coverage simulation by subsampling higher-coverage perturbations; "
    "performance does not establish applicability to truly fewer-than-five-cell targets or unmeasured perturbations.",
    "The 5-cell and 10-cell cohorts and reference cell sets differ; their comparison is sensitivity, not a causal budget effect.",
    "Calr is not a low-coverage query and cannot be counted as a denoising success.",
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def nanmean(values, axis=0):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    n = valid.sum(axis=axis)
    numerator = np.where(valid, values, 0.0).sum(axis=axis)
    out = np.full(np.shape(numerator), np.nan, dtype=float)
    return np.divide(numerator, n, out=out, where=n > 0)


def interval(draws):
    draws = np.asarray(draws, dtype=float)
    good = draws[np.isfinite(draws)]
    if not len(good):
        return (np.nan, np.nan, 0)
    lo, hi = np.quantile(good, [0.025, 0.975])
    return float(lo), float(hi), len(good)


def result_fields(point_and_draws, prefix):
    x = np.asarray(point_and_draws, dtype=float)
    lo, hi, n = interval(x[1:])
    return {prefix: float(x[0]), prefix + "_ci_low": lo,
            prefix + "_ci_high": hi, prefix + "_valid_bootstraps": n}


def weighted_midranks(values, weights):
    """Ranks of each original item under explicit integer replication.

    Returns one rank per item for every bootstrap draw.  Zero-weight items have
    ranks but cannot affect any subsequent weighted calculation.  Ties receive
    the mean rank of all replicated observations in the tied value group.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=np.int64)
    if weights.ndim != 2 or weights.shape[1] != len(values):
        raise ValueError("weights must have shape [draw, item]")
    if np.any(weights < 0) or not np.isfinite(values).all():
        raise ValueError("nonnegative integer weights and finite scores required")
    if not len(values):
        return np.empty_like(weights, dtype=float)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    groups = np.cumsum(np.r_[True, ordered[1:] != ordered[:-1]]) - 1
    group_weights = np.add.reduceat(weights[:, order], starts, axis=1)
    before = np.cumsum(group_weights, axis=1) - group_weights
    grouped_ranks = before + (group_weights + 1.0) / 2.0
    ranks = np.empty(weights.shape, dtype=float)
    ranks[:, order] = grouped_ranks[:, groups]
    return ranks


def weighted_spearman(prediction, reference, weights):
    """Explicit-replication Spearman; constant prediction=0, truth=NaN.

    Reference-constant draws are excluded jointly across methods.  Prediction
    constants with variable reference contribute zero information rather than
    being silently dropped from the macro statistic.
    """
    w = np.asarray(weights, dtype=np.int64)
    rp = weighted_midranks(prediction, w)
    rt = weighted_midranks(reference, w)
    total = w.sum(axis=1)
    mean = (total + 1.0) / 2.0
    dp = rp - mean[:, None]
    dt = rt - mean[:, None]
    vp = np.sum(w * dp * dp, axis=1)
    vt = np.sum(w * dt * dt, axis=1)
    cov = np.sum(w * dp * dt, axis=1)
    reference_valid = (total >= 2) & (vt > 0)
    pred_constant = reference_valid & (vp == 0)
    rho = np.full(len(w), np.nan)
    usable = reference_valid & (vp > 0)
    rho[usable] = cov[usable] / np.sqrt(vp[usable] * vt[usable])
    rho[pred_constant] = 0.0
    rho[usable] = np.clip(rho[usable], -1.0, 1.0)
    return rho, reference_valid, pred_constant


def weighted_top_counts(scores, weights, tie_order=None):
    """Copy counts selected by fixed top ceil(0.2*M), descending score.

    Ties use the declared target-name order (passed as integer tie_order), never
    truth or another method's scores.  Each duplicate target's copies are kept
    contiguous, which is identical to expanding the weighted sample explicitly.
    """
    scores = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=np.int64)
    if not np.isfinite(scores).all() or np.any(w < 0):
        raise ValueError("finite scores and nonnegative weights required")
    if tie_order is None:
        tie_order = np.arange(len(scores))
    order = np.lexsort((np.asarray(tie_order), -scores))
    total = w.sum(axis=1)
    quota = np.ceil(TOP_FRACTION * total).astype(int)
    ordered = w[:, order]
    before = np.cumsum(ordered, axis=1) - ordered
    chosen = np.minimum(np.maximum(quota[:, None] - before, 0), ordered)
    counts = np.zeros_like(w)
    counts[:, order] = chosen
    assert np.array_equal(counts.sum(axis=1), quota)
    return counts, quota, total


def pool_statistics(prediction, reference, weights, tie_order=None):
    """Point (first weights row) and bootstrap statistics for one pool."""
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    weights = np.asarray(weights, dtype=np.int64)
    rho, valid, constant = weighted_spearman(prediction, reference, weights)
    selected, quota, total = weighted_top_counts(prediction, weights, tie_order)
    truth_selected, _, _ = weighted_top_counts(reference, weights, tie_order)
    denom = np.maximum(quota, 1)
    n_denom = np.maximum(total, 1)
    overlap_count = np.minimum(selected, truth_selected).sum(axis=1)
    overlap = overlap_count / denom
    chance = quota / n_denom
    utility = (selected @ reference) / denom
    pool_mean = (weights @ reference) / n_denom
    ref_positive = reference > 0
    positive_truth_selected = truth_selected * ref_positive[None, :]
    positive_total = positive_truth_selected.sum(axis=1)
    positive_overlap = np.minimum(selected, positive_truth_selected).sum(axis=1)
    positive_recall = np.divide(positive_overlap, positive_total,
                                out=np.full(len(weights), np.nan), where=positive_total > 0)
    pred_mean = (weights @ prediction) / n_denom
    pred_var = np.sum(weights * (prediction[None, :] - pred_mean[:, None]) ** 2, axis=1) / n_denom
    truth_var = np.sum(weights * (reference[None, :] - pool_mean[:, None]) ** 2, axis=1) / n_denom
    amplitude = np.divide(np.sqrt(pred_var), np.sqrt(truth_var),
                          out=np.full(len(weights), np.nan), where=truth_var > 0)
    stats = {
        "rho": rho,
        "top20_overlap": overlap,
        "top20_random_expectation": chance,
        "top20_overlap_excess": overlap - chance,
        "top20_reference_utility": utility,
        "top20_reference_utility_above_pool": utility - pool_mean,
        "top20_positive_reference_recall": positive_recall,
        "top20_selected_reference_positive_fraction": (selected @ ref_positive.astype(float)) / denom,
        "pred_reference_sd_ratio": amplitude,
    }
    # The same reference-constant pools are removed for every method and every
    # decision-information statistic; no method-specific favorable omissions.
    for x in stats.values():
        x[~valid] = np.nan
    metadata = {
        "reference_valid": valid, "prediction_constant": constant,
        "selected_counts": selected, "reference_selected_counts": truth_selected,
        "quota": quota, "total_weight": total,
        "reference_positive_top_count": positive_total,
    }
    return stats, metadata


def target_macro(case_values, case_target_index, n_targets):
    """First mean over splits per case, then cases/batches per target."""
    values = np.asarray(case_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("case_values must be [split, case]")
    per_case = values.mean(axis=0)
    return np.array([per_case[np.asarray(case_target_index) == i].mean()
                     for i in range(n_targets)])


def macro_pool_statistics(by_batch):
    """Mean split statistics within each batch, then equally across batches."""
    return nanmean(np.stack([nanmean(np.stack(rows), axis=0)
                             for rows in by_batch.values()]), axis=0)


def verify_seal(root):
    """Fail closed before any reference file can be opened."""
    root = Path(root)
    pred = root / "predictions"
    path = pred / "PREDICTIONS_SEALED.json"
    if not path.is_file():
        raise RuntimeError("Complete PREDICTIONS_SEALED.json is absent; reference access prohibited")
    seal = json.loads(path.read_text())
    if seal.get("status") != "SEALED" or seal.get("reference_values_read") is not False:
        raise RuntimeError("Prediction seal does not certify reference-blind completion")
    required = {f"n{b}/{m}.npy" for b in BUDGETS for m in METHODS}
    if set(seal.get("prediction_sha256", {})) != required:
        raise RuntimeError("Seal must contain exactly both budgets and all three methods")
    expected_files = {
        "protocol_sha256": root / "protocol.json",
        "preparation_manifest_sha256": root / "prepared" / "COMPLETE.json",
        "model_sha256": root / "src" / "model.py",
        "predictor_sha256": root / "src" / "run_predictions.py",
        "fit_audit_sha256": pred / "fit_audits.json",
    }
    for field, file in expected_files.items():
        if sha256(file) != seal.get(field):
            raise RuntimeError(f"Sealed {field} no longer matches {file}")
    hashes = {}
    for name in sorted(required):
        hashes[name] = sha256(pred / name)
        if hashes[name] != seal["prediction_sha256"][name]:
            raise RuntimeError(f"Prediction checksum mismatch: {name}")
    preparation = json.loads((root / "prepared" / "COMPLETE.json").read_text())
    if preparation.get("status") != "COMPLETE" or preparation.get("protocol_sha256") != seal["protocol_sha256"]:
        raise RuntimeError("Preparation manifest and protocol do not match the seal")
    return seal, hashes, preparation


def full_expression_errors(prediction, reference, input_control, reference_control, mask):
    prediction = np.asarray(prediction[:, mask], dtype=float)
    reference = np.asarray(reference[:, mask], dtype=float)
    absolute = prediction - reference
    delta = absolute + (reference_control[mask] - input_control[mask])[None, :]
    return np.mean(absolute ** 2, axis=1), np.mean(delta ** 2, axis=1)


def evaluate_budget(root, budget, protocol, axis, weights, out, query_writer, reference_audits):
    """Reference access occurs only inside this function, after verify_seal."""
    prep = root / "prepared"
    prediction_dir = root / "predictions" / f"n{budget}"
    cases = pd.read_csv(prediction_dir / "cases.csv").sort_values("case_id").reset_index(drop=True)
    prepared_cases = pd.read_csv(prep / "cases.csv")
    prepared_cases = prepared_cases.loc[prepared_cases.budget == budget].sort_values("case_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(cases, prepared_cases)
    repeats = int(protocol["outer_splits"])
    genes = axis["genes"].astype(str)
    batches = axis["batches"].astype(str)
    mask = axis["evaluation_gene_mask"].astype(bool)
    targets = np.array(sorted(cases.target.unique()), dtype=str)
    target_to_index = {target: i for i, target in enumerate(targets)}
    target_index = np.array([target_to_index[t] for t in cases.target])
    programs = list(protocol["programs"])
    columns = {p: np.flatnonzero(np.isin(genes, protocol["programs"][p])) for p in programs}
    n, g = len(cases), len(genes)
    expected = protocol["expected_coverage"][str(budget)]
    assert n == expected["queries"] and len(targets) == expected["targets"]
    assert np.array_equal(cases.case_id.to_numpy(), np.arange(n))
    assert "Calr" not in set(targets)
    assert weights.shape == (BOOTSTRAPS + 1, len(targets))
    assert np.all(weights[0] == 1)
    assert np.all(weights.sum(axis=1) == len(targets))
    assert np.all(cases.n_input == budget) and np.all(cases.n_reference >= protocol["reference_min_cells"])
    for p, col in columns.items():
        assert len(col) == len(protocol["programs"][p])
        assert set(genes[col]) == set(protocol["programs"][p])
    predictions = {m: np.load(prediction_dir / f"{m}.npy", mmap_mode="r") for m in METHODS}
    for prediction in predictions.values():
        assert prediction.shape == (repeats, n, g) and prediction.dtype == np.dtype("float32")
    reference_abs = np.full((repeats, n, len(programs)), np.nan)
    reference_delta = np.full_like(reference_abs, np.nan)
    pred_abs = {m: np.full_like(reference_abs, np.nan) for m in METHODS}
    pred_delta = {m: np.full_like(reference_abs, np.nan) for m in METHODS}
    full_errors = {m: {scale: np.full((repeats, n), np.nan) for scale in ("absolute", "delta")}
                   for m in METHODS}
    ctrl_input_program = np.full_like(reference_abs, np.nan)
    ctrl_reference_program = np.full_like(reference_abs, np.nan)
    seen = np.zeros((repeats, n), dtype=int)
    split_audit = json.loads((prep / "split_audit.json").read_text())
    audit_lookup = {(int(a["budget"]), int(a["repeat"]), str(a["batch"])): a for a in split_audit}
    fit_audit = json.loads((root / "predictions" / "fit_audits.json").read_text())
    fit_lookup = {(int(a["budget"]), int(a["repeat"]), str(a["batch"])): a for a in fit_audit}

    for repeat in range(repeats):
        for b, batch in enumerate(batches):
            source = prep / f"n{budget}" / f"repeat_{repeat:02d}" / f"batch_{b}"
            input_path, reference_path = source / "input_meta.npz", source / "reference.npz"
            expected_audit = audit_lookup[(budget, repeat, str(batch))]
            input_hash, ref_hash = sha256(input_path), sha256(reference_path)
            assert input_hash == expected_audit["input_meta_sha256"]
            assert ref_hash == expected_audit["reference_sha256"]
            fit_record = fit_lookup[(budget, repeat, str(batch))]
            assert fit_record["reference_read"] is False
            sealed_input_hashes = [v for k, v in fit_record["input_hashes"].items()
                                   if Path(k).name == "input_meta.npz"]
            assert sealed_input_hashes == [input_hash]
            with np.load(input_path) as inp, np.load(reference_path) as ref:
                ids = inp["case_ids"].astype(int)
                assert np.array_equal(ids, ref["case_ids"])
                assert np.array_equal(cases.iloc[ids].target.to_numpy(dtype=str), inp["query_targets"].astype(str))
                assert np.array_equal(inp["query_targets"], ref["query_targets"])
                assert np.all(cases.iloc[ids].batch_index == b)
                assert np.array_equal(cases.iloc[ids].n_reference.to_numpy(), inp["query_n_reference"])
                in_cells, ref_cells = inp["input_cell_rows"], ref["reference_cell_rows"]
                in_controls, ref_controls = inp["ctrl_input_rows"], ref["ctrl_reference_rows"]
                assert not np.intersect1d(in_cells, ref_cells).size
                assert not np.intersect1d(in_controls, ref_controls).size
                assert not np.intersect1d(np.r_[in_cells, ref_cells], np.r_[in_controls, ref_controls]).size
                assert len(in_controls) + len(ref_controls) > 0
                assert len(ref_controls) == int(inp["ctrl_reference_n"])
                assert np.array_equal(np.bincount(ref["program_reference_group"], minlength=len(ids)), inp["query_n_reference"])
                truth = np.asarray(ref["absolute_mean"], dtype=float)
                control_i = np.asarray(inp["ctrl_input_mean"], dtype=float)
                control_r = np.asarray(ref["ctrl_reference_mean"], dtype=float)
                assert truth.shape == (len(ids), g) and control_i.shape == control_r.shape == (g,)
                assert np.isfinite(truth).all() and np.isfinite(control_i).all() and np.isfinite(control_r).all()
                for j, program in enumerate(programs):
                    cols = columns[program]
                    reference_abs[repeat, ids, j] = truth[:, cols].mean(axis=1)
                    reference_delta[repeat, ids, j] = truth[:, cols].mean(axis=1) - control_r[cols].mean()
                    ctrl_input_program[repeat, ids, j] = control_i[cols].mean()
                    ctrl_reference_program[repeat, ids, j] = control_r[cols].mean()
                for method, prediction in predictions.items():
                    values = np.asarray(prediction[repeat, ids], dtype=float)
                    assert np.isfinite(values).all() and np.min(values) >= 0
                    for j, program in enumerate(programs):
                        pred_abs[method][repeat, ids, j] = values[:, columns[program]].mean(axis=1)
                        pred_delta[method][repeat, ids, j] = (values[:, columns[program]].mean(axis=1)
                                                                          - control_i[columns[program]].mean())
                    absolute_error, delta_error = full_expression_errors(values, truth, control_i, control_r, mask)
                    full_errors[method]["absolute"][repeat, ids] = absolute_error
                    full_errors[method]["delta"][repeat, ids] = delta_error
                seen[repeat, ids] += 1
                reference_audits.append({"budget": budget, "repeat": repeat, "batch": str(batch),
                                         "reference_sha256": ref_hash, "input_meta_sha256": input_hash,
                                         "case_ids": ids.tolist(), "input_reference_overlap": 0,
                                         "input_reference_control_overlap": 0,
                                         "n_input_control": len(in_controls), "n_reference_control": len(ref_controls)})
        print(f"REFERENCE_SCORED budget={budget} repeat={repeat:02d}", flush=True)
    assert np.all(seen == 1)
    for x in [reference_abs, reference_delta, *pred_abs.values(), *pred_delta.values()]:
        assert np.isfinite(x).all()

    method_rows, target_rows, paired_rows, selection_rows = [], [], [], []
    estimates = {m: {} for m in METHODS}
    cardinal = programs.index("cardiac")
    for method in METHODS:
        case_metrics = {}
        for j, program in enumerate(programs):
            for scale, predicted, truth in (("absolute", pred_abs[method], reference_abs),
                                             ("delta", pred_delta[method], reference_delta)):
                case_metrics[f"{program}_{scale}_mse"] = (predicted[:, :, j] - truth[:, :, j]) ** 2
            case_metrics[f"{program}_direction_agreement"] = (
                np.sign(pred_delta[method][:, :, j]) == np.sign(reference_delta[:, :, j])).astype(float)
            case_metrics[f"{program}_predicted_delta_mean"] = pred_delta[method][:, :, j]
            case_metrics[f"{program}_reference_delta_mean"] = reference_delta[:, :, j]
        for scale in ("absolute", "delta"):
            case_metrics[f"non_target_expression_{scale}_mse"] = full_errors[method][scale]
        metrics_by_target = {name: target_macro(values, target_index, len(targets))
                             for name, values in case_metrics.items()}
        estimates[method] = {name: weights @ values / len(targets) for name, values in metrics_by_target.items()}
        for ti, target in enumerate(targets):
            target_rows.append({"budget": budget, "method": method, "target": target,
                                "n_cases": int(np.sum(target_index == ti)), "n_splits": repeats,
                                **{name: values[ti] for name, values in metrics_by_target.items()}})
        for repeat in range(repeats):
            for case_id, case in cases.iterrows():
                for j, program in enumerate(programs):
                    query_writer.writerow({
                        "budget": budget, "repeat": repeat, "case_id": case_id, "target": case.target,
                        "batch": case.batch, "n_input": case.n_input, "n_reference": case.n_reference,
                        "method": method, "program": program,
                        "predicted_absolute": pred_abs[method][repeat, case_id, j],
                        "reference_absolute": reference_abs[repeat, case_id, j],
                        "input_nt": ctrl_input_program[repeat, case_id, j],
                        "reference_nt": ctrl_reference_program[repeat, case_id, j],
                        "predicted_delta": pred_delta[method][repeat, case_id, j],
                        "reference_delta": reference_delta[repeat, case_id, j],
                        "absolute_squared_error": case_metrics[f"{program}_absolute_mse"][repeat, case_id],
                        "delta_squared_error": case_metrics[f"{program}_delta_mse"][repeat, case_id],
                        "direction_agreement": case_metrics[f"{program}_direction_agreement"][repeat, case_id],
                        "non_target_expression_absolute_mse": full_errors[method]["absolute"][repeat, case_id],
                        "non_target_expression_delta_mse": full_errors[method]["delta"][repeat, case_id],
                    })

        pool_values = {}
        reference_constant_count = prediction_constant_count = point_pool_count = 0
        bootstrap_reference_constant_count = np.zeros(BOOTSTRAPS, int)
        bootstrap_prediction_constant_count = np.zeros(BOOTSTRAPS, int)
        for b, batch in enumerate(batches):
            ids = np.flatnonzero(cases.batch_index.to_numpy() == b)
            if not len(ids):
                continue
            pool_targets = cases.iloc[ids].target.to_numpy(dtype=str)
            pool_weights = weights[:, target_index[ids]]
            tie_order = np.argsort(np.argsort(pool_targets, kind="stable"), kind="stable")
            for repeat in range(repeats):
                prediction = pred_delta[method][repeat, ids, cardinal]
                truth = reference_delta[repeat, ids, cardinal]
                stats, info = pool_statistics(prediction, truth, pool_weights, tie_order)
                point_pool_count += 1
                reference_constant_count += int(not info["reference_valid"][0])
                prediction_constant_count += int(info["prediction_constant"][0])
                bootstrap_reference_constant_count += ~info["reference_valid"][1:]
                bootstrap_prediction_constant_count += info["prediction_constant"][1:]
                for metric, values in stats.items():
                    pool_values.setdefault(metric, {}).setdefault(str(batch), []).append(values)
                selection_rows.append({
                    "budget": budget, "method": method, "batch": str(batch), "repeat": repeat,
                    "n_targets": len(ids), "quota": int(info["quota"][0]),
                    "reference_constant": not bool(info["reference_valid"][0]),
                    "prediction_constant": bool(info["prediction_constant"][0]),
                    "predicted_top_targets": "|".join(pool_targets[info["selected_counts"][0] > 0]),
                    "reference_top_targets": "|".join(pool_targets[info["reference_selected_counts"][0] > 0]),
                    "reference_positive_top_count": int(info["reference_positive_top_count"][0]),
                    **{metric: values[0] for metric, values in stats.items()},
                })
        for metric, by_batch in pool_values.items():
            estimates[method]["cardiac_" + metric] = macro_pool_statistics(by_batch)
        row = {"budget": budget, "method": method, "n_targets": len(targets), "n_cases": n,
               "n_splits": repeats, "n_batches": cases.batch.nunique(), "n_pools": point_pool_count,
               "reference_constant_pools": reference_constant_count,
               "prediction_constant_pools": prediction_constant_count,
               "prediction_constant_fraction_among_valid_reference": prediction_constant_count / max(1, point_pool_count - reference_constant_count),
               "bootstrap_reference_constant_pools_mean": bootstrap_reference_constant_count.mean(),
               "bootstrap_prediction_constant_pools_mean": bootstrap_prediction_constant_count.mean()}
        for name, values in estimates[method].items():
            row.update(result_fields(values, name))
        method_rows.append(row)

    benefit_metrics = ("cardiac_direction_agreement", "cardiac_rho", "cardiac_top20_overlap",
                       "cardiac_top20_overlap_excess", "cardiac_top20_reference_utility",
                       "cardiac_top20_reference_utility_above_pool", "cardiac_top20_positive_reference_recall")
    for baseline in ("raw", "shrinkage"):
        for name in estimates["KiPERT"]:
            loss_metric = name.endswith("_mse")
            if not loss_metric and name not in benefit_metrics:
                continue
            model, comparator = estimates["KiPERT"][name], estimates[baseline][name]
            gain = comparator - model if loss_metric else model - comparator
            relative = np.divide(gain, comparator, out=np.full_like(gain, np.nan), where=comparator != 0) if loss_metric else np.full_like(gain, np.nan)
            paired_rows.append({"budget": budget, "baseline": baseline, "metric": name,
                                "gain_definition": "baseline_minus_KiPERT" if loss_metric else "KiPERT_minus_baseline",
                                "KiPERT_point": model[0], "baseline_point": comparator[0],
                                **result_fields(gain, "gain"), **result_fields(relative, "relative_gain"),
                                "ci_supports_improvement": bool(interval(gain[1:])[0] > 0)})

    # Scalar summaries are useful for auditing the CSVs; large predicted gene
    # matrices remain read-only in their original seal and are not duplicated.
    np.savez_compressed(out / f"program_scores_n{budget}.npz", targets=targets, case_ids=cases.case_id.to_numpy(),
                        case_targets=cases.target.to_numpy(dtype=str), case_batches=cases.batch.to_numpy(dtype=str),
                        programs=np.array(programs), reference_absolute=reference_abs, reference_delta=reference_delta,
                        input_nt=ctrl_input_program, reference_nt=ctrl_reference_program,
                        **{m + "_absolute": pred_abs[m] for m in METHODS},
                        **{m + "_delta": pred_delta[m] for m in METHODS})
    return method_rows, target_rows, paired_rows, selection_rows


def make_decision(method_rows, paired_rows):
    methods = {(int(r["budget"]), r["method"]): r for r in method_rows}
    paired = {(int(r["budget"]), r["baseline"], r["metric"]): r for r in paired_rows}
    baselines = ("raw", "shrinkage")
    primary = methods[(5, "KiPERT")]
    sensitivity = methods[(10, "KiPERT")]
    gates = {
        "primary_delta_cardiac_mse_interval_improves_both": all(
            paired[(5, base, "cardiac_delta_mse")]["gain_ci_low"] > 0 for base in baselines),
        "primary_absolute_cardiac_mse_same_direction_both": all(
            paired[(5, base, "cardiac_absolute_mse")]["gain"] > 0 for base in baselines),
        "primary_cardiac_rank_interval_positive": primary["cardiac_rho_ci_low"] > 0,
        "primary_top20_overlap_excess_interval_positive": primary["cardiac_top20_overlap_excess_ci_low"] > 0,
        "sensitivity_delta_and_absolute_mse_not_reversed": all(
            paired[(10, base, metric)]["gain"] >= 0
            for base in baselines for metric in ("cardiac_delta_mse", "cardiac_absolute_mse")),
        "sensitivity_information_point_direction_not_reversed": (
            sensitivity["cardiac_rho"] > 0 and sensitivity["cardiac_top20_overlap_excess"] > 0),
    }
    # Direction/large-response preservation has no predeclared biological
    # noninferiority margin.  Report signs and intervals, and never infer
    # equivalence from a nonsignificant difference or invent a margin now.
    diagnostic_metrics = ("cardiac_direction_agreement", "cardiac_rho",
                          "cardiac_top20_positive_reference_recall", "cardiac_top20_reference_utility")
    diagnostics = []
    for budget in BUDGETS:
        for base in baselines:
            for metric in diagnostic_metrics:
                r = paired[(budget, base, metric)]
                diagnostics.append({"budget": budget, "baseline": base, "metric": metric,
                                    "gain": r["gain"], "ci_low": r["gain_ci_low"], "ci_high": r["gain_ci_high"],
                                    "interval_supports_deterioration": bool(r["gain_ci_high"] < 0),
                                    "point_deterioration": bool(r["gain"] < 0)})
    supported_harm = any(d["interval_supports_deterioration"] for d in diagnostics)
    quantitative_gate = all(gates.values())
    stronger = {}
    for metric in diagnostic_metrics:
        stronger[metric] = all(paired[(5, base, metric)]["gain_ci_low"] > 0 for base in baselines)
    # This is a transparent decision aid.  The frozen protocol's qualitative
    # phrase "strong responses retained" is not converted to a post-hoc
    # numerical noninferiority threshold.
    if not quantitative_gate:
        status = "INSUFFICIENT_FOR_POSITIVE_CARDIAC_APPLICATION"
    elif supported_harm:
        status = "ERROR_AND_INFORMATION_GATES_PASS_BUT_DECISION_HARM_REQUIRES_QUALIFIED_INTERPRETATION"
    else:
        status = "ERROR_AND_INFORMATION_GATES_PASS_FOR_LOW_COVERAGE_ESTIMATION"
    return {
        "status": status,
        "quantitative_estimation_and_information_gates_pass": quantitative_gate,
        "gates": gates,
        "decision_preservation_diagnostics": diagnostics,
        "any_interval_supported_direction_rank_retention_or_utility_harm": supported_harm,
        "stronger_primary_claim_supported_against_both_comparators": stronger,
        "retention_noninferiority_claim": "Not assessed: no biological noninferiority margin was prespecified.",
        "interpretation_rule": "Generic error reduction alone is insufficient. Report all cardiac decision diagnostics. "
                               "Only paired interval-supported gains justify ranking, direction, or selection-improvement claims.",
        "allowed_scope_if_supported": "Retrospective estimation of mean cardiac-program responses for already measured "
                                      "low-coverage perturbations in these D14 samples, beyond raw means and matched simple shrinkage.",
        "excluded_claims": ["unseen-target discovery", "independent animal-level validation", "functional iCM yield prediction",
                            "new Calr discovery", "noninferiority from nonsignificant differences"],
        "limitations": LIMITATIONS,
    }


def run_evaluation(root):
    root = Path(root).resolve()
    # Keep the barrier above every axis/prediction/reference read.  Reading the
    # manifest/hash files inside verify_seal does not disclose reference values.
    seal, prediction_hashes_before, preparation = verify_seal(root)
    protocol = json.loads((root / "protocol.json").read_text())
    if int(protocol["outer_splits"]) != 20 or int(protocol["primary_budget"]) != 5 or int(protocol["sensitivity_budget"]) != 10:
        raise RuntimeError("Unexpected frozen protocol; do not silently evaluate a different task")
    out = root / "evaluation"
    if out.exists():
        raise RuntimeError("Refusing to overwrite an existing evaluation directory")
    out.mkdir()
    evaluation_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with np.load(root / "prepared" / "axis.npz") as handle:
        axis = {k: handle[k] for k in handle.files}
    assert len(axis["genes"]) == int(preparation["n_genes"])
    assert int(axis["evaluation_gene_mask"].sum()) == int(preparation["evaluation_genes"])
    all_methods, all_targets, all_pairs, all_selections, reference_audits = [], [], [], [], []
    query_fields = ["budget", "repeat", "case_id", "target", "batch", "n_input", "n_reference", "method", "program",
                    "predicted_absolute", "reference_absolute", "input_nt", "reference_nt", "predicted_delta", "reference_delta",
                    "absolute_squared_error", "delta_squared_error", "direction_agreement",
                    "non_target_expression_absolute_mse", "non_target_expression_delta_mse"]
    bootstrap_metadata = {}
    with (out / "query_split_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=query_fields)
        writer.writeheader()
        for budget in BUDGETS:
            targets = np.array(sorted(pd.read_csv(root / "predictions" / f"n{budget}" / "cases.csv").target.unique()))
            seed = [int(protocol["seed"]), 2041, budget]
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            bootstrap_weights = rng.multinomial(len(targets), np.full(len(targets), 1.0 / len(targets)), size=BOOTSTRAPS)
            weights = np.vstack([np.ones(len(targets), dtype=int), bootstrap_weights])
            np.savez_compressed(out / f"target_bootstrap_n{budget}.npz", targets=targets, weights=bootstrap_weights, seed=np.array(seed))
            bootstrap_metadata[str(budget)] = {"seed_sequence": seed, "n_targets": len(targets), "draws": BOOTSTRAPS,
                                              "target_order": targets.tolist()}
            methods, target_rows, paired, selections = evaluate_budget(root, budget, protocol, axis, weights, out, writer, reference_audits)
            all_methods.extend(methods)
            all_targets.extend(target_rows)
            all_pairs.extend(paired)
            all_selections.extend(selections)
    for name, rows in (("method_summary", all_methods), ("target_summary", all_targets),
                       ("paired_checks", all_pairs), ("selection_summary", all_selections)):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    decision = make_decision(all_methods, all_pairs)
    dump(out / "decision.json", decision)
    prediction_hashes_after = {name: sha256(root / "predictions" / name) for name in prediction_hashes_before}
    assert prediction_hashes_before == prediction_hashes_after == seal["prediction_sha256"]
    assert sha256(root / "protocol.json") == seal["protocol_sha256"]
    dump(out / "reference_access_audit.json", reference_audits)
    integrity = {
        "status": "PASS", "script_version": SCRIPT_VERSION, "evaluation_started_utc": evaluation_start,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seal_time_utc": seal.get("time_utc"), "complete_seal_verified_before_reference_access": True,
        "all_prediction_hashes_unchanged": True, "prediction_sha256_before": prediction_hashes_before,
        "prediction_sha256_after": prediction_hashes_after,
        "protocol_sha256": seal["protocol_sha256"], "evaluation_script_sha256": sha256(__file__),
        "seal_sha256": sha256(root / "predictions" / "PREDICTIONS_SEALED.json"),
        "reference_files_verified_against_preparation_audit": len(reference_audits),
        "bootstrap": bootstrap_metadata,
        "mse_aggregation": "Mean splits within target-by-batch case; mean eligible batches within target; mean targets equally.",
        "ranking_aggregation": "Rank independently per batch/split; mean valid splits within batch; mean batches equally. "
                               "Each target-bootstrap draw retains its multiplicity in every batch/split.",
        "constant_prediction_rule": "Spearman=0 when reference varies; reported, never silently dropped.",
        "constant_reference_rule": "Exclude identical pools jointly for all methods and decision-information metrics.",
        "top_selection": "ceil(0.2*M), descending prediction/reference, ties by target name; integer copies in bootstrap. "
                         "Overlap is selected-copy intersection/K; random expectation K/M recalculated per draw/pool.",
        "paired_gain_orientation": "Positive is better: baseline-model for MSE; model-baseline for rank/direction/selection.",
        "intervals": "Two-sided percentile 95% conditional target-cluster intervals; no multiplicity-adjusted exploratory claims.",
        "limitations": LIMITATIONS,
        "outputs_sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    }
    dump(out / "integrity.json", integrity)
    print("EVALUATION_COMPLETE", decision["status"], flush=True)


def self_test():
    """All checks use synthetic data and temporary directories only."""
    def explicit_spearman(x, y):
        # Independent ordinary-sample rank implementation.  It neither uses
        # bootstrap weights nor calls any weighted evaluator routine.
        def rank(a):
            ordered = np.sort(a)
            return (np.searchsorted(ordered, a, side="left") +
                    np.searchsorted(ordered, a, side="right") + 1.0) / 2.0
        return float(np.corrcoef(rank(x), rank(y))[0, 1])
    rng = np.random.default_rng(81)
    weights = rng.multinomial(9, np.full(6, 1 / 6), size=200)
    weights = np.vstack([weights, [0, 0, 0, 0, 0, 0], [9, 0, 0, 0, 0, 0]])
    truth = np.array([0, 1, 1, -2, 3, 3.0])
    examples = [np.array([2, 2, 1, 0, 3, -1.0]), np.ones(6), truth.copy(), -truth]
    checked_ranks = checked_top = 0
    for prediction in examples:
        stats, metadata = pool_statistics(prediction, truth, weights)
        for i, w in enumerate(weights):
            indices = np.repeat(np.arange(len(w)), w)
            xp, yt = prediction[indices], truth[indices]
            if len(indices) < 2 or not np.ptp(yt):
                assert not metadata["reference_valid"][i] and np.isnan(stats["rho"][i])
            else:
                expected_rho = 0.0 if not np.ptp(xp) else explicit_spearman(xp, yt)
                np.testing.assert_allclose(stats["rho"][i], expected_rho, rtol=0, atol=2e-15)
            checked_ranks += 1
            k = math.ceil(TOP_FRACTION * len(indices))
            po = np.lexsort((indices, -xp))[:k]
            to = np.lexsort((indices, -yt))[:k]
            pc = np.bincount(indices[po], minlength=len(w))
            tc = np.bincount(indices[to], minlength=len(w))
            assert np.array_equal(metadata["selected_counts"][i], pc)
            assert np.array_equal(metadata["reference_selected_counts"][i], tc)
            if metadata["reference_valid"][i]:
                np.testing.assert_allclose(stats["top20_overlap"][i], len(np.intersect1d(po, to)) / k)
                np.testing.assert_allclose(stats["top20_reference_utility"][i], yt[po].mean())
                np.testing.assert_allclose(stats["top20_overlap_excess"][i], len(np.intersect1d(po, to)) / k - k / len(indices))
                positive_top = to[yt[to] > 0]
                if len(positive_top):
                    np.testing.assert_allclose(stats["top20_positive_reference_recall"][i],
                                               len(np.intersect1d(po, positive_top)) / len(positive_top))
            checked_top += 1
    # Unequal batch coverage must not give targets with more eligible cases more
    # weight.  Three cases belong to A and one to B; target mean is 6, not 4.
    case_values = np.array([[1, 3, 2, 10], [1, 3, 2, 10]], dtype=float)
    target_values = target_macro(case_values, np.array([0, 0, 0, 1]), 2)
    np.testing.assert_array_equal(target_values, [2, 10])
    assert target_values.mean() == 6
    # Macro pools likewise average repeats before batches, not valid row counts.
    macro = macro_pool_statistics({"a": [np.array([1., 2.]), np.array([3., np.nan])],
                                   "b": [np.array([10., 8.])]})
    np.testing.assert_allclose(macro, [6, 5])
    # Delta error must include the independent-control difference.  Shared NT
    # errors do not algebraically cancel between squared-error comparisons.
    pred = np.array([[2., 4., 8.]])
    ref = np.array([[1., 3., 4.]])
    ci, cr, mask = np.array([1., 1., 1.]), np.array([2., 0., 3.]), np.array([True, False, True])
    a, d = full_expression_errors(pred, ref, ci, cr, mask)
    np.testing.assert_allclose(a, [(1 + 16) / 2])
    np.testing.assert_allclose(d, [(4 + 36) / 2])
    # A missing seal must fail without requiring even a prepared/ directory.
    with tempfile.TemporaryDirectory(prefix="cai_eval_no_seal_") as tmp:
        try:
            verify_seal(tmp)
        except RuntimeError as error:
            assert "reference access prohibited" in str(error)
        else:
            raise AssertionError("Unsealed evaluation was not blocked")
    return {"status": "PASS", "synthetic_only": True, "real_reference_values_read": False,
            "weighted_spearman_expansion_checks": checked_ranks,
            "weighted_top_selection_expansion_checks": checked_top,
            "target_macro_unequal_batch_coverage": "PASS", "pool_macro_equal_batch_weight": "PASS",
            "independent_control_delta_error": "PASS", "missing_seal_barrier": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        report = self_test()
        if args.self_test_output:
            dump(args.self_test_output, report)
        print(json.dumps(report, indent=2))
        return
    if args.self_test_output:
        parser.error("--self-test-output requires --self-test")
    run_evaluation(args.root)


if __name__ == "__main__":
    main()
