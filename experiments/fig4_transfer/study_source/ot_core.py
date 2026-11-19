"""Small, deterministic optimal-transport maps for control-cell embeddings.

Only NumPy is required.  Sinkhorn uses uniform empirical marginals and the
entropic objective <P,C> + epsilon * sum(P * (log(P) - 1)), where C is squared
Euclidean distance divided by the median strictly positive training distance.

Entropic transport of a distribution to itself need not be the identity map.
Responses are transported as differences of mapped controls, never as mapped
response vectors or as uncentered target barycenters.
"""

import numpy as np


def _points(value, name, dimension=None, min_rows=1):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] < min_rows or out.shape[1] < 1:
        raise ValueError("%s must be a nonempty two-dimensional matrix" % name)
    if dimension is not None and out.shape[1] != dimension:
        raise ValueError("%s has a different embedding dimension" % name)
    if not np.isfinite(out).all():
        raise ValueError("%s must contain only finite values" % name)
    return out


def _squared_distances(a, b):
    # A common origin limits cancellation for coordinates with a large offset.
    origin = b.mean(axis=0)
    aa, bb = a - origin, b - origin
    cost = (aa * aa).sum(axis=1)[:, None]
    cost = cost + (bb * bb).sum(axis=1)[None, :] - 2.0 * (aa @ bb.T)
    np.maximum(cost, 0.0, out=cost)
    return cost


def _logsumexp(a, axis):
    maximum = np.max(a, axis=axis, keepdims=True)
    value = maximum + np.log(np.exp(a - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(value, axis=axis)


def _softmax(a):
    z = a - np.max(a, axis=-1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=-1, keepdims=True)
    return z


def fit_sinkhorn(X, Y, epsilon=0.05, max_iter=2000, tol=1e-8,
                 check_every=10):
    """Fit a log-domain, uniform-marginal entropic transport plan.

    `tol` is the maximum *relative* marginal error.  A nonconverged fit is
    returned with converged=False; callers should report this and decide whether
    to increase max_iter.  `cost_scale` is fitted once and reused for queries.

    Returns a dictionary containing X, Y, epsilon, cost_scale, coupling, log_u,
    log_v, barycenter (also training_barycenter), n_iter, converged,
    marginal_error (absolute), marginal_error_relative, row/column marginal
    errors, and the normalized/unscaled transport costs.
    """
    X = _points(X, "X").copy()
    Y = _points(Y, "Y", X.shape[1]).copy()
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if int(check_every) != check_every or check_every < 1:
        raise ValueError("check_every must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    max_iter, check_every = int(max_iter), int(check_every)
    raw_cost = _squared_distances(X, Y)
    positive = raw_cost[raw_cost > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    cost = raw_cost / scale
    log_kernel = -cost / epsilon
    if not np.isfinite(log_kernel).all():
        raise ValueError("cost / epsilon exceeds floating-point range")
    n, m = len(X), len(Y)
    log_a, log_b = -np.log(float(n)), -np.log(float(m))
    log_u, log_v = np.zeros(n), np.zeros(m)
    converged = False
    for iteration in range(1, max_iter + 1):
        log_u = log_a - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_kernel + log_u[:, None], axis=0)
        # Gauge changes leave the coupling exactly unchanged algebraically.
        gauge = log_u.mean()
        log_u -= gauge
        log_v += gauge
        if iteration % check_every == 0 or iteration == max_iter:
            lp = log_kernel + log_u[:, None] + log_v[None, :]
            rows = np.exp(_logsumexp(lp, axis=1))
            columns = np.exp(_logsumexp(lp, axis=0))
            row_error = float(np.max(np.abs(rows - 1.0 / n)))
            column_error = float(np.max(np.abs(columns - 1.0 / m)))
            relative_error = max(row_error * n, column_error * m)
            if relative_error <= tol:
                converged = True
                break
    coupling = np.exp(log_kernel + log_u[:, None] + log_v[None, :])
    row_mass, column_mass = coupling.sum(axis=1), coupling.sum(axis=0)
    row_error = float(np.max(np.abs(row_mass - 1.0 / n)))
    column_error = float(np.max(np.abs(column_mass - 1.0 / m)))
    relative_error = max(row_error * n, column_error * m)
    converged = bool(converged and relative_error <= tol * (1.0 + 1e-7))
    # Conditional normalization gives exact training-query map parity even if
    # a caller deliberately stops Sinkhorn before its marginals converge.
    conditional = coupling / row_mass[:, None]
    y_origin = Y.mean(axis=0)
    barycenter = conditional @ (Y - y_origin) + y_origin
    return {
        "kind": "sinkhorn", "X": X, "Y": Y, "epsilon": epsilon,
        "cost_scale": scale, "coupling": coupling,
        "log_u": log_u, "log_v": log_v,
        "barycenter": barycenter, "training_barycenter": barycenter,
        "n_iter": iteration, "max_iter": max_iter, "tol": float(tol),
        "converged": converged,
        "marginal_error": max(row_error, column_error),
        "marginal_error_relative": relative_error,
        "row_marginal_error": row_error,
        "column_marginal_error": column_error,
        "transport_cost": float(np.sum(coupling * cost)),
        "transport_cost_unscaled": float(np.sum(coupling * raw_cost)),
        "cost_scale_rule": "median strictly positive squared training distance",
    }


def _regularized_covariance(points, shrinkage, eigen_floor):
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / (len(points) - 1)
    covariance = (covariance + covariance.T) * 0.5
    mean_variance = float(np.trace(covariance) / covariance.shape[0])
    regularized = ((1.0 - shrinkage) * covariance
                   + shrinkage * mean_variance * np.eye(covariance.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    # Relative floor in covariance units, with a small absolute floor for a
    # completely degenerate control cloud.  No mapping-gain truncation is used.
    floor = max(float(eigen_floor) * mean_variance, 1e-12)
    clipped = np.maximum(eigenvalues, floor)
    regularized = (eigenvectors * clipped) @ eigenvectors.T
    return regularized, clipped, eigenvectors, floor, covariance


def fit_gaussian_bures(X, Y, shrinkage=0.1, eigen_floor=1e-8):
    """Fit the Gaussian W2/Bures affine map T(x)=(x-mu_X)@A+mu_Y.

    Covariances use n-1 degrees of freedom, isotropic shrinkage, then a positive
    eigenvalue floor relative to their own mean variance (absolute minimum
    1e-12).  Mapping gains are never clipped.  Alignment diagnostics compare
    A.T @ regularized_cov_X @ A to regularized_cov_Y, not the raw covariances.
    """
    X = _points(X, "X", min_rows=2).copy()
    Y = _points(Y, "Y", X.shape[1], min_rows=2).copy()
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie between zero and one")
    if not np.isfinite(eigen_floor) or eigen_floor <= 0.0:
        raise ValueError("eigen_floor must be finite and positive")
    cs, values, vectors, floor_s, raw_s = _regularized_covariance(
        X, float(shrinkage), float(eigen_floor))
    ct, _, _, floor_t, raw_t = _regularized_covariance(
        Y, float(shrinkage), float(eigen_floor))
    root_s = (vectors * np.sqrt(values)) @ vectors.T
    inverse_root_s = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    middle = root_s @ ct @ root_s
    middle = (middle + middle.T) * 0.5
    values_m, vectors_m = np.linalg.eigh(middle)
    # Roundoff can make a theoretically positive eigenvalue slightly negative.
    root_m = (vectors_m * np.sqrt(np.maximum(values_m, 0.0))) @ vectors_m.T
    A = inverse_root_s @ root_m @ inverse_root_s
    A = (A + A.T) * 0.5
    error = A.T @ cs @ A - ct
    alignment_absolute = float(np.linalg.norm(error, ord="fro"))
    alignment_relative = alignment_absolute / max(
        float(np.linalg.norm(ct, ord="fro")), np.finfo(np.float64).tiny)
    return {
        "kind": "gaussian_bures", "X": X, "Y": Y,
        "A": A, "mean_X": X.mean(axis=0), "mean_Y": Y.mean(axis=0),
        "cov_X": cs, "cov_Y": ct, "raw_cov_X": raw_s, "raw_cov_Y": raw_t,
        "shrinkage": float(shrinkage), "eigen_floor": float(eigen_floor),
        "source_covariance_floor": floor_s, "target_covariance_floor": floor_t,
        "alignment_error": alignment_relative,
        "alignment_error_relative": alignment_relative,
        "alignment_error_absolute": alignment_absolute,
        "gain_eigenvalues": np.linalg.eigvalsh(A),
    }


def transform(state, query, mode="continuous", batch_size=256):
    """Map rows of query; one-dimensional input returns one vector.

    Sinkhorn modes:
      continuous: softmax(log_v - cost(query,Y)/epsilon) @ Y.
      ferradans: query + barycenter[nearest_X] - X[nearest_X].
    Gaussian states use their affine map; mode may be continuous or gaussian.
    """
    q = np.asarray(query, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None, :]
    q = _points(q, "query", state["X"].shape[1])
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if state["kind"] == "gaussian_bures":
        if mode not in ("continuous", "gaussian", "bures"):
            raise ValueError("Gaussian states require mode='gaussian' or 'continuous'")
        mapped = (q - state["mean_X"]) @ state["A"] + state["mean_Y"]
    else:
        if mode not in ("continuous", "ferradans"):
            raise ValueError("unknown Sinkhorn map mode: %s" % mode)
        mapped = np.empty_like(q)
        Y = state["Y"]
        origin = Y.mean(axis=0)
        shifts = state["barycenter"] - state["X"]
        for start in range(0, len(q), int(batch_size)):
            stop = min(start + int(batch_size), len(q))
            if mode == "continuous":
                logits = (state["log_v"][None, :]
                          - _squared_distances(q[start:stop], Y)
                          / (state["cost_scale"] * state["epsilon"]))
                mapped[start:stop] = _softmax(logits) @ (Y - origin) + origin
            else:
                nearest = np.argmin(_squared_distances(q[start:stop], state["X"]), axis=1)
                mapped[start:stop] = q[start:stop] + shifts[nearest]
    return mapped[0] if single else mapped


def transport_response(state, delta_latent, anchors, mode="continuous",
                       query_batch_size=16, anchor_batch_size=128):
    """Return mean_a[T(a+delta)-T(a)] with uniform supplied anchor weights.

    Supports one delta vector or a matrix of deltas.  Blocks over both queries
    and anchors; no query x anchor x target x dimension array is materialized.
    For continuous entropic maps the average conditional target probability is
    accumulated before multiplying by target coordinates.  This avoids a costly
    matrix product for every anchor.  Exact zero deltas return exact zeros.
    """
    delta = np.asarray(delta_latent, dtype=np.float64)
    single = delta.ndim == 1
    if single:
        delta = delta[None, :]
    delta = _points(delta, "delta_latent", state["X"].shape[1])
    anchors = _points(anchors, "anchors", state["X"].shape[1])
    for name, value in (("query_batch_size", query_batch_size),
                        ("anchor_batch_size", anchor_batch_size)):
        if int(value) != value or value < 1:
            raise ValueError("%s must be a positive integer" % name)
    qb, ab = int(query_batch_size), int(anchor_batch_size)
    zero = np.all(delta == 0.0, axis=1)
    if state["kind"] == "gaussian_bures":
        if mode not in ("continuous", "gaussian", "bures"):
            raise ValueError("Gaussian states require mode='gaussian' or 'continuous'")
        result = delta @ state["A"]
    elif mode == "continuous":
        Y = state["Y"]
        centered_y = Y - Y.mean(axis=0)
        denominator = state["cost_scale"] * state["epsilon"]
        base_logits = state["log_v"][None, :] - _squared_distances(anchors, Y) / denominator
        baseline_probability = _softmax(base_logits).mean(axis=0)
        result = np.zeros_like(delta)
        active = np.flatnonzero(~zero)
        for start in range(0, len(active), qb):
            rows = active[start:start + qb]
            # All other delta terms in squared distance are constant across
            # target cells within a row and therefore cancel inside softmax.
            bias = 2.0 * (delta[rows] @ centered_y.T) / denominator
            probability_sum = np.zeros((len(rows), len(Y)))
            for offset in range(0, len(anchors), ab):
                logits = base_logits[None, offset:offset + ab, :] + bias[:, None, :]
                probability_sum += _softmax(logits).sum(axis=1)
            average_probability = probability_sum / len(anchors)
            result[rows] = (average_probability - baseline_probability[None, :]) @ centered_y
    elif mode == "ferradans":
        X = state["X"]
        centered_x = X - X.mean(axis=0)
        shifts = state["barycenter"] - X
        base_cost = _squared_distances(anchors, X)
        baseline_shift = shifts[np.argmin(base_cost, axis=1)].mean(axis=0)
        result = np.zeros_like(delta)
        active = np.flatnonzero(~zero)
        for start in range(0, len(active), qb):
            rows = active[start:start + qb]
            bias = -2.0 * (delta[rows] @ centered_x.T)
            shift_sum = np.zeros((len(rows), X.shape[1]))
            for offset in range(0, len(anchors), ab):
                cost = base_cost[None, offset:offset + ab, :] + bias[:, None, :]
                nearest = np.argmin(cost, axis=-1)
                shift_sum += shifts[nearest].sum(axis=1)
            result[rows] = delta[rows] + shift_sum / len(anchors) - baseline_shift
    else:
        raise ValueError("unknown map mode: %s" % mode)
    result[zero] = 0.0
    return result[0] if single else result
