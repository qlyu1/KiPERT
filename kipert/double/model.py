from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse, linalg
from . import pairhead

VIEWS = [
    "DepMap",
    "GenePT",
    "STRING",
    "Reactome",
    "GO_BP",
    "CORUM",
    "CollecTRI",
    "HPA",
    "GO_MF",
    "GO_CC",
    "HPO",
    "WikiPathways",
]


def parts(t):
    return pairhead.parts(t)


def center(k):
    return k - k.mean(0)[None, :] - k.mean(1)[:, None] + k.mean()


class Bank:
    def __init__(self, root):
        root = Path(root)
        self.targets = np.load(root / "targets.npy").astype(str)
        self.index = {t: i for i, t in enumerate(self.targets)}
        self.raw = {}
        self.presence = {}
        self.sizes = {}
        for v in VIEWS:
            if v in ["DepMap", "GenePT", "STRING", "HPA"]:
                z = np.load(root / (v + ".npz"))
                self.raw[v] = z["x"].astype(float)
                self.presence[v] = z["present"]
            else:
                self.raw[v] = sparse.load_npz(root / (v + ".npz"))
                self.sizes[v] = np.load(root / (v + "_metadata.npz"))[
                    "public_set_sizes"
                ]

    def kernels(self, conditions):
        train = sorted({g for t in conditions for g in parts(t)})
        ix = np.array([self.index[t] for t in train])
        kernels = []
        coverage = np.zeros(len(self.targets), int)
        for v in VIEWS:
            raw = self.raw[v]
            if not sparse.issparse(raw):
                present = self.presence[v]
                use = ix[present[ix]]
                z = np.zeros(raw.shape)
                if len(use):
                    z[present] = (raw[present] - raw[use].mean(0)) / np.maximum(
                        raw[use].std(0), 1e-6
                    )
                norm = np.linalg.norm(z, axis=1)
                z = np.divide(
                    z, norm[:, None], out=np.zeros_like(z), where=norm[:, None] > 1e-12
                )
                kv = z @ z.T
            else:
                freq = (
                    np.asarray(raw[ix].sum(0)).ravel()
                    if v in ["Reactome", "GO_BP", "CORUM", "CollecTRI"]
                    else self.sizes[v]
                )
                z = raw.multiply(1 / np.sqrt(np.maximum(freq, 1))).tocsr()
                norm = np.sqrt(np.asarray(z.multiply(z).sum(1)).ravel())
                z = (
                    sparse.diags(
                        np.divide(1, norm, out=np.zeros_like(norm), where=norm > 1e-12)
                    )
                    @ z
                )
                kv = (z @ z.T).toarray()
            kernels.append(kv)
            coverage += norm > 1e-12
        return np.stack(kernels), coverage, train


def hweights(conditions, cfg):
    arity = np.array([len(parts(t)) for t in conditions])
    pair = bool(cfg["pair_intercept"] and np.any(arity == 2))
    h = (
        np.column_stack([arity, arity == 2]).astype(float)
        if pair
        else arity[:, None].astype(float)
    )
    w = np.ones(len(arity))
    for a in np.unique(arity):
        w[arity == a] = len(arity) / (len(np.unique(arity)) * sum(arity == a))
    return h, w, pair


def lift(node, components, query, train, cfg):
    return pairhead.condition_kernel(
        node, components, query, train, cfg["lambda_pair"], cfg["lambda_int"]
    )


def prepare(bank, conditions, Y, genes, cfg):
    views, coverage, components = bank.kernels(conditions)
    h, w, pair = hweights(conditions, cfg)
    b = linalg.solve(h.T @ (w[:, None] * h), h.T @ (w[:, None] * Y), assume_a="sym")
    residual = Y - h @ b
    keep = ~np.isin(genes, bank.targets)
    z = residual[:, keep]
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    unit = np.divide(z, norm, out=np.zeros_like(z), where=norm > 1e-12)
    target = center(unit @ unit.T)
    mask = ~np.eye(len(conditions), dtype=bool)
    tv = target[mask]
    align = []
    for node in views:
        a = center(lift(node, bank.targets, conditions, conditions, cfg))[mask]
        den = np.linalg.norm(a) * np.linalg.norm(tv)
        align.append(max(0.0, float(a @ tv / den)) if den > 1e-12 else 0.0)
    align = np.array(align)
    weights = 0.5 / 12 + 0.5 * (
        align / align.sum() if align.sum() > 1e-12 else np.ones(12) / 12
    )
    node = np.einsum("v,vij->ij", weights, views)
    K = lift(node, bank.targets, conditions, conditions, cfg)
    assert np.allclose(weights.sum(), 1) and np.min(weights) >= 1 / 24 - 1e-12
    return dict(
        node=node,
        K=K,
        h=h,
        w=w,
        pair=pair,
        alignment=align,
        weights=weights,
        coverage=coverage,
        training_components=components,
        conditions=conditions,
        Y=Y,
        cfg=cfg,
    )


def fit(pre, alpha):
    n = len(pre["Y"])
    h = pre["h"]
    A = np.block(
        [
            [pre["K"] + alpha * np.diag(1 / pre["w"]), h],
            [h.T, np.zeros((h.shape[1], h.shape[1]))],
        ]
    )
    rhs = np.vstack([pre["Y"], np.zeros((h.shape[1], pre["Y"].shape[1]))])
    sol = linalg.solve(A, rhs, assume_a="sym")
    err = np.linalg.norm(A @ sol - rhs) / max(np.linalg.norm(rhs), 1e-30)
    assert err < 1e-7
    return dict(
        pre=pre, alpha=alpha, coef=sol[:n], intercept=sol[n:], solve_error=float(err)
    )


def predict(bank, state, queries, control, return_parts=False):
    pre = state["pre"]
    cfg = pre["cfg"]
    _, shared, pair, tensor = pairhead.condition_kernel(
        pre["node"],
        bank.targets,
        queries,
        pre["conditions"],
        cfg["lambda_pair"],
        cfg["lambda_int"],
        True,
    )
    arity = np.array([len(parts(t)) for t in queries])
    first = shared @ state["coef"] + arity[:, None] * state["intercept"][0]
    cal = pair @ state["coef"]
    inter = tensor @ state["coef"]
    if pre["pair"]:
        cal += (arity == 2)[:, None] * state["intercept"][1]
    out = np.maximum(control + first + cal + inter, 0)
    assert np.isfinite(out).all()
    return (
        (out, dict(shared=first, calibration=cal, interaction=inter))
        if return_parts
        else out
    )


def context_data(panel, c):
    t = panel["train"]
    ix = t["context"] == c
    ctrl = panel["controls_fit"]
    ci = np.flatnonzero(ctrl["context"] == c)
    assert len(ci) == 1
    control = ctrl["expression"][ci[0]]
    conditions = t["target"][ix].astype(str).tolist()
    y = (t["expression"][ix] - control).astype(float)
    return conditions, y, control, t["deg_mask"][ix]


def score_validation(pred, truth, conditions, contexts, control, keep):
    rows = []
    for c in sorted(set(contexts)):
        ix = np.flatnonzero(contexts == c)
        pp = pred[ix][:, keep] - control[c][keep]
        yy = truth[ix][:, keep] - control[c][keep]
        n = len(ix)
        den = np.linalg.norm(pp, axis=1)[:, None] * np.linalg.norm(yy, axis=1)[None, :]
        cos = np.divide(pp @ yy.T, den, out=np.zeros((n, n)), where=den > 1e-15)
        l2 = np.maximum(
            np.sum(pp * pp, axis=1)[:, None]
            + np.sum(yy * yy, axis=1)[None, :]
            - 2 * pp @ yy.T,
            0,
        )
        for j, k in enumerate(ix):
            scores = []
            for dist in [1 - cos, l2]:
                d = dist[j]
                v = d[j]
                scores.append(
                    1
                    - ((d < v - 1e-12).sum() + 0.5 * ((abs(d - v) <= 1e-12).sum() - 1))
                    / (n - 1)
                    if n > 1
                    else np.nan
                )
            rows.append(
                dict(
                    target=conditions[k],
                    pds=np.mean(scores),
                    mse=np.mean((pred[k] - truth[k]) ** 2),
                )
            )
    f = pd.DataFrame(rows).groupby("target")[["pds", "mse"]].mean()
    return {
        "pds": float(f.pds.mean()) if f.pds.notna().any() else None,
        "mse": float(f.mse.mean()),
    }
