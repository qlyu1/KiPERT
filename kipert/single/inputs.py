"""Prepared-prior reader and original pseudobulk aggregation for selected replay."""
from pathlib import Path
import numpy as np
from scipy import sparse, stats

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


class Bank:
    def __init__(self, prepared_prior_directory):
        self.path = Path(prepared_prior_directory)
        self.targets = np.load(self.path / "targets.npy")
        self.index = {g: i for i, g in enumerate(self.targets)}
        self.raw = {}
        self.presence = {}
        self.sizes = {}
        for v in VIEWS:
            if v in ["DepMap", "GenePT", "STRING", "HPA"]:
                z = np.load(self.path / (v + ".npz"))
                self.raw[v] = z["x"].astype(float)
                self.presence[v] = z["present"]
            else:
                self.raw[v] = sparse.load_npz(self.path / (v + ".npz"))
                self.sizes[v] = np.load(self.path / (v + "_metadata.npz"))[
                    "public_set_sizes"
                ]


def make_rows(x, m, ctrl, cv, cn):
    out = []
    labels = []
    for t, ii in m.groupby("target", sort=True).indices.items():
        xx = x[ii]
        mu = np.asarray(xx.mean(0)).ravel()
        var = np.maximum(np.asarray(xx.power(2).mean(0)).ravel() - mu**2, 0)
        se = np.sqrt(var / len(ii) + cv / cn)
        stat = np.divide(abs(mu - ctrl), se, out=np.zeros_like(mu), where=se > 1e-12)
        out.append((t, mu - ctrl, np.argsort(-stat, kind="stable")[:100]))
        v1 = var / max(len(ii) - 1, 1)
        v0 = cv / max(cn - 1, 1)
        s = v1 + v0
        tt = np.divide(
            abs(mu - ctrl), np.sqrt(s), out=np.zeros_like(mu), where=s > 1e-20
        )
        tt[(s <= 1e-20) & (abs(mu - ctrl) > 1e-12)] = np.inf
        df = np.divide(
            s * s,
            v1 * v1 / max(len(ii) - 1, 1) + v0 * v0 / max(cn - 1, 1),
            out=np.ones_like(mu),
            where=s > 1e-20,
        )
        p = 2 * stats.t.sf(tt, df)
        order = np.argsort(p)
        adj = np.empty(len(p))
        adj[order] = np.minimum.accumulate(
            (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
        )[::-1]
        labels.append((adj < 0.05) & (abs(mu - ctrl) > 0.1))
    return out, np.stack(labels) if labels else np.empty((0, x.shape[1]), bool)
