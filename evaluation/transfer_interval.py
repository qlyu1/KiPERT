"""Re-run the paired gene-cluster bootstrap used for the Fig. 4 PDS comparison."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data/fig4"
LINES = ("k562", "hepg2", "rpe1", "jurkat")


def check():
    settings = json.loads((DATA / "paired_bootstrap_reference.json").read_text())
    rng = np.random.default_rng(20261113)
    draws = 5000
    scores = {}
    for line in LINES:
        for method in ("UC", "PMean"):
            df = pd.read_csv(DATA / f"{method}_{line}_per_condition.csv").set_index(
                "perturbation"
            )
            scores[line, method] = df.pds_l1_exclude_target.sort_index()
        delta = scores[line, "UC"] - scores[line, "PMean"]
        # The original analysis drew within-line intervals before the joint bootstrap.
        indices = rng.integers(len(delta), size=(draws, len(delta)))
        within = np.quantile(delta.to_numpy()[indices].mean(1), [0.025, 0.975])
        row = next(r for r in settings["by_context"] if r["context"] == line)
        np.testing.assert_allclose(
            within, [row["mean_delta_ci_low"], row["mean_delta_ci_high"]], atol=1e-12
        )
    genes = sorted(set.union(*(set(s.index) for s in scores.values())))
    weights = rng.multinomial(
        len(genes), np.full(len(genes), 1 / len(genes)), size=draws
    )
    means = {}
    for key, s in scores.items():
        values = s.reindex(genes).to_numpy()
        present = np.isfinite(values)
        w = weights[:, present]
        means[key] = w @ values[present] / w.sum(1)
    model, baseline = (
        np.mean([means[line, m] for line in LINES], axis=0) for m in ("UC", "PMean")
    )
    difference = np.quantile(model - baseline, [0.025, 0.975])
    np.testing.assert_allclose(
        difference, settings["equal_context_macro"]["mean_delta_ci"], atol=1e-12
    )
    ratio = np.quantile(100 * (model / baseline - 1), [0.025, 0.975]).tolist()
    manuscript = json.loads(Path(__file__).with_name("manuscript_version.json").read_text())
    paper = manuscript["fig4g_relative_pds_95ci_percent"]
    return dict(
        draws=draws,
        seed=20261113,
        unit="perturbation gene across target lines",
        difference_ci_verified=difference.tolist(),
        ratio_ci_percent_this_replay=ratio,
        ratio_ci_percent_manuscript=paper,
        ratio_endpoints_match_at_printed_precision=[
            round(x, 2) == y for x, y in zip(ratio, paper, strict=True)
        ],
    )
