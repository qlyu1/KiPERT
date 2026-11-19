from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

DS = ["S3", "S4", "S5", "S6", "S7"]
DN = dict(
    zip(DS, ["Tian CRISPRi", "Tian CRISPRa", "Adamson", "Norman singles", "Frangieh"])
)
CC = ["k562", "hepg2", "rpe1", "jurkat"]
CN = dict(zip(CC, ["K562", "HepG2", "RPE1", "Jurkat"]))
V = [
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
N_BOOT = 2000


def compare(
    frame,
    model,
    reference,
    metric,
    label,
    group="dataset",
    unit="target",
    denominator=None,
):
    """Same paired-ratio bootstrap as the existing manuscript percentage table."""
    keys = [group, unit]
    a = frame[frame.variant.eq(model)].set_index(keys)
    b = frame[frame.variant.eq(reference)].set_index(keys)
    assert a.index.is_unique and b.index.is_unique
    assert set(a.index) == set(b.index)
    b = b.reindex(a.index)
    ok = np.isfinite(a[metric]) & np.isfinite(b[metric])
    a, b = a.loc[ok], b.loc[ok]
    if denominator:
        assert np.allclose(a[denominator], b[denominator])
    groups = sorted(a.index.get_level_values(group).unique())
    ids = sorted(a.index.get_level_values(unit).unique())

    def matrix(s):
        return s.unstack(group).reindex(index=ids, columns=groups).to_numpy()

    av, bv = matrix(a[metric]), matrix(b[metric])
    dv = (
        matrix(a[denominator])
        if denominator
        else np.where(np.isfinite(av), 1.0, np.nan)
    )
    mask = np.isfinite(av) & np.isfinite(bv) & np.isfinite(dv)
    av, bv, dv = [np.where(mask, v, 0.0) for v in [av, bv, dv]]

    def macro(v, d):
        return np.nanmean(
            np.divide(v, d, out=np.full_like(v, np.nan), where=d > 0), axis=-1
        )

    ma, mb = float(macro(av.sum(0), dv.sum(0))), float(macro(bv.sum(0), dv.sum(0)))
    assert mb > 0
    seed = 2026111801 + int(
        hashlib.sha256(
            "|".join([label, model, reference, metric]).encode()
        ).hexdigest()[:7],
        16,
    )
    rng = np.random.default_rng(seed)
    rels = []
    for start in range(0, N_BOOT, 100):
        ix = rng.integers(len(ids), size=(min(100, N_BOOT - start), len(ids)))
        aa, bb = macro(av[ix].sum(1), dv[ix].sum(1)), macro(
            bv[ix].sum(1), dv[ix].sum(1)
        )
        rels.extend(100 * (aa / bb - 1))
    assert np.isfinite(rels).all()
    low, high = np.quantile(rels, [0.025, 0.975])
    return dict(
        comparison=label,
        model=model,
        reference=reference,
        metric="nMSE20" if denominator else metric,
        model_mean=ma,
        reference_mean=mb,
        relative_change_percent=100 * (ma / mb - 1),
        relative_ci_low_percent=low,
        relative_ci_high_percent=high,
        n_clusters=len(ids),
        n_paired_units=int(mask.sum()),
        groups=len(groups),
        bootstrap_seed=seed,
    )


def per_dataset(frame, model, reference, metric, label, group="dataset", unit="target"):
    order = DS if group == "dataset" else CC
    names = DN if group == "dataset" else CN
    rows = []
    for k in order:
        r = compare(
            frame[frame[group].eq(k)],
            model,
            reference,
            metric,
            label + "_" + k,
            group,
            unit,
        )
        r["dataset"] = k
        r["display_label"] = names[k]
        rows.append(r)
    r = compare(frame, model, reference, metric, label, group, unit)
    r["dataset"] = "macro"
    r["display_label"] = (
        "Equal-dataset mean" if group == "dataset" else "Equal-cell-line mean"
    )
    rows.append(r)
    return pd.DataFrame(rows)


def recalculate():
    root = Path(__file__).resolve().parent
    A = root / "inputs/supplement"
    C = root / "inputs/priors"
    prior = pd.read_csv(C / "comparison_target_metrics.csv.gz", dtype={"target": str})
    single = pd.read_csv(A / "single_target_metrics.csv.gz", dtype={"target": str})
    raw = pd.read_csv(
        A / "condition_metrics.csv.gz", dtype={"target": str, "context": str}
    )
    raw = (
        raw.groupby(["dataset", "variant", "target", "split"])[
            ["pds_query_cosine", "auprc_dg"]
        ]
        .mean()
        .groupby(["dataset", "variant", "target"])
        .mean()
        .reset_index()
    )
    strata = pd.read_csv(C / "stratified_target_metrics.csv.gz", dtype={"target": str})
    double = pd.read_csv(A / "double_pair_metrics.csv.gz", dtype={"target": str})
    transfer = pd.read_csv(
        A / "transfer_condition_metrics.csv.gz", dtype={"perturbation": str}
    )
    transfer = transfer[transfer["mode"].eq("adapted")]
    data = {}
    data["1a"] = per_dataset(
        prior,
        "full_retuned",
        "joint_shuffle_mean_retuned",
        "pds_query_cosine",
        "full_vs_joint_shuffle_mean_retuned",
    )
    data["1b"] = per_dataset(
        prior,
        "full_retuned",
        "joint_shuffle_mean_retuned",
        "auprc_dg",
        "full_vs_joint_shuffle_mean_retuned",
    )
    q = pd.read_csv(C / "quantity_curves.csv")
    for panel, metric in [("1c", "cosine_PDS"), ("1d", "DEG_AUPRC")]:
        data[panel] = (
            q[
                q.scope.eq("macro")
                & q["mode"].eq("retuned")
                & q.metric.eq(metric)
                & q["size"].ge(1)
            ]
            .sort_values("size")
            .copy()
        )

    # All twelve leave-one-view-out rows are explicitly described in the manuscript.
    rows = []
    for view in V:
        model = "drop_" + view + "_retuned"
        reference = "full_beta0p5_retuned"
        for ds in DS + ["macro"]:
            f = single if ds == "macro" else single[single.dataset.eq(ds)]
            r = compare(
                f,
                model,
                reference,
                "pds_query_cosine",
                "removing_" + view + ("" if ds == "macro" else "_" + ds),
            )
            r.update(
                view=view,
                dataset=ds,
                display_label=DN.get(ds, "Equal-dataset mean"),
                relative_loss_percent=-r["relative_change_percent"],
                relative_loss_ci_low_percent=-r["relative_ci_high_percent"],
                relative_loss_ci_high_percent=-r["relative_ci_low_percent"],
            )
            rows.append(r)
    data["2a"] = pd.DataFrame(rows)
    data["2b"] = per_dataset(
        single,
        "full_beta0p5_retuned",
        "three_prior_retuned",
        "auprc_dg",
        "full_vs_three_prior",
    )
    data["3a"] = per_dataset(
        prior, "full_retuned", "dense4_retuned", "auprc_dg", "full_vs_dense4_retuned"
    )
    rows = []
    for tier in ["low", "middle", "high"]:
        r = compare(
            strata[strata.proximity_bin.eq(tier)],
            "full_retuned",
            "joint_shuffle_mean_retuned",
            "auprc_dg",
            "proximity_" + tier,
        )
        r["proximity_group"] = tier
        rows.append(r)
    data["3b"] = pd.DataFrame(rows)
    data["4a"] = per_dataset(
        raw, "aligned_raw", "uniform_raw", "auprc_dg", "aligned_vs_uniform_raw"
    )
    beta = pd.read_csv(A / "key_beta_sensitivity.csv")
    data["4b"] = beta[["beta", "pds_query_cosine"]].copy()
    data["4c"] = beta[["beta", "auprc_dg"]].copy()
    rows = []
    for arm in ["seen0", "seen1", "seen2"]:
        r = compare(
            double[double.arm.eq(arm)],
            "drop_CORUM_retuned",
            "full_beta0p5_retuned",
            "mse20",
            "removing_CORUM_" + arm,
            group="dataset_fold",
            denominator="nochange20",
        )
        r["setting"] = arm
        rows.append(r)
    data["5"] = pd.DataFrame(rows)
    data["6a"] = per_dataset(
        transfer,
        "weighted_OT",
        "weighted_identity",
        "pds_l1_exclude_target",
        "weighted_OT_vs_weighted_identity",
        group="context",
        unit="perturbation",
    )
    retuned = compare(
        transfer,
        "weighted_OT",
        "weighted_identity_retuned",
        "pds_l1_exclude_target",
        "weighted_OT_vs_weighted_identity_retuned",
        group="context",
        unit="perturbation",
    )
    retuned.update(dataset="macro_retuned", display_label="Mean, reselected temp.")
    data["6a"] = pd.concat([data["6a"], pd.DataFrame([retuned])], ignore_index=True)
    tm = pd.read_csv(A / "transfer_macro.csv")
    tm = tm[tm["mode"].eq("adapted")]
    variants = [
        "weighted_identity",
        "weighted_eta0p25",
        "weighted_eta0p5",
        "weighted_eta0p75",
        "weighted_OT",
    ]
    eta = tm.set_index("variant").loc[variants].reset_index()
    eta.insert(0, "eta", [0, 0.25, 0.5, 0.75, 1])
    data["6b"] = eta[["eta", "variant", "pds_l1_exclude_target"]].copy()
    data["6c"] = eta[["eta", "variant", "mse_delta"]].copy()
    tuned = tm[
        tm.variant.isin(
            ["weighted_identity", "weighted_identity_retuned", "weighted_OT"]
        )
    ].copy()
    data["6_temperature"] = tuned

    return data
