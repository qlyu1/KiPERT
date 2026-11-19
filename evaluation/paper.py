"""Recalculate the manuscript's numerical results from the released source data."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from experiments.extended_data.recalculate import recalculate

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"


def figure2():
    base = EXP / "fig2_single"
    data = ROOT / "source_data/fig2"
    target = pd.read_csv(data / "target_metrics.csv")
    condition = pd.read_csv(data / "condition_metrics.csv")
    metrics = json.loads((base / "figure2_panels.json").read_text())["a"]["metrics"]
    scores = target.groupby(["dataset", "method"])[metrics].mean()
    ranks = scores.copy()
    for metric in metrics:
        ranks[metric] = (
            scores[metric]
            .groupby(level="dataset")
            .rank(ascending=metric in ("mse", "sp_mse100"))
        )
    ranks = ranks.groupby(level="method").mean().T
    expected = pd.read_csv(data / "ranks.csv").pivot(
        index="metric", columns="method", values="mean_rank"
    )
    np.testing.assert_allclose(
        ranks, expected.loc[ranks.index, ranks.columns], atol=1e-9, rtol=0
    )
    best = int(np.isclose(ranks.UC, ranks.min(axis=1)).sum())
    assert best == 9
    split = condition.groupby(
        ["dataset", "method", "split", "target"]
    ).systema_cell_pearson.mean()
    pearson = (
        split.groupby(["dataset", "method", "split"])
        .mean()
        .groupby(["dataset", "method"])
        .mean()
    )
    adamson = {m: float(pearson.loc["S5", m]) for m in ("UC", "gears")}
    np.testing.assert_allclose(
        list(adamson.values()), [0.472, 0.304], atol=0.0005, rtol=0
    )
    pds = scores.pds_query_cosine.unstack()
    values = pds.loc[["S5", "S6", "S7"], "UC"]
    runners = pds.drop(columns="UC").loc[values.index].max(axis=1)
    np.testing.assert_allclose(values, [0.656, 0.576, 0.622], atol=0.0005, rtol=0)
    np.testing.assert_allclose(runners, [0.631, 0.565, 0.550], atol=0.0005, rtol=0)
    with np.load(data / "landscapes.npz") as z:
        for ds, n in [("S4", 19), ("S6", 14)]:
            for method in ("truth", "UC"):
                matrix = z[f"{ds}_{method}"]
                assert matrix.shape == (n, n) and np.isfinite(matrix).all()
                np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)
    return dict(
        best_or_tied=best,
        metrics=len(metrics),
        KiPERT_average_ranks=ranks.UC.to_dict(),
        Adamson_corrected_Pearson=adamson,
        KiPERT_cosine_PDS=values.to_dict(),
        best_comparator_cosine_PDS=runners.to_dict(),
        landscape_axes_checked={"S4": 19, "S6": 14},
    )


def figure3():
    data = ROOT / "source_data/fig3"
    f = pd.read_csv(data / "Figure3_metric_source.csv")
    opposite = (
        f[f.metric.eq("opposite20")].groupby(["arm", "model"]).value.mean().unstack()
    )
    np.testing.assert_allclose(
        opposite.kipert, [0.081, 0.069, 0.059], atol=0.0005, rtol=0
    )
    best_other = opposite.drop(columns="kipert").min(axis=1)
    np.testing.assert_allclose(best_other, [0.099, 0.095, 0.084], atol=0.0005, rtol=0)
    nmse = (
        f[f.metric.eq("nmse20") & f.model.eq("kipert")]
        .groupby(["dataset", "arm"])
        .value.mean()
        .unstack()
    )
    np.testing.assert_allclose(
        nmse.loc[["D1", "D2"]],
        [[0.250, 0.177, 0.110], [0.232, 0.206, 0.189]],
        atol=0.0005,
        rtol=0,
    )
    retrieval = pd.read_csv(data / "retrieval_target.csv")
    hits = (
        retrieval[retrieval.dataset.eq("D1") & retrieval.arm.eq("seen0")]
        .groupby(["model", "k"])
        .hit.mean()
        .unstack()
    )
    np.testing.assert_allclose(
        hits.loc["kipert", [1, 4, 8]], [0.710, 0.935, 0.974], atol=0.0005, rtol=0
    )
    np.testing.assert_allclose(
        hits.loc[["scouter", "txpert", "gears"], 1],
        [0.684, 0.600, 0.338],
        atol=0.0005,
        rtol=0,
    )
    case = (
        pd.read_csv(data / "Figure3_case_source.csv")
        .query("dataset=='D1' and gene=='ARHGDIB'")
        .iloc[0]
    )
    case_values = [case.single_a + case.single_b, case.observed, case.KiPERT]
    np.testing.assert_allclose(case_values, [0.894, 0.266, 0.175], atol=0.0005, rtol=0)
    scatter = pd.read_csv(data / "Figure3_scatter_source.csv.gz")
    assert not scatter.duplicated(["model", "index"]).any()
    pivot = scatter.pivot(index="index", columns="model", values="observed")
    assert len(pivot) == 6000 and pivot.notna().all().all()
    np.testing.assert_allclose(pivot.iloc[:, 0], pivot.iloc[:, 1])
    mse = (scatter.predicted - scatter.observed).pow(2).groupby(scatter.model).mean()
    gain = 100 * (1 - mse.kipert / mse.available_evidence_additive)
    np.testing.assert_allclose(
        [mse.kipert, mse.available_evidence_additive],
        [0.00112, 0.00229],
        atol=0.000005,
        rtol=0,
    )
    assert round(gain, 1) == 51.0
    return dict(
        opposite20=opposite.kipert.to_dict(),
        best_other=best_other.to_dict(),
        nMSE20=nmse.T.to_dict("list"),
        Norman_seen0_topk=hits.loc["kipert", [1, 4, 8]].to_dict(),
        ARHGDIB=case_values,
        contrast_MSE=mse.to_dict(),
        contrast_reduction_percent=gain,
    )


def figure4():
    data = ROOT / "source_data/fig4"
    ab = pd.read_csv(data / "Figure4_abcde_summary.csv")
    for _, frame in ab.groupby("metric"):
        assert frame.nlargest(1, "mean").iloc[0].model == "KiPERT"
    contexts = pd.read_csv(data / "state_per_context.csv")
    means = contexts.groupby("model").de_overlap_N.mean()
    np.testing.assert_allclose(
        means.loc[["KiPERT", "TxPert", "STATE"]],
        [0.217, 0.134, 0.066],
        atol=0.0005,
        rtol=0,
    )
    selection = pd.read_csv(data / "SEH1L_correlation_reference.csv").set_index("context")
    r2, frames = {}, []
    for line, n in [("k562", 884), ("hepg2", 867), ("rpe1", 983), ("jurkat", 978)]:
        with np.load(data / f"SEH1L_{line}.npz") as z:
            r2[line] = np.corrcoef(z["true_mean"], z["pred_mean"])[0, 1] ** 2
        np.testing.assert_allclose(
            r2[line], selection.loc[line, "current_squared_pearson"], atol=1e-6, rtol=0
        )
        sides = [
            pd.read_csv(data / f"{m}_{line}_per_condition.csv") for m in ("UC", "PMean")
        ]
        pair = sides[0].merge(
            sides[1],
            on=["context", "perturbation"],
            validate="one_to_one",
            suffixes=("_KiPERT", "_PMean"),
        )
        assert len(pair) == len(sides[0]) == len(sides[1]) == n
        frames.append(pair)
    paired = pd.concat(frames)
    assert paired.perturbation.nunique() == 1595
    fields = ["pds_l1_exclude_target", "mse_delta"]
    effects = {}
    for metric in fields:
        a, b = [
            paired.groupby("context")[metric + "_" + m].mean().mean()
            for m in ("KiPERT", "PMean")
        ]
        effects[metric] = dict(
            KiPERT=a, PMean=b, relative_change_percent=100 * (a / b - 1)
        )
    np.testing.assert_allclose(
        [effects[m]["relative_change_percent"] for m in fields],
        [5.88, 7.38],
        atol=0.005,
        rtol=0,
    )
    wins = (
        (paired.pds_l1_exclude_target_KiPERT > paired.pds_l1_exclude_target_PMean)
        .groupby(paired.context)
        .mean()
    )
    return dict(
        conditions=len(paired),
        targets=paired.perturbation.nunique(),
        macro_overlap=means.to_dict(),
        SEH1L_squared_Pearson=r2,
        KiPERT_higher_PDS_fraction=wins.to_dict(),
        **effects,
    )


def figure5():
    base = ROOT / "configs/cardiac"
    data = ROOT / "source_data/fig5"
    rows = pd.read_csv(data / "query_split_results.csv")
    rows = rows[rows.program.eq("cardiac") & rows.method.isin(["raw", "KiPERT"])].copy()
    config = json.loads((base / "protocol.json").read_text())
    truth = pd.read_csv(data / "panel_c_paired_reductions.csv").set_index(
        ["budget", "metric"]
    )
    effects = {}
    for budget, targets in [(5, 33), (10, 24)]:
        frame = rows[rows.budget.eq(budget)]
        assert frame.target.nunique() == targets and frame.repeat.nunique() == 20
        rng = np.random.default_rng(
            np.random.SeedSequence([int(config["seed"]), 2041, budget])
        )
        draws = rng.multinomial(targets, np.full(targets, 1 / targets), size=2000)
        for quantity in ["delta", "absolute"]:
            mse = (frame["predicted_" + quantity] - frame["reference_" + quantity]).pow(
                2
            )
            batch = mse.groupby([frame.target, frame.batch, frame.method]).mean()
            tar = batch.groupby(["target", "method"]).mean().unstack().sort_index()
            raw, model = tar.raw.to_numpy(), tar.KiPERT.to_numpy()
            red = 1 - model.mean() / raw.mean()
            lo, hi = np.quantile(1 - (draws @ model) / (draws @ raw), [0.025, 0.975])
            expected = truth.loc[(budget, "cardiac_" + quantity + "_mse")]
            np.testing.assert_allclose(
                [raw.mean(), model.mean(), red, lo, hi],
                expected[
                    [
                        "baseline_point",
                        "KiPERT_point",
                        "relative_gain",
                        "relative_gain_ci_low",
                        "relative_gain_ci_high",
                    ]
                ].astype(float),
                atol=2e-12,
                rtol=0,
            )
            effects[f"{budget}_cells_{quantity}"] = dict(
                raw_MSE=raw.mean(),
                KiPERT_MSE=model.mean(),
                reduction_percent=100 * red,
                CI95_percent=[100 * lo, 100 * hi],
                targets=targets,
            )
    land = pd.read_csv(data / "panel_d_predicted_landscape.csv")
    assert len(land) == 211 and not land.duplicated(["target", "batch"]).any()
    rank = land.prediction.groupby(land.batch).rank(ascending=False, method="min")
    ranks = dict(
        zip(land.loc[land.target.eq("Calr"), "batch"], rank[land.target.eq("Calr")])
    )
    assert len(ranks) == 4 and all(x == 1 for x in ranks.values())
    e = pd.read_csv(data / "panel_e_Calr_gene_responses.csv")
    assert len(e) == 32 and not e.duplicated(["gene", "batch"]).any()
    directions = e[e.gene.isin(["Tnnt2", "Ryr2", "Ttn", "Myl4"])]
    assert (
        len(directions) == 16
        and (directions[["observed", "KiPERT_OT_prediction"]] > 0).all().all()
    )
    f = (
        pd.read_csv(data / "panel_f_Calr_module.csv")
        .query("method=='KiPERT_OT'")
        .set_index("batch")
    )
    averaged = (
        e.groupby("batch")[["observed", "KiPERT_OT_prediction"]].mean().sort_index()
    )
    np.testing.assert_allclose(
        averaged,
        f.loc[averaged.index, ["module_observed", "module_prediction"]],
        atol=1e-7,
    )
    return dict(
        **effects,
        Calr_batch_ranks=ranks,
        matched_upregulation_genes=["Tnnt2", "Ryr2", "Ttn", "Myl4"],
    )


def extended():
    tables = recalculate()
    for panel, frame in tables.items():
        expected = pd.read_csv(EXP / "extended_data/reference" / f"{panel}.csv")
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            expected,
            check_exact=False,
            check_dtype=False,
            rtol=1e-9,
            atol=1e-10,
        )
    return {panel: len(frame) for panel, frame in tables.items()}
