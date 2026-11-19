# KiPERT

KiPERT predicts mean expression after unseen single- and double-gene perturbations and transfers measured perturbation responses to a held-out cell line. This repository accompanies the manuscript by Qiantong Liang, Yi Wang and Qing Rex Lyu.

**Start with the [executed tutorial](tutorial/quickstart.ipynb).** It refits the single-gene model on a real Tian CRISPRi split, prints the 12 learned prior weights for each context, predicts a held-out response and checks all 481,173 expression values against the archived run. The example inputs, including the two additional examples, are in this repository.

## Run the tutorial

From the repository root, use Python 3.10 in a new environment. The reference run used macOS 15.6.1 on ARM64 with Python 3.10.15. Model fitting and numerical verification require no GPU or external datasets.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/core.lock.txt
python -m pip install -e .
python -m tutorial.run
python -m tutorial.execute rendered/quickstart.ipynb
```

The expected terminal output gives 33 query rows, 14,581 genes, four contexts and a maximum absolute difference of zero on the tested machine. The numeric acceptance tolerance for the single- and double-gene predictions is 0.000002. To interact with the notebook, open it in a Jupyter frontend using this environment. The alternative Conda specification is [environment.yml](environment.yml).

## Reproduce the figures

```bash
python -m evaluation.verify --output rerun.json
python -m evaluation.verify --only fig3h_refit
python -m evaluation.verify --only fig4_interval
python figures/render.py rendered
```

All 10 numerical checks passed in the reference run. For Fig. 4g, the paired bootstrap reproduced the manuscript’s 95% interval of 5.23–6.57% around the 5.88% relative increase in L1 PDS. The first command records the numerical results, runtime per check and software versions. See the [result ledger](REPRODUCIBILITY.md) and [machine-readable reference run](evaluation/verified_run.json).

| Task | Inputs and implementation | What the reviewer can run |
|---|---|---|
| Fig. 2, single gene | Five datasets, three target splits, six comparator methods. The model learns a training-response alignment for 12 prior views and fits centered kernel ridge regression. | Tian CRISPRi split 101 refit, average ranks, shared-response-corrected Pearson and PDS. |
| Fig. 3, gene pairs | A shared first-order kernel, pair calibration and a symmetric interaction kernel. The seen-0 validation choice is reused for all three arms. | Refit Replogle exp6 fold 0 in seen 0/1/2. Reproduce the Fig. 3h contrast directly from the fit. Recompute other benchmark summaries. |
| Fig. 4, cell-line transfer | Control-cell Gaussian transport maps and measured-source weights based on cell-line similarity and source GP uncertainty. | Replay SEH1L for all four held-out lines with fitted maps and weights. Evaluate paired PDS and MSE over 3,712 conditions. |
| Fig. 5, cardiac responses | Five- and ten-cell measured-target fits with disjoint references, plus a separate Calr cross-batch example. | Recalculate errors from predictions and observations, repeat the target bootstrap and recompute Calr ranks. |
| Extended Data 1–6 | Prior permutations and subset sizes, view ablations, proximity strata, alignment sensitivity, CORUM and OT. | Regenerate and compare 16 numerical tables with the figure references. |

## Repository layout

```text
kipert/                   single and double models, prior normalization, transport and inference
tutorial/                 executed notebook, command-line tutorial and notebook runner
examples/                 single_small, double_small and transfer_small with reference predictions
configs/                  frozen selections and protocols for all four tasks and Extended Data
data/                     resource manifests, preparation code and split membership
experiments/              model entry points, archived training and scoring implementations
baselines/                comparator adapters, frozen configurations and upstream version records
evaluation/               numeric checks, reference run and panel-level result ledger
source_data/              main-figure tables, expression arrays and scoring references
figures/                  editable Fig. 1, PDF rendering and source-workbook export
environments/             tested CPU dependencies and historical comparator versions
```

For the complete experiment inputs, recorded source identifiers and run instructions, see [Data and full runs](data/README.md). The full prepared expression matrices, unrestricted-size prior banks and third-party model installations are separate from this review bundle. Data access for each level is documented there.
