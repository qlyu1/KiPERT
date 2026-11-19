# Data, splits and full runs

The examples run offline. The [dataset manifest](manifests/datasets.tsv) gives upstream records and exact filenames where recorded. The [prior manifest](manifests/priors.tsv) lists the 12 views and their preprocessing. The [packaged artifact inventory](manifests/artifacts.tsv) gives byte counts and SHA-256 hashes for the example inputs, source tables, split files, frozen configurations and numerical evidence.

## Raw data and larger prepared inputs

Use the following commands to list the single-gene record or fetch one explicitly selected raw file. The downloader checks the recorded size and checksum before moving the file into place. GEO and PIE entries link to their original providers.

```bash
python data/fetch.py --dataset S3
python data/fetch.py --dataset S3 --file TianKampmann2021_CRISPRi.h5ad --output downloads
```

The five single-gene datasets use scPerturb record 13350497. Combination benchmarks use Norman, Replogle exp6, Wessels lane L4 and Dixit K562 high-MOI. The four-line benchmark uses the Replogle–Nadig essential-gene release and PIE unseen-context splits. The cardiac data are the day-14 matrices, features, barcodes and cell metadata from the Cai study. [scPerturb](https://zenodo.org/records/13350497) · [Wessels GEO record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213957) · [Dixit GEO record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063) · [Cai data](https://zenodo.org/records/19185135)

The full experiment inputs are currently identified by provider records, split membership, and study-side inventories. A versioned public archive of the complete prepared inputs and all baseline predictions has yet to be assigned. The small examples and data for the numerical checks are bundled in full.

## Matching inputs to the original benchmark

- **Single.** Each of S3–S7 has splits 101, 202 and 303. The split folders contain the output genes, per-role target/context counts and query metadata. The [full input inventory](manifests/single_prepared_inventory.json) identifies the original prepared matrices and prior-bank files by checksum. For a complete prepared split, place role-specific expression.npz and metadata.tsv under DATASET/splitNNN and place the 12-view bank under PRIORS/DATASET. The portable entry point uses the recorded validation choice and refits on fit plus validation.
- **Double.** Four datasets × five folds × three availability arms are in splits/double. The structure and validation grid are in configs/double. A full prepared fold contains genes.tsv and train.npz, validation.npz, controls_fit.npz, query.npz. The example needs only its cached training kernels. The full-fold entry point accepts the prepared component banks. The original seen-0 alpha can be reselected with --reselect.
- **Transfer.** splits/transfer contains the original train/validation/test JSON for each held-out line. Source selection, temperature grids and protocols are under configs/transfer. The original fitting code is in experiments/fig4_transfer/study_source, including train_current.py, current_engine.py, adaptive_engine.py and ot_core.py. The packaged transfer example begins after source-GP fitting and map estimation.
- **Cardiac.** splits/cardiac contains eligibility, axes and the actual input/reference indices and cross-validation fold labels for 160 budget/repeat/batch combinations. The row indices refer to the study's prepared matrix in its original order. configs/cardiac contains the protocol, outcomes of model selection, eight-gene program and Calr protocol. The original preparation, prediction and evaluation code is under experiments/fig5_cardiac/study_source.

```bash
python -m kipert.single.replay --formal-data DATA --priors PRIORS --dataset S3 --split 101 --output single-run
python -m experiments.fig3_double.full_fold --data DATA --priors PRIORS --dataset D2 --fold 0 --output double-run
```

Replace DATA and PRIORS with the paths to the **full prepared** data and prior banks for each task. The [verification report](../evaluation/verified_run.json) specifies the inputs exercised in the review run.

## Archived implementations

The study_source directories and baseline adapters preserve the preparation, training, selection and scoring routines used in the study. Historical absolute study-root paths appear as `@STUDY@`. To recreate the working directory layout, run `python data/prepare/stage_study.py NEW_DIRECTORY`. The command stages the code and selected protocols. Full native training additionally needs task-specific data, upstream code, weights and original environments. Some historical Linux workers check the original file digests and execution environment. The self-contained review entry points use the portable modules directly.

Under `uc_cross_context_transfer/experiments/`, the cardiac scripts use `cai_low_coverage_20261113` for the sparse-measurement experiments and `cai_d14_application_20261112` for the day-14 application. If the full day-14 archive is stored elsewhere, set `KIPERT_CAI_D14_DIR` to its absolute directory. Day-14 preparation reads prior inputs from `cai_stage_transfer_20261112`; set `KIPERT_CAI_STAGE_DIR` to use an existing archive in a different location.

The three scripts in prepare/examples document how the example arrays were exported from the original study data. They accept the native study root and stream a tar archive to standard output.
