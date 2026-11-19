# Comparator implementations

Figure 2 uses biolord, Scouter, TxPert, GEARS, STAMP–Geneformer and RegFormer–GEARS. The 90 dataset/split/method selection records are in config/fig2. The worker files retain their original content-derived names so that recorded worker IDs can be located. Some workers import the original method repositories.

Figure 3 includes GEARS, Scouter, TxPert, STAMP and GPerturb plus expression baselines. The training adapters and common fold code are under adapters/fig3. Figure 4 uses STATE, STACK, TxPert and CMean for the five-method comparison and PMean for the measured-source comparison. Adapters and recorded STATE/STACK settings are in adapters/fig4 and config/fig4. STATE and TxPert used random initialization. STACK used local pretraining followed by posttraining.

Known upstream revisions are in [upstream_versions.tsv](upstream_versions.tsv). Revisions absent from the source records are marked unrecorded. The repository includes the study's adapters. Upstream installations and trained comparator weights are external. Benchmark comparisons here were recomputed from released numerical evidence, with no new comparator training.
