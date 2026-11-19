# Numerical sources

The panel index in [figures/panel_index.tsv](../figures/panel_index.tsv) maps panels to their exact source directories and verification levels. Identifier strings in release tables preserve benchmark names such as UC for the published KiPERT model. Equal-dataset and equal-target-line averages use the aggregation specified for the corresponding panel. Undefined statistics retain their recorded missing values.

Fig. 3h's gene and pair IDs are stored in GI_D2_fold0_seen2.npz. Flatten the pair-by-gene matrix in row-major order to align the 6,000 indices per method with the scatter table. Fig. 4f uses the current model's adapted gene means. Fig. 4g joins on both cell line and perturbation. The Fig. 5 tables distinguish errors before and after non-targeting-control subtraction.

Reference tables for Extended Data 1–6 are in [experiments/extended_data/reference](../experiments/extended_data/reference). Their inputs and the original bootstrap calculation are in the adjacent inputs and recalculate.py. To generate a source workbook for each figure, install the workbooks extra and run `python figures/export_source_data.py rendered/workbooks`.
