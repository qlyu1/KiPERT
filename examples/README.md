# Example data

| Folder | Fit inputs | Reference, opened after prediction |
|---|---|---|
| single_small | axes.npz contains the ordered query and gene axes and the alignment mask. Each context file contains pseudobulk training changes, a matched control mean, target identities, prior names, coverage and 12 training-normalized Gram matrices. Training and query targets are disjoint. | expected_predictions.npz |
| double_small | seen0, seen1 and seen2 each contain six contexts. Each context supplies training condition names, mean changes from control, the control mean, component coverage and 12 component-level kernels. The pair kernels, intercepts, alignment weights and ridge coefficients are rebuilt at run time. | expected.npz inside each arm |
| transfer_small | For SEH1L, each held-out line supplies the measured source changes, fitted source weights, source-measurement mask, target control mean and fitted OT maps. The six basis files are shared between directions. | expected_k562, expected_hepg2, expected_rpe1 and expected_jurkat NPZ files |

The example matrices preserve their original gene ordering. Single and double outputs are nonnegative absolute mean expression. The transfer output is the target control mean plus the weighted transported changes in the original model space. Fig. 4f uses the subsequent cell-level scoring adapter. Its gene-level source table also includes the observed mean.
