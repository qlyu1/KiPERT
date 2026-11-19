"""Subset original twelve prior kernels after training-only feature normalization."""
import numpy as np
from prior_inputs import Bank, view_kernels
_BANK = None

def kernels(train_targets, all_targets):
    global _BANK
    if _BANK is None:
        _BANK = Bank()
    all_targets = list(all_targets)
    ix = np.array([_BANK.index[str(t)] for t in all_targets],dtype=int)
    K, _, coverage, views = view_kernels(_BANK,[str(t) for t in train_targets])
    return K[:,ix][:,:,ix], views, coverage[ix]
