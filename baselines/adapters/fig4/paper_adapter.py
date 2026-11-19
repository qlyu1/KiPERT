import numpy as np

def shift_cells(controls,shift):
 return np.maximum(np.asarray(controls,dtype=np.float64)+np.asarray(shift,dtype=np.float64),0)

def q_to_pde(q):
 q=np.asarray(q,dtype=np.float64)
 assert np.isfinite(q).all() and np.all((q>=0)&(q<=1))
 s=-np.log10(np.maximum(q,1e-10))
 return (s/(s-np.log10(.05))).astype(np.float32)
