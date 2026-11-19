"""Frozen raw prior bank; original view_kernels transforms within provided training targets."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
class Bank:
    def __init__(self, root=None):
        root=Path(root) if root else Path(__file__).resolve().parents[1]/'prepared/prior_bank'
        meta=json.loads((root/'metadata.json').read_text())
        self.targets=np.load(root/'targets.npy').astype(str)
        self.index={t:i for i,t in enumerate(self.targets)}
        self.raw={}; self.presence={}; self.sizes={}
        for view in meta['views']:
            if view in meta['sparse_views']: self.raw[view]=sparse.load_npz(root/(view+'.npz'))
            else:
                with np.load(root/(view+'.npz')) as z:
                    self.raw[view]=z['raw'];self.presence[view]=z['present']
            if view in meta['sizes_views']:self.sizes[view]=np.load(root/(view+'_sizes.npy'))

def view_kernels(bank,train):
 ix=np.array([bank.index[g] for g in train]);kernels=[];coverage=np.zeros(len(bank.targets),int)
 for v,raw in bank.raw.items():
  if not sparse.issparse(raw):
   present=bank.presence[v];use=ix[present[ix]];z=np.zeros(raw.shape)
   if len(use):z[present]=(raw[present]-raw[use].mean(0))/np.maximum(raw[use].std(0),1e-6)
   norm=np.linalg.norm(z,axis=1);z=np.divide(z,norm[:,None],out=np.zeros_like(z),where=norm[:,None]>1e-12);kv=z@z.T
  else:
   freq=np.asarray(raw[ix].sum(0)).ravel() if v in ['Reactome','GO_BP','CORUM','CollecTRI'] else bank.sizes[v]
   z=raw.multiply(1/np.sqrt(np.maximum(freq,1))).tocsr();norm=np.sqrt(np.asarray(z.multiply(z).sum(1)).ravel());z=sparse.diags(np.divide(1,norm,out=np.zeros_like(norm),where=norm>1e-12))@z;kv=(z@z.T).toarray()
  kernels.append(kv);coverage+=norm>1e-12
 return np.stack(kernels),ix,coverage,list(bank.raw)
