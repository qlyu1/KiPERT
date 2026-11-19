"""Bounded sparse implementation of author's absolute Pearson top21 coexpression graph."""
import numpy as np,torch,pandas as pd
from scipy import sparse

def coexpression(x,axis_native,threshold=.4,k=20):
 from threadpoolctl import threadpool_limits
 threadpool_limits(limits=4)
 x=x.toarray().astype(np.float64);n=x.shape[0];mu=x.mean(0);sd=x.std(0);x-=mu;np.divide(x,sd[None,:],out=x,where=sd[None,:]>1e-12);x[:,sd<=1e-12]=0;edges=[];weights=[]
 for lo in range(0,x.shape[1],256):
  hi=min(x.shape[1],lo+256);corr=np.abs(x[:,lo:hi].T@x/n);np.clip(corr,0,1,out=corr)
  for j,row in enumerate(corr):
   # Stable tie ordering is recorded; independent of held-out expression.
   ix=np.argsort(row,kind='stable')[-(k+1):];ix=ix[row[ix]>threshold]
   edges.extend((int(axis_native[s]),int(axis_native[lo+j])) for s in ix);weights.extend(row[ix])
 return torch.tensor(np.asarray(edges).T,dtype=torch.long),torch.tensor(weights,dtype=torch.float32)
