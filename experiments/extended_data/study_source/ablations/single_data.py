import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import zipfile,shutil,json,gc
import numpy as np,pandas as pd
from scipy import sparse,stats
O=Path('@STUDY@/kipert_supplement_20261117');R=Path('@STUDY@/uc_fig2_benchmark_20260911/data/formal')
def aggregated(d,seed,role):
 out=O/'single/aggregates'/d/f'split{seed}'/role
 if (out/'COMPLETE.json').exists():return out
 out.mkdir(parents=True,exist_ok=True);base=R/d/f'split{seed}';m=pd.read_csv(base/role/'metadata.tsv',sep='\t',dtype=str);tmp=out/'temporary_csr';tmp.mkdir(exist_ok=True)
 with zipfile.ZipFile(base/role/'expression.npz') as zz:
  for name in ['data.npy','indices.npy','indptr.npy','shape.npy']:
   with zz.open(name) as a,(tmp/name).open('wb') as b:shutil.copyfileobj(a,b,1<<20)
 data=np.load(tmp/'data.npy',mmap_mode='r');ind=np.load(tmp/'indices.npy',mmap_mode='r');ptr=np.load(tmp/'indptr.npy',mmap_mode='r');shape=np.load(tmp/'shape.npy');x=sparse.csr_matrix((data,ind,ptr),shape=shape,copy=False)
 groups=list(m.groupby(['context','target'],sort=True).indices.items());means=np.lib.format.open_memmap(out/'mean.npy',mode='w+',dtype=float,shape=(len(groups),int(shape[1])));sq=np.lib.format.open_memmap(out/'mean_sq.npy',mode='w+',dtype=float,shape=means.shape);records=[]
 for j,((c,t),ii) in enumerate(groups):
  a=np.zeros(shape[1]);b=np.zeros(shape[1])
  for lo in range(0,len(ii),128):
   block=x[ii[lo:lo+128]].astype(float);a+=np.asarray(block.sum(0)).ravel();b+=np.asarray(block.power(2).sum(0)).ravel()
  means[j]=a/len(ii);sq[j]=b/len(ii);records.append(dict(context=c,target=t,n=len(ii)))
 means.flush();sq.flush();pd.DataFrame(records).to_csv(out/'groups.csv',index=False);del x,data,ind,ptr,means,sq;gc.collect();shutil.rmtree(tmp);(out/'COMPLETE.json').write_text(json.dumps(dict(groups=len(groups),cells=len(m),genes=int(shape[1]))));print('AGGREGATED',d,seed,role,len(groups),flush=True);return out
def load(d,seed,role):
 out=aggregated(d,seed,role);return pd.read_csv(out/'groups.csv',dtype={'target':str,'context':str}),np.load(out/'mean.npy',mmap_mode='r'),np.load(out/'mean_sq.npy',mmap_mode='r')
def rows(m,mean,meansq,c,ctrl,cv,cn):
 ii=np.flatnonzero(m.context.to_numpy()==c);ii=sorted(ii,key=lambda i:m.target.iloc[i]);res=[];lab=[]
 for i in ii:
  n=int(m.n.iloc[i]);mu=np.asarray(mean[i]);var=np.maximum(meansq[i]-mu**2,0);se=np.sqrt(var/n+cv/cn);stat=np.divide(abs(mu-ctrl),se,out=np.zeros_like(mu),where=se>1e-12);res.append((m.target.iloc[i],mu-ctrl,np.argsort(-stat,kind='stable')[:100]));v1=var/max(n-1,1);v0=cv/max(cn-1,1);s=v1+v0;tt=np.divide(abs(mu-ctrl),np.sqrt(s),out=np.zeros_like(mu),where=s>1e-20);tt[(s<=1e-20)&(abs(mu-ctrl)>1e-12)]=np.inf;df=np.divide(s*s,v1*v1/max(n-1,1)+v0*v0/max(cn-1,1),out=np.ones_like(mu),where=s>1e-20);p=2*stats.t.sf(tt,df);order=np.argsort(p);adj=np.empty(len(p));adj[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];lab.append((adj<.05)&(abs(mu-ctrl)>.1))
 return res,np.stack(lab) if lab else np.empty((0,len(ctrl)),bool)
