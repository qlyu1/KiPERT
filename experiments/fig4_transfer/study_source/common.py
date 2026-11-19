import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,json,hashlib,datetime,time,sys,gc,itertools
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'prepared';R=ROOT/'results';CACHE=ROOT/'cache';SEED=20261101
EXP=ROOT.parent/'uc_10context_8prior_20261031'
GATES=[dict(temperature=t,measured_only=m,precision=p) for t,m,p in itertools.product([None,.25,1.,4.],[True,False],[False,True])]
ALPHAS=[0.,.25,.5,.75,1.]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def log(**x):print(json.dumps(dict(utc=now(),**x)),flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def seed_for(s):return SEED+int(hashlib.sha256(s.encode()).hexdigest()[:7],16)
def guard(fold):
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   p=str(args[0])
   if any(s in p for s in ['/official/','/evaluation_only/',f'/prepared/{fold}/','per_target_metrics','comparison.csv','paired_comparisons','metrics.json']):raise PermissionError('Held-out outcomes blocked while fitting '+fold+': '+p)
 sys.addaudithook(audit)
def corr(a,b,axis=-1):
 x=a-a.mean(axis,keepdims=True);y=b-b.mean(axis,keepdims=True);den=np.linalg.norm(x,axis=axis)*np.linalg.norm(y,axis=axis)
 return np.divide((x*y).sum(axis),den,out=np.zeros_like(den),where=den>1e-15)
def column(x):
 import h5py
 if isinstance(x,h5py.Group):
  a=column(x['categories']);c=x['codes'][:];assert (c>=0).all();return a[c]
 return x.asstr()[:].astype(str) if x.dtype.kind in 'OS' else x[:]
def official(model,target,kind):return ROOT/'official'/('ST-'+model+'-Replogle')/'zeroshot'/target/'eval_best.ckpt'/('adata_'+kind+'.h5ad')

def model_gene_names(model,target):
 import pickletools
 file=ROOT/'official'/('ST-'+model+'-Replogle')/'zeroshot'/target/'var_dims.pkl'
 strings=[arg for op,arg,pos in pickletools.genops(file.read_bytes()) if op.name in ['SHORT_BINUNICODE','BINUNICODE']]
 start=strings.index('gene_names')+1;end=strings.index('batch_dim',start);genes=np.array(strings[start:end]);assert len(genes)==2000 and len(set(genes))==2000,(model,target,len(genes));return genes
