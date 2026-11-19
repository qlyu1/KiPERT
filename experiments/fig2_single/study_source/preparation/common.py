import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import json, hashlib, datetime
import numpy as np
R=Path('@STUDY@/uc_fig2_benchmark_20260911')
ROOT=Path(__file__).resolve().parents[1]
VIEWS=['DepMap','GenePT','STRING','Reactome','GO_BP','CORUM','CollecTRI','HPA','GO_MF','GO_CC','HPO','WikiPathways']
DATASETS=['S3','S4','S5','S6','S7']
SEEDS=[101,202,303]
VERSION='UC_Fig2_equal12_trainOnly_rank512_jointSingles_20261105'
CONFIGS=[dict(alpha=a,response=r) for r in ['full','deg_subspace'] for a in [1.,10.]]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def log(**v):print(json.dumps(dict(utc=now(),**v)),flush=True)
