"""Shared legal configuration, condition-balanced fixed-budget sampling."""
import json
from pathlib import Path
import numpy as np

def get_force(base,method,seed):
 base=Path(base);arm=base.parent.name
 if arm=='seen0' and seed==17:return None
 p=base/'p0_final_configs'/f'{method}_{seed}.json'
 if not p.exists():raise RuntimeError('Common seen0 selection must finish first: '+str(p))
 return json.loads(p.read_text())

def balanced_order(meta,base,batch,rng,control_meta=None,pseudobulk=False):
 sp=json.loads((Path(base)/'split.json').read_text())
 labels=meta.target.astype(str).tolist();ctx=meta.context.astype(str).tolist()
 if control_meta is not None:
  labels+=['CTRL']*len(control_meta);ctx+=control_meta.context.astype(str).tolist()
 kinds=np.array([0 if t=='CTRL' else len(t.split('+')) for t in labels])
 groups={}
 for i,k in enumerate(zip(kinds,labels,ctx)):groups.setdefault(k,[]).append(i)
 prob=np.zeros(len(labels),float);types=sorted(set(kinds))
 mass={0:.1,1:.45,2:.45} if 0 in types else {1:.5,2:.5}
 for typ in types:
  gg=[ix for (ar,t,c),ix in groups.items() if ar==typ]
  for ix in gg:prob[ix]=mass[typ]/len(gg)/len(ix)
 prob/=prob.sum()
 size=sp['base_conditions'] if pseudobulk else sp['base_cells']+(len(control_meta) if control_meta is not None else 0)
 cap=16 if pseudobulk else 128
 steps=max(1,min(cap,int(np.ceil(size/batch))))
 return rng.choice(len(labels),size=steps*batch,replace=True,p=prob)
