"""Frozen UC v3 numerical sources, new legal fits, context-matched selection."""
from pathlib import Path
import sys,argparse,json,time,pickle
from resources import R,digest
from isolation import confine
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=17);p.add_argument('--dataset',required=True);p.add_argument('--protocol',default='seen0');p.add_argument('--split',type=int,required=True);p.add_argument('--phase',choices=['smoke','train'],default='train');a=p.parse_args();B=R/'data/formal'/a.dataset/a.protocol/f'fold{a.split}';O=R/'runs'/('integration' if a.phase=='smoke' else 'formal')/a.dataset/a.protocol/f'fold{a.split}'/'seed17'/'uc';confine(B,O)
from worker_common import check_gate
check_gate(B,a.phase,'uc')
import numpy as np,pandas as pd
from scipy import sparse
sys.path.insert(0,str(R/'assets/uc'));import mean_model
from common_seen import get_force
bank=dict(np.load(R/'assets/prior_banks'/(a.dataset+'.npz')));start=time.time()
def load(role):
 x=sparse.load_npz(B/role/'expression.npz');m=pd.read_csv(B/role/'metadata.tsv',sep='\t',dtype=str);d={}
 for (t,c),ix in m.groupby(['target','context'],sort=True).indices.items():d[(t,c)]=np.asarray(x[ix].mean(0)).ravel()
 return d
tr=load('fit');va=load('validation');ctrl={c:y for (t,c),y in load('controls_fit').items()};deg=json.loads((B/'fit/deg_indices.json').read_text());sp=json.loads((B/'split.json').read_text());forbidden=sp.get('test_genes',[])+sp.get('validation_genes',[])
q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str)
training={c:[t for t,cc in tr if c==cc] for c in ctrl}
def fit(config):
 states={}
 for c,tt in training.items():
  if not tt:raise ValueError('context with no legal training conditions: '+c)
  delta=np.stack([tr[(t,c)]-ctrl[c] for t in tt]);ix=[np.array(deg[t+'|'+c],int) for t in tt];m=mean_model.fit(bank,tt,delta,ctrl[c],ix,config,forbidden_genes=forbidden);states[c]=(m,tt,delta)
 return states
def prediction(states,t,c,nonnegative=True):
 m,tt,y=states[c];delta=mean_model.predict(m,[t],nonnegative=nonnegative);return ctrl[c]+delta[0]
def val_score(states):
 by={}
 for (t,c),y in va.items():by.setdefault(t,[]).append(float(np.mean((prediction(states,t,c)-y)**2)))
 return float(np.mean([np.mean(v) for v in by.values()]))
force=get_force(B,'uc',17);configs=[force['config']] if force else list(mean_model.CONFIGS);selection=[]
for i,cfg in enumerate(configs):
 states=fit(cfg);score=force['cv_mse'] if force else val_score(states);selection.append({'candidate':i,'config':cfg,'validation_macro_mse':score});print(json.dumps(selection[-1]),flush=True)
 if a.phase=='smoke':
  t,c=next((z.target,z.context) for z in q.itertuples() if '+' in z.target);y=prediction(states,t,c);assert np.isfinite(y).all() and np.allclose(y,prediction(states,'+'.join(t.split('+')[::-1]),c));
  if not any('+' in u for u in training[c]) and all((g,c) in tr for g in t.split('+')):assert np.allclose(y,np.maximum(sum(tr[(g,c)] for g in t.split('+'))-ctrl[c],0))
  (O/'result.json').write_text(json.dumps({'status':'SMOKE_PASS','finite':True,'swap_invariant':True,'formal_training':False}));sys.exit(0)
 with (O/f'candidate_{i}.pkl').open('wb') as f:pickle.dump(states,f)
best=min(selection,key=lambda z:(z['validation_macro_mse'],z['candidate']));(O/'selection.json').write_text(json.dumps({'candidates':selection,'chosen':best},indent=2));states=pickle.load(open(O/f"candidate_{best['candidate']}.pkl",'rb'));yp=np.stack([prediction(states,z.target,z.context) for z in q.itertuples()]);np.savez_compressed(O/'predictions_sealed.npz',prediction=yp.astype(np.float32),target=q.target.to_numpy(dtype=str),context=q.context.to_numpy(dtype=str));(O/'model_final.pkl').write_bytes((O/f"candidate_{best['candidate']}.pkl").read_bytes());q.to_csv(O/'query_order.tsv',sep='\t',index=False)
result={'status':'PREDICTIONS_SEALED','formal_training':True,'method':'UC Balanced v3','selection':best,'queries':len(q),'prediction_sha256':digest(O/'predictions_sealed.npz'),'weight_sha256':digest(O/'model_final.pkl'),'seconds':time.time()-start,'test_metrics_computed':False};(O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
