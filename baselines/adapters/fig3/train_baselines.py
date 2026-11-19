from pathlib import Path
import argparse,json,sys,time,pickle
from resources import R,digest
from isolation import confine
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--protocol',required=True);p.add_argument('--split',type=int,required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--phase',default='train');a=p.parse_args()
B=R/f'data/formal/{a.dataset}/{a.protocol}/fold{a.split}';O=R/f'runs/formal/{a.dataset}/{a.protocol}/fold{a.split}/seed17/baselines';confine(B,O)
from worker_common import check_gate
check_gate(B,a.phase,'baselines')
import numpy as np,pandas as pd
from scipy import sparse
from common_seen import get_force
sys.path.insert(0,str(R/'assets/uc'));from pair_research import pairhead
start=time.time();force=get_force(B,'baselines',17);bank=dict(np.load(R/'assets/prior_banks'/f'{a.dataset}.npz'))
def load(role):
 x=sparse.load_npz(B/role/'expression.npz');m=pd.read_csv(B/role/'metadata.tsv',sep='\t',dtype=str)
 return {(t,c):np.asarray(x[ix].mean(0)).ravel() for (t,c),ix in m.groupby(['target','context'],sort=True).indices.items()}
tr=load('fit');va=load('validation');ctrl={c:y for (t,c),y in load('controls_fit').items()};q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str)
targets=sorted(pd.read_csv(B/'target_prior_mapping.tsv',sep='\t').target);ti={g:i for i,g in enumerate(targets)}
def features(tt):
 z=np.zeros((len(tt),len(targets)),float)
 for i,t in enumerate(tt):
  for g in t.split('+'):z[i,ti[g]]=1
 return z
def fit(kind,alpha):
 states={}
 for c in ctrl:
  tt=[t for t,cc in tr if cc==c];y=np.stack([tr[t,c]-ctrl[c] for t in tt]);ar=np.array([len(t.split('+')) for t in tt]);w=np.array([len(ar)/(2*sum(ar==k)) for k in ar])
  if kind=='multi_hot_ridge':
   z=features(tt);states[c]=np.linalg.solve(z.T@(w[:,None]*z)+alpha*np.eye(z.shape[1]),z.T@(w[:,None]*y))
  else:states[c]=pairhead.fit_joint(bank,tt,y,alpha=alpha,lambda_pair=0.,lambda_int=0.,node_mode='linear' if kind=='same_prior_linear' else 'hybrid',pair_intercept=False,balance_arities=True)
 return states
def predict(states,kind,t,c):
 delta=features([t])@states[c] if kind=='multi_hot_ridge' else pairhead.predict_joint(states[c],[t])
 return np.maximum(ctrl[c]+delta[0],0)
def score(states,kind):
 rows={}
 for (t,c),truth in va.items():rows.setdefault(t,[]).append(float(np.mean((predict(states,kind,t,c)-truth)**2)))
 return float(np.mean([np.mean(v) for v in rows.values()]))
alphas={};selections={};fitted={};preds={}
for kind in ['multi_hot_ridge','same_prior_linear','same_prior_KRR']:
 candidates=[force['alphas'][kind]] if force else ([.1,1.,10.,100.] if kind=='multi_hot_ridge' else [1.,10.])
 best=None
 for alpha in candidates:
  states=fit(kind,alpha);mse=force['cv_mse'] if force else score(states,kind);print(kind,alpha,mse,flush=True)
  if best is None or mse<best[0]:best=(mse,alpha,states)
 alphas[kind]=best[1];selections[kind]=dict(alpha=best[1],validation_macro_mse=best[0]);fitted[kind]=best[2]
 preds[kind]=np.stack([predict(best[2],kind,z.target,z.context) for z in q.itertuples()])
preds['no_change']=np.stack([ctrl[z.context] for z in q.itertuples()])
double_mean={c:np.mean([v for (t,cc),v in tr.items() if cc==c and '+' in t],axis=0) for c in ctrl}
preds['double_train_mean']=np.stack([double_mean[z.context] for z in q.itertuples()])
add=[];matching=[]
for z in q.itertuples():
 g1,g2=z.target.split('+');mu=ctrl[z.context];ss=fitted['multi_hot_ridge'][z.context]
 effects=[tr[g,z.context]-mu if (g,z.context) in tr else (features([g])@ss)[0] for g in [g1,g2]]
 add.append(np.maximum(mu+sum(effects),0))
 matching.append(np.mean([tr.get((g,z.context),double_mean[z.context]) for g in [g1,g2]],axis=0))
preds['available_evidence_additive']=np.stack(add);preds['matching_mean']=np.stack(matching)
for kind,y in preds.items():
 assert np.isfinite(y).all();np.savez_compressed(O/f'{kind}.npz',prediction=y.astype(np.float32),target=q.target.to_numpy(str),context=q.context.to_numpy(str))
(O/'selection.json').write_text(json.dumps(dict(chosen=dict(alphas=alphas,validation_macro_mse=float(np.mean([z['validation_macro_mse'] for z in selections.values()]))),details=selections),indent=2))
(O/'result.json').write_text(json.dumps(dict(status='BASELINES_SEALED',seconds=time.time()-start,test_metrics_computed=False,baselines={k:dict(sha256=digest(O/f'{k}.npz')) for k in preds}),indent=2));print('BASELINES_SEALED',flush=True)
