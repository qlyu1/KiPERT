"""KiPERT priors and full responses, preserving the frozen UC double head."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,time,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse,linalg
R=Path('@STUDY@/uc_seen012_benchmark_20261005')
E=R/'experiments/kipert_double_20261106'
BUNDLE=Path('@STUDY@/uc_structure_attribution_export_20261007/bundle')
P=Path('@STUDY@/autodl-fs/kipert_double_priors_20261106/priors').resolve()
sys.path.insert(0,str(R/'assets/uc'))
import mean_model
from pair_research import pairhead
VIEWS=['DepMap','GenePT','STRING','Reactome','GO_BP','CORUM','CollecTRI','HPA','GO_MF','GO_CC','HPO','WikiPathways']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False))
def parts(t):return pairhead.parts(t)
def center(k):return k-k.mean(0)[None,:]-k.mean(1)[:,None]+k.mean()
class Bank:
 def __init__(self,ds):
  root=P/ds;self.targets=np.load(root/'targets.npy').astype(str);self.index={t:i for i,t in enumerate(self.targets)};self.raw={};self.presence={};self.sizes={}
  for v in VIEWS:
   if v in ['DepMap','GenePT','STRING','HPA']:
    z=np.load(root/(v+'.npz'));self.raw[v]=z['x'].astype(float);self.presence[v]=z['present']
   else:self.raw[v]=sparse.load_npz(root/(v+'.npz'));self.sizes[v]=np.load(root/(v+'_metadata.npz'))['public_set_sizes']
 def kernels(self,conditions):
  train=sorted({g for t in conditions for g in parts(t)});ix=np.array([self.index[t] for t in train]);kernels=[];coverage=np.zeros(len(self.targets),int)
  for v in VIEWS:
   raw=self.raw[v]
   if not sparse.issparse(raw):
    present=self.presence[v];use=ix[present[ix]];z=np.zeros(raw.shape)
    if len(use):z[present]=(raw[present]-raw[use].mean(0))/np.maximum(raw[use].std(0),1e-6)
    norm=np.linalg.norm(z,axis=1);z=np.divide(z,norm[:,None],out=np.zeros_like(z),where=norm[:,None]>1e-12);kv=z@z.T
   else:
    freq=np.asarray(raw[ix].sum(0)).ravel() if v in ['Reactome','GO_BP','CORUM','CollecTRI'] else self.sizes[v]
    z=raw.multiply(1/np.sqrt(np.maximum(freq,1))).tocsr();norm=np.sqrt(np.asarray(z.multiply(z).sum(1)).ravel());z=sparse.diags(np.divide(1,norm,out=np.zeros_like(norm),where=norm>1e-12))@z;kv=(z@z.T).toarray()
   kernels.append(kv);coverage+=norm>1e-12
  return np.stack(kernels),coverage,train
def hweights(conditions,cfg):
 arity=np.array([len(parts(t)) for t in conditions]);pair=bool(cfg['pair_intercept'] and np.any(arity==2));h=np.column_stack([arity,arity==2]).astype(float) if pair else arity[:,None].astype(float);w=np.ones(len(arity))
 for a in np.unique(arity):w[arity==a]=len(arity)/(len(np.unique(arity))*sum(arity==a))
 return h,w,pair
def lift(node,components,query,train,cfg):return pairhead.condition_kernel(node,components,query,train,cfg['lambda_pair'],cfg['lambda_int'])
def prepare(bank,conditions,Y,genes,cfg):
 views,coverage,components=bank.kernels(conditions);h,w,pair=hweights(conditions,cfg);b=linalg.solve(h.T@(w[:,None]*h),h.T@(w[:,None]*Y),assume_a='sym');residual=Y-h@b
 keep=~np.isin(genes,bank.targets);z=residual[:,keep];norm=np.linalg.norm(z,axis=1,keepdims=True);unit=np.divide(z,norm,out=np.zeros_like(z),where=norm>1e-12);target=center(unit@unit.T);mask=~np.eye(len(conditions),dtype=bool);tv=target[mask];align=[]
 for node in views:
  a=center(lift(node,bank.targets,conditions,conditions,cfg))[mask];den=np.linalg.norm(a)*np.linalg.norm(tv);align.append(max(0.,float(a@tv/den)) if den>1e-12 else 0.)
 align=np.array(align);weights=.5/12+.5*(align/align.sum() if align.sum()>1e-12 else np.ones(12)/12);node=np.einsum('v,vij->ij',weights,views);K=lift(node,bank.targets,conditions,conditions,cfg)
 assert np.allclose(weights.sum(),1) and np.min(weights)>=1/24-1e-12
 return dict(node=node,K=K,h=h,w=w,pair=pair,alignment=align,weights=weights,coverage=coverage,training_components=components,conditions=conditions,Y=Y,cfg=cfg)
def fit(pre,alpha):
 n=len(pre['Y']);h=pre['h'];A=np.block([[pre['K']+alpha*np.diag(1/pre['w']),h],[h.T,np.zeros((h.shape[1],h.shape[1]))]]);rhs=np.vstack([pre['Y'],np.zeros((h.shape[1],pre['Y'].shape[1]))]);sol=linalg.solve(A,rhs,assume_a='sym');err=np.linalg.norm(A@sol-rhs)/max(np.linalg.norm(rhs),1e-30);assert err<1e-7
 return dict(pre=pre,alpha=alpha,coef=sol[:n],intercept=sol[n:],solve_error=float(err))
def predict(bank,state,queries,control,return_parts=False):
 pre=state['pre'];cfg=pre['cfg'];_,shared,pair,tensor=pairhead.condition_kernel(pre['node'],bank.targets,queries,pre['conditions'],cfg['lambda_pair'],cfg['lambda_int'],True);arity=np.array([len(parts(t)) for t in queries]);first=shared@state['coef']+arity[:,None]*state['intercept'][0];cal=pair@state['coef'];inter=tensor@state['coef']
 if pre['pair']:cal+=(arity==2)[:,None]*state['intercept'][1]
 out=np.maximum(control+first+cal+inter,0)
 assert np.isfinite(out).all()
 return (out,dict(shared=first,calibration=cal,interaction=inter)) if return_parts else out
def loadpanel(ds,arm,fold):
 base=E/f'data/{ds}/{arm}/fold{fold}';z={name:dict(np.load(base/(name+'.npz'))) for name in ['train','validation','controls_fit','query']};z['genes']=pd.read_csv(base/'genes.tsv',sep='\t').gene_id.astype(str).to_numpy();return z
def context_data(panel,c):
 t=panel['train'];ix=t['context']==c;ctrl=panel['controls_fit'];ci=np.flatnonzero(ctrl['context']==c);assert len(ci)==1;control=ctrl['expression'][ci[0]];conditions=t['target'][ix].astype(str).tolist();y=(t['expression'][ix]-control).astype(float)
 return conditions,y,control,t['deg_mask'][ix]
def score_validation(pred,truth,conditions,contexts,control,keep):
 rows=[]
 for c in sorted(set(contexts)):
  ix=np.flatnonzero(contexts==c);pp=pred[ix][:,keep]-control[c][keep];yy=truth[ix][:,keep]-control[c][keep];n=len(ix);den=np.linalg.norm(pp,axis=1)[:,None]*np.linalg.norm(yy,axis=1)[None,:];cos=np.divide(pp@yy.T,den,out=np.zeros((n,n)),where=den>1e-15);l2=np.maximum(np.sum(pp*pp,axis=1)[:,None]+np.sum(yy*yy,axis=1)[None,:]-2*pp@yy.T,0)
  for j,k in enumerate(ix):
   scores=[]
   for dist in [1-cos,l2]:
    d=dist[j];v=d[j];scores.append(1-((d<v-1e-12).sum()+.5*((abs(d-v)<=1e-12).sum()-1))/(n-1) if n>1 else np.nan)
   rows.append(dict(target=conditions[k],pds=np.mean(scores),mse=np.mean((pred[k]-truth[k])**2)))
 f=pd.DataFrame(rows).groupby('target')[['pds','mse']].mean();return {'pds':float(f.pds.mean()) if f.pds.notna().any() else None,'mse':float(f.mse.mean())}
def run(ds,fold):
 from isolation_double import confine
 out=E/f'runs/{ds}/fold{fold}';out.mkdir(parents=True,exist_ok=True);confine(ds,fold,out)
 start=time.time();bank=Bank(ds);old=json.loads((E/f'provenance/old_configs/{ds}_{fold}.json').read_text());cfg={k:old[k] for k in ['lambda_pair','lambda_int','pair_intercept']};panel=loadpanel(ds,'seen0',fold);v=panel['validation'];queries=v['target'].astype(str);contexts=v['context'].astype(str);truth=v['expression'].astype(float);states={};ctrl={}
 for c in sorted(set(panel['train']['context'])):
  conditions,y,control,degs=context_data(panel,c);states[c]=prepare(bank,conditions,y,panel['genes'],cfg);ctrl[c]=control
 selection=[]
 for alpha in [.01,.1,1.]:
  pred=np.empty_like(truth)
  for c,pre in states.items():
   ix=np.flatnonzero(contexts==c)
   if len(ix):pred[ix]=predict(bank,fit(pre,alpha),queries[ix].tolist(),ctrl[c])
  s=score_validation(pred,truth,queries,contexts,ctrl,~np.isin(panel['genes'],bank.targets));selection.append(dict(alpha=alpha,**s))
 best=min(selection,key=lambda s:(-(s['pds'] if s['pds'] is not None else -1),s['mse'],s['alpha']));save(out/'selection.json',dict(candidates=selection,chosen=best,structure=cfg,source_old_config=old,selection_scope='seen0 validation only; frozen across arms; no validation refit'))
 del states
 for arm in ['seen0','seen1','seen2']:
  panel=loadpanel(ds,arm,fold);q=panel['query'];qt=q['target'].astype(str);qc=q['context'].astype(str);pred=np.empty((len(qt),len(panel['genes'])),np.float32);dest=out/arm;dest.mkdir(exist_ok=True);audits=[]
  for ci,c in enumerate(sorted(set(qc))):
   conditions,y,control,degs=context_data(panel,c);pre=prepare(bank,conditions,y,panel['genes'],cfg);state=fit(pre,best['alpha']);ix=np.flatnonzero(qc==c);qq=qt[ix].tolist();yp,pieces=predict(bank,state,qq,control,True);pred[ix]=yp
   swapped=['+'.join(reversed(parts(t))) for t in qq];np.testing.assert_allclose(yp,predict(bank,state,swapped,control),atol=1e-8,rtol=1e-8)
   np.savez_compressed(dest/f'model_context{ci}.npz',components=bank.targets,node=pre['node'],train_conditions=np.array(conditions),coef=state['coef'],intercept=state['intercept'],control=control,weights=pre['weights'],alignment=pre['alignment'],training_weights=pre['w'])
   audits.append(dict(context=c,training_conditions=len(conditions),training_components=len(pre['training_components']),query_count=len(ix),weights=pre['weights'].tolist(),alignment=pre['alignment'].tolist(),solve_error=state['solve_error'],pair_intercept=pre['pair'],swap_invariant=True,component_rms={k:float(np.sqrt(np.mean(v*v))) for k,v in pieces.items()},model_sha256=sha(dest/f'model_context{ci}.npz')))
  np.savez_compressed(dest/'predictions_sealed.npz',prediction=pred,target=qt,context=qc);save(dest/'result.json',dict(status='PREDICTIONS_SEALED',model='KiPERT-double',dataset=ds,fold=fold,arm=arm,alpha=best['alpha'],structure=cfg,queries=len(qt),prediction_sha256=sha(dest/'predictions_sealed.npz'),audits=audits,test_metrics_computed=False));print(json.dumps(dict(stage='SEALED',dataset=ds,fold=fold,arm=arm,seconds=time.time()-start)),flush=True)
 save(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.time()-start))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--fold',type=int,required=True);a=p.parse_args();run(a.dataset,a.fold)
