import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,ast,gc,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
from single_score import O,E,R,save,sha,corr,unit,load_truth,score
from single_data import load,rows
sys.path.insert(0,'@STUDY@/uc_single_perturbation/submission/figure2_code')
from kipert.inputs import Bank,VIEWS
from kipert.core import view_kernels,center
cor=corr
tree=ast.parse((E/'alpha_constrained_20261105/run.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metrics','aggregate']],type_ignores=[]),'frozen_validation_helpers','exec'))
AL=[.01,.1,1.]
def variants():
 vv=[dict(name='full_beta'+str(b).replace('.','p'),active=list(range(12)),beta=b) for b in [0.,.25,.5,.75,1.]]
 vv += [dict(name='drop_'+v,active=[j for j in range(12) if j!=i],beta=.5) for i,v in enumerate(VIEWS)]
 vv += [dict(name='only_'+v,active=[i],beta=.5) for i,v in enumerate(VIEWS)]
 vv += [dict(name='three_prior',active=[0,1,2],beta=.5)]
 return vv
def prepare(bank,tr,queries,keep):
 views,ix,coverage,names=view_kernels(bank,[r[0] for r in tr]);qi=np.array([bank.index[t] for t in queries]);y=np.stack([r[1] for r in tr]);b=y.mean(0);res=y-b;z=unit(res[:,keep]);tv=center(z@z.T);mask=~np.eye(len(tr),dtype=bool);a=[];vv=[]
 for node in views:
  v=center(node[np.ix_(ix,ix)])[mask];t=tv[mask];den=np.linalg.norm(v)*np.linalg.norm(t);a.append(max(0,float(v@t/den)) if den>1e-12 else 0);vv.append(v)
 sim=np.array([unit(v) for v in vv])@np.array([unit(v) for v in vv]).T
 return dict(views=views,ix=ix,qi=qi,b=b,R=res,alignment=np.array(a),similarity=sim)
def kernel(state,cfg):
 ids=cfg['active'];a=state['alignment'][ids];a=a/a.sum() if a.sum()>1e-12 else np.ones(len(ids))/len(ids);w=(1-cfg['beta'])/len(ids)+cfg['beta']*a;k=np.einsum('v,vij->ij',w,state['views'][ids]);ix=state['ix'];qi=state['qi'];kt=k[np.ix_(ix,ix)];cross=k[np.ix_(qi,ix)];qc=cross-kt.mean(0)[None,:]-cross.mean(1)[:,None]+kt.mean();ev,u=np.linalg.eigh(center(kt));ev=np.maximum(ev,0);missing=np.diag(k)[qi]<=1e-24
 assert abs(w.sum()-1)<1e-12
 return dict(ev=ev,u=u,qc=qc,missing=missing,weights=w)
def predict(s,k,ctrl,alpha):
 weight=(k['qc']@k['u']/(k['ev']+alpha))@k['u'].T;delta=s['b']+weight@s['R'];delta[k['missing']]=s['b'];return np.maximum(ctrl+delta,0).astype(np.float32).astype(float)
def context_rows(d,seed,c):
 m,x,xx=load(d,seed,'controls_fit');ii=np.flatnonzero(m.context.to_numpy()==c);n=m.n.iloc[ii].to_numpy();ctrl=np.average(x[ii],axis=0,weights=n);cv=np.maximum(np.average(xx[ii],axis=0,weights=n)-ctrl**2,0);cn=n.sum();tr,_=rows(*load(d,seed,'fit'),c,ctrl,cv,cn);va,deg=rows(*load(d,seed,'validation'),c,ctrl,cv,cn);rg=sorted(tr+va,key=lambda r:r[0]);return tr,va,rg,deg,ctrl
def run(d,seed):
 dest=O/'single/ablation'/d/f'split{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return
 cfgs=variants();base=R/d/f'split{seed}';q=pd.read_csv(base/'query/metadata.tsv',sep='\t',dtype=str);genes=pd.read_csv(base/'genes.tsv',sep='\t',dtype=str).gene_id.to_numpy();bank=Bank(E/'priors'/d);keep=~np.isin(genes,bank.targets);contexts=sorted(set(q.context));old=json.loads((E/'specific_methods_20261105'/d/f'split{seed}/SELECTIONS_FROZEN.json').read_text());full_alpha=float(old['aligned_raw'].split('_a')[1].split('_')[0]);details={(cfg['name'],alpha):([],[]) for cfg in cfgs for alpha in AL};weights=[]
 if not (dest/'SELECTIONS_FROZEN.json').exists():
  for c in contexts:
   tr,va,rg,deg,ctrl=context_rows(d,seed,c)
   assert not set(r[0] for r in rg)&set(q.target)
   if not va:continue
   s=prepare(bank,tr,[r[0] for r in va],keep);truth=np.stack([r[1]+ctrl for r in va]);shared=s['b']
   np.save(dest/('kernel_similarity_fit_'+str(contexts.index(c))+'.npy'),s['similarity'])
   for cfg in cfgs:
    k=kernel(s,cfg)
    for alpha in AL:
     rr,mm=metrics(predict(s,k,ctrl,alpha),truth,ctrl,keep,shared,deg,[r[0] for r in va],c);details[(cfg['name'],alpha)][0].append(rr);details[(cfg['name'],alpha)][1].append(mm)
   print('SINGLE VAL',d,seed,c,flush=True);del s;gc.collect()
  grid=pd.DataFrame([dict(variant=name,alpha=alpha,**aggregate(*det)) for (name,alpha),det in details.items()]);grid['objective']=(grid.pds_l2+grid.pds_cosine)/2;selected={}
  for cfg in cfgs:
   g=grid[grid.variant==cfg['name']].copy();g['candidate']=g.alpha.map(lambda a:f'a{a:g}');best=g.sort_values(['objective','matrix_spearman','specific_pearson','rmse','candidate'],ascending=[False,False,False,True,True],na_position='last').iloc[0];selected[cfg['name']]=float(best.alpha)
  grid.to_csv(dest/'validation_grid.csv',index=False);save(dest/'SELECTIONS_FROZEN.json',dict(selected=selected,fixed_alpha=full_alpha,full_alpha_selection_matches=selected['full_beta0p5']==full_alpha,configuration=cfgs));assert selected['full_beta0p5']==full_alpha,(selected,old)
 else:selected=json.loads((dest/'SELECTIONS_FROZEN.json').read_text())['selected']
 del details;gc.collect()
 if not (dest/'PREDICTIONS_SEALED.json').exists():
  pred={}
  for cfg in cfgs:
   for mode in ['retuned','fixed']:
    path=dest/(cfg['name']+'_'+mode+'.npy');pred[cfg['name'],mode]=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(q),len(genes)))
  for c in contexts:
   tr,va,rg,deg,ctrl=context_rows(d,seed,c);ii=np.flatnonzero(q.context.to_numpy()==c);s=prepare(bank,rg,q.target.iloc[ii].tolist(),keep)
   for cfg in cfgs:
    k=kernel(s,cfg);a=selected[cfg['name']];pp=predict(s,k,ctrl,a);pred[cfg['name'],'retuned'][ii]=pp;pred[cfg['name'],'fixed'][ii]=pp if a==full_alpha else predict(s,k,ctrl,full_alpha)
    for i,w in zip(cfg['active'],k['weights']):weights.append(dict(context=c,variant=cfg['name'],view=VIEWS[i],weight=float(w),alignment=float(s['alignment'][i]),queries=len(ii),uncovered_queries=int(k['missing'].sum())))
   print('SINGLE REFIT',d,seed,c,flush=True);del s;gc.collect()
  for a in pred.values():a.flush()
  pred={};pd.DataFrame(weights).to_csv(dest/'prior_weights.csv',index=False);p=np.load(dest/'full_beta0p5_retuned.npy',mmap_mode='r');ref=np.load(E/'specific_methods_20261105'/d/f'split{seed}/aligned_raw_predictions.npz')['prediction'];err=float(np.max(abs(p-ref)));assert err<2e-6,err
  uniform=np.load(dest/'full_beta0p0_retuned.npy',mmap_mode='r');uf=np.load(E/'specific_methods_20261105'/d/f'split{seed}/uniform_raw_predictions.npz')['prediction'];ue=float(np.max(abs(uniform-uf)));assert ue<2e-6,ue
  save(dest/'PREDICTIONS_SEALED.json',dict(full_replay_max_error=err,uniform_replay_max_error=ue,predictions={p.name:sha(p) for p in dest.glob('*.npy') if p.name.endswith(('_retuned.npy','_fixed.npy'))}));del p,ref,uniform,uf;gc.collect()
 z=load_truth(d,seed)
 for cfg in cfgs:
  for mode in ['retuned','fixed']:
   name=cfg['name']+'_'+mode;path=dest/(name+'_metrics.csv')
   if not path.exists():score(np.load(dest/(name+'.npy'),mmap_mode='r'),z,d,seed,name).to_csv(path,index=False)
 save(dest/'COMPLETE.json',dict(status='PASS',variants=len(cfgs),selection_modes=2,queries=len(q)));print('SINGLE COMPLETE',d,seed,flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:run(sys.argv[1],int(sys.argv[2]))
 else:
  for d in ['S3','S4','S5','S6','S7']:
   for seed in [101,202,303]:run(d,seed)
