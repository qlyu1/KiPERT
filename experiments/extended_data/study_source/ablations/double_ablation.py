import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,ast,gc,zipfile,tempfile,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse,stats,linalg
from single_score import save,sha,corr,auprc
ROOT=Path('@STUDY@/kipert_supplement_20261117');R=Path('@STUDY@/uc_seen012_benchmark_20261005');E=R/'experiments/kipert_double_20261106';O=ROOT/'double/truth';O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(E/'src'));import kipert_double as kd
tree=ast.parse((E/'literature_panels_20261106/extract_literature_panels.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read'],type_ignores=[]),'original_double_truth_reader','exec'))
def variants():
 return [dict(name='full_beta'+str(b).replace('.','p'),active=list(range(12)),beta=b) for b in [0.,.25,.5,.75,1.]]+[dict(name='drop_'+v,active=[j for j in range(12) if j!=i],beta=.5) for i,v in enumerate(kd.VIEWS)]
def prepare(bank,conditions,Y,genes,cfg):
 views,coverage,components=bank.kernels(conditions);h,w,pair=kd.hweights(conditions,cfg);b=linalg.solve(h.T@(w[:,None]*h),h.T@(w[:,None]*Y),assume_a='sym');res=Y-h@b;keep=~np.isin(genes,bank.targets);z=res[:,keep];norm=np.linalg.norm(z,axis=1,keepdims=True);z=np.divide(z,norm,out=np.zeros_like(z),where=norm>1e-12);target=kd.center(z@z.T);mask=~np.eye(len(conditions),dtype=bool);tv=target[mask];align=[]
 for node in views:
  a=kd.center(kd.lift(node,bank.targets,conditions,conditions,cfg))[mask];den=np.linalg.norm(a)*np.linalg.norm(tv);align.append(max(0.,float(a@tv/den)) if den>1e-12 else 0.)
 return dict(views=views,h=h,w=w,pair=pair,alignment=np.array(align),coverage=coverage,training_components=components,conditions=conditions,Y=Y,cfg=cfg)
def kernel(bank,p,cfg):
 ids=cfg['active'];a=p['alignment'][ids];a=a/a.sum() if a.sum()>1e-12 else np.ones(len(ids))/len(ids);weights=(1-cfg['beta'])/len(ids)+cfg['beta']*a;node=np.einsum('v,vij->ij',weights,p['views'][ids]);q={k:v for k,v in p.items() if k!='views'};q.update(node=node,K=kd.lift(node,bank.targets,p['conditions'],p['conditions'],p['cfg']),weights=weights);assert abs(weights.sum()-1)<1e-12;return q
def truth(ds,fold):
 path=O/f'{ds}_{fold}.npz'
 if not path.exists():
  base=R/f'data/formal/{ds}/seen0/fold{fold}';x,m,t=read('evaluation_only',base);cx,cm,c=read('controls_score',base);keys=sorted(t);y=np.stack([t[k][0] for k in keys]);cd={cc:next(v for kk,v in c.items() if kk[1]==cc) for cc in {k[1] for k in keys}};control=np.stack([cd[k[1]][0] for k in keys]);dy=y-control;top20=np.argsort(-abs(dy),axis=1,kind='stable')[:,:20];de100=[];labels=[]
  for k in keys:
   a=t[k];b=cd[k[1]];pv=stats.ttest_ind_from_stats(a[0],np.sqrt(a[1]),a[2],b[0],np.sqrt(b[1]),b[2],equal_var=False).pvalue;pv=np.nan_to_num(pv,nan=1);de100.append(np.lexsort((np.arange(len(pv)),-abs(a[0]-b[0]),pv))[:100]);order=np.argsort(pv);adj=np.empty(len(pv));adj[order]=np.minimum.accumulate((pv[order]*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1];labels.append((adj<.05)&(abs(a[0]-b[0])>.1))
  np.savez_compressed(path,truth=y,control=control,target=np.array([k[0] for k in keys]),context=np.array([k[1] for k in keys]),top20=top20,top100=np.array(de100),deg=np.stack(labels));del x,cx,m,cm;gc.collect()
 return dict(np.load(path))
def score(ds,fold,arm,name,pred,qt,qc,z):
 lookup={(t,c):i for i,(t,c) in enumerate(zip(z['target'],z['context']))};ii=np.array([lookup[t,c] for t,c in zip(qt,qc)]);y=z['truth'][ii];c=z['control'][ii];top=z['top20'][ii];top100=z['top100'][ii];labs=z['deg'][ii];pred=np.maximum(pred.astype(float),0);dp=pred-c;dy=y-c;err=(pred-y)**2;j=np.arange(len(y))[:,None]
 return pd.DataFrame(dict(dataset=ds,fold=fold,arm=arm,variant=name,target=qt,context=qc,mse=err.mean(1),mse20=err[j,top].mean(1),nochange20=(dy[j,top]**2).mean(1),opposite20=((dp[j,top]*dy[j,top])<0).mean(1),pcc_delta=corr(dp,dy),pcc_delta20=corr(dp[j,top],dy[j,top]),mse100=err[j,top100].mean(1),deg_auprc_dg=[auprc(l,abs(a)) for l,a in zip(labs,dp)]))
def run(ds,fold):
 dest=ROOT/'double/ablation'/ds/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return
 cfgs=variants();bank=kd.Bank(ds);orig=json.loads((E/f'runs/{ds}/fold{fold}/selection.json').read_text());structure=orig['structure'];full_alpha=orig['chosen']['alpha'];panel=kd.loadpanel(ds,'seen0',fold)
 if not (dest/'SELECTIONS_FROZEN.json').exists():
  v=panel['validation'];qt=v['target'].astype(str);qc=v['context'].astype(str);yp=v['expression'].astype(float);pres={};ctrl={}
  for c in sorted(set(panel['train']['context'])):
   cond,y,mu,_=kd.context_data(panel,c);pres[c]=prepare(bank,cond,y,panel['genes'],structure);ctrl[c]=mu
  allrows=[];selected={}
  for cfg in cfgs:
   ks={c:kernel(bank,p,cfg) for c,p in pres.items()};rr=[]
   for alpha in [.01,.1,1.]:
    pp=np.empty_like(yp)
    for c,pre in ks.items():
     ii=np.flatnonzero(qc==c)
     if len(ii):pp[ii]=kd.predict(bank,kd.fit(pre,alpha),qt[ii].tolist(),ctrl[c])
    r=dict(variant=cfg['name'],alpha=alpha,**kd.score_validation(pp,yp,qt,qc,ctrl,~np.isin(panel['genes'],bank.targets)));rr.append(r)
   best=min(rr,key=lambda s:(-(s['pds'] if s['pds'] is not None else -1),s['mse'],s['alpha']));selected[cfg['name']]=best['alpha'];allrows.extend(rr)
  assert selected['full_beta0p5']==full_alpha,(selected,full_alpha);pd.DataFrame(allrows).to_csv(dest/'validation_grid.csv',index=False);save(dest/'SELECTIONS_FROZEN.json',dict(selected=selected,fixed_alpha=full_alpha,structure=structure,variants=cfgs));del pres,ks;gc.collect()
 else:selected=json.loads((dest/'SELECTIONS_FROZEN.json').read_text())['selected']
 for arm in ['seen0','seen1','seen2']:
  folder=dest/arm;folder.mkdir(exist_ok=True);panel=kd.loadpanel(ds,arm,fold);qt=panel['query']['target'].astype(str);qc=panel['query']['context'].astype(str);ww=[]
  if not (folder/'PREDICTIONS_SEALED.json').exists():
   preds={(cfg['name'],mode):np.lib.format.open_memmap(folder/(cfg['name']+'_'+mode+'.npy'),mode='w+',dtype=np.float32,shape=(len(qt),len(panel['genes']))) for cfg in cfgs for mode in ['retuned','fixed']}
   for c in sorted(set(qc)):
    cond,y,mu,_=kd.context_data(panel,c);p=prepare(bank,cond,y,panel['genes'],structure);ii=np.flatnonzero(qc==c)
    for cfg in cfgs:
     pre=kernel(bank,p,cfg);a=selected[cfg['name']];yp=kd.predict(bank,kd.fit(pre,a),qt[ii].tolist(),mu);preds[cfg['name'],'retuned'][ii]=yp;preds[cfg['name'],'fixed'][ii]=yp if a==full_alpha else kd.predict(bank,kd.fit(pre,full_alpha),qt[ii].tolist(),mu)
     ww.extend(dict(context=c,variant=cfg['name'],view=kd.VIEWS[j],weight=float(w),alignment=float(p['alignment'][j])) for j,w in zip(cfg['active'],pre['weights']))
    print('DOUBLE FIT',ds,fold,arm,c,flush=True)
   for a in preds.values():a.flush()
   preds={};ref=np.load(E/f'runs/{ds}/fold{fold}/{arm}/predictions_sealed.npz');np.testing.assert_array_equal(qt,ref['target']);np.testing.assert_array_equal(qc,ref['context']);pp=np.load(folder/'full_beta0p5_retuned.npy');err=float(np.max(abs(pp-ref['prediction'])));assert err<2e-6,err;pd.DataFrame(ww).to_csv(folder/'weights.csv',index=False);save(folder/'PREDICTIONS_SEALED.json',dict(full_replay_error=err,files={p.name:sha(p) for p in folder.glob('*.npy')}));del pp,ref
  z=truth(ds,fold)
  for cfg in cfgs:
   for mode in ['retuned','fixed']:
    name=cfg['name']+'_'+mode;path=folder/(name+'_metrics.csv')
    if not path.exists():score(ds,fold,arm,name,np.load(folder/(name+'.npy'),mmap_mode='r'),qt,qc,z).to_csv(path,index=False)
  gold=pd.read_csv(E/'literature_panels_20261106/data/query_metrics.csv.gz',dtype={'target':str,'context':str});gold=gold[(gold.model=='kipert')&(gold.dataset==ds)&(gold.fold==fold)&(gold.arm==arm)].set_index(['target','context']);new=pd.read_csv(folder/'full_beta0p5_retuned_metrics.csv',dtype={'target':str,'context':str}).set_index(['target','context']).reindex(gold.index);errors={k:float(np.nanmax(abs(new[k]-gold[k]))) for k in ['mse','mse20','nochange20','opposite20','pcc_delta','pcc_delta20']};assert max(errors.values())<1e-8,errors;save(folder/'SCORE_PARITY.json',errors)
 save(dest/'COMPLETE.json',dict(status='PASS',variants=len(cfgs),arms=3,modes=2));print('DOUBLE COMPLETE',ds,fold,flush=True)
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
