import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,hashlib,ast,inspect,gc
import numpy as np,pandas as pd
from scipy import sparse,stats
from sklearn.metrics import average_precision_score
R=Path('@STUDY@/uc_fig2_benchmark_20260911');E=R/'experiments/uc_fig2_latest_within_context_20261105';O=E/'alpha_constrained_20261105';O.mkdir(exist_ok=True)
sys.path.insert(0,str(E/'src'));from model import Bank,response_subspace
# Reuse only tested numerical helpers, without executing earlier experiments.
tree=ast.parse((E/'parameter_sensitivity_20261105/experiment.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['transform','features','predict','cor','unit']],type_ignores=[]),'helpers','exec'))
AL=np.logspace(-5,2,15).tolist();D=['S3','S4','S5','S6','S7'];SEEDS=[101,202,303]
protocol=dict(scope='Five datasets, three original splits, all available backgrounds; one alpha per dataset/split',alpha_grid=AL,structure='Only alpha changes. Original response mode, rank cap16 and tail0.25 (or full), 12 equal priors, free intercept, controls and nonnegative projection retained.',primary='Minimum validation off-diagonal cosine-matrix RMSE; full target-panel exclusion, same as original response-structure pages.',constraints=dict(rmse_max_ratio=1.01,pds_l2='>=baseline',pds_cosine='>=baseline',specific_pearson='>=baseline',deg_ap='>=baseline',matrix_spearman='>=baseline'),missing='If any gate or primary metric undefined at baseline, keep original alpha; undefined candidate metrics are ineligible. Matrix Spearman only contexts with >=3 targets. Backgrounds with no validation targets supply no validation score, but retain refit/test predictions under the dataset-shared alpha.',aggregation='Per-target metrics: equal contexts within target then equal targets. Matrix metrics: equal eligible contexts. One alpha shared across all contexts in each dataset/split.',tie='Within 1e-12 primary tie: keep original alpha if eligible, otherwise largest alpha.',freeze='All fifteen selections and refit predictions sealed before test truth is loaded. All test configurations reported; no test-based promotion.',metric_note='PDS excludes full target panel; specific Pearson uses equal-target fit-only reference; DEG AP uses mean-output abs(delta) ranking with original Welch/BH truth labels, not nonlinear AUPRC.',status='RETROSPECTIVE_PROTOCOL_FROZEN_BEFORE_GRID')
assert not (O/'PROTOCOL.json').exists(),'Existing run must be inspected, not overwritten'
(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));ph=hashlib.sha256((O/'PROTOCOL.json').read_bytes()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
GATES=['pds_l2','pds_cosine','specific_pearson','deg_ap','matrix_spearman']
def metrics(p,y,ctrl,keep,shared,deg,targets,c):
 dp=p-ctrl;dy=y-ctrl;dp[np.sqrt(np.mean(dp*dp,axis=1))<1e-7]=0;pp=dp[:,keep];yy=dy[:,keep];n=len(y)
 r=pd.DataFrame(dict(target=targets,context=c,rmse=np.sqrt(np.mean((p-y)**2,axis=1)),specific_pearson=cor(dp-shared,dy-shared)))
 r['deg_ap']=[average_precision_score(l,abs(x)) if 0<np.sum(l)<len(l) else np.nan for l,x in zip(deg,dp)]
 for name,dist in [('l2',np.maximum(np.sum(pp*pp,axis=1)[:,None]+np.sum(yy*yy,axis=1)[None,:]-2*pp@yy.T,0)),('cosine',1-unit(pp)@unit(yy).T)]:
  v=[]
  for i in range(n):
   other=np.arange(n)!=i;v.append(1-(np.sum(dist[i,other]<dist[i,i]-1e-12)+.5*np.sum(abs(dist[i,other]-dist[i,i])<=1e-12))/(n-1) if n>1 else np.nan)
  r['pds_'+name]=v
 a=unit(yy)@unit(yy).T;b=unit(pp)@unit(pp).T;ix=np.triu_indices(n,1)
 den=np.sum((yy-yy.mean(0))**2)
 m=dict(context=c,n=n,matrix_rmse=float(np.sqrt(np.mean((a[ix]-b[ix])**2))) if n>1 else np.nan,matrix_spearman=float(stats.spearmanr(a[ix],b[ix]).statistic) if n>=3 and np.std(b[ix])>0 else np.nan,pair_cosine=float(b[ix].mean()) if n>1 else np.nan,energy_ratio=float(np.sum((pp-pp.mean(0))**2)/den) if den>0 else np.nan)
 return r,m
def aggregate(rr,mm):
 t=pd.concat(rr).groupby('target')[['rmse','specific_pearson','pds_l2','pds_cosine','deg_ap']].mean();m=pd.DataFrame(mm);a=t.mean().to_dict();a.update(m[['matrix_rmse','matrix_spearman','pair_cosine','energy_ratio']].mean().to_dict());a.update(n_targets=len(t),deg_targets=int(t.deg_ap.notna().sum()),matrix_contexts=int(m.matrix_rmse.notna().sum()));return a
def make_rows(x,m,ctrl,cv,cn):
 out=[];labels=[]
 for t,ii in m.groupby('target',sort=True).indices.items():
  xx=x[ii];mu=np.asarray(xx.mean(0)).ravel();var=np.maximum(np.asarray(xx.power(2).mean(0)).ravel()-mu**2,0);se=np.sqrt(var/len(ii)+cv/cn);stat=np.divide(abs(mu-ctrl),se,out=np.zeros_like(mu),where=se>1e-12);out.append((t,mu-ctrl,np.argsort(-stat,kind='stable')[:100]))
  v1=var/max(len(ii)-1,1);v0=cv/max(cn-1,1);s=v1+v0;tt=np.divide(abs(mu-ctrl),np.sqrt(s),out=np.zeros_like(mu),where=s>1e-20);tt[(s<=1e-20)&(abs(mu-ctrl)>1e-12)]=np.inf;df=np.divide(s*s,v1*v1/max(len(ii)-1,1)+v0*v0/max(cn-1,1),out=np.ones_like(mu),where=s>1e-20);p=2*stats.t.sf(tt,df);order=np.argsort(p);adj=np.empty(len(p));adj[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];labels.append((adj<.05)&(abs(mu-ctrl)>.1))
 return out,np.stack(labels) if labels else np.empty((0,x.shape[1]),bool)
V=[];SEL=[];RECEIPTS=[];PARITY=[]
for d in D:
 bank=Bank(d)
 for seed in SEEDS:
  B=R/'data/formal'/d/f'split{seed}';dest=O/d/f'split{seed}';dest.mkdir(parents=True);arrays={};meta={}
  for role in ['fit','validation','controls_fit']:
   arrays[role]=sparse.load_npz(B/role/'expression.npz').tocsr();meta[role]=pd.read_csv(B/role/'metadata.tsv',sep='\t',dtype=str)
  q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str);genes=pd.read_csv(B/'genes.tsv',sep='\t').gene_id.astype(str).to_numpy();keep=~np.isin(genes,bank.targets.astype(str));base=json.loads((E/'runs'/d/f'split{seed}/UC/selection.json').read_text())['chosen']['config'];cfg=dict(alpha=base['alpha'],rank=16,tail=.25 if base['response']=='deg_subspace' else 1.)
  caches={};details={a:([],[]) for a in AL}
  for c in sorted(set(q.context)):
   sub={};ms={}
   for role in arrays:
    ii=np.flatnonzero(meta[role].context.to_numpy()==c);sub[role]=arrays[role][ii].astype(float);ms[role]=meta[role].iloc[ii].reset_index(drop=True)
   xc=sub['controls_fit'];ctrl=np.asarray(xc.mean(0)).ravel();cv=np.maximum(np.asarray(xc.power(2).mean(0)).ravel()-ctrl**2,0);cn=len(ms['controls_fit']);tr,_=make_rows(sub['fit'],ms['fit'],ctrl,cv,cn);va,deg=make_rows(sub['validation'],ms['validation'],ctrl,cv,cn)
   rg,_=make_rows(sparse.vstack([sub['fit'],sub['validation']],format='csr'),pd.concat([ms['fit'],ms['validation']],ignore_index=True),ctrl,cv,cn);shared=np.stack([r[1] for r in tr]).mean(0);caches[c]=(rg,ctrl,shared)
   if not va:continue
   queries=[r[0] for r in va];cache=features(bank,tr,queries);trans={};y=np.stack([r[1]+ctrl for r in va])
   for a in AL:
    p=predict(tr,queries,ctrl,dict(cfg,alpha=a),cache,trans);rr,mm=metrics(p,y,ctrl,keep,shared,deg,queries,c);details[a][0].append(rr);details[a][1].append(mm)
   caches[c]=(rg,ctrl,shared);del sub,ms
  for a in AL:V.append(dict(dataset=d,split=seed,alpha=a,**aggregate(*details[a])))
  grid=pd.DataFrame(V);grid=grid[(grid.dataset==d)&(grid.split==seed)].copy();baseline=grid[np.isclose(grid.alpha,base['alpha'],atol=0,rtol=1e-12)].iloc[0];eligible=np.isfinite(grid[['matrix_rmse','rmse']+GATES]).all(axis=1)&(grid.rmse<=baseline.rmse*1.01+1e-12)
  for key in GATES:eligible&=grid[key]>=baseline[key]-1e-12
  if not np.isfinite(baseline[['matrix_rmse','rmse']+GATES].astype(float)).all():chosen=baseline;reason='undefined_baseline_gate'
  else:
   g=grid[eligible];assert len(g)>0;best=g[g.matrix_rmse<=g.matrix_rmse.min()+1e-12];same=best[np.isclose(best.alpha,base['alpha'],rtol=1e-12,atol=0)];chosen=same.iloc[0] if len(same) else best.sort_values('alpha',ascending=False).iloc[0];reason='constrained_minimum_matrix_error'
  record=dict(dataset=d,split=seed,original_alpha=base['alpha'],selected_alpha=float(chosen.alpha),response=base['response'],eligible_candidates=int(eligible.sum()),reason=reason,**{f'validation_{key}':float(chosen[key]) for key in ['matrix_rmse','rmse']+GATES});SEL.append(record)
  grid['eligible']=eligible.to_numpy();grid.to_csv(dest/'validation.csv',index=False);(dest/'SELECTION_FROZEN.json').write_text(json.dumps(record,indent=2,allow_nan=True))
  oldfile=E/'runs'/d/f'split{seed}/UC/predictions_sealed.npz';old=np.load(oldfile)['prediction'];assert sha(oldfile)==json.loads((oldfile.parent/'result.json').read_text())['prediction_sha256'];yp=np.empty_like(old);ypbase=np.empty_like(old);refs={}
  for ci,(c,(rg,ctrl,shared)) in enumerate(caches.items()):
   ii=np.flatnonzero(q.context.to_numpy()==c);queries=q.target.iloc[ii].tolist();cache=features(bank,rg,queries);trans={};yp[ii]=predict(rg,queries,ctrl,dict(cfg,alpha=float(chosen.alpha)),cache,trans);ypbase[ii]=predict(rg,queries,ctrl,cfg,cache,trans);refs[f'shared_{ci}']=shared;refs[f'control_{ci}']=ctrl;refs[f'context_{ci}']=np.array([c])
  err=float(np.max(abs(ypbase-old)));assert err<2e-6;PARITY.append(dict(dataset=d,split=seed,max_error=err));np.savez_compressed(dest/'predictions_sealed.npz',prediction=yp,target=q.target.to_numpy(dtype=str),context=q.context.to_numpy(dtype=str));np.savez_compressed(dest/'references.npz',**refs,keep=keep)
  RECEIPTS.append(dict(dataset=d,split=seed,selection_sha256=sha(dest/'SELECTION_FROZEN.json'),prediction_sha256=sha(dest/'predictions_sealed.npz'),original_prediction_sha256=sha(oldfile)))
  pd.DataFrame(SEL).to_csv(O/'selections.csv',index=False);pd.DataFrame(V).to_csv(O/'validation_grid.csv',index=False);print('SEALED',d,seed,base['alpha'],'->',chosen.alpha,'eligible',eligible.sum(),flush=True);del arrays,meta,caches,details;gc.collect()
(O/'ALL_SELECTIONS_AND_PREDICTIONS_SEALED.json').write_text(json.dumps(RECEIPTS,indent=2));testrows=[];contexts=[];targets=[]
for item in RECEIPTS:
 d=item['dataset'];seed=item['split'];dest=O/d/f'split{seed}';assert sha(dest/'SELECTION_FROZEN.json')==item['selection_sha256'];assert sha(dest/'predictions_sealed.npz')==item['prediction_sha256'];refs=np.load(dest/'references.npz');z=np.load(E/'specificity_extension_20261105/truth_cache'/f'{d}_{seed}.npz');keep=refs['keep'];np.testing.assert_array_equal(keep,z['gene_keep'])
 for arm,pf in [('baseline',E/'runs'/d/f'split{seed}/UC/predictions_sealed.npz'),('selected',dest/'predictions_sealed.npz')]:
  pred=np.load(pf);np.testing.assert_array_equal(pred['target'].astype(str),z['target'].astype(str));np.testing.assert_array_equal(pred['context'].astype(str),z['context'].astype(str));rr=[];mm=[]
  for ci in range(len([k for k in refs.files if k.startswith('context_')])):
   c=str(refs[f'context_{ci}'][0]);ii=np.flatnonzero(z['common']&(z['context'].astype(str)==c))
   if not len(ii):continue
   r,m=metrics(pred['prediction'][ii].astype(float),z['truth'][ii].astype(float),z['control'][ii].astype(float),keep,refs[f'shared_{ci}'],z['deg'][ii],z['target'][ii].astype(str),c);rr.append(r);mm.append(m);contexts.append(dict(dataset=d,split=seed,arm=arm,**m));targets.append(r.assign(dataset=d,split=seed,arm=arm))
  testrows.append(dict(dataset=d,split=seed,arm=arm,**aggregate(rr,mm)))
 pd.DataFrame(testrows).to_csv(O/'test_split_summary.csv',index=False);pd.DataFrame(contexts).to_csv(O/'test_contexts.csv',index=False);pd.concat(targets).to_csv(O/'test_target_context_scores.csv',index=False);print('EVALUATED',d,seed,flush=True)
assert ph==sha(O/'PROTOCOL.json')
(O/'COMPLETE.json').write_text(json.dumps(dict(status='PASS',dataset_splits=15,alpha_candidates=15,validation_rows=len(V),test_summary_rows=len(testrows),baseline_reproduction=PARITY,protocol_sha256=ph,structure_unchanged=True,formal_benchmark_replaced=False),indent=2));print('COMPLETE',flush=True)
