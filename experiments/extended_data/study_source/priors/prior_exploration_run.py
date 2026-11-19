import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,gc,shutil
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,'@STUDY@/kipert_supplement_20261117/code')
from single_ablation import Bank,VIEWS,prepare,kernel,predict,context_rows,metrics,aggregate,unit,center,E,R,AL
from single_score import score,load_truth,sha,save
N=Path('@STUDY@/kipert_prior_exploration_20261117');OLD=Path('@STUDY@/kipert_supplement_20261117')
def seed(*parts):return 2026111702+int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:7],16)
def permute_strata(signatures,rng):
 p=np.arange(len(signatures));groups={}
 for i,row in enumerate(signatures):groups.setdefault(tuple(row),[]).append(i)
 for ids in groups.values():
  if len(ids)<2:continue
  original=np.array(ids)
  while True:
   trial=rng.permutation(original)
   if np.all(trial!=original):break
  p[original]=trial
 assert np.array_equal(np.sort(p),np.arange(len(p)))
 return p
def make_permutations(states,bank,d,s,cfgs,dest):
 coverage=[np.stack([np.diag(v)>1e-24 for v in x['views']],axis=1) for x in states]
 signatures=np.concatenate(coverage,axis=1);maps={};rows=[]
 for cfg in cfgs:
  family=cfg['family']
  if not family.endswith('shuffle'):continue
  rng=np.random.default_rng(seed(d,s,cfg['name']))
  if family=='joint_shuffle':p=permute_strata(signatures,rng);pp=np.tile(p,(12,1))
  else:pp=np.stack([permute_strata(np.stack([cov[:,v] for cov in coverage],axis=1),rng) for v in range(12)])
  for cov in coverage:
   for v in range(12):assert np.array_equal(cov[:,v],cov[pp[v],v]),(cfg['name'],v)
  maps[cfg['name']]=pp
  for v in range(12):rows.append(dict(variant=cfg['name'],view=VIEWS[v],bank_targets=len(bank.targets),moved=int(np.sum(pp[v]!=np.arange(len(pp[v])))),covered=int(coverage[-1][:,v].sum()),covered_moved=int(np.sum((pp[v]!=np.arange(len(pp[v])))&coverage[-1][:,v]))))
 np.savez_compressed(dest/'permutation_mappings.npz',targets=bank.targets,**maps);pd.DataFrame(rows).to_csv(dest/'permutation_diagnostics.csv',index=False);return maps
def transformed(s,cfg,maps):
 if not cfg['family'].endswith('shuffle'):return s
 pp=maps[cfg['name']];t=s.copy();t['views']=np.stack([v[np.ix_(p,p)] for v,p in zip(s['views'],pp)]);z=unit(s['R'][:,s['keep']]);response=center(z@z.T);mask=~np.eye(len(z),dtype=bool);target=response[mask];ix=s['ix'];align=[]
 for original,v,p in zip(s['views'],t['views'],pp):
  np.testing.assert_allclose(np.diag(v),np.diag(original)[p],rtol=0,atol=1e-12)
  a=center(v[np.ix_(ix,ix)])[mask];den=np.linalg.norm(a)*np.linalg.norm(target);align.append(max(0,float(a@target/den)) if den>1e-12 else 0.)
 t['alignment']=np.array(align);return t
def make_model(s,cfg,maps):
 if cfg['family']=='centroid':return s,None
 t=transformed(s,cfg,maps);return t,kernel(t,cfg)
def prediction(s,k,ctrl,alpha):
 if k is None:return np.repeat(np.maximum(ctrl+s['b'],0).astype(np.float32)[None,:],len(s['qi']),axis=0).astype(float)
 return predict(s,k,ctrl,alpha)
def run(d,split):
 dest=N/'runs'/d/f'split{split}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return
 assert shutil.disk_usage(N).free>7*1024**3
 protocol=json.loads((N/'PROTOCOL.json').read_text());cfgs=protocol['configurations'];base=R/d/f'split{split}';q=pd.read_csv(base/'query/metadata.tsv',sep='\t',dtype=str);genes=pd.read_csv(base/'genes.tsv',sep='\t',dtype=str).gene_id.to_numpy();bank=Bank(E/'priors'/d);keep=~np.isin(genes,bank.targets);full_alpha=json.loads((OLD/'single/ablation'/d/f'split{split}/SELECTIONS_FROZEN.json').read_text())['fixed_alpha'];contexts={};allstates=[];qdiag=[]
 for c in sorted(set(q.context)):
  tr,va,rg,deg,ctrl=context_rows(d,split,c);ii=np.flatnonzero(q.context.to_numpy()==c);assert not set(q.target)&set(x[0] for x in rg)
  fit=prepare(bank,rg,q.target.iloc[ii].tolist(),keep);fit['keep']=keep
  val=prepare(bank,tr,[x[0] for x in va],keep) if va else None
  if val is not None:val['keep']=keep;allstates.append(val)
  allstates.append(fit);contexts[c]=dict(fit=fit,val=val,ctrl=ctrl,truth=np.stack([x[1]+ctrl for x in va]) if va else None,deg=deg,va=va,ii=ii)
  covered=np.stack([np.diag(v)>1e-24 for v in fit['views']],axis=1);uniform=fit['views'].mean(0);near=uniform[np.ix_(fit['qi'],fit['ix'])].max(1)
  for j,row in enumerate(ii):qdiag.append(dict(dataset=d,split=split,context=c,target=q.target.iloc[row],covered_views=int(covered[fit['qi'][j]].sum()),nearest_train_similarity=float(near[j]),fit_targets=len(rg)))
 pd.DataFrame(qdiag).to_csv(dest/'query_prior_diagnostics.csv',index=False)
 maps=make_permutations(allstates,bank,d,split,cfgs,dest)
 if not (dest/'SELECTIONS_FROZEN.json').exists():
  details={(cfg['name'],a):([],[]) for cfg in cfgs for a in ([None] if cfg['family']=='centroid' else AL)}
  for c,z in contexts.items():
   if z['val'] is None:continue
   for cfg in cfgs:
    s,k=make_model(z['val'],cfg,maps)
    for a in ([None] if cfg['family']=='centroid' else AL):
     rr,mm=metrics(prediction(s,k,z['ctrl'],a),z['truth'],z['ctrl'],keep,s['b'],z['deg'],[x[0] for x in z['va']],c);details[cfg['name'],a][0].append(rr);details[cfg['name'],a][1].append(mm)
   print('EXPLORATION VAL',d,split,c,flush=True)
  grid=pd.DataFrame([dict(variant=name,alpha=a,**aggregate(*vals)) for (name,a),vals in details.items()]);grid['objective']=(grid.pds_l2+grid.pds_cosine)/2;selected={}
  for cfg in cfgs:
   if cfg['family']=='centroid':selected[cfg['name']]=None;continue
   g=grid[grid.variant==cfg['name']].copy();g['candidate']=g.alpha.map(lambda a:f'a{a:g}');best=g.sort_values(['objective','matrix_spearman','specific_pearson','rmse','candidate'],ascending=[False,False,False,True,True],na_position='last').iloc[0];selected[cfg['name']]=float(best.alpha)
  assert selected['full']==full_alpha,(d,split,selected['full'],full_alpha);grid.to_csv(dest/'validation_grid.csv',index=False);save(dest/'SELECTIONS_FROZEN.json',dict(selected=selected,fixed_alpha=full_alpha,centroid_has_no_ridge_parameter=True,protocol_sha256=sha(N/'PROTOCOL.json')));del details;gc.collect()
 else:selected=json.loads((dest/'SELECTIONS_FROZEN.json').read_text())['selected']
 if not (dest/'PREDICTIONS_SEALED.json').exists():
  predictions={(cfg['name'],mode):np.lib.format.open_memmap(dest/(cfg['name']+'_'+mode+'.npy'),mode='w+',dtype=np.float32,shape=(len(q),len(genes))) for cfg in cfgs for mode in ['retuned','fixed']};ww=[]
  for c,z in contexts.items():
   for cfg in cfgs:
    s,k=make_model(z['fit'],cfg,maps);a=selected[cfg['name']];p=prediction(s,k,z['ctrl'],a);assert np.isfinite(p).all();predictions[cfg['name'],'retuned'][z['ii']]=p;predictions[cfg['name'],'fixed'][z['ii']]=p if a==full_alpha or k is None else prediction(s,k,z['ctrl'],full_alpha)
    if k is not None:
     for v,w in zip(cfg['active'],k['weights']):ww.append(dict(context=c,variant=cfg['name'],view=VIEWS[v],weight=float(w),alignment=float(s['alignment'][v]),uncovered_queries=int(k['missing'].sum())))
   print('EXPLORATION REFIT',d,split,c,flush=True)
  for a in predictions.values():a.flush()
  predictions={};pd.DataFrame(ww).to_csv(dest/'weights.csv',index=False);old=OLD/'single/ablation'/d/f'split{split}/full_beta0p5_retuned.npy';error=float(np.max(abs(np.load(dest/'full_retuned.npy',mmap_mode='r')-np.load(old,mmap_mode='r'))));assert error<2e-6,error
  save(dest/'PREDICTIONS_SEALED.json',dict(full_replay_error=error,reference_path=str(old),reference_sha256=sha(old),predictions={p.name:sha(p) for p in dest.glob('*.npy')}))
 # Test outcomes first enter the new experiment here, after all selections and predictions are frozen.
 z=load_truth(d,split)
 for cfg in cfgs:
  for mode in ['retuned','fixed']:
   name=cfg['name']+'_'+mode;p=dest/(name+'_metrics.csv')
   if not p.exists():score(np.load(dest/(name+'.npy'),mmap_mode='r'),z,d,split,name).to_csv(p,index=False)
 gold=pd.read_csv(OLD/'single/ablation'/d/f'split{split}/full_beta0p5_retuned_metrics.csv',dtype={'target':str,'context':str}).set_index(['target','context']);new=pd.read_csv(dest/'full_retuned_metrics.csv',dtype={'target':str,'context':str}).set_index(['target','context']).reindex(gold.index);errs={m:float(np.nanmax(abs(gold[m]-new[m]))) for m in ['mse','pds_query_cosine','auprc_dg','systema_cell_pearson']};assert max(errs.values())<1e-9,errs
 save(dest/'COMPLETE.json',dict(status='PASS',variants=len(cfgs),modes=2,queries=len(q),scored_queries=len(new),score_replay_errors=errs));print('EXPLORATION COMPLETE',d,split,flush=True)
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
