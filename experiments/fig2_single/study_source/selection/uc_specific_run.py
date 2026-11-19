import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,hashlib,ast,inspect,gc
import numpy as np,pandas as pd
from scipy import sparse,stats
from sklearn.metrics import average_precision_score
R=Path('@STUDY@/uc_fig2_benchmark_20260911');E=R/'experiments/uc_fig2_latest_within_context_20261105';O=E/'specific_methods_20261105';O.mkdir(exist_ok=True)
sys.path.insert(0,str(E/'src'));from model import Bank,response_subspace
from uc_specific_methods import build,predict_candidate,configurations
for source,names in [(E/'parameter_sensitivity_20261105/experiment.py',['transform','features','predict','cor','unit']),(E/'alpha_constrained_20261105/run.py',['sha','metrics','aggregate','make_rows'])]:
 tree=ast.parse(source.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'frozen_helpers','exec'))
CONFIGS=configurations();IDS=['baseline']+[x['id'] for x in CONFIGS];BYID={x['id']:x for x in CONFIGS};FAMILIES=list(dict.fromkeys(x['family'] for x in CONFIGS));D=['S3','S4','S5','S6','S7'];SEEDS=[101,202,303]
protocol=dict(scope='All five datasets and three original splits; within-context unseen single perturbation',candidates=CONFIGS,baseline='Original frozen UC; raw uniform KRR is a distinct full-response control',alignment='Train-only positive off-diagonal centred kernel alignment to unit-residual response geometry excluding all panel target genes; weights=0.5*uniform+0.5*normalized_positive_alignment; no held labels',direction='Train centred response rows normalized to median training residual norm, then centred again; no predicted-direction renormalization; gains1/2',neighbors='Top3/10/30 cosine prior neighbors; softmax temperature0.1; transfer measured training residuals',validation_objective='Maximize average L2/Cosine PDS; ties by structural Spearman, specific Pearson, smaller expression RMSE, candidate ID',safe_constraints='Expression RMSE <=1.01 baseline; both PDS, specific Pearson, DEG AP and structural Spearman >=baseline; matrix RMSE <=baseline. Undefined necessary baseline metric retains baseline; undefined candidate fails.',safe_stability='Among constrained nonbaseline candidates select maximal 10th percentile of 2000 target-cluster bootstrap paired mean PDS gains; require >0. This is a selection heuristic, not multiplicity-adjusted significance.',diagnostic_arms='Also freeze best candidate of EACH of five families by validation objective, irrespective of safe gates; evaluate all and report tradeoffs. Diagnostic arms are not approved replacements.',freeze='All selections and refit predictions sealed before loading any test truth',prior_task_constraints='Same masks, control references, gene axes, data splits and comparator predictions. Shared-centred Pearson uses equal-target fit-only reference; DEG AP, not nonlinear AUPRC.',status='RETROSPECTIVE_PROTOCOL_DEFINED_BEFORE_NEW_RUN')
assert not (O/'PROTOCOL.json').exists();(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));ph=sha(O/'PROTOCOL.json');V=[];SEL=[];RECEIPTS=[];WEIGHTS=[];PARITY=[]
for d in D:
 bank=Bank(d)
 for seed in SEEDS:
  B=R/'data/formal'/d/f'split{seed}';dest=O/d/f'split{seed}';dest.mkdir(parents=True);arrays={};meta={}
  for role in ['fit','validation','controls_fit']:arrays[role]=sparse.load_npz(B/role/'expression.npz').tocsr();meta[role]=pd.read_csv(B/role/'metadata.tsv',sep='\t',dtype=str)
  q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str);genes=pd.read_csv(B/'genes.tsv',sep='\t').gene_id.astype(str).to_numpy();keep=~np.isin(genes,bank.targets.astype(str));bc=json.loads((E/'runs'/d/f'split{seed}/UC/selection.json').read_text())['chosen']['config'];basecfg=dict(alpha=bc['alpha'],rank=16,tail=.25 if bc['response']=='deg_subspace' else 1.)
  caches={};details={key:([],[]) for key in IDS}
  for c in sorted(set(q.context)):
   sub={};ms={}
   for role in arrays:
    ii=np.flatnonzero(meta[role].context.to_numpy()==c);sub[role]=arrays[role][ii].astype(float);ms[role]=meta[role].iloc[ii].reset_index(drop=True)
   xc=sub['controls_fit'];ctrl=np.asarray(xc.mean(0)).ravel();cv=np.maximum(np.asarray(xc.power(2).mean(0)).ravel()-ctrl**2,0);cn=len(ms['controls_fit']);tr,_=make_rows(sub['fit'],ms['fit'],ctrl,cv,cn);va,deg=make_rows(sub['validation'],ms['validation'],ctrl,cv,cn);rg,_=make_rows(sparse.vstack([sub['fit'],sub['validation']],format='csr'),pd.concat([ms['fit'],ms['validation']],ignore_index=True),ctrl,cv,cn);shared=np.stack([r[1] for r in tr]).mean(0);caches[c]=(rg,ctrl,shared)
   if not va:continue
   queries=[r[0] for r in va];y=np.stack([r[1]+ctrl for r in va]);state=build(bank,tr,queries,keep);basecache=features(bank,tr,queries);trans={};p=predict(tr,queries,ctrl,basecfg,basecache,trans);r,m=metrics(p,y,ctrl,keep,shared,deg,queries,c);details['baseline'][0].append(r);details['baseline'][1].append(m)
   check=predict_candidate(state,ctrl,dict(kernel='uniform',head='raw',alpha=bc['alpha'],gain=1.));ref=predict(tr,queries,ctrl,dict(basecfg,tail=1.),basecache,trans);assert np.max(abs(check-ref))<2e-6
   for cfg in CONFIGS:
    p=predict_candidate(state,ctrl,cfg);r,m=metrics(p,y,ctrl,keep,shared,deg,queries,c);details[cfg['id']][0].append(r);details[cfg['id']][1].append(m)
   for v,w,a in zip(state['view_names'],state['weights'],state['alignment']):WEIGHTS.append(dict(dataset=d,split=seed,context=c,stage='fit',view=v,weight=w,alignment=a))
  for key in IDS:V.append(dict(dataset=d,split=seed,candidate=key,family='baseline' if key=='baseline' else BYID[key]['family'],**aggregate(*details[key])))
  grid=pd.DataFrame(V);grid=grid[(grid.dataset==d)&(grid.split==seed)].copy();grid['objective']=(grid.pds_l2+grid.pds_cosine)/2;baseline=grid[grid.candidate=='baseline'].iloc[0];gates=['pds_l2','pds_cosine','specific_pearson','deg_ap','matrix_spearman'];required=['rmse','matrix_rmse']+gates
  eligible=np.isfinite(grid[required]).all(axis=1)&(grid.rmse<=1.01*baseline.rmse+1e-12)&(grid.matrix_rmse<=baseline.matrix_rmse+1e-12)
  for k in gates:eligible&=grid[k]>=baseline[k]-1e-12
  base_targets=pd.concat(details['baseline'][0]).groupby('target')[['pds_l2','pds_cosine']].mean().mean(1)
  lower={};rng=np.random.default_rng(seed+20261105)
  samples=rng.integers(0,len(base_targets),(2000,len(base_targets)))
  for key in grid[eligible&grid.candidate.ne('baseline')].candidate:
   tt=pd.concat(details[key][0]).groupby('target')[['pds_l2','pds_cosine']].mean().mean(1);delta=(tt.reindex(base_targets.index)-base_targets).to_numpy()
   lower[key]=float(np.quantile(np.nanmean(delta[samples],axis=1),.1)) if np.isfinite(delta).all() else -np.inf
  safe=max(lower,key=lambda k:(lower[k],float(grid[grid.candidate==k].objective.iloc[0]))) if lower else 'baseline'
  if safe!='baseline' and lower[safe]<=0:safe='baseline'
  if not np.isfinite(baseline[required].astype(float)).all():safe='baseline'
  selected={'baseline':'baseline','safe_selected':safe}
  for family in FAMILIES:
   g=grid[grid.family==family].sort_values(['objective','matrix_spearman','specific_pearson','rmse','candidate'],ascending=[False,False,False,True,True],na_position='last');selected[family]=str(g.iloc[0].candidate)
  grid['eligible']=eligible.to_numpy();grid['bootstrap_lower_gain']=grid.candidate.map(lower);grid.to_csv(dest/'validation.csv',index=False)
  for arm,key in selected.items():
   r=grid[grid.candidate==key].iloc[0];SEL.append(dict(dataset=d,split=seed,arm=arm,candidate=key,safe_eligible=bool(r.eligible),bootstrap_lower_gain=lower.get(key,np.nan),validation_objective=float(r.objective),**{f'validation_{k}':float(r[k]) for k in required}))
  (dest/'SELECTIONS_FROZEN.json').write_text(json.dumps(selected,indent=2));oldfile=E/'runs'/d/f'split{seed}/UC/predictions_sealed.npz';old=np.load(oldfile)['prediction'];assert sha(oldfile)==json.loads((oldfile.parent/'result.json').read_text())['prediction_sha256'];unique=list(dict.fromkeys(selected.values()));preds={k:np.empty_like(old) for k in unique};refs={}
  for ci,(c,(rg,ctrl,shared)) in enumerate(caches.items()):
   ii=np.flatnonzero(q.context.to_numpy()==c);queries=q.target.iloc[ii].tolist();state=build(bank,rg,queries,keep);basecache=features(bank,rg,queries);trans={}
   for key in unique:preds[key][ii]=predict(rg,queries,ctrl,basecfg,basecache,trans) if key=='baseline' else predict_candidate(state,ctrl,BYID[key])
   refs[f'context_{ci}']=np.array([c]);refs[f'shared_{ci}']=shared;refs[f'control_{ci}']=ctrl
   for v,w,a in zip(state['view_names'],state['weights'],state['alignment']):WEIGHTS.append(dict(dataset=d,split=seed,context=c,stage='refit',view=v,weight=w,alignment=a))
  err=float(np.max(abs(preds['baseline']-old)));assert err<2e-6;PARITY.append(dict(dataset=d,split=seed,max_error=err));np.savez_compressed(dest/'references.npz',**refs,keep=keep)
  for arm,key in selected.items():
   pf=dest/f'{arm}_predictions.npz';np.savez_compressed(pf,prediction=preds[key],target=q.target.to_numpy(dtype=str),context=q.context.to_numpy(dtype=str));RECEIPTS.append(dict(dataset=d,split=seed,arm=arm,candidate=key,prediction_sha256=sha(pf),selection_sha256=sha(dest/'SELECTIONS_FROZEN.json')))
  pd.DataFrame(SEL).to_csv(O/'selections.csv',index=False);pd.DataFrame(V).to_csv(O/'validation_grid.csv',index=False);pd.DataFrame(WEIGHTS).to_csv(O/'prior_weights.csv',index=False);print('SEALED',d,seed,'SAFE',safe,'FAMILIES',selected,flush=True);del arrays,meta,caches,details,preds;gc.collect()
(O/'ALL_PREDICTIONS_SEALED.json').write_text(json.dumps(RECEIPTS,indent=2));tr=[];cr=[];trows=[]
for d in D:
 for seed in SEEDS:
  dest=O/d/f'split{seed}';refs=np.load(dest/'references.npz');z=np.load(E/'specificity_extension_20261105/truth_cache'/f'{d}_{seed}.npz');keep=refs['keep'];np.testing.assert_array_equal(keep,z['gene_keep'])
  for receipt in [r for r in RECEIPTS if r['dataset']==d and r['split']==seed]:
   arm=receipt['arm'];pf=dest/f'{arm}_predictions.npz';assert sha(pf)==receipt['prediction_sha256'];assert sha(dest/'SELECTIONS_FROZEN.json')==receipt['selection_sha256'];p=np.load(pf);np.testing.assert_array_equal(p['target'].astype(str),z['target'].astype(str));np.testing.assert_array_equal(p['context'].astype(str),z['context'].astype(str));rr=[];mm=[]
   for ci in range(sum(k.startswith('context_') for k in refs.files)):
    c=str(refs[f'context_{ci}'][0]);ii=np.flatnonzero(z['common']&(z['context'].astype(str)==c))
    if not len(ii):continue
    r,m=metrics(p['prediction'][ii].astype(float),z['truth'][ii].astype(float),z['control'][ii].astype(float),keep,refs[f'shared_{ci}'],z['deg'][ii],z['target'][ii].astype(str),c);rr.append(r);mm.append(m);cr.append(dict(dataset=d,split=seed,arm=arm,**m));trows.append(r.assign(dataset=d,split=seed,arm=arm))
   tr.append(dict(dataset=d,split=seed,arm=arm,**aggregate(rr,mm)))
  pd.DataFrame(tr).to_csv(O/'test_split_summary.csv',index=False);pd.DataFrame(cr).to_csv(O/'test_contexts.csv',index=False);pd.concat(trows).to_csv(O/'test_target_context_scores.csv',index=False);print('TESTED',d,seed,flush=True)
assert sha(O/'PROTOCOL.json')==ph
(O/'COMPLETE.json').write_text(json.dumps(dict(status='PASS',dataset_splits=15,candidate_count=len(CONFIGS),validation_rows=len(V),sealed_prediction_files=len(RECEIPTS),test_rows=len(tr),baseline_parity=PARITY,protocol_sha256=ph,formal_benchmark_replaced=False),indent=2));print('COMPLETE',flush=True)
