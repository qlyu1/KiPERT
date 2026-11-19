"""Build response-isolated matched seen0/1/2 panels from frozen count-normalized data."""
from pathlib import Path
import json, hashlib, shutil, os, sys
import numpy as np
import pandas as pd
from scipy import sparse

OLD=Path('@STUDY@/uc_fig3_benchmark_20260912')
R=Path('@STUDY@/uc_seen012_benchmark_20261005')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def link(src,dst):
 if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src)

for name in ['envs','repos']:link(OLD/name,R/name)
for p in (OLD/'assets').iterdir():link(p,R/'assets'/p.name)
for d in ['data/formal','data/shared','runs/formal','logs/jobs','reports','manifests','protocol']:(R/d).mkdir(parents=True,exist_ok=True)
protocol=dict(id='SEEN012_V1',created='2026-10-05',datasets=['D1','D2','D3','D4'],arms=['seen0','seen1','seen2'],
 seen1='Only L-side singleton targets supplied; exactly one fit per split/seed. No A/B two-direction runs.',
 split_seeds=[101,211,307,401,503],model_seeds=[17,29,43],models=['uc','gears','scouter','txpert','stamp','gperturb'],
 tuning='seen0 seed17 only; same selected learning rate and update count for every other arm/seed; no validation refit',
 training='cell methods: condition-balanced replacement sampling, fixed arity/control mass and draws per virtual epoch; all legal cells eligible',
 max_cell_steps_per_virtual_epoch=128,max_pseudobulk_steps_per_virtual_epoch=16,
 metrics=['macro_MSE','nMSE_vs_no_change','PCC_delta','specific_MRR','paired_gain01','paired_gain12'],
 scoring='all registered predictions sealed before final scoring; no test-driven hyperparameters',
 checkpoint_retention='all seed17 final checkpoints; other seeds retain predictions, hashes, configs and logs; intermediate candidates deleted only after sealing',
 scope='retrospective public-data benchmark; historical results have been inspected; no old response-trained checkpoints reused')
dump(R/'protocol/PROTOCOL.json',protocol)
dump(R/'manifests/FORMAL_RUN_GATE.json',dict(ready=True,models=protocol['models']+['baselines'],datasets=protocol['datasets'],protocol='SEEN012_V1',authorization='Server run requested on 2026-10-05'))
summary=[]
for dataset in protocol['datasets']:
 src=OLD/'data/formal'/dataset/'P1/fold0'
 ms=[];xs=[]
 for role in ['fit','validation','evaluation_only']:
  ms.append(pd.read_csv(src/role/'metadata.tsv',sep='\t',dtype=str));xs.append(sparse.load_npz(src/role/'expression.npz'))
 m=pd.concat(ms,ignore_index=True);x=sparse.vstack(xs,format='csr');assert m.cell_id.is_unique
 if dataset=='D3':keep=m.context.eq('lane=L4').to_numpy();m=m.loc[keep].reset_index(drop=True);x=x[keep]
 contexts=sorted(m.context.unique())
 groups=m.groupby(['target','context'],sort=True).indices
 singles={c:{t for t,cc in groups if cc==c and '+' not in t} for c in contexts}
 pairs=sorted({t for t,c in groups if '+' in t})
 matched={t:{c for c in contexts if (t,c) in groups and set(t.split('+'))<=singles[c]} for t in pairs}
 matched={t:cs for t,cs in matched.items() if cs}
 common=set.intersection(*(singles[c] for c in contexts));assert len(common)>=4
 deg={}
 for pp in sorted((OLD/'data/formal'/dataset/'P1').glob('fold*/fit/deg_indices.json')):deg.update(json.loads(pp.read_text()))
 assert all(t+'|'+c in deg for t,c in groups)
 shared=R/'data/shared'/dataset;shared.mkdir(parents=True,exist_ok=True)
 for file in ['genes.tsv','genes_all.tsv','target_prior_mapping.tsv']:shutil.copyfile(src/file,shared/file)
 for role in ['controls_fit','controls_val','controls_score']:
  if dataset!='D3':link(src/role,shared/role)
  else:
   dest=shared/role;dest.mkdir(exist_ok=True);mm=pd.read_csv(src/role/'metadata.tsv',sep='\t',dtype=str);ii=mm.context.eq('lane=L4').to_numpy()
   mm.loc[ii].to_csv(dest/'metadata.tsv',sep='\t',index=False)
   sparse.save_npz(dest/'expression.npz',sparse.load_npz(src/role/'expression.npz')[ii],compressed=False)
 nc=max(5,int(np.ceil(.2*len(matched))));nv=max(3,int(np.ceil(.1*len(matched))))
 degree={g:sum(g in t.split('+') for t in matched) for g in common}
 for fold,split_seed in enumerate(protocol['split_seeds']):
  rng=np.random.RandomState(split_seed);solution=None
  for attempt in range(10000):
   # Metadata-only preference for anchors that remove fewer potential query edges.
   ranked=sorted(common,key=lambda g:(degree[g]+rng.uniform(0,3),g));anchors=set(ranked[:2])
   endpoint=sorted({g for t in matched for g in t.split('+')}-anchors);rng.shuffle(endpoint)
   left=set(endpoint[:len(endpoint)//2]);right=set(endpoint[len(endpoint)//2:])
   cross=[t for t in matched if len(set(t.split('+'))&left)==1 and len(set(t.split('+'))&right)==1]
   if len(cross)<nc+nv:continue
   rng.shuffle(cross);test=sorted(cross[:nc]);val=sorted(cross[nc:nc+nv]);fit=sorted(set(pairs)-set(test)-set(val))
   if not all(sum(c in matched[t] for t in test)>=5 and sum(c in matched[t] for t in val)>=3 and sum((t,c) in groups for t in fit)>=10 for c in contexts):continue
   endpoint={g for t in test+val for g in t.split('+')};left&=endpoint;right&=endpoint
   solution=(anchors,left,right,test,val,fit,attempt);break
  if solution is None:raise RuntimeError(f'No compliant metadata-only graph partition {dataset} fold {fold}')
  anchors,left,right,test,val,fit,attempt=solution
  fold_shared=shared/f'fold{fold}';fold_shared.mkdir(exist_ok=True)
  query=pd.DataFrame([dict(target=t,context=c,condition_id=t+'|'+c) for t in test for c in sorted(matched[t])])
  valkeys={(t,c) for t in val for c in matched[t]};testkeys={(t,c) for t in test for c in matched[t]}
  for role,keys in [('validation',valkeys),('evaluation_only',testkeys)]:
   ix=np.concatenate([groups[k] for k in sorted(keys)]);dest=fold_shared/role;dest.mkdir(exist_ok=True)
   m.iloc[ix].to_csv(dest/'metadata.tsv',sep='\t',index=False);sparse.save_npz(dest/'expression.npz',x[ix],compressed=False)
  (fold_shared/'query').mkdir(exist_ok=True);query.to_csv(fold_shared/'query/metadata.tsv',sep='\t',index=False)
  fixed_double_ids=sorted(m.loc[m.target.isin(fit),'cell_id']);double_hash=hashlib.sha256('\n'.join(fixed_double_ids).encode()).hexdigest()
  common_note=dict(dataset=dataset,fold=fold,split_seed=split_seed,anchors=sorted(anchors),left=sorted(left),right=sorted(right),train_pairs=fit,validation_pairs=val,test_pairs=test,
   training_double_cell_sha256=double_hash,partition_attempt=attempt,matched_pairs_available=len(matched),contexts=contexts,
   double_exposure={g:sum(g in t.split('+') for t in fit) for g in sorted(left|right)},seen1_policy='retain L only')
  dump(fold_shared/'partition.json',common_note)
  for arm in protocol['arms']:
   base=R/'data/formal'/dataset/arm/f'fold{fold}';base.mkdir(parents=True,exist_ok=True)
   supplied=set() if arm=='seen0' else left if arm=='seen1' else left|right
   allowed_singles={g for ss in singles.values() for g in ss if g not in left|right}|supplied
   keep=m.target.isin(set(fit)|allowed_singles).to_numpy();fitm=m.loc[keep].reset_index(drop=True);fitx=x[keep]
   assert sorted(fitm.loc[fitm.target.isin(fit),'cell_id'])==fixed_double_ids
   for row in query.itertuples():assert sum(g in set(fitm.loc[fitm.context.eq(row.context),'target']) for g in row.target.split('+'))==int(arm[-1])
   assert not set(fitm.target)&set(test+val)
   for c in contexts:assert len(set(fitm.loc[fitm.context.eq(c),'target'])&singles[c])>=2
   dest=base/'fit';dest.mkdir(exist_ok=True);fitm.to_csv(dest/'metadata.tsv',sep='\t',index=False);sparse.save_npz(dest/'expression.npz',fitx,compressed=False)
   dump(dest/'deg_indices.json',{t+'|'+c:deg[t+'|'+c] for t,c in fitm.groupby(['target','context']).indices})
   for role in ['validation','evaluation_only','query']:link(fold_shared/role,base/role)
   for role in ['controls_fit','controls_val','controls_score','genes.tsv','genes_all.tsv','target_prior_mapping.tsv']:link(shared/role,base/role)
   if arm=='seen0':base_cells=len(fitm);base_conditions=len(fitm.groupby(['target','context']))
   dump(base/'split.json',{**common_note,'arm':arm,'test_genes':[],'validation_genes':[],'hidden_single_targets':sorted((left|right)-supplied),'supplied_single_targets':sorted(allowed_singles),'base_cells':base_cells,'base_conditions':base_conditions})
   ready=dict(protocol='SEEN012_V1',dataset=dataset,arm=arm,fold=fold,fit_response_sha256=sha(dest/'expression.npz'),axis_sha256=sha(base/'genes.tsv'),split_sha256=sha(base/'split.json'),query_sha256=sha(base/'query/metadata.tsv'),training_double_cell_sha256=double_hash,training_cells=len(fitm),test_queries=len(query),test_pairs=len(test),validation_pairs=len(val),training_pairs=len(fit))
   dump(base/'READY.json',ready);summary.append(ready)
  print(json.dumps(dict(dataset=dataset,fold=fold,test_pairs=len(test),validation_pairs=len(val),training_pairs=len(fit),query_contexts=len(query),anchors=sorted(anchors)),ensure_ascii=False),flush=True)
 del x,xs,ms,m
dump(R/'manifests/DATA_READY.json',dict(status='READY',protocol_sha256=sha(R/'protocol/PROTOCOL.json'),panels=summary))
print('ALL_DATA_READY',flush=True)
