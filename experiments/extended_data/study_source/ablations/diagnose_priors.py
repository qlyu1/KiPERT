import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json
import numpy as np,pandas as pd
R=Path('@STUDY@/kipert_supplement_20261117');O=R/'reports'
V=['DepMap','GenePT','STRING','Reactome','GO_BP','CORUM','CollecTRI','HPA','GO_MF','GO_CC','HPO','WikiPathways']
weights=[];sim=[];alpha=[]
for p in sorted((R/'single/ablation').glob('S*/split*/prior_weights.csv')):
 d=p.parents[1].name;s=int(p.parent.name[5:]);w=pd.read_csv(p);weights.append(w.assign(dataset=d,split=s))
 for f in p.parent.glob('kernel_similarity_fit_*.npy'):
  a=np.load(f)
  for i,v in enumerate(V):
   for j,u in enumerate(V):sim.append(dict(dataset=d,split=s,context_index=int(f.stem.rsplit('_',1)[1]),view1=v,view2=u,kernel_cosine=a[i,j]))
 z=json.loads((p.parent/'SELECTIONS_FROZEN.json').read_text())
 alpha.extend(dict(task='single',dataset=d,split=s,variant=v,retuned=a,fixed=z['fixed_alpha']) for v,a in z['selected'].items())
w=pd.concat(weights,ignore_index=True);w.to_csv(O/'single_weights_and_coverage.csv.gz',index=False)
cov=w[w.variant.str.startswith('only_')].groupby(['dataset','view'])[['queries','uncovered_queries']].sum().reset_index();cov['query_coverage']=1-cov.uncovered_queries/cov.queries;cov.to_csv(O/'single_query_coverage.csv',index=False)
f=w[w.variant=='full_beta0p5'];f.groupby(['dataset','view'])[['weight','alignment']].agg(['mean','min','max']).to_csv(O/'single_full_weights_summary.csv')
eff=f.groupby(['dataset','split','context']).weight.apply(lambda a:1/np.sum(a*a)).reset_index(name='effective_views');eff.to_csv(O/'single_effective_views.csv',index=False)
pd.DataFrame(sim).to_csv(O/'single_fit_kernel_similarity.csv.gz',index=False)
for p in (R/'double/ablation').glob('D*/fold*/SELECTIONS_FROZEN.json'):
 z=json.loads(p.read_text());alpha.extend(dict(task='double',dataset=p.parents[1].name,split=int(p.parent.name[4:]),variant=v,retuned=a,fixed=z['fixed_alpha']) for v,a in z['selected'].items())
pd.DataFrame(alpha).to_csv(O/'ridge_selection_summary.csv',index=False)
temps=[];drifts=[]
for c in ['k562','hepg2','rpe1','jurkat']:
 p=R/'transfer'/c;z=json.loads((p/'IDENTITY_TEMPERATURE_SEALED.json').read_text());temps.append(dict(context=c,ot_temperature=z['original_ot_temperature'],identity_temperature=z['selected']['temperature'],source_validation_nmse=z['selected']['weighted']['nmse'],source_validation_rank=z['selected']['weighted']['rank_score']))
 ref=np.load(p/'axes.npz')['weights']
 for v in V:
  w=np.load(p/('drop_'+v+'_weights.npz'))['weights'];a=np.abs(w-ref)
  drifts.append(dict(context=c,removed_view=v,mean_total_variation=float(a.sum(0).mean()/2),maximum_absolute_weight_change=float(a.max())))
pd.DataFrame(temps).to_csv(O/'identity_temperature_selection.csv',index=False);pd.DataFrame(drifts).to_csv(O/'transfer_prior_weight_changes.csv',index=False)
print('PRIOR DIAGNOSTICS COMPLETE')
