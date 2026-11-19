import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
N=Path('@STUDY@/kipert_prior_exploration_20261117');O=N/'reports';OLD=Path('@STUDY@/kipert_supplement_20261117');NB=2000
DEFS=[('cosine_PDS','pds_query_cosine',False),('DEG_AUPRC','auprc_dg',False),('MSE','mse',True),('shared_Pearson','systema_cell_pearson',False)]
def effect(f,base,cand,metric,group,lower):
 key=[group,'target'];a=f[f.variant==base].set_index(key)[metric];b=f[f.variant==cand].set_index(key)[metric];assert a.index.is_unique and b.index.is_unique and set(a.index)==set(b.index);b=b.reindex(a.index);ok=np.isfinite(a)&np.isfinite(b);delta=(a[ok]-b[ok])*(-1 if lower else 1)
 if not len(delta):return dict(effect=np.nan,ci_low=np.nan,ci_high=np.nan,n_clusters=0,n_pairs=0,groups=0)
 x=delta.unstack(group).sort_index();ids=x.index;mask=np.isfinite(x.to_numpy());x=np.nan_to_num(x.to_numpy(),nan=0);den=mask.astype(float)
 def mean(v,d):return np.nanmean(np.divide(v,d,out=np.full_like(v,np.nan),where=d>0),axis=-1)
 point=float(mean(x.sum(0),den.sum(0)));rng=np.random.default_rng(2026111702+int(hashlib.sha256((base+cand+metric+','.join(map(str,delta.index.get_level_values(group).unique()))).encode()).hexdigest()[:7],16));values=[]
 for start in range(0,NB,100):
  ii=rng.integers(len(ids),size=(min(100,NB-start),len(ids)));values.extend(mean(x[ii].sum(1),den[ii].sum(1)))
 lo,hi=np.quantile(values,[.025,.975]);return dict(effect=point,ci_low=float(lo),ci_high=float(hi),n_clusters=len(ids),n_pairs=int(mask.sum()),groups=mask.shape[1])
def contrasts(frame,base,cands,strata,group,scope):
 out=[];outer=[((),frame)] if not strata else frame.groupby(strata,sort=True,observed=True)
 for keys,f in outer:
  keys=keys if isinstance(keys,tuple) else (keys,)
  for cand in cands:
   if cand==base or cand not in set(f.variant) or base not in set(f.variant):continue
   for label,m,lower in DEFS:out.append(dict(scope=scope,baseline=base,candidate=cand,metric=label,positive_means='baseline_better',**dict(zip(strata,keys)),**effect(f,base,cand,m,group,lower)))
 return pd.DataFrame(out)
def main():
 assert (N/'EXPERIMENTS_COMPLETE.json').exists();cfg=json.loads((N/'PROTOCOL.json').read_text())['configurations'];catalog=[]
 for c in cfg:
  for mode in ['retuned','fixed']:catalog.append(dict(variant=c['name']+'_'+mode,base_variant=c['name'],mode=mode,family=c['family'],size=len(c['active']),replicate=c.get('replicate',np.nan),source='new'))
 files=sorted((N/'runs').glob('S*/split*/*_metrics.csv'));assert len(files)==4320,len(files);f=pd.concat([pd.read_csv(p,dtype={'target':str,'context':str}) for p in files],ignore_index=True);f.to_csv(O/'condition_metrics.csv.gz',index=False);mets=[c for c in f.select_dtypes(include='number') if c!='split'];t=f.groupby(['dataset','variant','target','split'])[mets].mean().groupby(['dataset','variant','target']).mean().reset_index();t.to_csv(O/'new_target_metrics.csv.gz',index=False)
 old=pd.read_csv(OLD/'reports/single_target_metrics.csv.gz',dtype={'target':str});old=old[old.variant.str.startswith(('only_','drop_','three_prior_'))]
 for v in sorted(set(old.variant)):
  name,mode=v.rsplit('_',1);family='single_view' if name.startswith('only_') else 'leave_one_out' if name.startswith('drop_') else 'three_prior_reference';catalog.append(dict(variant=v,base_variant=name,mode=mode,family=family,size=1 if family=='single_view' else 11 if family=='leave_one_out' else 3,replicate=np.nan,source='previous'))
 catalog=pd.DataFrame(catalog);catalog.to_csv(O/'model_catalog.csv',index=False);t=pd.concat([t,old[['dataset','variant','target']+mets]],ignore_index=True);assert not t.duplicated(['dataset','variant','target']).any();t.to_csv(O/'all_target_metrics.csv.gz',index=False);s=t.groupby(['dataset','variant'])[mets].mean().reset_index().merge(catalog,on='variant',validate='many_to_one');s.to_csv(O/'dataset_summary.csv',index=False);macro=s.groupby('variant')[mets].mean().reset_index().merge(catalog,on='variant',validate='one_to_one');macro.to_csv(O/'macro_summary.csv',index=False)
 # Average scores, not predictions, over the predefined subset or permutation panel.
 collapsed=[];curve=[]
 for mode in ['retuned','fixed']:
  ct=catalog[catalog['mode']==mode];keep=ct[ct.family.isin(['full','centroid','representation_group','three_prior_reference'])].variant;collapsed.append(t[t.variant.isin(keep)])
  for family in ['joint_shuffle','independent_shuffle']:
   names=ct[ct.family==family].variant;a=t[t.variant.isin(names)].groupby(['dataset','target'])[mets].mean().reset_index();a['variant']=family+'_mean_'+mode;collapsed.append(a)
  for size in [0,1,2,4,6,8,10,11,12]:
   names=ct[(ct['size']==size)&ct.family.isin(['centroid','single_view','subset','leave_one_out','full'])].variant
   a=t[t.variant.isin(names)].groupby(['dataset','target'])[mets].mean().reset_index();a['variant']=f'quantity_m{size:02d}_'+mode;collapsed.append(a)
   for scope,z in [('macro',macro[macro.variant.isin(names)]),*[(d,s[(s.dataset==d)&s.variant.isin(names)]) for d in ['S3','S4','S5','S6','S7']]]:
    for label,m,_ in DEFS:curve.append(dict(scope=scope,mode=mode,size=size,metric=label,n_models=len(z),mean=z[m].mean(),median=z[m].median(),q25=z[m].quantile(.25),q75=z[m].quantile(.75),minimum=z[m].min(),maximum=z[m].max()))
 collapsed=pd.concat(collapsed,ignore_index=True);collapsed['all']='all';collapsed.to_csv(O/'comparison_target_metrics.csv.gz',index=False);pd.DataFrame(curve).to_csv(O/'quantity_curves.csv',index=False);ef=[]
 for mode in ['retuned','fixed']:
  sub=collapsed[collapsed.variant.str.endswith('_'+mode)];base='full_'+mode;cands=sorted(set(sub.variant)-{base,'quantity_m12_'+mode,'quantity_m00_'+mode})
  for strata,g,scope in [(['dataset'],'all','dataset'),([],'dataset','macro')]:ef.append(contrasts(sub,base,cands,strata,g,scope))
  for base,cand in [('joint_shuffle_mean_'+mode,'independent_shuffle_mean_'+mode),('dense4_'+mode,'quantity_m04_'+mode),('annotation8_'+mode,'quantity_m08_'+mode)]:
   for strata,g,scope in [(['dataset'],'all','dataset'),([],'dataset','macro')]:ef.append(contrasts(sub,base,[cand],strata,g,scope))
 pd.concat(ef).to_csv(O/'paired_effects.csv',index=False);print('CORE EXPLORATION SUMMARY COMPLETE',flush=True)
 diagnostics=pd.concat([pd.read_csv(p,dtype={'target':str,'context':str}) for p in (N/'runs').glob('S*/split*/query_prior_diagnostics.csv')],ignore_index=True);valid=f[f.variant=='full_retuned'][['dataset','split','context','target']];diagnostics=diagnostics.merge(valid,on=['dataset','split','context','target'],validate='one_to_one');diag=diagnostics.groupby(['dataset','target']).agg(covered_views=('covered_views','min'),nearest_train_similarity=('nearest_train_similarity','mean'),fit_targets=('fit_targets','mean')).reset_index();diag['coverage_bin']=pd.cut(diag.covered_views,[-1,6,9,12],labels=['0-6','7-9','10-12']).astype(str);diag['proximity_bin']=diag.groupby('dataset').nearest_train_similarity.transform(lambda x:pd.qcut(x,3,labels=['low','middle','high']).astype(str));diag.to_csv(O/'target_prior_diagnostics.csv',index=False)
 strat=collapsed.merge(diag,on=['dataset','target'],validate='many_to_one');strat.to_csv(O/'stratified_target_metrics.csv.gz',index=False);res=[];counts=[]
 for dimension,bins in [('coverage_bin',['0-6','7-9','10-12']),('proximity_bin',['low','middle','high'])]:
  for d in ['S3','S4','S5','S6','S7']:
   for b in bins:counts.append(dict(dimension=dimension,dataset=d,bin=b,targets=int(((diag.dataset==d)&(diag[dimension]==b)).sum())))
  for mode in ['retuned','fixed']:
   sub=strat[strat.variant.str.endswith('_'+mode)];cands=[v+'_'+mode for v in ['centroid','joint_shuffle_mean','independent_shuffle_mean','dense4','annotation8']];rr=contrasts(sub,'full_'+mode,cands,['dataset',dimension],'all','stratified_dataset');res.append(rr.rename(columns={dimension:'bin'}).assign(dimension=dimension))
   if dimension=='proximity_bin':rr=contrasts(sub,'full_'+mode,cands,[dimension],'dataset','stratified_macro');res.append(rr.rename(columns={dimension:'bin'}).assign(dimension=dimension))
 pd.concat(res).to_csv(O/'stratified_effects.csv',index=False);pd.DataFrame(counts).to_csv(O/'stratum_counts.csv',index=False)
 permutations=[];selection=[]
 for p in (N/'runs').glob('S*/split*/permutation_diagnostics.csv'):
  permutations.append(pd.read_csv(p).assign(dataset=p.parents[1].name,split=int(p.parent.name[5:])));z=json.loads((p.parent/'SELECTIONS_FROZEN.json').read_text());selection.extend(dict(dataset=p.parents[1].name,split=int(p.parent.name[5:]),variant=k,selected_alpha=v,fixed_alpha=z['fixed_alpha']) for k,v in z['selected'].items())
 pd.concat(permutations).to_csv(O/'permutation_diagnostics.csv',index=False);pd.DataFrame(selection).to_csv(O/'parameter_selection.csv',index=False)
 (O/'STATISTICS_PROTOCOL.json').write_text(json.dumps(dict(bootstrap=NB,seed=2026111702,unit='Paired target clusters; repeats retained; fixed datasets equally weighted',interval='Pointwise percentile conditional on fitted models, fixed dataset panel and sampled model compositions; no biological-replicate or retraining uncertainty',quantity='Arithmetic mean model scores per target; variability across compositions reported as IQR and range, not a biological CI',null='Mean score of ten permutations per target; replicate scores retained; not an exhaustive randomization test',coverage='Minimum covered-view count over fit/refit-derived test prediction contexts and splits for the same target; bins 0-6,7-9,10-12',proximity='Mean nearest-training similarity across contexts/splits, then tertiles within dataset using priors only',coverage_macro='Not aggregated across datasets because low-coverage bins may be absent in some datasets',p_values='No confirmatory p values or significance stars; all contrasts exploratory'),indent=2))
 print('ALL EXPLORATION SUMMARIES COMPLETE',flush=True)
if __name__=='__main__':main()
