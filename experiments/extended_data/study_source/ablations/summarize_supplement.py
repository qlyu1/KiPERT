import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd
R=Path('@STUDY@/kipert_supplement_20261117');OUT=R/'reports';OUT.mkdir(exist_ok=True)
NB=2000
def interval(frame,metric,base,candidate,group,idcol,lower=False,den=None):
 key=[group,idcol];cols=key+[metric]+([den] if den else [])
 a=frame[frame.variant==base][cols].set_index(key);b=frame[frame.variant==candidate][cols].set_index(key);assert a.index.is_unique and b.index.is_unique;assert set(a.index)==set(b.index)
 b=b.reindex(a.index);valid=np.isfinite(a[metric])&np.isfinite(b[metric]);a=a[valid];b=b[valid];groups=sorted(set(a.index.get_level_values(group)));ids=sorted(set(a.index.get_level_values(idcol)));A=a[metric].unstack(group).reindex(index=ids,columns=groups).to_numpy();B=b[metric].unstack(group).reindex(index=ids,columns=groups).to_numpy();D=a[den].unstack(group).reindex(index=ids,columns=groups).to_numpy() if den else np.where(np.isfinite(A),1.,np.nan)
 if not len(ids):return dict(effect=np.nan,ci_low=np.nan,ci_high=np.nan,n_clusters=0,n_pairs=0)
 assert np.array_equal(np.isnan(A),np.isnan(B));mask=np.isfinite(A)&np.isfinite(B)&np.isfinite(D);A=np.where(mask,A,0);B=np.where(mask,B,0);D=np.where(mask,D,0)
 def calc(a,b,d):
  v=np.divide(a-b,d,out=np.full_like(d,np.nan,dtype=float),where=d>0);return np.nanmean(v,axis=-1)*(-1 if lower else 1)
 point=float(calc(A.sum(0),B.sum(0),D.sum(0)));seed=20261117+int(hashlib.sha256((str(metric)+base+candidate+','.join(map(str,groups))).encode()).hexdigest()[:7],16);rng=np.random.default_rng(seed);values=[]
 for start in range(0,NB,100):
  ii=rng.integers(len(ids),size=(min(100,NB-start),len(ids)));values.extend(calc(A[ii].sum(1),B[ii].sum(1),D[ii].sum(1)))
 low,high=np.quantile(values,[.025,.975]);return dict(effect=point,ci_low=float(low),ci_high=float(high),n_clusters=len(ids),n_pairs=int(mask.sum()),groups=len(groups))
def paired_table(frame,base,candidates,metric_defs,strata,group,idcol,scope):
 results=[]
 outer=[((),frame)] if not strata else list(frame.groupby(strata,sort=True))
 for keys,f in outer:
  keys=keys if isinstance(keys,tuple) else (keys,)
  for cand in candidates:
   if cand==base or cand not in set(f.variant):continue
   for name,metric,lower,den in metric_defs:
    r=interval(f,metric,base,cand,group,idcol,lower,den);results.append(dict(scope=scope,baseline=base,candidate=cand,metric=name,positive_means='baseline_better',**dict(zip(strata,keys)),**r))
 return pd.DataFrame(results)
def single():
 files=list((R/'single/ablation').glob('S*/split*/*_metrics.csv'));assert len(files)==15*30*2,len(files)
 f=pd.concat([pd.read_csv(p,dtype={'target':str,'context':str}) for p in sorted(files)],ignore_index=True);f.to_csv(OUT/'single_condition_metrics.csv.gz',index=False)
 mets=[c for c in f.select_dtypes(include='number') if c!='split'];t=f.groupby(['dataset','variant','target','split'])[mets].mean().groupby(['dataset','variant','target']).mean().reset_index();t['all']='all';t.to_csv(OUT/'single_target_metrics.csv.gz',index=False);summary=t.groupby(['dataset','variant'])[mets].mean().reset_index();summary.to_csv(OUT/'single_summary.csv',index=False);summary.groupby('variant')[mets].mean().to_csv(OUT/'single_macro.csv')
 defs=[('cosine_PDS','pds_query_cosine',False,None),('mean_PDS','mean_query_pds',False,None),('MSE','mse',True,None),('DEG_AUPRC','auprc_dg',False,None),('shared_Pearson_target_mean','systema_cell_pearson',False,None)]
 eff=[]
 for mode in ['retuned','fixed']:
  sub=t[t.variant.str.endswith('_'+mode)];base='full_beta0p5_'+mode;candidates=sorted(set(sub.variant));eff.append(paired_table(sub,base,candidates,defs,['dataset'],'all','target','single_dataset'));eff.append(paired_table(sub,base,candidates,defs,[],'dataset','target','single_macro'))
 pd.concat(eff).to_csv(OUT/'single_paired_effects.csv',index=False)
 # Priority re-scoring kept separate, including exact frozen-prediction comparison.
 f=pd.read_csv(R/'single/rescore/condition_metrics.csv.gz',dtype={'target':str,'context':str});t=f.groupby(['dataset','variant','target','split'])[mets].mean().groupby(['dataset','variant','target']).mean().reset_index();t['all']='all';pd.concat([paired_table(t,'aligned_raw',['uniform_raw'],defs,['dataset'],'all','target','single_rescore'),paired_table(t,'aligned_raw',['uniform_raw'],defs,[],'dataset','target','single_rescore_macro')]).to_csv(OUT/'single_rescore_paired_effects.csv',index=False)
 sels=[];weights=[]
 for p in (R/'single/ablation').glob('S*/split*/SELECTIONS_FROZEN.json'):
  z=json.loads(p.read_text());sels.extend(dict(dataset=p.parents[1].name,split=int(p.parent.name[5:]),variant=v,alpha=a,fixed_alpha=z['fixed_alpha']) for v,a in z['selected'].items());w=pd.read_csv(p.parent/'prior_weights.csv');weights.append(w.assign(dataset=p.parents[1].name,split=int(p.parent.name[5:])))
 pd.DataFrame(sels).to_csv(OUT/'single_selections.csv',index=False);pd.concat(weights).to_csv(OUT/'single_weights_and_coverage.csv.gz',index=False);print('SINGLE SUMMARY COMPLETE',flush=True)
 original_split_shared()
def original_split_shared():
 files=list((R/'single/ablation').glob('S*/split*/*_metrics.csv'));frames=[pd.read_csv(R/'single/rescore/condition_metrics.csv.gz',dtype={'target':str,'context':str})]
 if files:frames.extend(pd.read_csv(p,dtype={'target':str,'context':str}) for p in sorted(files))
 f=pd.concat(frames,ignore_index=True);s=f.groupby(['dataset','variant','split','target']).systema_cell_pearson.mean().reset_index();s['split_group']=s.split.astype(str);s['dataset_split']=s.dataset+'_'+s.split_group;summary=s.groupby(['dataset','variant','split']).systema_cell_pearson.mean().groupby(['dataset','variant']).mean().reset_index();summary.to_csv(OUT/'single_original_split_shared_Pearson_summary.csv',index=False);defs=[('shared_Pearson_original_split_mean','systema_cell_pearson',False,None)];effects=[]
 for base,candidates in [('aligned_raw',['uniform_raw']),('full_beta0p5_retuned',sorted(v for v in set(s.variant) if v.endswith('_retuned'))),('full_beta0p5_fixed',sorted(v for v in set(s.variant) if v.endswith('_fixed')))]:
  if base not in set(s.variant):continue
  sub=s[s.variant.isin([base]+candidates)];effects.extend([paired_table(sub,base,candidates,defs,['dataset'],'split_group','target','single_original_split_dataset'),paired_table(sub,base,candidates,defs,[],'dataset_split','target','single_original_split_macro')])
 pd.concat(effects).to_csv(OUT/'single_original_split_shared_Pearson_effects.csv',index=False)
def double():
 files=list((R/'double/ablation').glob('D*/fold*/seen*/*_metrics.csv'));assert len(files)==20*3*17*2,len(files);f=pd.concat([pd.read_csv(p,dtype={'target':str,'context':str}) for p in sorted(files)],ignore_index=True);f.to_csv(OUT/'double_condition_metrics.csv.gz',index=False);mets=[c for c in f.select_dtypes(include='number') if c!='fold'];t=f.groupby(['dataset','arm','variant','fold','target'])[mets].mean().reset_index();t['fold_group']=t.fold.astype(str);t['dataset_fold']=t.dataset+'_'+t.fold.astype(str);t.to_csv(OUT/'double_pair_metrics.csv.gz',index=False);fold=t.groupby(['dataset','arm','variant','fold'])[mets].mean().reset_index();fold['nmse20']=fold.mse20/fold.nochange20;fold.to_csv(OUT/'double_fold_summary.csv',index=False);summary=fold.groupby(['dataset','arm','variant'])[[*mets,'nmse20']].mean().reset_index();summary.to_csv(OUT/'double_summary.csv',index=False)
 defs=[('opposite20','opposite20',True,None),('nMSE20','mse20',True,'nochange20'),('MSE','mse',True,None),('Pearson_delta','pcc_delta',False,None),('DEG_AUPRC','deg_auprc_dg',False,None)];eff=[]
 for mode in ['retuned','fixed']:
  sub=t[t.variant.str.endswith('_'+mode)];base='full_beta0p5_'+mode;candidates=sorted(set(sub.variant));eff.append(paired_table(sub,base,candidates,defs,['dataset','arm'],'fold_group','target','double_dataset'));eff.append(paired_table(sub,base,candidates,defs,['arm'],'dataset_fold','target','double_macro'))
 pd.concat(eff).to_csv(OUT/'double_paired_effects.csv',index=False);print('DOUBLE SUMMARY COMPLETE',flush=True)
def transfer():
 frames=[];qa=[]
 gold=pd.read_csv('@STUDY@/uc_cross_context_transfer/experiments/pie_uc_state_benchmark_20261106/benchmark_final_20261107/state_per_condition.csv.gz')
 for c in ['k562','hepg2','rpe1','jurkat']:
  root=R/'transfer'/c
  for p in sorted(root.glob('*_native.npy')):
   name=p.name[:-11]
   for mode in ['native','adapted']:
    f=pd.read_csv(root/(name+'_'+mode+'_metrics.csv'));assert len(f)==dict(k562=884,hepg2=867,rpe1=983,jurkat=978)[c]
    if mode=='adapted':
     de=pd.read_csv(root/(name+'_de_metrics.csv'));f=f.merge(de,on=['context','perturbation','variant'],validate='one_to_one')
     if name in ['weighted_OT','uniform_identity']:
      model='UC' if name=='weighted_OT' else 'PMean';a=f.set_index('perturbation');b=gold[(gold.model==model)&(gold.context==c)].set_index('perturbation').reindex(a.index);b['direction_true_de']=b.direction_true_de.fillna(0);errors={k:float(np.nanmax(abs(a[k]-b[k]))) for k in ['mse_delta','pds_l1_exclude_target','pearson_delta','direction_true_de','lfc_spearman_true_de','de_overlap_N','auprc']};assert max(errors.values())<1e-6,(c,model,errors);qa.append(dict(context=c,model=model,errors=errors))
    frames.append(f)
 f=pd.concat(frames,ignore_index=True);f.to_csv(OUT/'transfer_condition_metrics.csv.gz',index=False);mets=list(f.select_dtypes(include='number'));summary=f.groupby(['context','mode','variant'])[mets].mean().reset_index();summary.to_csv(OUT/'transfer_context_summary.csv',index=False);summary.groupby(['mode','variant'])[mets].mean().to_csv(OUT/'transfer_macro.csv');eff=[]
 defs=[('L1_PDS','pds_l1_exclude_target',False,None),('MSE','mse_delta',True,None),('Pearson_delta','pearson_delta',False,None),('direction_DE','direction_true_de',False,None),('LFC_Spearman','lfc_spearman_true_de',False,None),('DE_overlap','de_overlap_N',False,None),('DE_AUPRC','auprc',False,None)]
 f['all']='all'
 for mode in ['native','adapted']:
  sub=f[f['mode']==mode];dd=defs if mode=='adapted' else defs[:3];candidates=sorted(set(sub.variant));base='weighted_OT'
  for q in [paired_table(sub,base,candidates,dd,['context'],'all','perturbation','transfer_context'),paired_table(sub,base,candidates,dd,[],'context','perturbation','transfer_macro')]:eff.append(q.assign(mode=mode))
  # Factorial conditional effects additionally isolate source weighting with and without OT.
  for base,cand in [('uniform_OT','uniform_identity'),('weighted_identity','uniform_identity')]:
   for q in [paired_table(sub,base,[cand],dd,['context'],'all','perturbation','transfer_factorial_context'),paired_table(sub,base,[cand],dd,[],'context','perturbation','transfer_factorial_macro')]:eff.append(q.assign(mode=mode))
 pd.concat(eff).to_csv(OUT/'transfer_paired_effects.csv',index=False);(OUT/'transfer_score_parity.json').write_text(json.dumps(qa,indent=2));print('TRANSFER SUMMARY COMPLETE',flush=True)
def priority():
 assert (R/'transfer/PRIORITY_COMPLETE.json').exists()
 names=['weighted_OT','weighted_identity','uniform_OT','uniform_identity'];frames=[]
 for c in ['k562','hepg2','rpe1','jurkat']:
  for name in names:
   for mode in ['native','adapted']:
    f=pd.read_csv(R/'transfer'/c/(name+'_'+mode+'_metrics.csv'))
    if mode=='adapted':f=f.merge(pd.read_csv(R/'transfer'/c/(name+'_de_metrics.csv')),on=['context','perturbation','variant'],validate='one_to_one')
    frames.append(f)
 f=pd.concat(frames,ignore_index=True);f.to_csv(OUT/'priority_transfer_conditions.csv.gz',index=False);mets=list(f.select_dtypes(include='number'));s=f.groupby(['context','mode','variant'])[mets].mean().reset_index();s.to_csv(OUT/'priority_transfer_summary.csv',index=False);s.groupby(['mode','variant'])[mets].mean().to_csv(OUT/'priority_transfer_macro.csv');f['all']='all';eff=[]
 defs=[('L1_PDS','pds_l1_exclude_target',False,None),('MSE','mse_delta',True,None),('Pearson_delta','pearson_delta',False,None),('direction_DE','direction_true_de',False,None),('LFC_Spearman','lfc_spearman_true_de',False,None),('DE_overlap','de_overlap_N',False,None),('DE_AUPRC','auprc',False,None)]
 for mode in ['native','adapted']:
  sub=f[f['mode']==mode];dd=defs if mode=='adapted' else defs[:3]
  for base,candidates in [('weighted_OT',names[1:]),('uniform_OT',['uniform_identity']),('weighted_identity',['uniform_identity'])]:
   eff.extend([paired_table(sub,base,candidates,dd,['context'],'all','perturbation','context').assign(mode=mode),paired_table(sub,base,candidates,dd,[],'context','perturbation','macro').assign(mode=mode)])
 pd.concat(eff).to_csv(OUT/'priority_transfer_effects.csv',index=False);print('PRIORITY SUMMARY COMPLETE',flush=True)
if __name__=='__main__':
 which=sys.argv[1:] or ['single','double','transfer']
 for s in which:globals()[s]()
 (OUT/'STATISTICS_PROTOCOL.json').write_text(json.dumps(dict(bootstrap=NB,seed=20261117,uncertainty='Pointwise paired cluster percentile intervals conditional on frozen predictions; no retraining or biological-replicate uncertainty',direction='Positive effect means named baseline better; lower-is-better metrics are sign-reversed',single_aggregation='Mean contexts within target/split, mean splits within target, mean targets within dataset, macro equal datasets',double_aggregation='Mean contexts within pair/fold, mean pairs within fold, mean folds; nMSE20 ratio within each fold then mean folds',transfer_aggregation='Mean perturbations within context, macro equal four contexts; resample perturbation clusters jointly across contexts',missing='Secondary metrics paired complete-case within each comparison; denominators saved; direction DE with no observed DE is 0 to match current manuscript',p_values='Not reported; intervals are pointwise, not simultaneous'),indent=2))
