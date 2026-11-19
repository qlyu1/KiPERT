"""One-time independent unblinding after all predictions are sealed. No training imports."""
from pathlib import Path
import json,hashlib,time,argparse
import numpy as np,pandas as pd
from scipy import sparse,stats
R=Path('@STUDY@/uc_fig2_benchmark_20260911');D=['S3','S4','S5','S6','S7'];SEEDS=[101,202,303];NEURAL=['biolord','scouter','txpert','gears','stamp','regformer'];SIMPLE=['zero','mean','knn','ridge','full_krr','lowrank_krr'];ABL=['UC_without_depmap','UC_without_string','UC_without_genept'];METHODS=NEURAL+['UC']+SIMPLE+ABL;OUT=R/'evaluation/formal';BOOT=10000

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def directory(d,s,m):
 root=R/'runs/formal'/d/f'split{s}'
 return root/m if m in NEURAL else root/'uc_and_baselines'/m

def verify_all():
 receipt=[];missing=[]
 for d in D:
  for s in SEEDS:
   for m in METHODS:
    path=directory(d,s,m);res=path/'result.json';pred=path/'predictions_sealed.npz'
    if not res.exists() or not pred.exists():missing.append(f'{d}/{s}/{m}');continue
    z=json.loads(res.read_text());assert z['status']=='PREDICTIONS_SEALED';assert not z.get('test_metrics_computed',False);assert sha(pred)==z['prediction_sha256'];receipt.append({'dataset':d,'split':s,'method':m,'prediction_sha256':z['prediction_sha256']})
 if missing:raise RuntimeError('NO UNBLINDING: incomplete predictions '+str(missing))
 return receipt

def corr(a,b):
 aa=a-a.mean();bb=b-b.mean();den=np.linalg.norm(aa)*np.linalg.norm(bb)
 return float(aa@bb/den) if den>1e-12 else np.nan

def retrieval(p,y):
 dist=np.maximum((p*p).mean(1)[:,None]+(y*y).mean(1)[None,:]-2*(p@y.T)/p.shape[1],0);rows=[];N=len(p)
 for i in range(N):
  v=dist[i,i];better=int((dist[i]<v-1e-12).sum());ties=int((np.abs(dist[i]-v)<=1e-12).sum());rank=better+(ties+1)/2;rows.append({'retrieval_top1':1/ties if better==0 else 0.,'retrieval_mrr':float(np.mean(1/np.arange(better+1,better+ties+1))),'retrieval_rank':(rank-1)/(N-1) if N>1 else np.nan,'retrieval_candidates':N,'chance_top1':1/N,'chance_mrr':float(np.mean(1/np.arange(1,N+1)))})
 return rows

def bh(p):
 order=np.argsort(p,kind='stable');vals=p[order]*len(p)/np.arange(1,len(p)+1);adj=np.minimum.accumulate(vals[::-1])[::-1];out=np.empty(len(p));out[order]=np.minimum(adj,1);return out

def test():
 rng=np.random.RandomState(12);y=rng.normal(size=(11,30));p=np.tile(rng.normal(size=(1,30)),(11,1));rows=retrieval(p,y);assert abs(np.mean([z['retrieval_top1'] for z in rows])-1/11)<1e-12;assert abs(np.mean([z['retrieval_mrr'] for z in rows])-np.mean(1/np.arange(1,12)))<1e-12
 rows=retrieval(np.zeros((4,3)),np.zeros((4,3)));assert all(abs(z['retrieval_top1']-.25)<1e-12 for z in rows);assert all(abs(z['retrieval_mrr']-np.mean(1/np.arange(1,5)))<1e-12 for z in rows);assert all(z['retrieval_top1']==1 for z in retrieval(y,y));assert np.isnan(corr(np.ones(3),np.arange(3)));print('SCORER_SYNTHETIC_TESTS_PASS')

def main():
 if (R/'manifests/SCORING_HOLD.json').exists():raise RuntimeError('Scoring hold: independent full-matrix verification required')
 receipt=verify_all();OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'UNBLINDING_LOCK.json'
 if lock.exists():raise RuntimeError('Unblinding already started; inspect outputs, do not silently repeat')
 lock.write_text(json.dumps({'unix_time':time.time(),'worker_sha256':sha(__file__),'predictions':receipt,'cohort':'five eligible datasets from four source studies; original42 plus21 newly added comparisons; retired comparisons retained at p=1','bootstrap':BOOT},indent=2));rows=[];repeat=[];variance=[];baseline_selection=[]
 for d in D:
  for s in SEEDS:
   base=R/'data/formal'/d/f'split{s}';q=pd.read_csv(base/'query/metadata.tsv',sep='\t',dtype=str);x=sparse.load_npz(base/'evaluation_only/expression.npz').tocsr();meta=pd.read_csv(base/'evaluation_only/metadata.tsv',sep='\t',dtype=str);xc=sparse.load_npz(base/'controls_fit/expression.npz');mc=pd.read_csv(base/'controls_fit/metadata.tsv',sep='\t',dtype=str);xt=sparse.load_npz(base/'fit/expression.npz');mt=pd.read_csv(base/'fit/metadata.tsv',sep='\t',dtype=str);G=x.shape[1]
   # Training-defined HVG5000. Independent of evaluation labels and expression.
   import scanpy as sc,anndata as ad
   av=ad.AnnData(sparse.vstack([xt,xc],format='csr'));sc.pp.highly_variable_genes(av,flavor='seurat',n_top_genes=min(5000,G),subset=False);hvg=np.flatnonzero(av.var.highly_variable.to_numpy());np.save(OUT/f'{d}_{s}_training_hvg_indices.npy',hvg);del av
   ctrl={};cstats={}
   for c,ix in mc.groupby('context',sort=True).indices.items():
    z=xc[ix].astype(float);mu=np.asarray(z.mean(0)).ravel();var=np.maximum(np.asarray(z.power(2).mean(0)).ravel()-mu**2,0);ctrl[c]=mu;cstats[c]=(var,len(ix))
   shared={}
   for (t,c),ix in mt.groupby(['target','context'],sort=True).indices.items():shared.setdefault(c,[]).append(np.asarray(xt[ix].mean(0)).ravel()-ctrl[c])
   shared={c:np.mean(v,axis=0) for c,v in shared.items()};truth={};degs={};statinfo={};indices=meta.groupby(['target','context'],sort=True).indices
   for (t,c),ix in indices.items():
    z=x[ix].astype(float);mu=np.asarray(z.mean(0)).ravel();var=np.maximum(np.asarray(z.power(2).mean(0)).ravel()-mu**2,0);cv,cn=cstats[c];v1=var/max(len(ix)-1,1);v0=cv/max(cn-1,1);se2=v1+v0;tt=np.divide(np.abs(mu-ctrl[c]),np.sqrt(se2),out=np.zeros(G),where=se2>1e-20);df=np.divide(se2**2,v1**2/max(len(ix)-1,1)+v0**2/max(cn-1,1),out=np.ones(G),where=se2>1e-20);tt[(se2<=1e-20)&(np.abs(mu-ctrl[c])>1e-12)]=np.inf;pval=2*stats.t.sf(tt,df);adj=bh(pval);deg=np.flatnonzero((adj<.05)&(np.abs(mu-ctrl[c])>.1));top=np.argsort(-tt,kind='stable')[:100];truth[t,c]=mu;degs[t,c]=(deg,top);statinfo[t,c]={'cells':len(ix),'degs_fdr05_delta01':len(deg),'effect_rms':float(np.sqrt(np.mean((mu-ctrl[c])**2)))}
    # Fixed hashed split-half reference, never used by selection.
    order=sorted(ix,key=lambda i:hashlib.sha256(('fig2-replicate-'+str(meta.cell_id.iloc[i])).encode()).digest());half=len(order)//2
    if half:
     h1=np.asarray(x[order[:half]].mean(0)).ravel();h2=np.asarray(x[order[half:]].mean(0)).ravel();repeat.append({'dataset':d,'split':s,'target':t,'context':c,'split_half_rmse':float(np.sqrt(np.mean((h1-h2)**2))),'n_first':half,'n_second':len(order)-half})
   best=[]
   for m in SIMPLE:
    z=json.loads((directory(d,s,m)/'selection.json').read_text())['chosen'];best.append((z['validation_macro_rmse'],SIMPLE.index(m),m))
   winner=min(best)[2];baseline_selection.append({'dataset':d,'split':s,'best_simple':winner,'validation_score':min(best)[0]})
   y=np.stack([truth[z.target,z.context] for z in q.itertuples()]);mu=np.stack([ctrl[z.context] for z in q.itertuples()]);common=q.common_supported.eq('True').to_numpy();queries=list(zip(q.target,q.context));context=q.context.to_numpy()
   for m in METHODS:
    predfile=directory(d,s,m)/'predictions_sealed.npz';z=np.load(predfile,allow_pickle=False);yp=z['prediction'].astype(float);assert yp.shape==y.shape;assert z['target'].astype(str).tolist()==q.target.tolist();assert z['context'].astype(str).tolist()==q.context.tolist();assert np.isfinite(yp).all();dp=yp-mu;dy=y-mu
    for cohort,keep in [('common',common),('full',np.ones(len(q),bool))]:
     ret={}
     for c in sorted(set(context[keep])):
      ix=np.flatnonzero(keep&(context==c));retr=retrieval(dp[ix],dy[ix]);ret.update({int(i):v for i,v in zip(ix,retr)});truevar=float(np.mean(np.var(dy[ix],axis=0)));predvar=float(np.mean(np.var(dp[ix],axis=0)));variance.append({'dataset':d,'split':s,'method':m,'context':c,'cohort':cohort,'variance_ratio':predvar/truevar if truevar>1e-15 else np.nan,'true_variance':truevar,'predicted_variance':predvar})
     for i in np.flatnonzero(keep):
      t,c=queries[i];err=yp[i]-y[i];deg,top=degs[t,c];row={'dataset':d,'split':s,'method':m,'cohort':cohort,'target':t,'context':c,'rmse':float(np.sqrt(np.mean(err**2))),'mse':float(np.mean(err**2)),'hvg5000_rmse':float(np.sqrt(np.mean(err[hvg]**2))),'pearson_delta':corr(dp[i],dy[i]),'spearman_delta':corr(stats.rankdata(dp[i]),stats.rankdata(dy[i])),'specific_pearson':corr(dp[i]-shared[c],dy[i]-shared[c]),'top100_rmse':float(np.sqrt(np.mean(err[top]**2))),'deg_direction_accuracy':float(np.mean(np.sign(dp[i,deg])==np.sign(dy[i,deg]))) if len(deg) else np.nan,'native_fallback':bool(z['native_fallback'][i]) if 'native_fallback' in z else False,'selected_simple_baseline':winner,**statinfo[t,c],**ret[i]};rows.append(row)
   print(json.dumps({'scored_dataset':d,'split':s,'rows_so_far':len(rows)}),flush=True)
 frame=pd.DataFrame(rows);baseline_rows=frame[frame.method==frame.selected_simple_baseline][['dataset','split','cohort','target','context','mse']].rename(columns={'mse':'baseline_mse'});frame=frame.merge(baseline_rows,on=['dataset','split','cohort','target','context'],validate='many_to_one');frame['mse_skill']=np.where(frame.baseline_mse>1e-15,1-frame.mse/frame.baseline_mse,np.nan);frame.to_csv(OUT/'condition_metrics.tsv',sep='\t',index=False);pd.DataFrame(repeat).to_csv(OUT/'split_half_reference.tsv',sep='\t',index=False);pd.DataFrame(variance).to_csv(OUT/'variance_diagnostics.tsv',sep='\t',index=False);pd.DataFrame(baseline_selection).to_csv(OUT/'validation_selected_baselines.tsv',sep='\t',index=False)
 numeric=['rmse','mse_skill','mse','hvg5000_rmse','pearson_delta','spearman_delta','specific_pearson','top100_rmse','deg_direction_accuracy','retrieval_top1','retrieval_mrr','retrieval_rank','effect_rms','cells'];target=frame.groupby(['dataset','split','method','cohort','target'],sort=True)[numeric].mean().reset_index();target.to_csv(OUT/'target_metrics.tsv',sep='\t',index=False);unique_target=target.groupby(['dataset','method','cohort','target'],sort=True)[numeric].mean().reset_index();unique_target.to_csv(OUT/'unique_target_metrics.tsv',sep='\t',index=False);summary=unique_target.groupby(['dataset','method','cohort'],sort=True)[numeric].mean().reset_index();summary.to_csv(OUT/'summary.tsv',sep='\t',index=False)
 # Paired effects average repeat splits within target before bootstrap.
 main=target[target.cohort=='common'];comparisons=[];rng=np.random.RandomState(240911)
 for d in ['S1','S2',*D]:
  uc=main[(main.dataset==d)&(main.method=='UC')][['split','target','rmse']].rename(columns={'rmse':'uc'})
  for other in ['biolord','scouter','scgpt','scfoundation','txpert','gears','stamp','regformer','best_simple']:
   if d not in D or other in ['scgpt','scfoundation']:comparisons.append({'dataset':d,'comparator':other,'status':'RETIRED' if other in ['scgpt','scfoundation'] else 'NOT_RUN','n_targets':0,'p':1.});continue
   if other=='best_simple':
    parts=[]
    for s in SEEDS:
     winner=next(z['best_simple'] for z in baseline_selection if z['dataset']==d and z['split']==s);parts.append(main[(main.dataset==d)&(main.split==s)&(main.method==winner)])
    cmp=pd.concat(parts)
   else:cmp=main[(main.dataset==d)&(main.method==other)]
   j=uc.merge(cmp[['split','target','rmse']],on=['split','target'],validate='one_to_one');j['difference']=j.uc-j.rmse;v=j.groupby('target').difference.mean().to_numpy();estimate=float(v.mean());samples=v[rng.randint(0,len(v),(BOOT,len(v)))].mean(1);lo,hi=np.quantile(samples,[.025,.975]);pval=(1+np.sum(np.abs(samples-estimate)>=abs(estimate)))/(BOOT+1);comparisons.append({'dataset':d,'comparator':other,'status':'COMPLETE','n_targets':len(v),'uc_minus_comparator_rmse':estimate,'ci_low':float(lo),'ci_high':float(hi),'p':float(pval)})
 cmp=pd.DataFrame(comparisons);order=np.argsort(cmp.p.to_numpy(),kind='stable');adjusted=np.maximum.accumulate((len(cmp)-np.arange(len(cmp)))*cmp.p.to_numpy()[order]);cmp['p_holm']=np.nan;cmp.loc[order,'p_holm']=np.minimum(adjusted,1);assert len(cmp)==63;cmp.to_csv(OUT/'paired_comparisons_holm63.tsv',sep='\t',index=False)
 study_map={'S3':'Tian2021','S4':'Tian2021','S5':'Adamson2016','S6':'Norman2019','S7':'Frangieh2021'};ss=summary.copy();ss['study']=ss.dataset.map(study_map);study=ss.groupby(['study','method','cohort'])[numeric].mean().reset_index();study.to_csv(OUT/'study_metrics.tsv',sep='\t',index=False);study.groupby(['method','cohort'])[numeric].mean().to_csv(OUT/'study_equal_weight_summary.tsv',sep='\t');cs=cmp[cmp.status=='COMPLETE'].copy();cs['study']=cs.dataset.map(study_map);study_comparisons=[]
 for other in NEURAL+['best_simple']:
  values=cs[cs.comparator==other].groupby('study').uc_minus_comparator_rmse.mean().to_numpy();samples=values[rng.randint(0,len(values),(BOOT,len(values)))].mean(1);lo,hi=np.quantile(samples,[.025,.975]);study_comparisons.append({'comparator':other,'n_studies':len(values),'effect':float(values.mean()),'ci_low':float(lo),'ci_high':float(hi),'interpretation':'four-study cluster bootstrap; exploratory limited generalization'})
 pd.DataFrame(study_comparisons).to_csv(OUT/'study_cluster_comparisons.tsv',sep='\t',index=False)
 nanrates=frame.groupby(['dataset','method','cohort'])[['pearson_delta','spearman_delta','deg_direction_accuracy']].agg(lambda x:int(x.isna().sum()));nanrates.to_csv(OUT/'undefined_metric_counts.tsv',sep='\t')
 (OUT/'COMPLETE.json').write_text(json.dumps({'status':'FORMAL_EVALUATION_COMPLETE','datasets':D,'source_studies':4,'splits':SEEDS,'methods':METHODS,'prediction_sets':len(receipt),'primary_hypothesis_family':63,'unrun_comparisons_p':1,'bootstrap':BOOT,'test_used_for_selection':False,'note':'Primary paired errors cancel shared C_fit reference algebraically; primary CI resamples target clusters, not cells/seeds; split-half reference describes cell sampling noise. Supplemental correlations remain reference-sensitive.'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args();test() if a.self_test else main()
