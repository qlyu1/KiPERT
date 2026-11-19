"""Paper-matched retrospective metrics. All inputs are frozen; no model is fitted."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy import sparse,stats
from scipy.spatial.distance import cdist,squareform
from scipy.cluster.hierarchy import linkage,leaves_list
import statsmodels.api as sm
import statsmodels.formula.api as smf
R=Path('@STUDY@/uc_fig2_benchmark_20260911');E=R/'experiments/uc_fig2_latest_within_context_20261105'
S=E/'specificity_extension_20261105';OLD=E/'figures/comprehensive_20261105';O=E/'figures/weighted_full_atlas_20261106';OUT=O/'source_data';OUT.mkdir(parents=True,exist_ok=True)
M=['UC','biolord','scouter','txpert','gears','stamp','regformer'];D=['S3','S4','S5','S6','S7'];SEEDS=[101,202,303]
SCALES=np.array([0,1,2,4,8,16,25,40,60,80,100,125,150,175,200,250,300,350,400,450,500.])
def corr(a,b):
 a=np.asarray(a);b=np.asarray(b);a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True);den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
 return np.divide(np.sum(a*b,-1),den,out=np.full(np.shape(den),np.nan),where=den>1e-12)
def unit(a):
 n=np.linalg.norm(a,axis=-1,keepdims=True);return np.divide(a,n,out=np.zeros_like(a),where=n>1e-12)
def pds(x,strict=False):
 if len(x)<2:return np.full(len(x),np.nan)
 v=np.diag(x)[:,None];close=(x<v-1e-12).sum(1);ties=(abs(x-v)<=1e-12).sum(1)-1
 return 1-(close+(ties if strict else .5*ties))/(len(x)-1)
def auprc(y,s):
 """Davis-Goadrich nonlinear TP/FP interpolation, grouped tied scores."""
 if y.sum()==0 or y.sum()==len(y):return np.nan
 ix=np.argsort(-s,kind='stable');s=s[ix];y=y[ix].astype(float);ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
 tp=np.cumsum(y)[ends];fp=ends+1-tp;a=np.r_[0,tp[:-1]];b=np.r_[0,fp[:-1]];dt=tp-a;df=fp-b;ok=dt>0
 a=a[ok];b=b[ok];dt=dt[ok];df=df[ok];k=1+df/dt;v=dt/k;positive=(a+b)>0
 v[positive]+=(a[positive]-(a[positive]+b[positive])/k[positive])/k[positive]*np.log1p(k[positive]*dt[positive]/(a[positive]+b[positive]))
 return float(v.sum()/y.sum())
def curve(y,s):
 ix=np.argsort(-s,kind='stable');s=s[ix];y=y[ix];ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1];tp=np.cumsum(y)[ends];fp=ends+1-tp
 out=[];a=b=0.
 for t,f in zip(tp,fp):
  if t>a:
   tt=np.linspace(a,t,8);ff=b+(f-b)*(tt-a)/(t-a);pre=np.divide(tt,tt+ff,out=np.ones_like(tt)*(t/(t+f)),where=(tt+ff)>0)
   out.extend(zip(tt/y.sum(),pre))
  elif out:out.append((t/y.sum(),t/(t+f)))
  a,b=t,f
 return np.array(out)
def aggregate(f,keys=('dataset','method','target')):
 met=[k for k in f if k not in ['dataset','method','target','context','split']]
 return f.groupby([*keys,'split'])[met].mean().reset_index().groupby(list(keys))[met].mean().reset_index()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

ZERO_RMS=1e-7;TIE=1e-12
from sklearn.metrics import average_precision_score,roc_auc_score
def cosine_dist(a,b):
    na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1)
    aa=np.divide(a,na[:,None],out=np.zeros_like(a),where=na[:,None]>ZERO_RMS*np.sqrt(a.shape[1]))
    bb=np.divide(b,nb[:,None],out=np.zeros_like(b),where=nb[:,None]>ZERO_RMS*np.sqrt(b.shape[1]))
    return np.clip(1-aa@bb.T,0,2)
def ranks(dist,axis=1):
    a=dist if axis==1 else dist.T;n=len(a)
    if n<2:return np.full(n,np.nan),np.full(n,np.nan),np.full(n,np.nan)
    correct=np.diag(a)[:,None]
    better=(a<correct-TIE).sum(1);tied=(np.abs(a-correct)<=TIE).sum(1)-1
    assert np.all(tied>=0)
    return (better+.5*tied)/(n-1),(better+tied)/(n-1),(n-1-better-tied)/(n-1)
def top_weights(score,k):
    k=min(k,len(score));cut=np.partition(score,len(score)-k)[len(score)-k]
    above=score>cut;ties=score==cut;w=above.astype(float)
    w[ties]=(k-above.sum())/ties.sum()
    assert abs(w.sum()-k)<1e-9
    return w
def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-12 else np.nan
def retrieval(p,y):
 dist=np.maximum((p*p).mean(1)[:,None]+(y*y).mean(1)[None,:]-2*(p@y.T)/p.shape[1],0);rows=[];N=len(p)
 for i in range(N):
  v=dist[i,i];better=int((dist[i]<v-1e-12).sum());ties=int((np.abs(dist[i]-v)<=1e-12).sum());rank=better+(ties+1)/2;rows.append({'retrieval_top1':1/ties if better==0 else 0.,'retrieval_mrr':float(np.mean(1/np.arange(better+1,better+ties+1))),'retrieval_rank':(rank-1)/(N-1) if N>1 else np.nan,'retrieval_candidates':N,'chance_top1':1/N,'chance_mrr':float(np.mean(1/np.arange(1,N+1)))})
 return rows

def geo_unit(a):
 n=np.linalg.norm(a,axis=1);return np.divide(a,n[:,None],out=np.zeros_like(a),where=n[:,None]>1e-7*np.sqrt(a.shape[1]))
def extras(p,y,mu,shared,lab,top,ctx,keep,hvg,base_mse):
 raw=p-mu;dy=y-mu;dp=raw.copy();dp[np.sqrt(np.mean(dp*dp,1))<=1e-7]=0;G=y.shape[1];out=[]
 for i in range(len(y)):
  l=lab[i];n=int(l.sum());score=abs(dp[i]);t=top[i];err=p[i]-y[i];r=dict(rmse=float(np.sqrt(np.mean(err**2))),mse=float(np.mean(err**2)),hvg5000_rmse=float(np.sqrt(np.mean(err[hvg]**2))),top100_rmse=float(np.sqrt(np.mean(err[t]**2))),mse_skill=float(1-np.mean(err**2)/base_mse[i]) if base_mse[i]>1e-15 else np.nan,pearson_delta=float(corr(raw[i],dy[i])),spearman_delta=float(corr(stats.rankdata(raw[i]),stats.rankdata(dy[i]))),specific_pearson=float(corr(raw[i]-shared[i],dy[i]-shared[i])),deg_direction_accuracy=float(np.mean(np.sign(raw[i,l])==np.sign(dy[i,l]))) if n else np.nan,systema_pearson_top20=float(corr(raw[i,t[:20]]-shared[i,t[:20]],dy[i,t[:20]]-shared[i,t[:20]])),specific_cosine=cosine(raw[i]-shared[i],dy[i]-shared[i]),top100_direction=float(np.mean(np.sign(dp[i,t])==np.sign(dy[i,t]))),absolute_pearson=float(corr(p[i],y[i])),predicted_rms=float(np.sqrt(np.mean(dp[i]**2))),truth_rms=float(np.sqrt(np.mean(dy[i]**2))),shared_cosine=float(geo_unit(dy[i:i+1])[0]@geo_unit(shared[i:i+1])[0]),deg_prevalence=n/G,n_deg=n)
  r.update({k:np.nan for k in ['deg_ap','deg_ap_skill','deg_auroc','deg_precision100','deg_recall100','deg_overlap20_mean','deg_overlap100_mean']})
  if 0<n<G:
   ap=average_precision_score(l,score);hits=top_weights(score,100)@l;r.update(deg_ap=ap,deg_ap_skill=(ap-n/G)/(1-n/G),deg_auroc=roc_auc_score(l,score),deg_precision100=hits/100,deg_recall100=hits/n)
   for k in [20,100]:
    if n>=k:r[f'deg_overlap{k}_mean']=top_weights(score,k)@top_weights(np.where(l,abs(dy[i]),-np.inf),k)/k
  out.append(r)
 for c in sorted(set(ctx)):
  ii=np.flatnonzero(ctx==c);a=dp[ii];b=dy[ii];n=len(ii);l2=np.maximum(np.mean(raw[ii]**2,1)[:,None]+np.mean(b**2,1)[None,:]-2*raw[ii]@b.T/G,0);l1=cdist(a,b,'cityblock')/G;co=cosine_dist(a,b);cn=cosine_dist(a[:,keep],b[:,keep]);r2,_,strict=ranks(l2);bl,blo,_=ranks(l2,0);bc,bco,_=ranks(co,0);retr=retrieval(raw[ii],b)
  px=a[:,keep];yx=b[:,keep];sp=np.clip(geo_unit(px)@geo_unit(px).T,-1,1);sy=np.clip(geo_unit(yx)@geo_unit(yx).T,-1,1);err=(sp-sy)**2;np.fill_diagonal(err,0);np.fill_diagonal(sp,0)
  vals=dict(pds_l2=1-r2,systema_centroid_strict=strict,pds_l1=1-ranks(l1)[0],pds_cosine=1-ranks(co)[0],pds_cosine_exclude_targets=1-ranks(cn)[0],pb_reverse_rank_l2=bl,pb_reverse_rank_cosine=bc,pb_reverse_rank_l2_original=blo,pb_reverse_rank_cosine_original=bco,geometry_mse=err.sum(1)/(n-1) if n>1 else np.full(n,np.nan),mean_pred_pair_cos=sp.sum(1)/(n-1) if n>1 else np.full(n,np.nan),pred_var_contrib=np.mean((px-px.mean(0))**2,1) if n>1 else np.full(n,np.nan),truth_var_contrib=np.mean((yx-yx.mean(0))**2,1) if n>1 else np.full(n,np.nan))
  for j,i in enumerate(ii):
   out[i].update({k:float(v[j]) for k,v in vals.items()});out[i].update({k:retr[j][k] for k in ['retrieval_top1','retrieval_mrr','retrieval_rank']})
 return out

PRIOR=E/'figures/paper_matched_merged_20261105'
sealed=json.loads((E/'specific_methods_20261105/ALL_PREDICTIONS_SEALED.json').read_text())
original=pd.read_csv(E/'evaluation/condition_metrics.tsv',sep='\t',dtype={'target':str,'context':str});original=original[(original.method=='UC')&(original.cohort=='common')].set_index(['dataset','split','target','context'])
assert not (O/'COMPLETE.json').exists()
# Analytic checks separate nonlinear AUPRC from average precision.
assert abs(auprc(np.array([1,0,0,1]),np.ones(4))-.5)<1e-12
assert auprc(np.array([1,1,0,0]),np.array([4.,3,2,1]))==1
rows=[];scale_rows=[];truth_rows=[];ref_rows=[];raw_cases=dict(np.load(PRIOR/'source_data/cases.npz'));land=dict(np.load(PRIOR/'source_data/landscapes.npz'));pre_rows=[];receipts=[];cell_rows=[]
cases=json.loads((OLD/'CASE_SELECTION.json').read_text());case_lookup={x['dataset']:x for x in cases}
for d in D:
 for seed in SEEDS:
  base=R/'data/formal'/d/f'split{seed}';z=np.load(S/'truth_cache'/f'{d}_{seed}.npz');ix=np.flatnonzero(z['common'])
  y=z['truth'][ix].astype(float);mu=z['control'][ix].astype(float);dy=y-mu;target=z['target'][ix].astype(str);ctx=z['context'][ix].astype(str);lab=z['deg'][ix];top=z['top100'][ix];G=y.shape[1]
  genes=pd.read_csv(base/'genes.tsv',sep='\t',dtype=str);symbols=genes.symbol.fillna(genes.gene_id).to_numpy();lookup={t:i for i,t in enumerate(genes.gene_id)}
  xt=sparse.load_npz(base/'fit/expression.npz').astype(float);mt=pd.read_csv(base/'fit/metadata.tsv',sep='\t',dtype=str)
  ref={c:np.asarray(xt[j].mean(0)).ravel() for c,j in mt.groupby('context').indices.items()}
  yyref=np.stack([ref[c] for c in ctx]);rd=yyref-mu;corrected=y-yyref
  xc=sparse.load_npz(base/'controls_fit/expression.npz').astype(float);mc=pd.read_csv(base/'controls_fit/metadata.tsv',sep='\t',dtype=str)
  xe=sparse.load_npz(base/'evaluation_only/expression.npz').astype(float);me=pd.read_csv(base/'evaluation_only/metadata.tsv',sep='\t',dtype=str)
  cell_lookup=me.groupby(['target','context']).indices
  ci=next((i for i in range(len(y)) if target[i]==case_lookup[d]['target'] and ctx[i]==case_lookup[d]['context']),None) if seed==101 else None
  if ci is not None:
   jj=cell_lookup[target[ci],ctx[ci]];gi=top[ci,:10];cells=xe[jj][:,gi].toarray()
   raw_cases.update({d+'_truth':y[ci],d+'_control':mu[ci],d+'_label':lab[ci],d+'_symbols':symbols.astype(str),d+'_top':gi,d+'_cells':cells})
  for i in range(len(y)):
   c=ctx[i];n=len(cell_lookup[target[i],c]);tr=dict(dataset=d,split=seed,target=target[i],context=c,n_cells=n,n_deg=int(lab[i].sum()),n_genes=G,truth_rms=float(np.sqrt(np.mean(dy[i]**2))),systematic_cosine=float(unit(dy[i])@unit(rd[i])))
   truth_rows.append(tr)
  # Preprocessing diagnostic uses identical observed cells/features; the full-library
  # totals in metadata permit exact renormalization of the existing log1p matrix.
  if seed==101:
   med=float(np.median(np.r_[mt.raw_total.astype(float),mc.raw_total.astype(float),me.raw_total.astype(float)]))
   xm=xe.copy();xm.data=np.log1p(np.expm1(xm.data)*(med/10000));cm=xc.copy();cm.data=np.log1p(np.expm1(cm.data)*(med/10000));cmu={c:np.asarray(cm[j].mean(0)).ravel() for c,j in mc.groupby('context').indices.items()}
   for i in range(len(y)):
    ym=np.asarray(xm[cell_lookup[target[i],ctx[i]]].mean(0)).ravel()-cmu[ctx[i]];a=dy[i];mask=np.ones(G,bool)
    if target[i] in lookup:mask[lookup[target[i]]]=False
    a=a[mask];b=ym[mask]
    pre_rows.append(dict(dataset=d,target=target[i],context=ctx[i],median_library=med,l1_10k=float(abs(a).sum()),l1_median=float(abs(b).sum()),l2_10k=float(np.linalg.norm(a)),l2_median=float(np.linalg.norm(b)),cosine=float(unit(a)@unit(b)),sign_cosine=float(unit(np.sign(a))@unit(np.sign(b)))))
   contexts=sorted(set(ctx));lc=sorted(contexts,key=lambda c:(-sum(ctx==c),c))[0];li=np.flatnonzero(ctx==lc);uy=unit(dy[li][:,z['gene_keep']]);sy=np.clip(uy@uy.T,-1,1);dist=np.clip(1-sy,0,2);np.fill_diagonal(dist,0);order=leaves_list(linkage(squareform(dist,checks=False),method='average'))
   land.update({d+'_truth':sy[np.ix_(order,order)],d+'_target':target[li][order],d+'_context':np.array([lc]),d+'_symbols':np.array([symbols[lookup[t]] if t in lookup else t for t in target[li][order]])})
  # Training-cell weighted perturbation mean is a diagnostic reference, not a competitor.
  ref_ca=np.full(len(y),np.nan);pair_control=np.full(len(y),np.nan);pair_shift=np.full(len(y),np.nan)
  for c in sorted(set(ctx)):
   jj=np.flatnonzero(ctx==c);n=len(jj)
   ref_ca[jj]=pds(cdist(yyref[jj],y[jj],'sqeuclidean'),True)
   for dest,v in [(pair_control,dy[jj]),(pair_shift,corrected[jj])]:
    a=unit(v)@unit(v).T;dest[jj]=(a.sum(1)-np.diag(a))/(n-1) if n>1 else np.nan
  for i in range(len(y)):ref_rows.append(dict(dataset=d,split=seed,target=target[i],context=ctx[i],ref_ca=ref_ca[i],control_pair_cosine=pair_control[i],perturbed_pair_cosine=pair_shift[i]))
  for m in ['UC']:
   pf=E/'specific_methods_20261105'/d/f'split{seed}'/'aligned_raw_predictions.npz';h=sha(pf);rec=next(r for r in sealed if r['dataset']==d and r['split']==seed and r['arm']=='aligned_raw');assert h==rec['prediction_sha256'];receipts.append(dict(path=str(pf),sha256=h))
   pz=np.load(pf);assert np.array_equal(pz['target'][ix].astype(str),target);assert np.array_equal(pz['context'][ix].astype(str),ctx)
   p=pz['prediction'][ix].astype(float);dp=p-mu;dp[np.sqrt(np.mean(dp*dp,1))<1e-7]=0
   ext={k:np.full(len(y),np.nan) for k in ['pds_query_l1','pds_query_l2','pds_query_cosine','pds_query_sign','pds_oracle_l1','pds_oracle_l2']}
   for c in sorted(set(ctx)):
    ii=np.flatnonzero(ctx==c);n=len(ii)
    if n<2:continue
    a=dp[ii];b=dy[ii];target_ix=np.array([lookup.get(t,-1) for t in target[ii]])
    # Each query excludes its targeted gene from BOTH prediction and candidate truth.
    # Candidate sets remain in the same dataset/context/split.
    matrices={k:np.zeros((n,n)) for k in ['l1','l2','cosine','sign','oracle_l1','oracle_l2']}
    scaled=np.zeros((len(SCALES),2,n,n))
    for j in range(n):
     mask=np.ones(G,bool)
     if target_ix[j]>=0:mask[target_ix[j]]=False
     aa=a[j,mask];bb=b[:,mask];an=np.linalg.norm(aa);bn=np.linalg.norm(bb,axis=1)
     matrices['l1'][j]=cdist(aa[None],bb,'cityblock')[0];matrices['l2'][j]=cdist(aa[None],bb,'sqeuclidean')[0]
     matrices['cosine'][j]=1-unit(aa)@unit(bb).T;matrices['sign'][j]=1-unit(np.sign(aa))@unit(np.sign(bb)).T
     for norm,name in [(1,'l1'),(2,'l2')]:
      den=np.linalg.norm(aa,ord=norm);ratio=np.linalg.norm(bb[j],ord=norm)/den if den>1e-12 else 1.
      matrices['oracle_'+name][j]=cdist((aa*ratio)[None],bb,'cityblock' if norm==1 else 'sqeuclidean')[0]
     for k,scale in enumerate(SCALES):
      scaled[k,0,j]=cdist((aa*scale)[None],bb,'cityblock')[0]
      scaled[k,1,j]=np.maximum((scale*an)**2+bn**2-2*scale*(aa@bb.T),0)
    for k,v in matrices.items():ext['pds_'+('oracle_'+k[7:] if k.startswith('oracle_') else 'query_'+k)][ii]=pds(v)
    for k,scale in enumerate(SCALES):
     for distance in [0,1]:
      vv=pds(scaled[k,distance])
      for j,i in enumerate(ii):scale_rows.append(dict(dataset=d,method=m,split=seed,target=target[i],context=c,scale=float(scale),distance=['L1','L2'][distance],value=vv[j]))
   bm=np.array([original.loc[(d,seed,t,c),'baseline_mse'] for t,c in zip(target,ctx)]);hvg=np.load(E/'evaluation'/f'{d}_{seed}_training_hvg_indices.npy');more=extras(p,y,mu,z['shared'][ix].astype(float),lab,top,ctx,z['gene_keep'],hvg,bm)
   for i in range(len(y)):
    t=top[i];labels=lab[i];ng=int(labels.sum());predidx=np.argsort(-abs(dp[i]),kind='stable');n50=len(set(predidx[:50])&set(t[:50]))/50;n100=len(set(predidx[:100])&set(t))/100
    row=dict(dataset=d,split=seed,method=m,target=target[i],context=ctx[i],absolute_cosine=float(unit(p[i])@unit(y[i])),absolute_pearson20=float(corr(p[i,t[:20]],y[i,t[:20]])),delta_pearson20=float(corr(dp[i,t[:20]],dy[i,t[:20]])),systema_cell_pearson=float(corr(p[i]-yyref[i],corrected[i])),systema_cell_pearson20=float(corr(p[i,t[:20]]-yyref[i,t[:20]],corrected[i,t[:20]])),r2_all=float(corr(p[i],y[i])**2),r2_deg=float(corr(p[i,labels],y[i,labels])**2) if ng>=2 else np.nan,r2_all_deg_eligible=float(corr(p[i],y[i])**2) if ng>=2 else np.nan,auprc_dg=auprc(labels,abs(dp[i])),delta_direction_all=float(np.mean(np.sign(dp[i])==np.sign(dy[i]))),overlap50_rank=n50,overlap100_rank=n100,sp_mse100=float(np.mean((p[i,t]-y[i,t])**2)),sp_pcc100=float(corr(dp[i,t],dy[i,t])),**{k:float(v[i]) for k,v in ext.items()})
    row.update(more[i]);rows.append(row)
   if ci is not None:
    raw_cases[d+'_'+m]=p[ci]
   if seed==101:
    sp=np.clip(unit(dp[li][:,z['gene_keep']])@unit(dp[li][:,z['gene_keep']]).T,-1,1);land[d+'_'+m]=sp[np.ix_(order,order)]
    # Reverse-rank bars: hold each truth fixed and rank all predicted responses.
    sim=unit(dp[li][:,z['gene_keep']])@unit(dy[li][:,z['gene_keep']]).T
    raw_cases[d+'_'+m+'_retrieval_similarity']=sim
    raw_cases[d+'_retrieval_targets']=np.array([symbols[lookup[t]] if t in lookup else t for t in target[li]])
   print(d,seed,m,'done',flush=True)
  del xt,xc,xe
f=pd.DataFrame(rows);old=pd.read_csv(PRIOR/'source_data/condition_metrics.csv',dtype={'context':str,'target':str});assert set(f.columns)==set(old.columns),(set(old)-set(f),set(f)-set(old));f=f[old.columns];f=pd.concat([old[old.method!='UC'],f],ignore_index=True).sort_values(['dataset','split','method','target','context']);assert len(f)==6769
f.to_csv(OUT/'condition_metrics.csv',index=False);u=aggregate(f);u.to_csv(OUT/'target_metrics.csv',index=False);summary=u.groupby(['dataset','method']).mean(numeric_only=True).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
truth=pd.DataFrame(truth_rows);truth.to_csv(OUT/'truth_conditions.csv',index=False);pd.DataFrame(ref_rows).to_csv(OUT/'reference_conditions.csv',index=False)
np.savez_compressed(OUT/'cases.npz',**raw_cases);np.savez_compressed(OUT/'landscapes.npz',**land);(O/'CASE_SELECTION.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
pd.DataFrame(pre_rows).to_csv(OUT/'preprocessing.csv',index=False)
sc=pd.concat([pd.read_csv(PRIOR/'source_data/scaling_conditions.csv').query('method != "UC"'),pd.DataFrame(scale_rows)],ignore_index=True);sc.to_csv(OUT/'scaling_conditions.csv',index=False);sc=sc.groupby(['dataset','method','split','target','scale','distance']).value.mean().reset_index().groupby(['dataset','method','target','scale','distance']).value.mean().reset_index();sc.groupby(['dataset','method','scale','distance']).value.mean().reset_index().to_csv(OUT/'scaling_summary.csv',index=False)
rank=[]
for d in D:
 g=summary[summary.dataset==d]
 for met in [c for c in summary if c not in ['dataset','method']]:
  lower=met in ['mse','rmse','sp_mse100','hvg5000_rmse','top100_rmse','retrieval_rank','pb_reverse_rank_l2','pb_reverse_rank_cosine','pb_reverse_rank_l2_original','pb_reverse_rank_cosine_original','geometry_mse']
  for _,r in g.iterrows():rank.append(dict(dataset=d,method=r.method,metric=met,value=r[met],rank=1+int((g[met]<r[met]-1e-12).sum() if lower else (g[met]>r[met]+1e-12).sum()) if np.isfinite(r[met]) else np.nan))
pd.DataFrame(rank).to_csv(OUT/'rankings.csv',index=False)
# Author-style categorical Type II ANOVA, residual excluded from three-factor share.
# If the design is rank deficient the table is saved but is not used for claims.
anova=[]
for met in ['mse','absolute_pearson','pearson_delta','delta_direction_all','overlap50_rank']:
 a=u[['dataset','target','method',met]].dropna().rename(columns={met:'value'});fit=smf.ols('value ~ C(target) + C(dataset) + C(method)',data=a).fit();table=sm.stats.anova_lm(fit,typ=2);ss=table.loc[['C(target)','C(dataset)','C(method)'],'sum_sq'];rank_design=np.linalg.matrix_rank(fit.model.exog)
 for term,v in ss.items():anova.append(dict(metric=met,term=term,sum_sq=v,fraction=v/ss.sum(),design_columns=fit.model.exog.shape[1],design_rank=rank_design,n=len(a)))
pd.DataFrame(anova).to_csv(OUT/'anova_typeII.csv',index=False)
for receipt in receipts:assert sha(receipt['path'])==receipt['sha256']
(O/'INPUT_RECEIPTS.json').write_text(json.dumps(receipts,indent=2))
(O/'COMPLETE.json').write_text(json.dumps(dict(status='PASS',frozen_predictions=15,conditions=967,model_conditions=6769,unique_targets=283,methods=M,UC_version='Weighted full residual; validation selected; frozen aligned_raw predictions',scales=SCALES.tolist(),paper_auprc='Davis-Goadrich interpolation; benchmark labels retained, not AP',systema_reference='Training-cell-weighted mean in each context; no test labels',pds_target_mask='Query target excluded from prediction and every truth candidate; within-context pools; midrank ties',zero_vector_rule='Prediction RMS < 1e-7 treated as zero response; zero cosine similarity',preprocessing='Same cells/genes; log1p(per10k) versus log1p(per-median-library), median over fit/control/evaluation cells in split101',anova_rank_deficient=any(r['design_rank']<r['design_columns'] for r in anova)),indent=2))
print('COMPLETE',flush=True)

gold=pd.read_csv(E/'specific_methods_20261105/radar/summary.csv').query('method=="aligned_raw"').set_index('dataset');ucheck=summary.query('method=="UC"').set_index('dataset');errors={}
for k in gold.columns:
 if k=='method':continue
 errors[k]=float(np.max(abs(ucheck.loc[D,k]-gold.loc[D,k])));assert errors[k]<1e-10,(k,errors[k])
keys=['dataset','split','method','target','context'];pd.testing.assert_frame_equal(f.query('method!="UC"').sort_values(keys).reset_index(drop=True),old.query('method!="UC"').sort_values(keys).reset_index(drop=True),check_dtype=False)
(O/'SCORE_AUDIT.json').write_text(json.dumps(dict(status='PASS',independent_12_metric_max_errors=errors,other_six_condition_scores_unchanged=True,changed_uc_conditions=int(sum(f.method=='UC'))),indent=2))
