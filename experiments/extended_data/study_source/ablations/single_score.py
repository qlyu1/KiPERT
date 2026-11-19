import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,time,gc
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
O=Path('@STUDY@/kipert_supplement_20261117')
E=Path('@STUDY@/uc_single_perturbation/figure2/experiments/kipert')
R=Path('@STUDY@/uc_fig2_benchmark_20260911/data/formal')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def corr(a,b):
 a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True);den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
 return np.divide(np.sum(a*b,axis=-1),den,out=np.full(np.shape(den),np.nan),where=den>1e-12)
def unit(a):
 n=np.linalg.norm(a,axis=-1,keepdims=True);return np.divide(a,n,out=np.zeros_like(a),where=n>1e-12)
def pds(x):
 if len(x)<2:return np.full(len(x),np.nan)
 v=np.diag(x)[:,None];return 1-((x<v-1e-12).sum(1)+.5*((abs(x-v)<=1e-12).sum(1)-1))/(len(x)-1)
def auprc(y,s):
 if y.sum()==0 or y.sum()==len(y):return np.nan
 ix=np.argsort(-s,kind='stable');s=s[ix];y=y[ix].astype(float);ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
 tp=np.cumsum(y)[ends];fp=ends+1-tp;a=np.r_[0,tp[:-1]];b=np.r_[0,fp[:-1]];dt=tp-a;df=fp-b;ok=dt>0
 a=a[ok];b=b[ok];dt=dt[ok];df=df[ok];k=1+df/dt;v=dt/k;positive=(a+b)>0
 v[positive]+=(a[positive]-(a[positive]+b[positive])/k[positive])/k[positive]*np.log1p(k[positive]*dt[positive]/(a[positive]+b[positive]))
 return float(v.sum()/y.sum())
def load_truth(d,seed):
 base=R/d/f'split{seed}';z=dict(np.load(E/'specificity_extension_20261105/truth_cache'/f'{d}_{seed}.npz'));ix=np.flatnonzero(z['common'])
 z={k:(v[ix] if v.ndim and len(v)==len(z['common']) and k not in ['gene_keep'] else v) for k,v in z.items()};z['indices']=ix
 z['genes']=pd.read_csv(base/'genes.tsv',sep='\t',dtype=str).gene_id.to_numpy()
 cache=O/'single/reference'/f'{d}_{seed}.npz'
 if not cache.exists():
  from single_data import load
  m,x,_=load(d,seed,'fit');ref={c:np.average(x[j],axis=0,weights=m.n.iloc[j]) for c,j in m.groupby('context').indices.items()};cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,reference=np.stack([ref[str(c)] for c in z['context']]))
 z['cell_reference']=np.load(cache)['reference'];return z
def score(pred,z,d,seed,variant):
 p=pred[z['indices']].astype(float);y=z['truth'].astype(float);mu=z['control'].astype(float);dy=y-mu;dp=p-mu;dp[np.sqrt(np.mean(dp*dp,1))<1e-7]=0
 target=z['target'].astype(str);ctx=z['context'].astype(str);idx={g:i for i,g in enumerate(z['genes'])};lab=z['deg'];top=z['top100'];ref=z['cell_reference']
 out=pd.DataFrame(dict(dataset=d,split=seed,variant=variant,target=target,context=ctx,mse=np.mean((p-y)**2,1),rmse=np.sqrt(np.mean((p-y)**2,1)),pearson_delta=corr(p-mu,dy),systema_cell_pearson=corr(p-ref,y-ref),specific_pearson=corr(p-mu-z['shared'],dy-z['shared']),auprc_dg=[auprc(l,abs(a)) for l,a in zip(lab,dp)],deg_direction_accuracy=[np.mean(np.sign(a[l])==np.sign(b[l])) if l.any() else np.nan for l,a,b in zip(lab,dp,dy)],top100_mse=[np.mean((a[t]-b[t])**2) for a,b,t in zip(p,y,top)]))
 for c in sorted(set(ctx)):
  ii=np.flatnonzero(ctx==c);a=dp[ii];b=dy[ii];n=len(ii);mats={'l1':cdist(a,b,'cityblock'),'l2':cdist(a,b,'sqeuclidean'),'cosine':1-unit(a)@unit(b).T}
  for j,t in enumerate(target[ii]):
   g=idx.get(t)
   if g is not None:
    mats['l1'][j]-=abs(a[j,g]-b[:,g]);mats['l2'][j]-=(a[j,g]-b[:,g])**2
    mask=np.arange(len(z['genes']))!=g;mats['cosine'][j]=1-unit(a[j,mask])@unit(b[:,mask]).T
  for name,dist in mats.items():out.loc[ii,'pds_query_'+name]=pds(dist)
 out['mean_query_pds']=(out.pds_query_l2+out.pds_query_cosine)/2
 return out
def summaries(frame):
 metrics=[k for k in frame.select_dtypes(include='number') if k!='split']
 t=frame.groupby(['dataset','variant','target','split'])[metrics].mean().groupby(['dataset','variant','target']).mean()
 return t.reset_index(),t.groupby(['dataset','variant']).mean().reset_index()
def main():
 out=O/'single/rescore';out.mkdir(parents=True,exist_ok=True);receipts=[]
 for d in ['S3','S4','S5','S6','S7']:
  for seed in [101,202,303]:
   done=out/f'{d}_{seed}.csv'
   if done.exists():continue
   z=load_truth(d,seed);ff=[]
   for arm in ['aligned_raw','uniform_raw']:
    path=E/'specific_methods_20261105'/d/f'split{seed}'/(arm+'_predictions.npz');p=np.load(path);np.testing.assert_array_equal(p['target'][z['indices']].astype(str),z['target'].astype(str));np.testing.assert_array_equal(p['context'][z['indices']].astype(str),z['context'].astype(str));ff.append(score(p['prediction'],z,d,seed,arm));receipts.append(dict(path=str(path),sha256=sha(path)))
   pd.concat(ff).to_csv(done,index=False);print('SCORED',d,seed,flush=True);del z,ff;gc.collect()
 f=pd.concat([pd.read_csv(p,dtype={'target':str,'context':str}) for p in sorted(out.glob('S*.csv'))]);f.to_csv(out/'condition_metrics.csv.gz',index=False);t,s=summaries(f);t.to_csv(out/'target_metrics.csv',index=False);s.to_csv(out/'summary.csv',index=False)
 gold=pd.read_csv('@STUDY@/uc_single_perturbation/submission/figure2_code/figure_data/source_data/condition_metrics.csv',dtype={'target':str,'context':str});gold=gold[gold.method=='UC'].set_index(['dataset','split','target','context']);a=f[f.variant=='aligned_raw'].set_index(gold.index.names).reindex(gold.index)
 errors={k:float(np.nanmax(abs(a[k]-gold[k]))) for k in ['mse','pds_query_l1','pds_query_l2','pds_query_cosine','auprc_dg','systema_cell_pearson','pearson_delta']}
 assert max(errors.values())<1e-9,errors
 save(out/'COMPLETE.json',dict(status='PASS',conditions_per_arm=len(a),parity_max_errors=errors));save(out/'input_receipts.json',receipts);print(s.to_string(index=False),flush=True)
if __name__=='__main__':main()
