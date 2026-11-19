import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,hashlib,inspect,gc
import numpy as np,pandas as pd
from scipy import sparse,stats
from sklearn.metrics import average_precision_score
R=Path('@STUDY@/uc_fig2_benchmark_20260911');E=R/'experiments/uc_fig2_latest_within_context_20261105';O=E/'parameter_sensitivity_20261105';O.mkdir(exist_ok=True)
sys.path.insert(0,str(E/'src'));from model import Bank,response_subspace
land=np.load(E/'figures/paper_matched_merged_20261105/source_data/landscapes.npz')
D=['S3','S4','S5','S6','S7'];AL=[.001,.01,.1,1.,10.];SH=[(16,.25),(16,.5),(16,1.),(8,.25),(32,.25)]
configs=[dict(alpha=a,rank=r,tail=t) for a in AL for r,t in SH]
protocol=dict(scope='Retrospective mechanism sensitivity: split101, the five pre-existing illustrated contexts only; not full benchmark replacement',configs=configs,selection='Per-context validation: maximize mean of cosine and L2 PDS excluding full target panel, constrained to expression RMSE <=1.05 times original configuration; ties by shared-centred Pearson then smaller RMSE. DEG AP is measured, not optimized.',arms=['baseline','alpha_only','tail_only','rank_only','joint'],structure='Fixed 12 equal-prior kernel and unpenalized intercept; original response projection with numeric rank cap and residual-tail fraction exposed; unchanged nonnegative output. No test-selected parameters.',test='All five selections saved before loading any test truth; refit fit+validation; evaluate selected settings only',metric_note='PDS here uses full target-panel exclusion for both distances, unlike query-gene-only radar PDS. Specific Pearson uses equal-target fit-only mean, not cell-weighted Systema. DEG uses average precision, not nonlinear AUPRC.')
(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));saved=[];valrows=[];selections=[];diag=[]
def transform(y,targets,degs,rank,tail):
 src=inspect.getsource(response_subspace).replace('def response_subspace(', 'def custom(').replace('rank=min(16,','rank=min('+str(rank)+',').replace('target=h@intercept+.75*projected+.25*residual',f'target=h@intercept+{1-tail}*projected+{tail}*residual')
 import mean_model
 ns=dict(mean_model.__dict__);exec(src,ns);return ns['custom'](y,targets,degs)[0]
def cor(a,b):
 aa=a-a.mean(-1,keepdims=True);bb=b-b.mean(-1,keepdims=True);den=np.linalg.norm(aa,axis=-1)*np.linalg.norm(bb,axis=-1);return np.divide(np.sum(aa*bb,axis=-1),den,out=np.full(den.shape,np.nan),where=den>1e-12)
def unit(x):
 n=np.linalg.norm(x,axis=1,keepdims=True);return np.divide(x,n,out=np.zeros_like(x),where=n>1e-12)
def score(p,y,control,keep,shared,deg):
 dp=p-control;dy=y-control;pp=dp[:,keep];yy=dy[:,keep];n=len(y);ans=dict(rmse=float(np.sqrt(np.mean((p-y)**2,axis=1)).mean()),specific_pearson=float(np.nanmean(cor(dp-shared,dy-shared))))
 for label,dist in [('l2',np.maximum(np.sum(pp*pp,axis=1)[:,None]+np.sum(yy*yy,axis=1)[None,:]-2*pp@yy.T,0)),('cosine',1-unit(pp)@unit(yy).T)]:
  v=[]
  for i in range(n):
   other=np.arange(n)!=i;v.append(1-(np.sum(dist[i,other]<dist[i,i]-1e-12)+.5*np.sum(abs(dist[i,other]-dist[i,i])<=1e-12))/(n-1))
  ans['pds_'+label]=float(np.mean(v))
 ap=[average_precision_score(l,abs(x)) for l,x in zip(deg,dp) if 0<np.sum(l)<len(l)]
 ans['deg_ap']=float(np.mean(ap)) if ap else np.nan;ans['deg_valid']=len(ap)
 a=unit(yy)@unit(yy).T;b=unit(pp)@unit(pp).T;ix=np.triu_indices(n,1)
 ans.update(matrix_rmse=float(np.sqrt(np.mean((a[ix]-b[ix])**2))),matrix_spearman=float(stats.spearmanr(a[ix],b[ix]).statistic),pair_cosine=float(b[ix].mean()),energy_ratio=float(np.sum((pp-pp.mean(0))**2)/np.sum((yy-yy.mean(0))**2)))
 return ans
def features(bank,rows,queries,pre=None):
 feat=bank.kernel([r[0] for r in rows]) if pre is None else pre;ix=feat['ix'];f=feat['phi'][ix];K=f@f.T+np.diag(feat['rem'][ix]);qi=np.array([bank.index[t] for t in queries]);cross=feat['phi'][qi]@f.T
 kc=K-K.mean(0)[None,:]-K.mean(1)[:,None]+K.mean();qc=cross-K.mean(0)[None,:]-cross.mean(1)[:,None]+K.mean();ev,u=np.linalg.eigh(kc);ev=np.maximum(ev,0)
 return feat,ev,u,qc,feat['coverage'][qi]==0
def predict(rows,queries,control,cfg,cache,transforms):
 feat,ev,u,qc,missing=cache;key=(cfg['rank'],cfg['tail']);y=np.stack([r[1] for r in rows])
 if key not in transforms:transforms[key]=transform(y,[r[0] for r in rows],[r[2] for r in rows],*key)
 Y=transforms[key];w=(qc@u/(ev+cfg['alpha']))@u.T;delta=Y.mean(0)+w@(Y-Y.mean(0));delta[missing]=y.mean(0)
 return np.maximum(control+delta,0).astype(np.float32).astype(float)
for d in D:
 B=R/'data/formal'/d/'split101';c=str(land[d+'_context'][0]);arrays={};meta={}
 for role in ['fit','validation','controls_fit']:
  m=pd.read_csv(B/role/'metadata.tsv',sep='\t',dtype=str);ix=np.flatnonzero(m.context.to_numpy()==c);x=sparse.load_npz(B/role/'expression.npz').tocsr();arrays[role]=x[ix].astype(float);meta[role]=m.iloc[ix].reset_index(drop=True);del x
 xc=arrays['controls_fit'];ctrl=np.asarray(xc.mean(0)).ravel();cv=np.maximum(np.asarray(xc.power(2).mean(0)).ravel()-ctrl**2,0);cn=len(meta['controls_fit'])
 def group(x,m):
  out=[];labels=[]
  for t,ix in m.groupby('target',sort=True).indices.items():
   xx=x[ix];mu=np.asarray(xx.mean(0)).ravel();var=np.maximum(np.asarray(xx.power(2).mean(0)).ravel()-mu**2,0);se=np.sqrt(var/len(ix)+cv/cn);stat=np.divide(abs(mu-ctrl),se,out=np.zeros_like(mu),where=se>1e-12);top=np.argsort(-stat,kind='stable')[:100];out.append((t,mu-ctrl,top))
   v1=var/max(len(ix)-1,1);v0=cv/max(cn-1,1);s=v1+v0;tt=np.divide(abs(mu-ctrl),np.sqrt(s),out=np.zeros_like(mu),where=s>1e-20);tt[(s<=1e-20)&(abs(mu-ctrl)>1e-12)]=np.inf;df=np.divide(s*s,v1*v1/max(len(ix)-1,1)+v0*v0/max(cn-1,1),out=np.ones_like(mu),where=s>1e-20);p=2*stats.t.sf(tt,df);order=np.argsort(p);adj=np.empty(len(p));adj[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];labels.append((adj<.05)&(abs(mu-ctrl)>.1))
  return out,np.stack(labels)
 tr,_=group(arrays['fit'],meta['fit']);va,vd=group(arrays['validation'],meta['validation']);merged=sparse.vstack([arrays['fit'],arrays['validation']],format='csr');rg,_=group(merged,pd.concat([meta['fit'],meta['validation']],ignore_index=True));del merged,arrays;gc.collect()
 bank=Bank(d);genes=pd.read_csv(B/'genes.tsv',sep='\t').gene_id.astype(str).to_numpy();keep=~np.isin(genes,bank.targets.astype(str));shared=np.stack([r[1] for r in tr]).mean(0);queries=[r[0] for r in va];cache=features(bank,tr,queries);trans={};yy=np.stack([r[1]+ctrl for r in va]);base=json.loads((E/'runs'/d/'split101/UC/selection.json').read_text())['chosen']['config'];basecfg=dict(alpha=base['alpha'],rank=16,tail=.25 if base['response']=='deg_subspace' else 1.)
 for cfg in configs:
  p=predict(tr,queries,ctrl,cfg,cache,trans);s=score(p,yy,ctrl,keep,shared,vd);valrows.append(dict(dataset=d,context=c,n_validation=len(va),**cfg,**s))
 v=pd.DataFrame(valrows);v=v[v.dataset==d];bas=v[(v.alpha==basecfg['alpha'])&(v['rank']==16)&(v['tail']==basecfg['tail'])].iloc[0];eligible=v[v.rmse<=1.05*bas.rmse].copy();eligible['objective']=(eligible.pds_l2+eligible.pds_cosine)/2
 chosen={}
 for arm in ['baseline','alpha_only','tail_only','rank_only','joint']:
  g=eligible.copy()
  if arm in ['baseline','tail_only','rank_only']:g=g[g.alpha==basecfg['alpha']]
  if arm in ['baseline','alpha_only','tail_only']:g=g[g['rank']==16]
  if arm in ['baseline','alpha_only','rank_only']:g=g[g['tail']==basecfg['tail']]
  row=g.sort_values(['objective','specific_pearson','rmse','alpha','rank','tail'],ascending=[False,False,True,True,True,True]).iloc[0];chosen[arm]={k:float(row[k]) if k!='rank' else int(row[k]) for k in ['alpha','rank','tail']};selections.append(dict(dataset=d,context=c,arm=arm,**chosen[arm],validation_n=len(va),validation_rmse=float(row.rmse),validation_pds=float(row.objective)))
 saved.append((d,c,bank,tr,rg,ctrl,keep,shared,chosen));diag.append(dict(dataset=d,context=c,training_targets=len(tr),validation_targets=len(va),positive_centered_eigen_median=float(np.median(cache[1][cache[1]>1e-10])),eigen_max=float(cache[1].max()),original_alpha=basecfg['alpha'],original_effective_df=float(np.sum(cache[1]/(cache[1]+basecfg['alpha'])))))
 pd.DataFrame(valrows).to_csv(O/'validation_grid.csv',index=False);print('VALIDATION_DONE',d,chosen,flush=True)
pd.DataFrame(selections).to_csv(O/'SELECTIONS_FROZEN.csv',index=False);pd.DataFrame(diag).to_csv(O/'kernel_spectrum.csv',index=False);seal=hashlib.sha256((O/'SELECTIONS_FROZEN.csv').read_bytes()).hexdigest();results=[];parity=[]
for d,c,bank,tr,rg,ctrl,keep,shared,chosen in saved:
 z=np.load(E/'specificity_extension_20261105/truth_cache'/f'{d}_101.npz');ix=np.flatnonzero(z['common']&(z['context'].astype(str)==c));queries=z['target'][ix].astype(str).tolist();np.testing.assert_array_equal(keep,z['gene_keep']);yy=z['truth'][ix].astype(float);np.testing.assert_allclose(ctrl,z['control'][ix][0],atol=1e-12);cache=features(bank,rg,queries);trans={}
 for arm,cfg in chosen.items():
  p=predict(rg,queries,ctrl,cfg,cache,trans)
  if arm=='baseline':
   old=np.load(E/'runs'/d/'split101/UC/predictions_sealed.npz')['prediction'][ix];err=float(np.max(abs(p-old)));assert err<2e-6;parity.append(dict(dataset=d,max_prediction_error=err))
  s=score(p,yy,ctrl,keep,shared,z['deg'][ix]);results.append(dict(dataset=d,context=c,arm=arm,n_test=len(ix),**cfg,**s));np.savez_compressed(O/f'{d}_{arm}_predictions.npz',prediction=p,target=np.array(queries))
 pd.DataFrame(results).to_csv(O/'test_selected_only.csv',index=False);print('TEST_DONE',d,flush=True)
assert seal==hashlib.sha256((O/'SELECTIONS_FROZEN.csv').read_bytes()).hexdigest()
(O/'COMPLETE.json').write_text(json.dumps(dict(status='PASS',validation_candidates=125,selected_test_rows=25,selection_sha256=seal,baseline_reproduction=parity,model_structure_unchanged=True,formal_benchmark_replaced=False),indent=2))
print('COMPLETE',flush=True)
