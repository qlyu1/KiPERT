from engine import *
import csv

METRICS=['pearson_delta','mse','mae','mse_delta','mae_delta','discrimination_score_l1','discrimination_score_l2','discrimination_score_cosine']
OLDMETHODS=['UC','UC_OT','STATE-HVG','STATE-SE','source_measured_mean','zero_response']

def readcsv(p):return list(csv.DictReader(open(p)))
def writecsv(p,rows):
 with open(p,'w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def rowcorr(a,b):
 a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True);den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);return np.divide((a*b).sum(1),den,out=np.full(len(a),np.nan),where=den>0)

def evaluate_arrays(real,pred,cr,cp):
 delta_real=real-cr;delta_pred=pred-cp;err=pred-real;de=delta_pred-delta_real;n=len(real)
 out=dict(pearson_delta=rowcorr(delta_pred,delta_real),mse=np.mean(err*err,1),mae=np.mean(abs(err),1),mse_delta=np.mean(de*de,1),mae_delta=np.mean(abs(de),1))
 dist=cdist(delta_pred,delta_real,'cityblock');order=np.argsort(dist,axis=1);rank=np.argmax(order==np.arange(n)[:,None],axis=1);out['discrimination_score_l1']=1-rank/n;nearest=order[:,0];counts=np.bincount(nearest,minlength=n);freq=counts[counts>0]/n;hub=dict(l1_top1=float(np.mean(nearest==np.arange(n))),l1_top10=float(np.mean(rank<10)),l1_unique_nearest_matches=int((counts>0).sum()),l1_largest_hub_fraction=float(counts.max()/n),l1_effective_nearest_matches=float(np.exp(-np.sum(freq*np.log(freq)))))
 del dist,order;cross=delta_pred@delta_real.T;ds=np.maximum((delta_pred*delta_pred).sum(1)[:,None]+(delta_real*delta_real).sum(1)[None,:]-2*cross,0);order=np.argsort(ds,axis=1);out['discrimination_score_l2']=1-np.argmax(order==np.arange(n)[:,None],axis=1)/n;den=np.linalg.norm(delta_pred,axis=1)[:,None]*np.linalg.norm(delta_real,axis=1)[None,:];dist=1-np.divide(cross,den,out=np.zeros_like(cross),where=den>0);order=np.argsort(dist,axis=1);out['discrimination_score_cosine']=1-np.argmax(order==np.arange(n)[:,None],axis=1)/n
 # Direct numerical parity on real vectors for the BLAS-based L2/cosine computation.
 sample=np.unique(np.linspace(0,n-1,32,dtype=int));parity={}
 for metric in ['euclidean','cosine']:
  direct=cdist(delta_pred[sample],delta_real,metric);direct[~np.isfinite(direct)]=1.;ref=1-np.argmax(np.argsort(direct,axis=1)==sample[:,None],axis=1)/n;key='discrimination_score_'+('l2' if metric=='euclidean' else 'cosine');err=float(np.max(abs(ref-out[key][sample])));assert err<=1/n+1e-12;parity[metric]=err
 center_real=delta_real-delta_real.mean(0);center_pred=delta_pred-delta_pred.mean(0);total=float(np.mean(delta_pred**2));distinct=float(np.mean(center_pred**2));true_distinct=float(np.mean(center_real**2));norm_real=np.linalg.norm(delta_real,axis=1);norm_pred=np.linalg.norm(delta_pred,axis=1);ratio=norm_pred/np.maximum(norm_real,1e-20);center_corr=rowcorr(center_pred,center_real)
 diag=dict(response_energy=total,distinct_response_energy=distinct,distinct_energy_ratio_to_truth=distinct/max(true_distinct,1e-20),common_energy_fraction=float(np.mean(delta_pred.mean(0)**2)/max(total,1e-20)),median_response_norm_ratio=float(np.median(ratio)),q10_response_norm_ratio=float(np.quantile(ratio,.1)),q90_response_norm_ratio=float(np.quantile(ratio,.9)),centered_pearson_delta=float(np.nanmean(center_corr)) if np.isfinite(center_corr).any() else float('nan'),**hub)
 return out,diag,parity

def comparison_pairs(methods):
 pairs=[]
 for prior in [8,12]:
  for point in ['balanced','specificity']:
   u=f'UC_SP{prior}_{point}';o=f'UC_OT_SP{prior}_{point}';pairs.extend([(u,'UC'),(o,'UC_OT'),(o,u),(u,'STATE-HVG'),(o,'STATE-HVG'),(u,'source_measured_mean'),(o,'source_measured_mean')])
 for m in ['UC','UC_OT']:
  pairs.append((m+'_12prior_only',m));pairs.append((m+'_SP12_fixed8',m+'_SP8_balanced'))
  for point in ['balanced','specificity']:pairs.append((f'{m}_SP12_{point}',f'{m}_SP8_{point}'))
 for name in ['UC_no_amplitude','UC_measured_only','UC_residual_retained','UC_uniform_sources','UC_OT_no_amplitude']:pairs.append((name,'UC_OT' if name.startswith('UC_OT') else 'UC'))
 return list(dict.fromkeys(pairs))

def paired_intervals(target,allvals):
 descriptions=[];columns=[]
 for a,b in comparison_pairs(allvals):
  for metric in ['pearson_delta','mse_delta','discrimination_score_l1','discrimination_score_l2','discrimination_score_cosine']:
   descriptions.append(dict(target_context=target,method_a=a,method_b=b,metric=metric));columns.append(allvals[a][metric]-allvals[b][metric])
 D=np.stack(columns,axis=1);groups={}
 for j in range(D.shape[1]):groups.setdefault(np.isfinite(D[:,j]).tobytes(),[]).append(j)
 out=[None]*len(descriptions)
 for key,js in groups.items():
  mask=np.frombuffer(key,dtype=bool);d=D[mask][:,js];n=len(d);rng=np.random.default_rng(seed_for(target+'_paired_'+str(n)));W=np.empty((2000,n))
  for i in range(2000):W[i]=np.bincount(rng.integers(0,n,n),minlength=n)
  draws=W@d/n
  for k,j in enumerate(js):out[j]=dict(**descriptions[j],n=n,difference_a_minus_b=float(d[:,k].mean()),ci95_low=float(np.quantile(draws[:,k],.025)),ci95_high=float(np.quantile(draws[:,k],.975)))
 return out

def main():
 assert (ROOT/'ALL_PREDICTIONS_COMPLETE.json').exists();oldrows=readcsv(OLD/'results/per_target_metrics.csv');perrows=[];summaries=[];diagnostics=[];paired=[];sensitivity=[];checks={};start=time.time()
 for target in ['rpe1','jurkat']:
  out=R/target;seal=json.load(open(out/'PREDICTIONS_SEALED.json'));assert seal['protocol_sha256']==sha(ROOT/'protocol.json')
  for name,h in seal['prediction_sha256'].items():assert sha(out/name)==h
  for name,h in seal['code_sha256'].items():assert sha(ROOT/'src'/name)==h
  a=np.load(out/'axes.npz');q=a['targets'];G=a['genes'];supported=a['source_coverage'].any(0);truth=np.load(OLD/'evaluation_only'/target/'common_truth.npz');np.testing.assert_array_equal(q,truth['targets']);np.testing.assert_array_equal(G,truth['genes']);real=truth['mean'];cr=truth['control'];methods=OLDMETHODS+[p.name.removesuffix('_mean.npy') for p in sorted(out.glob('*_mean.npy'))];allvals={}
  for method in methods:
   if method.startswith('STATE-'):
    z=np.load(OLD/'evaluation_only'/target/(method.split('-')[1]+'_pred_bulk.npz'));mask=z['targets']!='non-targeting';np.testing.assert_array_equal(q,z['targets'][mask]);pred=z['mean'][mask];cp=z['mean'][~mask][0]
   else:pred=np.load((OLD/'results'/target if method in OLDMETHODS else out)/(method+'_mean.npy'));cp=cr
   vals,diag,parity=evaluate_arrays(real,pred,cr,cp);allvals[method]=vals;diagnostics.append(dict(target_context=target,method=method,**diag));checks[target+'/'+method]=parity
   if method in OLDMETHODS:
    old={r['perturbation']:r for r in oldrows if r['target_context']==target and r['method']==method}
    for metric in METRICS:
     expected=np.array([float(old[g][metric]) for g in q]);tol=1/len(q)+1e-12 if metric.startswith('discrimination') else 1e-10;np.testing.assert_allclose(vals[metric],expected,rtol=0,atol=tol,equal_nan=True)
   for i,g in enumerate(q):perrows.append(dict(target_context=target,method=method,perturbation=g,n_cells=int(truth['n'][i]),source_supported=bool(supported[i]),n_sources_observed=int(a['source_coverage'][:,i].sum()),**{k:float(vals[k][i]) for k in METRICS}))
   for cohort,mask in [('all_official_targets',np.ones(len(q),bool)),('source_supported',supported),('not_source_supported',~supported)]:
    for metric,v in vals.items():
     valid=mask&np.isfinite(v);summaries.append(dict(target_context=target,method=method,cohort=cohort,metric=metric,n_total=int(mask.sum()),n_finite=int(valid.sum()),mean=float(v[valid].mean()) if valid.any() else float('nan')))
   dr=real-cr;dp=pred-cp;er=[];pc=[]
   for i,g in enumerate(q):
    keep=G!=g;er.append(float(np.mean((dr[i,keep]-dp[i,keep])**2)));pc.append(rowcorr(dp[i:i+1,keep],dr[i:i+1,keep])[0])
   sensitivity.append(dict(target_context=target,method=method,n_target_gene_in_output=int(np.isin(q,G).sum()),pearson_delta=float(np.nanmean(pc)) if np.isfinite(pc).any() else float('nan'),mse_delta=float(np.mean(er))))
   log(stage='EVALUATED',target=target,method=method,pearson_delta=float(np.nanmean(vals['pearson_delta'])) if np.isfinite(vals['pearson_delta']).any() else None,mse_delta=float(vals['mse_delta'].mean()),l1=float(vals['discrimination_score_l1'].mean()),l2=float(vals['discrimination_score_l2'].mean()),cosine=float(vals['discrimination_score_cosine'].mean()))
  paired.extend(paired_intervals(target,allvals));writecsv(R/(target+'_summary.csv'),[x for x in summaries if x['target_context']==target]);log(stage='FOLD_EVALUATED',target=target)
 writecsv(R/'per_target_metrics.csv',perrows);writecsv(R/'comparison_long.csv',summaries);writecsv(R/'response_geometry_diagnostics.csv',diagnostics);writecsv(R/'paired_comparisons.csv',paired);writecsv(R/'target_gene_exclusion.csv',sensitivity);save(R/'METRIC_PARITY.json',checks);save(R/'EVALUATION_COMPLETE.json',dict(status='COMPLETE',utc=now(),elapsed_seconds=time.time()-start,n_methods=len(methods),files_sha256={p.name:sha(p) for p in R.glob('*.csv')}));log(stage='EVALUATION_COMPLETE',seconds=time.time()-start,n_methods=len(methods))
if __name__=='__main__':main()
