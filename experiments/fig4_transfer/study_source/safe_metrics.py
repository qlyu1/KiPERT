from evaluate_new import rowcorr,OLDMETHODS,METRICS,OLD,readcsv
from common import *
from scipy.spatial.distance import cdist

def small_rowcorr(a,b):
 result=np.empty(len(a))
 for lo in range(0,len(a),64):result[lo:lo+64]=rowcorr(a[lo:lo+64],b[lo:lo+64])
 return result

def norm(x):return np.sqrt(np.einsum('ij,ij->i',x,x))

def evaluate_arrays(real,pred,cr,cp,reference_values=None):
 dr=real-cr;dp=pred-cp;n=len(real);out=dict(pearson_delta=small_rowcorr(dp,dr),**{k:np.empty(n) for k in ['mse','mae','mse_delta','mae_delta']})
 for lo in range(0,n,64):
  sl=slice(lo,lo+64);err=pred[sl]-real[sl];de=dp[sl]-dr[sl];out['mse'][sl]=np.mean(err*err,1);out['mae'][sl]=np.mean(abs(err),1);out['mse_delta'][sl]=np.mean(de*de,1);out['mae_delta'][sl]=np.mean(abs(de),1)
 del err,de
 distances=[('l1','cityblock')] if reference_values is not None else [('l1','cityblock'),('l2','euclidean'),('cosine','cosine')]
 ranks={k:np.empty(n,int) for k,_ in distances};nearest=np.empty(n,int)
 real_norm2=np.einsum('ij,ij->i',dr,dr);pred_norm2=np.einsum('ij,ij->i',dp,dp)
 for lo in range(0,n,64):
  hi=min(lo+64,n);correct=np.arange(lo,hi)
  cross=dp[lo:hi]@dr.T if reference_values is None else None
  for key,distance in distances:
   if key=='l1':dist=cdist(dp[lo:hi],dr,distance)
   elif key=='l2':dist=np.maximum(pred_norm2[lo:hi,None]+real_norm2[None,:]-2*cross,0)
   else:
    den=np.sqrt(pred_norm2[lo:hi,None]*real_norm2[None,:]);dist=1-np.divide(cross,den,out=np.zeros_like(cross),where=den>0)
   dist[~np.isfinite(dist)]=1.;order=np.argsort(dist,axis=1);ranks[key][lo:hi]=np.argmax(order==correct[:,None],axis=1)
   if key=='l1':nearest[lo:hi]=order[:,0]
 parity={}
 if reference_values is None:
  sample=np.unique(np.linspace(0,n-1,32,dtype=int))
  for key,distance in [('l2','euclidean'),('cosine','cosine')]:
   dist=cdist(dp[sample],dr,distance);dist[~np.isfinite(dist)]=1.;direct=np.argmax(np.argsort(dist,axis=1)==sample[:,None],axis=1);difference=int(np.max(abs(direct-ranks[key][sample])));assert difference==0,(key,difference);parity[key+'_sample_max_rank_difference']=difference
 for key,r in ranks.items():out['discrimination_score_'+key]=1-r/n
 if reference_values is not None:
  for metric,v in out.items():np.testing.assert_allclose(v,reference_values[metric],rtol=0,atol=1e-10,equal_nan=True)
  out.update(reference_values)
 counts=np.bincount(nearest,minlength=n);freq=counts[counts>0]/n;total=float(np.einsum('ij,ij->',dp,dp)/dp.size);ratio=norm(dp)/np.maximum(norm(dr),1e-20);common=float(np.mean(dp.mean(0)**2));dr-=dr.mean(0);dp-=dp.mean(0);distinct=float(np.einsum('ij,ij->',dp,dp)/dp.size);true_distinct=float(np.einsum('ij,ij->',dr,dr)/dr.size);pcc=small_rowcorr(dp,dr)
 diag=dict(response_energy=total,distinct_response_energy=distinct,distinct_energy_ratio_to_truth=distinct/max(true_distinct,1e-20),common_energy_fraction=common/max(total,1e-20),median_response_norm_ratio=float(np.median(ratio)),q10_response_norm_ratio=float(np.quantile(ratio,.1)),q90_response_norm_ratio=float(np.quantile(ratio,.9)),centered_pearson_delta=float(np.nanmean(pcc)) if np.isfinite(pcc).any() else float('nan'),l1_top1=float(np.mean(nearest==np.arange(n))),l1_top10=float(np.mean(ranks['l1']<10)),l1_unique_nearest_matches=int((counts>0).sum()),l1_largest_hub_fraction=float(counts.max()/n),l1_effective_nearest_matches=float(np.exp(-np.sum(freq*np.log(freq)))))
 return out,diag,dict(implementation='64-row blocks: exact cityblock cdist; algebraic squared L2/cosine with 32-query direct-cdist rank parity; baseline vector scores independently verified',**parity)
