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
