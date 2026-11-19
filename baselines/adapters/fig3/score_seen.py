"""Final scorer, executed only after every registered model has sealed predictions."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
from scipy import sparse,stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('@STUDY@/uc_seen012_benchmark_20261005');O=R/'reports/final';O.mkdir(parents=True,exist_ok=True)
jobs=json.loads((R/'manifests/run_matrix.json').read_text())['jobs'];rows=[];cache={}
def load(role,base):
 x=sparse.load_npz(base/role/'expression.npz');m=pd.read_csv(base/role/'metadata.tsv',sep='\t',dtype=str)
 return {(t,c):np.asarray(x[ix].mean(0)).ravel().astype(float) for (t,c),ix in m.groupby(['target','context'],sort=True).indices.items()}
def moments(role,base):
 x=sparse.load_npz(base/role/'expression.npz').astype(float);m=pd.read_csv(base/role/'metadata.tsv',sep='\t',dtype=str);out={}
 for key,ix in m.groupby(['target','context'],sort=True).indices.items():
  xx=x[ix];mu=np.asarray(xx.mean(0)).ravel();n=len(ix);var=np.maximum(np.asarray(xx.power(2).mean(0)).ravel()-mu**2,0)*n/max(n-1,1);out[key]=(mu,var,n)
 return out
for j in jobs:
 p=R/f"runs/formal/{j['dataset']}/{j['arm']}/fold{j['fold']}/seed{j['seed']}/{j['method']}"
 z=json.loads((p/'result.json').read_text());assert z['status'] in ['PREDICTIONS_SEALED','BASELINES_SEALED'],j
release=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_registered_jobs=len(jobs),status='ALL_PREDICTIONS_SEALED',protocol_sha256=hashlib.sha256((R/'protocol/PROTOCOL.json').read_bytes()).hexdigest())
(R/'manifests/SCORING_RELEASE.json').write_text(json.dumps(release,indent=2))
for j in jobs:
 key=(j['dataset'],j['fold']);base=R/f"data/formal/{j['dataset']}/seen0/fold{j['fold']}"
 if key not in cache:
  truth=load('evaluation_only',base);ctrl={c:v for (t,c),v in load('controls_score',base).items()};train=load('fit',base)
  mean={c:np.mean([v for (t,cc),v in train.items() if cc==c and '+' in t],axis=0) for c in ctrl}
  tmom=moments('evaluation_only',base);cmom={c:v for (t,c),v in moments('controls_score',base).items()};deg={}
  for k,(mu,var,n) in tmom.items():
   cc,vv,nn=cmom[k[1]];pv=stats.ttest_ind_from_stats(mu,np.sqrt(var),n,cc,np.sqrt(vv),nn,equal_var=False).pvalue;pv=np.nan_to_num(pv,nan=1.);deg[k]=np.lexsort((np.arange(len(mu)),-np.abs(mu-cc),pv))[:100]
  cache[key]=(truth,ctrl,mean,deg)
 truth,ctrl,mean,deg=cache[key]
 p=R/f"runs/formal/{j['dataset']}/{j['arm']}/fold{j['fold']}/seed{j['seed']}/{j['method']}";result=json.loads((p/'result.json').read_text())
 files={k:p/f'{k}.npz' for k in result['baselines']} if j['method']=='baselines' else {j['method']:p/'predictions_sealed.npz'}
 for method,file in files.items():
  digest=hashlib.sha256(file.read_bytes()).hexdigest();expected=result['baselines'][method]['sha256'] if j['method']=='baselines' else result['prediction_sha256'];assert digest==expected
  z=np.load(file);ts=z['target'].astype(str);cs=z['context'].astype(str);pred=np.maximum(z['prediction'].astype(float),0);keys=list(zip(ts,cs));assert set(keys)==set(truth) and len(keys)==len(truth);assert np.isfinite(pred).all()
  y=np.stack([truth[k] for k in keys]);cm=np.stack([ctrl[c] for c in cs]);tm=np.stack([mean[c] for c in cs]);mse=np.mean((pred-y)**2,axis=1)
  aa=pred-cm;bb=y-cm;aa-=aa.mean(1,keepdims=True);bb-=bb.mean(1,keepdims=True);den=np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1);pcc=np.divide(np.sum(aa*bb,axis=1),den,out=np.full(len(den),np.nan),where=den>1e-15)
  mrr=np.full(len(ts),np.nan)
  for c in sorted(set(cs)):
   ix=np.flatnonzero(cs==c);a0=pred[ix]-tm[ix];b0=y[ix]-tm[ix];den=np.linalg.norm(a0,axis=1)[:,None]*np.linalg.norm(b0,axis=1)[None,:];sim=np.divide(a0@b0.T,den,out=np.zeros_like(den),where=den>1e-15)
   for k,ii in enumerate(ix):
    score=sim[k,k];better=int(sum(sim[k]>score+1e-12));ties=int(sum(abs(sim[k]-score)<=1e-12));mrr[ii]=float(np.mean(1/np.arange(better+1,better+ties+1)))
  for i,(t,c) in enumerate(keys):
   ix=deg[t,c];de_mse=float(np.mean((pred[i,ix]-y[i,ix])**2));sg=float(np.mean(np.sign(pred[i,ix]-cm[i,ix])==np.sign(y[i,ix]-cm[i,ix])))
   rows.append(dict(dataset=j['dataset'],fold=j['fold'],arm=j['arm'],seed=j['seed'],model=method,target=t,context=c,mse=float(mse[i]),pcc_delta=float(pcc[i]) if np.isfinite(pcc[i]) else None,specific_mrr=float(mrr[i]),posthoc_deg100_mse=de_mse,posthoc_deg100_sign=sg))
df=pd.DataFrame(rows);pair=df.groupby(['dataset','fold','arm','seed','model','target'],as_index=False)[['mse','pcc_delta','specific_mrr']].mean()
macro=pair.groupby(['dataset','fold','arm','seed','model'],as_index=False)[['mse','pcc_delta','specific_mrr']].mean()
den=macro[macro.model.eq('no_change')][['dataset','fold','arm','mse']].rename(columns={'mse':'no_change_mse'})
macro=macro.merge(den,on=['dataset','fold','arm'],validate='many_to_one');macro['nmse']=macro.mse/macro.no_change_mse
add=macro[macro.model.eq('available_evidence_additive')][['dataset','fold','arm','mse']].rename(columns={'mse':'additive_mse'})
macro=macro.merge(add,on=['dataset','fold','arm'],validate='many_to_one');macro['gain_vs_additive']=macro.additive_mse-macro.mse
summary=macro.groupby(['dataset','arm','model'],as_index=False)[['mse','nmse','pcc_delta','specific_mrr','gain_vs_additive']].mean()
seedmean=pair.groupby(['dataset','fold','arm','model','target'],as_index=False).mse.mean();wide=seedmean.pivot(index=['dataset','fold','model','target'],columns='arm',values='mse').reset_index();wide['gain01']=wide.seen0-wide.seen1;wide['gain12']=wide.seen1-wide.seen2
for name,frame in [('query_metrics',df),('pair_metrics',pair),('fold_metrics',macro),('summary_metrics',summary),('paired_information_gains',wide)]:
 (O/f'{name}.json').write_text(frame.to_json(orient='records',indent=2))
models=['uc','gears','scouter','txpert','stamp','gperturb','available_evidence_additive','multi_hot_ridge'];names=dict(D1='Norman',D2='Replogle exp6',D3='Wessels L4',D4='Dixit K562')
fig,axs=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
for ax,d in zip(axs.flat,names):
 for model in models:
  zz=summary[summary.dataset.eq(d)&summary.model.eq(model)].set_index('arm');ax.plot([0,1,2],[zz.loc[f'seen{i}','nmse'] for i in range(3)],marker='o',label=model)
 ax.axhline(1,color='gray',linestyle='--',linewidth=.8);ax.set(title=names[d],xlabel='Seen singleton targets',ylabel='MSE / no-change MSE',xticks=[0,1,2])
axs.flat[0].legend(fontsize=7,ncol=2);fig.savefig(O/'seen012_error_curves.png',dpi=180);fig.savefig(O/'seen012_error_curves.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,5),constrained_layout=True)
for i,ax in enumerate(axs):
 zz=summary[summary.arm.eq(f'seen{i}')].pivot(index='model',columns='dataset',values='nmse').reindex(index=models,columns=names);im=ax.imshow(zz.to_numpy(),aspect='auto',cmap='viridis_r');ax.set(title=f'seen{i}',xticks=range(4),xticklabels=list(names.values()),yticks=range(len(models)),yticklabels=models);ax.tick_params(axis='x',labelrotation=45)
 fig.colorbar(im,ax=ax,label='nMSE')
fig.savefig(O/'model_dataset_heatmap.png',dpi=180);fig.savefig(O/'model_dataset_heatmap.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
dedup=wide.groupby(['dataset','model','target'],as_index=False)[['gain01','gain12']].mean()
for ax,col,title in zip(axs,['gain01','gain12'],['First singleton information gain','Second singleton information gain']):
 ax.boxplot([dedup.loc[dedup.model.eq(m),col].dropna() for m in models],showfliers=False);ax.set_xticks(range(1,len(models)+1),models);ax.axhline(0,color='gray',linestyle='--');ax.set(title=title,ylabel='Paired MSE reduction');ax.tick_params(axis='x',labelrotation=45)
fig.savefig(O/'paired_information_gains.png',dpi=180);fig.savefig(O/'paired_information_gains.pdf');plt.close(fig)
lines=['双扰动 seen0/seen1/seen2 benchmark 最终报告','所有注册任务已封存后统一评分。seen1 每份划分仅保留 L 侧单扰动。','5 份重复划分；神经模型 3 个初始化种子；UC 和基线为确定性计算。','主指标为 5000 基因轴、非负预测、先背景后组合宏平均 MSE；nMSE 分母为相同数据集/划分的 No-change MSE。','以下均值用于回顾性描述；重复细胞和初始化不视为独立生物重复。','']
for d in names:
 lines.append(names[d]+' nMSE（越低越好）');lines.append(summary[summary.dataset.eq(d)].pivot(index='model',columns='arm',values='nmse').round(6).to_string());lines.append('')
lines+=['附：paired_information_gains.json 记录相同测试组合的 seen0→seen1、seen1→seen2 收益；覆盖全部注册预测，无共同成功子集筛选。','query_metrics.json 附事后 DEG Top100 MSE 和方向一致率，DEG 在冻结5000基因轴上用测试细胞及 C_score Welch 检验排序，仅评分器可读。','本轮不包含需要另行封存单扰动真值的非加和残差补充分析。']
(O/'最终报告.txt').write_text('\n'.join(lines));(O/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',jobs=len(jobs),query_metric_rows=len(rows),generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2));print('FINAL_REPORT_COMPLETE',flush=True)
