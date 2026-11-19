"""Post-hoc evaluation of frozen predictions; no training or model selection."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import json,hashlib,gc,zipfile,tempfile
import numpy as np,pandas as pd
from scipy import sparse,stats
R=Path('@STUDY@/uc_seen012_benchmark_20261005')
E=R/'experiments/kipert_double_20261106'
O=E/'literature_panels_20261106';(O/'data').mkdir(parents=True,exist_ok=True)
J=json.loads((R/'manifests/run_matrix.json').read_text())['jobs']
rows=[];inventory=[];hashes=[];cases=[];hitrows=[];gimanifest=[]
if (O/'inventory.json').exists():
 inventory=json.loads((O/'inventory.json').read_text());rows=pd.read_csv(O/'data/query_metrics.csv.gz').to_dict('records');hitrows=pd.read_csv(O/'data/retrieval.csv.gz').to_dict('records');hashes=json.loads((O/'prediction_hashes.json').read_text());cases=json.loads((O/'case_selection.json').read_text())
done={(r['dataset'],r['fold']) for r in inventory}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(role,base):
 # Disk-backed CSR avoids doubling a large NPZ's resident memory under 2GB cgroup.
 with tempfile.TemporaryDirectory(prefix='lit-csr-',dir=str(O)) as tmp:
  with zipfile.ZipFile(base/role/'expression.npz') as zz:
   for name in ['data.npy','indices.npy','indptr.npy','shape.npy','format.npy']:zz.extract(name,tmp)
  tmp=Path(tmp);fmt=np.load(tmp/'format.npy').item();assert fmt==b'csr'
  x=sparse.csr_matrix((np.load(tmp/'data.npy',mmap_mode='r'),np.load(tmp/'indices.npy',mmap_mode='r'),np.load(tmp/'indptr.npy',mmap_mode='r')),shape=tuple(np.load(tmp/'shape.npy')),copy=False)
 m=pd.read_csv(base/role/'metadata.tsv',sep='\t',dtype=str)
 out={}
 for key,ix in m.groupby(['target','context'],sort=True).indices.items():
  a=x[ix];mu=np.asarray(a.mean(0)).ravel().astype(float);n=len(ix);s=np.zeros(x.shape[1]);ss=np.zeros(x.shape[1])
  for start in range(0,n,256):
   aa=a[start:start+256].astype(float);s+=np.asarray(aa.sum(0)).ravel();ss+=np.asarray(aa.power(2).sum(0)).ravel()
  v=np.maximum(ss-s*s/n,0)/max(n-1,1)
  out[key]=(mu,v,n,s,ss)
 return x,m,out
def pear(a,b):
 a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True);den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
 return np.divide((a*b).sum(1),den,out=np.full(len(a),np.nan),where=den>1e-14)
def r2(p,y,w=None,clip=False):
 if w is None:w=np.ones_like(y)
 den=(w*(y-(w*y).sum(1,keepdims=True)/w.sum(1,keepdims=True))**2).sum(1)
 r=1-np.divide((w*(p-y)**2).sum(1),den,out=np.full(len(p),np.nan),where=den>1e-14)
 return np.maximum(r,-1) if clip else r
for ds in ['D1','D2','D3','D4']:
 for fold in range(5):
  if (ds,fold) in done:continue
  base=R/f'data/formal/{ds}/seen0/fold{fold}'
  x,m,truth=read('evaluation_only',base);cx,cm,ctrl=read('controls_score',base)
  keys=sorted(truth);ts=np.array([k[0] for k in keys]);cs=np.array([k[1] for k in keys]);y=np.stack([truth[k][0] for k in keys]);c=np.stack([ctrl[('control',k[1])][0] if ('control',k[1]) in ctrl else next(v[0] for kk,v in ctrl.items() if kk[1]==k[1]) for k in keys]);dy=y-c
  cdict={cc:next(v for kk,v in ctrl.items() if kk[1]==cc) for cc in set(cs)}
  genes=pd.read_csv(base/'genes.tsv',sep='\t',dtype=str).fillna('');top=np.argsort(-np.mean([v[0] for v in cdict.values()],axis=0),kind='stable')[:1000]
  # GEARS SI Note 8: rank by absolute expression change, not P value.
  de20=np.argsort(-abs(dy),axis=1,kind='stable')[:,:20]
  de100=[]
  for k in keys:
   a=truth[k];b=cdict[k[1]];pv=stats.ttest_ind_from_stats(a[0],np.sqrt(a[1]),a[2],b[0],np.sqrt(b[1]),b[2],equal_var=False).pvalue
   de100.append(np.lexsort((np.arange(len(pv)),-abs(a[0]-b[0]),np.nan_to_num(pv,nan=1)))[:100])
  de100=np.array(de100);ii=np.arange(len(keys))[:,None]
  # Retain only the chosen raw-cell example; release large matrices before fit data.
  if fold==0:
   case=int(np.argsort(np.linalg.norm(dy,axis=1),kind='stable')[len(keys)//2]);ct,casecontext=keys[case];idx=m.index[(m.target==ct)&(m.context==casecontext)].to_numpy();case_cells=x[idx][:,de20[case]].toarray()-c[case,de20[case]]
  del x,cx;gc.collect()
  # Pool available benchmark perturbation cells once, independent of model/arm.
  _,_,fulltrain=read('fit',R/f'data/formal/{ds}/seen2/fold{fold}');_,_,valid=read('validation',base)
  pool={**fulltrain,**valid,**truth};weights=[]
  for k in keys:
   other=[v for kk,v in pool.items() if kk[1]==k[1] and kk[0]!=k[0] and kk[0]!='control'];n=sum(v[2] for v in other);s=sum(v[3] for v in other);ss=sum(v[4] for v in other);mu=s/n;var=np.maximum(ss-s*s/n,0)/(n-1);a=truth[k]
   # scanpy t-test_overestim_var uses target sample size for both variance terms.
   den=np.sqrt((a[1]+var)/a[2]);t=np.divide(a[0]-mu,den,out=np.zeros_like(mu),where=den>0);z=abs(t);w=((z-z.min())/max(z.max()-z.min(),1e-30))**2;weights.append(w)
  weights=np.array(weights);assert (weights.sum(1)>0).all()
  targets=sorted(set(ts));ixs=[np.flatnonzero(ts==t) for t in targets]
  ypair=np.stack([y[z].mean(0) for z in ixs]);cpair=np.stack([c[z].mean(0) for z in ixs])
  singlesA=np.stack([np.mean([fulltrain[(t.split('+')[0],cs[j])][0] for j in z],axis=0) for t,z in zip(targets,ixs)])
  singlesB=np.stack([np.mean([fulltrain[(t.split('+')[1],cs[j])][0] for j in z],axis=0) for t,z in zip(targets,ixs)])
  symbol=dict(zip(genes.gene_id,genes.symbol));target_names=['+'.join(symbol.get(g,g) or g for g in t.split('+')) for t in targets]
  if fold==0:
   ct,cc=keys[case];g=de20[case];casepred=[];casemodel=[];casearm=[]
   cases.append(dict(dataset=ds,fold=fold,target=ct,context=cc,display='+'.join(symbol.get(t,t) or t for t in ct.split('+')),cells=len(idx),selection='Median L2 norm of true control-centred response among fold0 query-contexts; independent of predictions'))
  inventory.append(dict(dataset=ds,fold=fold,queries=len(keys),targets=len(targets),contexts=len(set(cs)),genes=y.shape[1],test_cells=len(m),weight_pool_conditions=len(pool)))
  for arm in ['seen0','seen1','seen2']:
   z=np.load(E/f'data/{ds}/{arm}/fold{fold}/train.npz');train={(str(t),str(cc)):v.astype(float) for t,cc,v in zip(z['target'],z['context'],z['expression'])};ref=np.stack([np.mean([v for (t,cc),v in train.items() if cc==k[1] and t!='control'],axis=0) for k in keys])
   pruns=[]
   for j in J:
    if j['dataset']!=ds or j['fold']!=fold or j['arm']!=arm:continue
    folder=R/f"runs/formal/{ds}/{arm}/fold{fold}/seed{j['seed']}/{j['method']}";result=json.loads((folder/'result.json').read_text())
    files={model:(folder/(model+'.npz'),info['sha256']) for model,info in result['baselines'].items()} if j['method']=='baselines' else {j['method']:(folder/'predictions_sealed.npz',result['prediction_sha256'])}
    for model,(p,h) in files.items():pruns.append((model,j['seed'],p,h))
   p=E/f'runs/{ds}/fold{fold}/{arm}/predictions_sealed.npz';pruns.append(('kipert',17,p,json.loads((p.parent/'result.json').read_text())['prediction_sha256']))
   gidata=[];gimodels=[];giseeds=[]
   for model,seed,p,h in pruns+[('paper_perturbed_mean',17,None,None)]:
    if p:
     hh=sha(p);assert hh==h;hashes.append(dict(file=str(p),sha256=hh));zz=np.load(p);lookup={(str(t),str(cc)):i for i,(t,cc) in enumerate(zip(zz['target'],zz['context']))};assert set(lookup)==set(keys);pred=np.maximum(zz['prediction'][[lookup[k] for k in keys]].astype(float),0)
    else:pred=ref.copy()
    assert np.isfinite(pred).all();dp=pred-c;err=(pred-y)**2
    wmse=(weights*err).sum(1)/weights.sum(1);wr2raw=r2(dp,dy,weights)
    vals=dict(mse=err.mean(1),mse20=err[ii,de20].mean(1),nochange20=(dy[ii,de20]**2).mean(1),pcc_delta=pear(dp,dy),pcc_delta20=pear(dp[ii,de20],dy[ii,de20]),opposite20=((dp[ii,de20]*dy[ii,de20])<0).mean(1),r2_all=r2(pred,y),r2_deg100=r2(pred[ii,de100],y[ii,de100]),l2_top1000=np.linalg.norm(pred[:,top]-y[:,top],axis=1),pcc_pert=pear(pred-ref,y-ref),wmse=wmse,weighted_r2_ctrl=np.maximum(wr2raw,-1),weighted_r2_ctrl_unclipped=wr2raw,mean_wmse=(weights*(ref-y)**2).sum(1)/weights.sum(1),mean_weighted_r2_ctrl=r2(ref-c,dy,weights,True),control_pcc_pert=pear(c-ref,y-ref),cell_count=np.array([truth[k][2] for k in keys]))
    for n,k in enumerate(keys):rows.append(dict(dataset=ds,fold=fold,arm=arm,model=model,seed=seed,target=k[0],context=k[1],**{v:float(a[n]) for v,a in vals.items()}))
    # Inverse retrieval within a context: rank candidate predicted combinations
    # by Pearson correlation to each observed response, tie-averaged top-K.
    for cc in sorted(set(cs)):
     ix=np.flatnonzero(cs==cc);aa=dp[ix]-dp[ix].mean(1,keepdims=True);bb=dy[ix]-dy[ix].mean(1,keepdims=True);den=np.linalg.norm(aa,axis=1)[:,None]*np.linalg.norm(bb,axis=1)[None,:];sim=np.divide(aa@bb.T,den,out=np.zeros((len(ix),len(ix))),where=den>1e-14)
     for n,jj in enumerate(ix):
      s=sim[:,n];score=s[n];better=sum(s>score+1e-12);tie=sum(abs(s-score)<=1e-12)
      for kk in range(1,min(8,len(ix))+1):hitrows.append(dict(dataset=ds,fold=fold,arm=arm,model=model,seed=seed,target=ts[jj],context=cc,k=kk,pool=len(ix),hit=float(np.clip((kk-better)/tie,0,1))))
    ppair=np.stack([pred[z].mean(0) for z in ixs]);gidata.append(ppair[:,top].astype('float32'));gimodels.append(model);giseeds.append(seed)
    if fold==0:
     casepred.append(pred[case].astype('float32'));casemodel.append(model);casearm.append(arm)
   np.savez_compressed(O/f'data/GI_{ds}_fold{fold}_{arm}.npz',prediction=np.stack(gidata),models=np.array(gimodels),seeds=np.array(giseeds),truth=ypair[:,top],control=cpair[:,top],single_a=singlesA[:,top],single_b=singlesB[:,top],target=np.array(targets),target_label=np.array(target_names),gene_id=genes.gene_id.to_numpy().astype(str)[top],symbol=genes.symbol.to_numpy().astype(str)[top])
  if fold==0:
   np.savez_compressed(O/f'data/case_{ds}.npz',prediction=np.stack(casepred),models=np.array(casemodel),arms=np.array(casearm),truth=y[case],control=c[case],de20=g,top1000=top,symbol=genes.symbol.to_numpy().astype(str),cell_values=case_cells,single_a=fulltrain[(ct.split('+')[0],keys[case][1])][0],single_b=fulltrain[(ct.split('+')[1],keys[case][1])][0])
  pd.DataFrame(rows).to_csv(O/'data/query_metrics.csv.gz',index=False,compression='gzip');pd.DataFrame(hitrows).to_csv(O/'data/retrieval.csv.gz',index=False,compression='gzip')
  (O/'inventory.json').write_text(json.dumps(inventory,indent=2));(O/'case_selection.json').write_text(json.dumps(cases,indent=2));(O/'prediction_hashes.json').write_text(json.dumps(hashes,indent=2));print('EXTRACTED',ds,fold,len(rows),flush=True)
  del m,cm,fulltrain,pool,valid;gc.collect()
(O/'EXTRACTION_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',rows=len(rows),prediction_files=len(hashes),datasets=4,folds=5,arms=3)))
