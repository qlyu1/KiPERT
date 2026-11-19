import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS']:os.environ[k]='1'
import sys,json,time,gc
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu,spearmanr
from sklearn.metrics import average_precision_score
from single_score import save,sha,corr
O=Path('@STUDY@/kipert_supplement_20261117');B=Path('@STUDY@/uc_cross_context_transfer/experiments/pie_uc_state_benchmark_20261106');P=B.parent/'pie_four_context_reproduction_20261106';D=P/'downloads/official_data/preprocessed'
C=['k562','hepg2','rpe1','jurkat'];meta=json.loads((D/'meta.json').read_text());genes=np.array(meta['genes']);gi={g:i for i,g in enumerate(genes)}
cid=np.load(D/'ctx_ids.npy');pid=np.load(D/'pert_ids.npy');rm={(int(c),int(p)):i for i,(c,p) in enumerate(zip(cid,pid))}
def row(c,p):return rm[(meta['context_to_id'][c],meta['pert_to_id'][p])]
def controls(c):
 ix=np.load(B/'runs/evaluation/UC'/c/(c+'_control_indices.npy'));x=np.load(B/'data/statistics'/c/'controls_all.npy',mmap_mode='r');return np.array(x[ix],dtype=float)
def infer(c):
 out=O/'transfer'/c;out.mkdir(parents=True,exist_ok=True)
 if (out/'factorial_SEALED.json').exists():return
 tr=np.load(D/'delta_p.npy',mmap_mode='r');z=np.load(B/'runs/evaluation/UC'/c/(c+'_adapter.npz'));perts=z['perts'].astype(str);r=B/'runs/uc'/c;ax=np.load(r/'results'/c/'axes.npz');wz=np.load(r/'results'/c/'UC_observed_no_shrink_adaptive_weights.npz');ix=np.array([{g:i for i,g in enumerate(ax['targets'])}[p] for p in perts]);w=wz['weights'][:,ix];known=wz['source_measured'][:,ix];uniform=known/known.sum(0);pred={m:np.zeros((len(perts),len(genes))) for m in ['weighted_identity','uniform_identity','weighted_OT','uniform_OT']};receipts=[]
 train=json.loads((P/'downloads/official_splits/replogle_wdataset/unseen_ctx'/c/'train.json').read_text());allowed={s.split('.',1)[1]:set(ps) for s,ps in train.items()}
 for si,s in enumerate(wz['sources'].astype(str)):
  assert s!=c;assert np.array_equal(known[si],[p in allowed[s] for p in perts]);effect=np.zeros_like(pred['weighted_identity']);jj=np.flatnonzero(known[si]);effect[jj]=tr[[row(s,perts[j]) for j in jj]]
  a,b=sorted([s,c]);key=a+'__'+b;bp=r/'cache/exact_maps'/key/'basis.npz';mp=r/'cache/no_shrink_maps'/key/'map.npz';u=np.load(bp)['U'][:,:128];m=np.load(mp)['ab' if s==a else 'ba'];mapped=effect+(effect@u@(m-np.eye(128)))@u.T
  for weight,label in [(w,'weighted'),(uniform,'uniform')]:pred[label+'_identity']+=weight[si,:,None]*effect;pred[label+'_OT']+=weight[si,:,None]*mapped
  receipts.extend(dict(path=str(p),sha256=sha(p)) for p in [bp,mp])
 original=np.asarray(np.load(r/'results'/c/'UC_observed_no_shrink_adaptive_mean.npy',mmap_mode='r')[ix],float)-ax['control'];err=float(np.max(abs(original-pred['weighted_OT'])));assert err<2e-10,err
 # Seal exact original endpoint and deterministic linear transport-strength interpolation.
 pred['weighted_OT']=original
 for eta in [.25,.5,.75]:pred['weighted_eta'+str(eta).replace('.','p')]=pred['weighted_identity']+eta*(pred['weighted_OT']-pred['weighted_identity'])
 for name,a in pred.items():np.save(out/(name+'_native.npy'),a)
 np.savez_compressed(out/'axes.npz',perts=perts,genes=genes,weights=w,uniform_weights=uniform,sources=wz['sources'])
 save(out/'factorial_SEALED.json',dict(status='SEALED',native_full_replay_max_error=err,queries=len(perts),genes=len(genes),receipts=receipts,predictions={m:sha(out/(m+'_native.npy')) for m in pred}));print('TRANSFER SEALED',c,err,flush=True)
def rank_primary(a,b,perts):
 out=np.empty(len(a));n=len(a)
 for start in range(0,n,32):
  end=min(start+32,n);dist=cdist(a[start:end].astype(float),b.astype(float),'cityblock')
  for j,i in enumerate(range(start,end)):
   g=gi.get(perts[i])
   if g is not None:dist[j]-=abs(a[i,g]-b[:,g])
  out[start:end]=1-np.argmax(np.argsort(dist,axis=1)==np.arange(start,end)[:,None],axis=1)/n
 return out
def primary(c,name,mode,dp):
 out=O/'transfer'/c;perts=np.load(out/'axes.npz')['perts'];y=np.asarray(np.load(D/'delta_p.npy',mmap_mode='r')[[row(c,p) for p in perts]],float)
 f=pd.DataFrame(dict(context=c,perturbation=perts,variant=name,mode=mode,mse_delta=np.mean((dp-y)**2,1),mae_delta=np.mean(abs(dp-y),1),pearson_delta=corr(dp,y),pds_l1_exclude_target=rank_primary(dp,y,perts)))
 f.to_csv(out/(name+'_'+mode+'_metrics.csv'),index=False);return f
def normalize(x):
 scale=(10000/np.where(x.sum(1)==0,1,x.sum(1))).astype(np.float32);return x.astype(np.float32)*scale[:,None]
_REF=None;_REF_NORM=None;_REF_SORTED=None;_REF_MEAN=None
def de_cpu(reference,shift):
 global _REF,_REF_NORM,_REF_SORTED,_REF_MEAN
 if _REF is not reference:
  _REF=reference;_REF_NORM=normalize(np.expm1(reference));_REF_SORTED=np.sort(_REF_NORM.T,axis=1);_REF_MEAN=_REF_NORM.mean(0,dtype=float)
 y=np.maximum(reference+shift,0);dp=(y.mean(0)-reference.mean(0)).astype(np.float32);yy=normalize(np.expm1(y));a=yy.mean(0,dtype=float);b=_REF_MEAN;keep=(a*100>5)|(b*100>5);q=np.ones(len(a));lfc=np.zeros(len(a),np.float32)
 try:
  from cpu_mwu import mwu_sorted_reference
  p=mwu_sorted_reference(np.ascontiguousarray(yy[:,keep].T),np.ascontiguousarray(_REF_SORTED[keep]))
 except ImportError:p=mannwhitneyu(yy[:,keep],_REF_NORM[:,keep],axis=0,method='asymptotic',use_continuity=True).pvalue
 order=np.argsort(p);adj=np.empty(len(p));adj[order]=np.minimum(np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1],1);q[keep]=adj;lfc[keep]=np.log2((a[keep]+1e-9)/(b[keep]+1e-9));return dp,lfc,q
def parity_de():
 result=[]
 for c in C:
  infer(c);ref=controls(c)
  for name,model in [('weighted_OT','UC'),('uniform_identity','PMean')]:
   z=np.load(B/'runs/evaluation'/model/c/(c+'_adapter.npz'));perts=np.load(O/'transfer'/c/'axes.npz')['perts'];np.testing.assert_array_equal(perts,z['perts']);x=np.load(O/'transfer'/c/(name+'_native.npy'),mmap_mode='r')
   for j in np.unique(np.linspace(0,len(perts)-1,6,dtype=int)):
    start=time.time();dp,lfc,q=de_cpu(ref,x[j]);dq=float(np.max(abs(q-z['q_value'][j])));dl=float(np.max(abs(lfc-z['lfc'][j])));dd=float(np.max(abs(dp-z['delta_p'][j])));nd=int(np.sum((q<.05)!=(z['q_value'][j]<.05)));r=dict(context=c,model=model,target=str(perts[j]),max_q_error=dq,max_lfc_error=dl,max_delta_error=dd,de_set_disagreements=nd,seconds=time.time()-start);result.append(r);print('CPU_DE_PARITY',r,flush=True)
    assert dq<1e-5 and dl<2e-6 and dd<2e-6 and nd==0,r
 save(O/'transfer/CPU_DE_PARITY.json',dict(status='PASS',checks=result,method='same float32 CP10k staging, arithmetic mean LFC, tie-corrected asymptotic Mann-Whitney U with continuity correction, CPM>5 either group and within-group BH over retained genes'))
def de_gpu(reference,shifts,perts):
 import torch,gpudge
 torch.set_num_threads(1)
 def source():
  for p,shift in zip(perts,shifts):yield gpudge.CellGroup(str(p),np.ascontiguousarray(np.expm1(np.maximum(reference+shift,0))))
 frame=gpudge.de(cell_source=source,targets=list(perts),var_names=list(genes),reference=np.expm1(reference),normalize_target_sum=10000,filter_gene_min_cpm_cell=5,gpu_gene_chunk_size=int(os.environ.get('KIPERT_GPU_CHUNK','512'))).to_pandas();lookup={str(p):i for i,p in enumerate(perts)};ii=np.array([lookup[str(p)] for p in frame.target]);jj=np.array([gi[g] for g in frame.feature]);q=np.ones((len(perts),len(genes)));lfc=np.zeros(q.shape,np.float32);q[ii,jj]=frame.p_adj;lfc[ii,jj]=frame.log2_fold_change;return lfc,q
def parity_gpu():
 checks=[]
 for c in C:
  infer(c);ref=controls(c)
  for name,model in [('weighted_OT','UC'),('uniform_identity','PMean')]:
   z=np.load(B/'runs/evaluation'/model/c/(c+'_adapter.npz'));perts=z['perts'];x=np.load(O/'transfer'/c/(name+'_native.npy'),mmap_mode='r');ii=np.unique(np.linspace(0,len(perts)-1,6,dtype=int));lfc,q=de_gpu(ref,x[ii],perts[ii]);e=dict(context=c,model=model,q_max_error=float(np.max(abs(q-z['q_value'][ii]))),lfc_max_error=float(np.max(abs(lfc-z['lfc'][ii]))),de_disagreements=int(np.sum((q<.05)!=(z['q_value'][ii]<.05))))
   assert e['q_max_error']<1e-5 and e['lfc_max_error']<2e-6 and e['de_disagreements']==0,e
   checks.append(e);print('GPU_PARITY',e,flush=True)
 save(O/'transfer/GPU_DE_PARITY.json',dict(status='PASS',checks=checks,implementation='Original gpudge GPU API and unchanged settings'))
def score_primary(c):
 infer(c);out=O/'transfer'/c;ref=controls(c)
 for path in sorted(out.glob('*_native.npy')):
  name=path.name[:-11];x=np.load(path,mmap_mode='r')
  if not (out/(name+'_native_metrics.csv')).exists():primary(c,name,'native',x)
  if not (out/(name+'_adapted.npy')).exists():
   if name in ['weighted_OT','uniform_identity']:
    model='UC' if name=='weighted_OT' else 'PMean';dp=np.load(B/'runs/evaluation'/model/c/(c+'_adapter.npz'))['delta_p']
   else:dp=np.stack([(np.maximum(ref+a,0).mean(0)-ref.mean(0)).astype(np.float32) for a in x])
   np.save(out/(name+'_adapted.npy'),dp)
  if not (out/(name+'_adapted_metrics.csv')).exists():primary(c,name,'adapted',np.load(out/(name+'_adapted.npy')))
  print('TRANSFER PRIMARY',c,name,flush=True)
def score_de(c,name):
 out=O/'transfer'/c;dp=np.load(out/(name+'_adapted.npy'));x=np.load(out/(name+'_native.npy'),mmap_mode='r');perts=np.load(out/'axes.npz')['perts'];ref=controls(c);gpu=os.environ.get('KIPERT_USE_GPU')=='1';checks=O/'transfer'/('GPU_DE_PARITY.json' if gpu else 'CPU_DE_PARITY.json');assert checks.exists()
 dest=out/(name+'_de_metrics.csv')
 if dest.exists():return
 model={'weighted_OT':'UC','uniform_identity':'PMean'}.get(name);rows=[];chunk=out/'de_chunks'/name;chunk.mkdir(parents=True,exist_ok=True)
 source=dict(np.load(B/'runs/evaluation'/model/c/(c+'_adapter.npz'))) if model else None
 rl=np.load(D/'lfc_true.npy',mmap_mode='r');rq=np.load(D/'fdr.npy',mmap_mode='r')
 for start in range(0,len(perts),25):
  file=chunk/(str(start)+'.npz')
  if not file.exists():
   if source:lfc=source['lfc'][start:start+25];q=source['q_value'][start:start+25]
   elif gpu:lfc,q=de_gpu(ref,x[start:start+25],perts[start:start+25])
   else:
    vals=[de_cpu(ref,x[j]) for j in range(start,min(start+25,len(perts)))];lfc=np.stack([v[1] for v in vals]);q=np.stack([v[2] for v in vals]);np.testing.assert_allclose(np.stack([v[0] for v in vals]),dp[start:start+25],atol=2e-6,rtol=0)
   np.savez_compressed(file,lfc=lfc,q=q)
  zz=np.load(file)
  for k,j in enumerate(range(start,min(start+25,len(perts)))):
   lfc=zz['lfc'][k].astype(np.float32);q=zz['q'][k].astype(np.float32);ii=row(c,perts[j]);tl=np.asarray(rl[ii],np.float32);y=rq[ii]<.05;sy=q<.05;ny=int(y.sum());scores=-np.log10(np.clip(q,1e-10,1));tr=np.flatnonzero(y);pr=np.flatnonzero(sy);tr=tr[np.argsort(-abs(tl[tr]),kind='stable')];pr=pr[np.argsort(-abs(lfc[pr]),kind='stable')]
   rows.append(dict(context=c,perturbation=perts[j],variant=name,n_true_de=ny,n_pred_de=int(sy.sum()),auprc=average_precision_score(y,scores) if 0<ny<len(genes) else np.nan,direction_true_de=float(np.mean(np.sign(tl[y])==np.sign(lfc[y]))) if ny else 0.,lfc_spearman_true_de=float(spearmanr(tl[y],lfc[y]).statistic) if ny>1 and np.ptp(tl[y])>0 and np.ptp(lfc[y])>0 else np.nan,de_overlap_N=len(set(tr[:ny])&set(pr[:ny]))/ny if ny and sy.any() else 0.))
  print('TRANSFER DE',c,name,min(start+25,len(perts)),len(perts),flush=True)
 pd.DataFrame(rows).to_csv(dest,index=False)
def main():
 stage=sys.argv[1]
 if stage=='parity':parity_de()
 elif stage=='parity_gpu':parity_gpu()
 elif stage=='primary':
  for c in (sys.argv[2:] or C):score_primary(c)
 elif stage=='de':
  for c in C:
   for name in ['weighted_OT','weighted_identity','uniform_identity','uniform_OT']:score_de(c,name)
 elif stage=='de_all':
  import subprocess,concurrent.futures
  tasks=[(c,p.name[:-11]) for c in C for p in sorted((O/'transfer'/c).glob('*_native.npy')) if not (O/'transfer'/c/(p.name[:-11]+'_de_metrics.csv')).exists()]
  def run_de(task):
   c,name=task
   with (O/'logs'/('de_'+c+'_'+name+'.log')).open('a') as f:r=subprocess.run([sys.executable,'-u',__file__,'de_one',c,name],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
   if r.returncode:raise RuntimeError('DE failed '+c+' '+name)
   print('DE COMPLETE',c,name,flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
   for f in [pool.submit(run_de,t) for t in tasks]:f.result()
 elif stage=='de_one':score_de(sys.argv[2],sys.argv[3])
if __name__=='__main__':main()
