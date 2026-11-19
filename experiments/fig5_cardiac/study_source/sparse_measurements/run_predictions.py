"""Input-only fitting and selection; reference files are never opened."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from scipy import sparse
from numpy.lib.format import open_memmap
from prior_inputs import Bank
from model import ModelContext,ALPHAS,SHRINKAGES,shrinkage_prediction,select_minimum_mse

ROOT=Path(__file__).resolve().parents[1];PREP=ROOT/'prepared';OUT=ROOT/'predictions'
assert (PREP/'COMPLETE.json').exists();assert not OUT.exists(),'Refusing to overwrite predictions'
OUT.mkdir();protocol=json.load(open(ROOT/'protocol.json'));manifest=json.load(open(PREP/'COMPLETE.json'));R=protocol['outer_splits']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 return x
def dump(p,x):Path(p).write_text(json.dumps(clean(x),indent=2)+'\n')
assert sha(ROOT/'protocol.json')==manifest['protocol_sha256']
axis=np.load(PREP/'axis.npz');genes=axis['genes'].astype(str);batches=axis['batches'].astype(str);mask=axis['evaluation_gene_mask'].astype(bool);G=len(genes)
card=np.flatnonzero(np.isin(genes,protocol['programs']['cardiac']));assert len(card)==8
cases=pd.read_csv(PREP/'cases.csv');bank=Bank();contexts={};audits=[];choices=[]
METHODS=['KiPERT','raw','shrinkage']
def means(X,group,n,use,columns=None):
 rows=np.flatnonzero(use);g=group[rows];cnt=np.bincount(g,minlength=n);assert (cnt>0).all()
 select=sparse.csr_matrix((1/cnt[g],(g,np.arange(len(g)))),shape=(n,len(g)))
 x=X[rows] if columns is None else X[rows][:,columns]
 return (select@x).toarray().astype(float)
for budget in [5,10]:
 directory=OUT/f'n{budget}';directory.mkdir();case=cases.query('budget==@budget').sort_values('case_id');N=len(case)
 assert np.array_equal(case.case_id,np.arange(N));case.to_csv(directory/'cases.csv',index=False)
 arrays={m:open_memmap(directory/(m+'.npy'),mode='w+',dtype=np.float32,shape=(R,N,G)) for m in METHODS}
 for r in range(R):
  for b,batch in enumerate(batches):
   src=PREP/f'n{budget}'/f'repeat_{r:02d}'/f'batch_{b}';allowed=[src/'input_cells.npz',src/'input_meta.npz']
   assert all('reference' not in p.name for p in allowed)
   X=sparse.load_npz(allowed[0]);z=np.load(allowed[1]);train=z['train_targets'].astype(str);group=z['cell_target'];fold=z['inner_fold'];qi=z['query_train_indices'];ids=z['case_ids'];n=len(train)
   key=tuple(train)
   if key not in contexts:contexts[key]=ModelContext(bank,train,mask,genes=genes)
   ctx=contexts[key]
   assert np.array_equal(case.iloc[ids].target.to_numpy(dtype=str),z['query_targets'])
   allmeans=means(X,group,n,np.ones(len(group),bool));kerrors=[];serrors=[]
   for f in range(5):
    fitmeans=means(X,group,n,fold!=f);validation=means(X,group,n,fold==f,card).mean(1)
    state=ctx.fit(fitmeans)
    kerrors.append([np.mean((state.predict_fitted(a,card).mean(1)-validation)**2) for a in ALPHAS])
    serrors.append([np.mean((shrinkage_prediction(fitmeans,c,card).mean(1)-validation)**2) for c in SHRINKAGES])
   km=np.mean(kerrors,axis=0);sm=np.mean(serrors,axis=0)
   alpha,_=select_minimum_mse(km,ALPHAS,'larger');coefficient,_=select_minimum_mse(sm,SHRINKAGES,'smaller')
   state=ctx.fit(allmeans)
   pred={'KiPERT':state.predict_fitted(alpha),'raw':np.maximum(allmeans,0).astype(np.float32).astype(float),'shrinkage':shrinkage_prediction(allmeans,coefficient)}
   for method,p in pred.items():
    assert np.isfinite(p[qi]).all();arrays[method][r,ids]=p[qi].astype(np.float32)
   choices.append(dict(budget=budget,repeat=r,batch=str(batch),n_training_targets=n,n_queries=len(qi),alpha=alpha,shrinkage_c=coefficient,**{f'alpha_{a:g}_inner_mse':v for a,v in zip(ALPHAS,km)},**{f'shrink_{c:g}_inner_mse':v for c,v in zip(SHRINKAGES,sm)}))
   audits.append(dict(budget=budget,repeat=r,batch=str(batch),paths_read=[str(p) for p in allowed],forbidden_reference_path=str(src/'reference.npz'),reference_read=False,input_hashes={str(p):sha(p) for p in allowed},
    inner_alignment='Recomputed independently from inner-training input means in every cell fold',inner_fit_n_per_target=[np.bincount(group[fold!=f],minlength=n) for f in range(5)],outer=state.audit(),case_ids=ids))
   del X,z,allmeans,state,pred,fitmeans
  for a in arrays.values():a.flush()
  print('PREDICTED',budget,r,flush=True)
 del arrays
pd.DataFrame(choices).to_csv(OUT/'hyperparameter_choices.csv',index=False)
dump(OUT/'fit_audits.json',audits);dump(OUT/'context_metadata.json',[c.metadata() for c in contexts.values()])
assert sha(ROOT/'protocol.json')==manifest['protocol_sha256']
seal=dict(status='SEALED',time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),reference_values_read=False,
 protocol_sha256=sha(ROOT/'protocol.json'),preparation_manifest_sha256=sha(PREP/'COMPLETE.json'),model_sha256=sha(ROOT/'src/model.py'),predictor_sha256=sha(__file__),
 prediction_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.glob('n*/*.npy')},fit_audit_sha256=sha(OUT/'fit_audits.json'))
dump(OUT/'PREDICTIONS_SEALED.json',seal);print('ALL_PREDICTIONS_SEALED',flush=True)
