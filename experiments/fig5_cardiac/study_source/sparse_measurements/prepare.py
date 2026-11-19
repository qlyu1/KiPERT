"""Physical input/reference isolation for retrospective low-coverage measurement."""
import os
print('PREPARATION_START',flush=True)
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
from scipy import sparse
from mmap_csr_reader import open_csr_npz_mmap

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'prepared';OUT.mkdir(exist_ok=True)
SRC=ROOT.parent/'cai_expression_programs_20261111/diagnostics_20261111/prepared'
OLD=Path(os.environ.get('KIPERT_CAI_D14_DIR', ROOT.parent/'cai_d14_application_20261112'))
protocol=json.load(open(ROOT/'protocol.json'));R=protocol['outer_splits'];SEED=protocol['seed']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
assert not (OUT/'COMPLETE.json').exists(),'Do not overwrite completed split preparation'
meta=pd.read_csv(SRC/'cell_metadata.csv');genes=np.load(SRC/'genes.npy').astype(str);X=open_csr_npz_mmap(SRC/'cells.npz',ROOT/'csr_unpacked')
print('SOURCE_READY',X.shape,len(X.data),flush=True)
assert X.shape==(len(meta),len(genes));assert np.array_equal(meta.csr_row,np.arange(len(meta)))
targets=np.array(sorted(set(meta.perturb_identity)-{'NT_73'}));batches=np.array(sorted(meta.sample_batch.unique()))
mask=~np.isin(genes,np.r_[targets,['Becn1','Ifnar2','Wwtr1']]);G=len(genes)
np.savez(OUT/'axis.npz',targets=targets,genes=genes,batches=batches,evaluation_gene_mask=mask)
programs=protocol['programs'];pn=list(programs);pcols={p:np.flatnonzero(np.isin(genes,v)) for p,v in programs.items()};assert all(len(pcols[p])==len(programs[p]) for p in pn)
program_scores=np.empty((len(meta),len(pn)))
for start in range(0,len(meta),128):
 ix=np.arange(start,min(start+128,len(meta)));chunk=X.take_rows(ix)
 for j,p in enumerate(pn):program_scores[ix,j]=np.asarray(chunk[:,pcols[p]].astype(float).mean(1)).ravel()
del chunk
counts=[];cases=[];sources={};nt={}
for b,batch in enumerate(batches):
 nt[b]=np.flatnonzero(meta.sample_batch.eq(batch)&meta.perturb_identity.eq('NT_73'))
 sources[b]={t:np.flatnonzero(meta.sample_batch.eq(batch)&meta.perturb_identity.eq(t)) for t in targets}
 for t,ix in sources[b].items():counts.append(dict(target=str(t),batch=str(batch),n=len(ix),nt_n=len(nt[b])))
 for budget in [5,10]:
  case=0
  for t,ix in sources[b].items():
   if len(ix)>=budget+10:cases.append(dict(budget=budget,batch=str(batch),batch_index=b,target=str(t),n_total=len(ix),n_input=budget,n_reference=len(ix)-budget))
pd.DataFrame(counts).to_csv(OUT/'counts.csv',index=False);cdf=pd.DataFrame(cases);cdf['case_id']=cdf.groupby('budget').cumcount();cdf.to_csv(OUT/'cases.csv',index=False)
assert cdf.query('budget==5').target.nunique()==33 and sum(cdf.budget==5)==77
assert cdf.query('budget==10').target.nunique()==24 and sum(cdf.budget==10)==53
split_audit=[]
for r in range(R):
 for b,batch in enumerate(batches):
  rng=np.random.default_rng(np.random.SeedSequence([SEED,r,b,0]));cp=rng.permutation(nt[b]);ci=cp[:len(cp)//2];cr=cp[len(cp)//2:]
  ctrl_i=np.asarray(X.take_rows(ci).astype(float).mean(0)).ravel();ctrl_r=np.asarray(X.take_rows(cr).astype(float).mean(0)).ravel()
  crscore=program_scores[cr]
  permutations={t:rng.permutation(ix) for t,ix in sources[b].items()}
  for budget in [5,10]:
   dest=OUT/f'n{budget}'/f'repeat_{r:02d}'/f'batch_{b}';dest.mkdir(parents=True,exist_ok=True)
   train=[t for t in targets if len(sources[b][t])>=5];tidx={t:i for i,t in enumerate(train)}
   c=cdf[(cdf.budget==budget)&cdf.batch_index.eq(b)];query=c.target.to_list()
   ir=[];group=[];fold=[];rr=[];rgroup=[];qmeans=[];qcounts=[]
   for i,t in enumerate(train):
    perm=permutations[t];ix=perm[:budget] if t in query else perm
    inner_rng=np.random.default_rng(np.random.SeedSequence([SEED,r,b,budget,i+1]));f=inner_rng.permutation(np.arange(len(ix))%5)
    ir.extend(ix.tolist());group.extend([i]*len(ix));fold.extend(f.tolist())
   for j,t in enumerate(query):
    ref=permutations[t][budget:];assert len(ref)>=10
    rr.extend(ref.tolist());rgroup.extend([j]*len(ref));qcounts.append(len(ref))
    qmeans.append(np.asarray(X.take_rows(ref).astype(float).mean(0)).ravel())
   ir=np.array(ir,int);rr=np.array(rr,int);group=np.array(group,int);fold=np.array(fold,int)
   assert not set(ir)&set(rr);assert not set(ci)&set(cr)
   assert not (set(ci)|set(cr))&(set(ir)|set(rr))
   assert all(set(fold[group==i])==set(range(5)) for i in range(len(train)))
   sparse.save_npz(dest/'input_cells.npz',X.take_rows(ir))
   np.savez_compressed(dest/'input_meta.npz',train_targets=np.array(train),cell_target=group,inner_fold=fold,
    ctrl_input_mean=ctrl_i,query_train_indices=np.array([tidx[t] for t in query],int),query_targets=np.array(query),case_ids=c.case_id.to_numpy(int),
    input_cell_rows=ir,ctrl_input_rows=ci,query_n_reference=np.array(qcounts),ctrl_reference_n=len(cr))
   reference_scores=program_scores[rr]
   np.savez_compressed(dest/'reference.npz',query_targets=np.array(query),absolute_mean=np.stack(qmeans),ctrl_reference_mean=ctrl_r,
    program_reference_scores=reference_scores,program_reference_group=np.array(rgroup),program_names=np.array(pn),ctrl_reference_program_scores=crscore,
    reference_cell_rows=rr,ctrl_reference_rows=cr,case_ids=c.case_id.to_numpy(int))
   split_audit.append(dict(budget=budget,repeat=r,batch=str(batch),n_train_targets=len(train),n_queries=len(query),input_cells=len(ir),reference_cells=len(rr),input_nt=len(ci),reference_nt=len(cr),input_reference_overlap=0,
    input_cells_sha256=sha(dest/'input_cells.npz'),input_meta_sha256=sha(dest/'input_meta.npz'),reference_sha256=sha(dest/'reference.npz')))
 print('PREPARED_REPEAT',r,flush=True)
shutil.copytree(OLD/'prepared/prior_bank',OUT/'prior_bank',dirs_exist_ok=True)
shutil.copyfile(OLD/'src/prior_inputs.py',ROOT/'src/prior_inputs.py')
dump(OUT/'split_audit.json',split_audit)
manifest=dict(status='COMPLETE',seed=SEED,outer_splits=R,n_genes=G,evaluation_genes=int(mask.sum()),counts_only_eligibility=True,
 source_files={str(SRC/f):sha(SRC/f) for f in ['cells.npz','cell_metadata.csv','genes.npy']},protocol_sha256=sha(ROOT/'protocol.json'),
 input_reference_isolation='Separate input_cells/input_meta versus reference.npz for every batch/split/budget; cell row intersections checked empty.',
 programs=programs,scale='Published normalized values; no new normalization or state filtering',
 cohorts={str(n):dict(queries=int(sum(cdf.budget==n)),targets=int(cdf.query('budget==@n').target.nunique())) for n in [5,10]})
dump(OUT/'COMPLETE.json',manifest);print('DATA_PREPARATION_COMPLETE',manifest['cohorts'],flush=True)
