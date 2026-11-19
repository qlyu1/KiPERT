"""Prepare existing D14 data by physical sample files; no prediction fitting."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'prepared';OUT.mkdir(exist_ok=True)
EXP=ROOT.parent
SRC=EXP/'cai_expression_programs_20261111/diagnostics_20261111/prepared'
OLD=Path(os.environ.get('KIPERT_CAI_STAGE_DIR', EXP/'cai_stage_transfer_20261112'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
meta=pd.read_csv(SRC/'cell_metadata.csv')
genes=np.load(SRC/'genes.npy').astype(str)
X=sparse.load_npz(SRC/'cells.npz')
assert X.shape==(len(meta),len(genes))
assert meta.perturb_captured.all() and not meta.predicted_doublet.any()
targets=np.array(sorted(set(meta.perturb_identity)-{'NT_73'}));batches=np.array(sorted(meta.sample_batch.unique()))
mask=~np.isin(genes,np.r_[targets,['Becn1','Ifnar2','Wwtr1']])
np.savez(OUT/'axis.npz',targets=targets,genes=genes,batches=batches,evaluation_gene_mask=mask)
programs=json.load(open(OLD/'prepared/programs.json'))
names=list(programs);gene_lookup={g:i for i,g in enumerate(genes)}
cols={p:np.array([gene_lookup[g] for g in programs[p]['used']],int) for p in names}
dump(OUT/'programs.json',programs)
rows=[];counts=[];ntcounts=[]
for s,b in enumerate(batches):
    dest=OUT/f'batch_{s}';dest.mkdir(exist_ok=True)
    ci=np.flatnonzero(meta.sample_batch.eq(b)&meta.perturb_identity.eq('NT_73'))
    C=X[ci].toarray().astype(float);np.save(dest/'controls.npy',C)
    n=np.zeros(len(targets),int);D=np.full((len(targets),len(genes)),np.nan);V=np.full_like(D,np.nan)
    for i,t in enumerate(targets):
        ix=np.flatnonzero(meta.sample_batch.eq(b)&meta.perturb_identity.eq(t));n[i]=len(ix)
        if not len(ix):continue
        a=X[ix].toarray().astype(float);D[i]=a.mean(0)-C.mean(0)
        if len(ix)>1:V[i]=a.var(0,ddof=1)
    np.save(dest/'n.npy',n);np.save(dest/'delta.npy',D);np.save(dest/'cell_var.npy',V)
    # Evaluation-only labels/cell scores; predictor must never load these.
    ix=np.flatnonzero(meta.sample_batch.eq(b))
    scores=np.column_stack([np.asarray(X[ix][:,cols[p]].mean(1)).ravel() for p in names])
    np.savez_compressed(dest/'evaluation_cells.npz',scores=scores,program_names=np.array(names),
        target=meta.iloc[ix].perturb_identity.to_numpy(dtype=str),is_cm=meta.iloc[ix].is_cm.to_numpy(dtype=bool),
        celltype=meta.iloc[ix].celltype.to_numpy(dtype=str))
    for i,t in enumerate(targets):rows.append(dict(target=str(t),batch=str(b),batch_index=s,n=int(n[i]),nt_n=len(C)))
    counts.append(n);ntcounts.append(len(C))
nmat=np.stack(counts,axis=1);pd.DataFrame(rows).to_csv(OUT/'target_batch_counts.csv',index=False)
elig=[]
for threshold in [5,10]:
    present=nmat>=threshold;valid=present&((present.sum(1)[:,None]-present)>=2)
    for i,t in enumerate(targets):elig.append(dict(target=str(t),min_cells_per_batch=threshold,n_total=int(nmat[i].sum()),
        source_supported_batches=int(present[i].sum()),valid_queries=int(valid[i].sum())))
pd.DataFrame(elig).to_csv(OUT/'eligibility.csv',index=False)
shutil.copytree(OLD/'prepared/prior_bank',OUT/'prior_bank',dirs_exist_ok=True)
for name in ['prior_adapter.py','prior_inputs.py']:shutil.copyfile(OLD/'src'/name,ROOT/'src'/name)
manifest=dict(status='PREPARED_NO_MODEL_FIT',n_targets=len(targets),n_genes=len(genes),evaluation_genes=int(mask.sum()),
    batch_names=batches.tolist(),nt_n=ntcounts,n_cells=len(meta),counts=nmat.tolist(),
    source='Existing full D14 QC cache, not restricted to D7 support or historical13.',
    scale='Published normalized real expression scale unchanged; no further normalization.',
    mask='All assayed displayed targets plus conservative alias candidates Becn1/Ifnar2/Wwtr1 excluded from scoring.',
    programs='Historical frozen 8/7/6 cardiac/fibrosis/injury genes; retrospective exploratory endpoints, not externally preregistered.',
    program_score='Unweighted mean gene expression change on the published scale; no data-dependent scaling/weights.',
    variance='Within-target within-batch cell variance ddof1; n<2 stays NaN; no cross-batch variance borrowing.',
    is_cm_definition='Existing source-cache is_cm field, evaluation only; verify provenance separately.',
    biological_replicates='sample_batch identities have not been established as independent animals.',
    inputs={str(SRC/f):sha(SRC/f) for f in ['cell_metadata.csv','cells.npz','genes.npy']},
    eligibility={str(v):dict(n_targets=int(sum(e['valid_queries']>0 for e in elig if e['min_cells_per_batch']==v)),
        n_queries=int(sum(e['valid_queries'] for e in elig if e['min_cells_per_batch']==v))) for v in [5,10]})
dump(OUT/'manifest.json',manifest)
print(json.dumps({k:manifest[k] for k in ['n_targets','n_genes','evaluation_genes','nt_n','n_cells','eligibility']},indent=2),flush=True)
