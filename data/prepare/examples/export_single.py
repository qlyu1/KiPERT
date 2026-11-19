import argparse
from pathlib import Path
p = argparse.ArgumentParser(description="Export the reviewer example from the original prepared study tree to a tar stream on stdout.")
p.add_argument("study_root", type=Path, help="Parent directory of the native study projects")
study_root = p.parse_args().study_root.resolve()
import io, sys, tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
proj=study_root / 'uc_single_perturbation'
sys.path.insert(0,str(proj/'submission/figure2_code'))
from kipert.core import view_kernels
from kipert.inputs import Bank
ds='S3'; fold='split101'
p=proj/'figure2/inputs/datasets'/ds/fold
bank=Bank(proj/'figure2/inputs/prepared_priors'/ds)
q=pd.read_csv(p/'query/metadata.tsv',sep='\t',dtype=str)
genes=pd.read_csv(p/'genes.tsv',sep='\t',dtype=str).gene_id.to_numpy(str)
keep=~np.isin(genes,bank.targets)
ctxs=sorted(set(q.context))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 def put(name,**values):
  buf=io.BytesIO();np.savez_compressed(buf,**values);hdr=tarfile.TarInfo(name);hdr.size=buf.tell();buf.seek(0);tar.addfile(hdr,buf)
 put('axes.npz',genes=genes,keep=keep,context=q.context.to_numpy(str),target=q.target.to_numpy(str),contexts=np.array(ctxs))
 roles=['fit','validation','controls_fit']
 xs={role:sparse.load_npz(p/role/'expression.npz').tocsr() for role in roles}
 ms={role:pd.read_csv(p/role/'metadata.tsv',sep='\t',dtype=str) for role in roles}
 for i,context in enumerate(ctxs):
  matrices={role:xs[role][np.flatnonzero(ms[role].context.to_numpy()==context)].astype(float) for role in roles}
  train_x=sparse.vstack([matrices['fit'],matrices['validation']],format='csr')
  train_meta=pd.concat([ms['fit'][ms['fit'].context==context],ms['validation'][ms['validation'].context==context]],ignore_index=True)
  ix_map=train_meta.groupby('target',sort=True).indices
  control=np.asarray(matrices['controls_fit'].mean(0)).ravel()
  train=list(ix_map); delta=np.stack([np.asarray(train_x[ix_map[g]].mean(0)).ravel()-control for g in train])
  test=q.loc[q.context==context,'target'].tolist()
  assert not set(test)&set(train)
  views,ix,coverage,names=view_kernels(bank,train)
  qi=[bank.index[g] for g in test];idx=list(ix)+qi
  put(f'context_{i}.npz',train=np.array(train),queries=np.array(test),views=views[:,idx][:,:,idx],coverage=coverage[idx],names=np.array(names),control=control,train_delta=delta)
  print('Exported',context,len(train),len(test),file=sys.stderr)
 ref=np.load(proj/'figure2/results/kipert'/ds/fold/'predictions.npz',allow_pickle=False)
 print('Reference fields',ref.files,file=sys.stderr)
 np.testing.assert_array_equal(q.target.to_numpy(str),ref['target'].astype(str))
 np.testing.assert_array_equal(q.context.to_numpy(str),ref['context'].astype(str))
 put('expected_predictions.npz',prediction=ref['prediction'].astype(np.float32),target=q.target.to_numpy(str),context=q.context.to_numpy(str))
