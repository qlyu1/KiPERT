import argparse
from pathlib import Path
p = argparse.ArgumentParser(description="Export the reviewer example from the original prepared study tree to a tar stream on stdout.")
p.add_argument("study_root", type=Path, help="Parent directory of the native study projects")
study_root = p.parse_args().study_root.resolve()
import sys,io,tarfile,json
from pathlib import Path
import numpy as np
d=study_root / 'uc_seen012_benchmark_20261005/experiments/kipert_double_20261106'
sys.path.insert(0,str(d/'src'))
import kipert_double as kd
ds='D2';fold=0;bank=kd.Bank(ds)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 def put(path,**arrays):
  out=io.BytesIO();np.savez_compressed(out,**arrays);h=tarfile.TarInfo(path);h.size=out.tell();out.seek(0);t.addfile(h,out)
 for arm in ['seen0','seen1','seen2']:
  panel=kd.loadpanel(ds,arm,fold);q=panel['query'];qt=q['target'].astype(str);qc=q['context'].astype(str);ctx=sorted(set(qc))
  put(arm+'/axes.npz',genes=np.asarray(panel['genes'],dtype=str),targets=bank.targets,query=qt,context=qc,contexts=np.array(ctx))
  for i,c in enumerate(ctx):
   conditions,y,control,_=kd.context_data(panel,c);views,cov,components=bank.kernels(conditions)
   put(arm+f'/context_{i}.npz',training=np.array(conditions),delta=y,control=control,views=views,coverage=cov,training_components=np.array(components))
  ref=np.load(d/f'runs/{ds}/fold{fold}/{arm}/predictions_sealed.npz')
  np.testing.assert_array_equal(ref['target'].astype(str),qt);np.testing.assert_array_equal(ref['context'].astype(str),qc)
  put(arm+'/expected.npz',prediction=ref['prediction'],target=qt,context=qc)
  print(arm,len(ctx),len(qt),len(panel['genes']),file=sys.stderr)
