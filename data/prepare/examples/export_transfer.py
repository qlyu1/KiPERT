import argparse
from pathlib import Path
p = argparse.ArgumentParser(description="Export the reviewer example from the original prepared study tree to a tar stream on stdout.")
p.add_argument("study_root", type=Path, help="Parent directory of the native study projects")
study_root = p.parse_args().study_root.resolve()
from pathlib import Path
import numpy as np,sys,tarfile,io,json,hashlib
T=study_root / 'uc_cross_context_transfer/experiments'
Y=T/'pie_uc_state_benchmark_20261106';D=T/'pie_four_context_reproduction_20261106/downloads/official_data/preprocessed'
meta=json.loads((D/'meta.json').read_text());rows=dict(((int(c),int(p)),i) for i,(c,p) in enumerate(zip(np.load(D/'ctx_ids.npy'),np.load(D/'pert_ids.npy'))))
tr=np.load(D/'delta_p.npy',mmap_mode='r');g='SEH1L';done=set()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 def write(name,**v):
  b=io.BytesIO();np.savez_compressed(b,**v);h=tarfile.TarInfo(name);h.size=b.tell();b.seek(0);tar.addfile(h,b)
 write('genes.npz',genes=np.asarray(meta['genes']))
 for c in ['k562','hepg2','rpe1','jurkat']:
  rt=Y/'runs/uc'/c;res=rt/'results'/c;wz=np.load(res/'UC_observed_no_shrink_adaptive_weights.npz');ax=np.load(res/'axes.npz')
  idx=list(ax['targets'].astype(str)).index(g); assert str(wz['targets'][idx])==g
  src=wz['sources'].astype(str);weights=wz['weights'][:,idx];known=wz['source_measured'][:,idx];delta=[];keys=[];maps=[]
  assert c not in src
  for si,s in enumerate(src):
   a,b=sorted([c,s]);key=a+'__'+b
   U=np.ascontiguousarray(np.load(rt/'cache/exact_maps'/key/'basis.npz')['U'][:,:128]); uid=hashlib.sha256(U.tobytes()).hexdigest()[:16];keys.append(uid)
   if uid not in done:
    write('basis_'+uid+'.npz',U=U);done.add(uid)
   maps.append(np.load(rt/'cache/no_shrink_maps'/key/'map.npz')['ab' if s==a else 'ba'])
   delta.append(np.asarray(tr[rows[(meta['context_to_id'][s],meta['pert_to_id'][g])]],float) if known[si] else np.zeros(U.shape[0]))
  write(c+'.npz',source=src,query=np.array(g),known=known,weights=weights,delta=np.stack(delta),control=ax['control'],maps=np.stack(maps),basis_keys=np.array(keys))
  frozen=np.load(res/'UC_observed_no_shrink_adaptive_mean.npy',mmap_mode='r')[idx]
  write('expected_'+c+'.npz',mean=frozen)
 print('Exported four measured-source transfers;',len(done),'unique OT bases',file=sys.stderr)
