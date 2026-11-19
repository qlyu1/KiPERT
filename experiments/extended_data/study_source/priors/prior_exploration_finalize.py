import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import time,subprocess,tarfile,json,hashlib,sys
import numpy as np,pandas as pd
N=Path('@STUDY@/kipert_prior_exploration_20261117');OLD=Path('@STUDY@/kipert_supplement_20261117');PY='@STUDY@/uc_cross_context_transfer/experiments/pie_uc_state_benchmark_20261106/env/bin/python'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
while not (N/'EXPERIMENTS_COMPLETE.json').exists():
 if (N/'FAILED.json').exists():raise RuntimeError((N/'FAILED.json').read_text())
 time.sleep(5)
print('ALL NEW EXPERIMENTS COMPLETE',flush=True)
if not (N/'reports/STATISTICS_PROTOCOL.json').exists():subprocess.run([PY,'-u',str(N/'code/prior_exploration_summary.py')],check=True)
count=0;checks=[]
for p in sorted((N/'runs').glob('S*/split*/PREDICTIONS_SEALED.json')):
 seal=json.loads(p.read_text());assert sha(seal['reference_path'])==seal['reference_sha256'];comp=json.loads((p.parent/'COMPLETE.json').read_text());assert comp['status']=='PASS'
 for name,h in seal['predictions'].items():
  f=p.parent/name;assert sha(f)==h;assert np.isfinite(np.load(f,mmap_mode='r')).all();count+=1
 checks.append(dict(dataset=p.parents[1].name,split=p.parent.name,full_replay_error=seal['full_replay_error'],score_errors=comp['score_replay_errors']))
assert len(checks)==15 and count==4320,(len(checks),count)
prior_archive=OLD/'KiPERT_supplement_delivery_20261117.zip';assert sha(prior_archive)=='a3b76727cb49fe55bebc57d138d9bc25450a1ac0a8605321fb7c79aa4044cb0e'
qa=dict(status='PASS',splits=15,configurations_per_split=144,selection_modes=2,sealed_finite_predictions=count,full_model_max_replay_error=max(x['full_replay_error'] for x in checks),checks=checks,previous_delivery_archive_unchanged=True,previous_full_prediction_references_unchanged=15,protocol_sha256=sha(N/'PROTOCOL.json'),code_sha256={p.name:sha(p) for p in (N/'code').glob('*.py')})
(N/'reports/COMPLETION_QA.json').write_text(json.dumps(qa,indent=2));print('EXPLORATION QA PASS',flush=True)
with tarfile.open(N/'exploration_tables_and_provenance.tar.gz','w:gz') as out:
 for p in (N/'reports').iterdir():
  if p.is_file():out.add(p,arcname=('tables/' if '.csv' in p.name else 'provenance/')+p.name)
 for p in (N/'code').glob('*.py'):out.add(p,arcname='code/'+p.name)
 for name in ['PROTOCOL.json','EXPERIMENTS_COMPLETE.json']:out.add(N/name,arcname='provenance/'+name)
 for p in (N/'runs').glob('S*/split*'):
  for name in ['SELECTIONS_FROZEN.json','PREDICTIONS_SEALED.json','permutation_mappings.npz','COMPLETE.json','validation_grid.csv']:
   out.add(p/name,arcname='provenance/runs/'+str(p.relative_to(N/'runs'))+'/'+name)
print('EXPLORATION ARCHIVE READY',flush=True)
