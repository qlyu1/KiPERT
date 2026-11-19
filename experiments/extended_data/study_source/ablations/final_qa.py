import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,sys,platform,subprocess,importlib.metadata
from pathlib import Path
import numpy as np,pandas as pd,scipy
from single_score import sha,save
O=Path('@STUDY@/kipert_supplement_20261117');S=Path('@STUDY@/uc_single_perturbation/figure2/experiments/kipert');D=Path('@STUDY@/uc_seen012_benchmark_20261005/experiments/kipert_double_20261106');T=Path('@STUDY@/uc_cross_context_transfer/experiments/pie_uc_state_benchmark_20261106')
def preserve():
 checked=[]
 records=json.loads((S/'specific_methods_20261105/ALL_PREDICTIONS_SEALED.json').read_text())
 for r in records:
  if r['arm'] not in ['aligned_raw','uniform_raw']:continue
  p=S/'specific_methods_20261105'/r['dataset']/('split'+str(r['split']))/(r['arm']+'_predictions.npz');assert sha(p)==r['prediction_sha256'];checked.append(str(p))
 for ds in ['D1','D2','D3','D4']:
  for f in range(5):
   for arm in ['seen0','seen1','seen2']:
    p=D/f'runs/{ds}/fold{f}/{arm}/predictions_sealed.npz';h=json.loads((p.parent/'result.json').read_text())['prediction_sha256'];assert sha(p)==h;checked.append(str(p))
 old=json.loads((T/'uc_context_specificity_20261107/input_manifest.json').read_text())
 for c in ['k562','hepg2','rpe1','jurkat']:
  root=T/'runs/uc'/c/'results'/c;p=root/'UC_observed_no_shrink_adaptive_mean.npy';s=json.loads((root/'PREDICTIONS_SEALED.json').read_text());assert sha(p)==s['prediction_sha256'][p.name];checked.append(str(p))
  for model in ['UC','PMean']:
   p=T/'runs/evaluation'/model/c/(c+'_adapter.npz');assert sha(p)==old[str(p)]['sha256'];checked.append(str(p))
 save(O/'manifests/ORIGINAL_RESULTS_PRESERVED.json',dict(status='PASS',checked_predictions=len(checked),files=checked,comparison='Original pre-existing sealed hashes; no replacement of manuscript outputs'))
def complete():
 assert (O/'ALL_EXPERIMENTS_COMPLETE.json').exists();records={}
 ss=list((O/'single/ablation').glob('S*/split*/COMPLETE.json'));dd=list((O/'double/ablation').glob('D*/fold*/COMPLETE.json'));assert len(ss)==15 and len(dd)==20
 for p in ss+dd:assert json.loads(p.read_text())['status']=='PASS'
 records['single_splits']=15;records['single_variants_per_split']=30;records['single_saved_predictions']=len(list((O/'single/ablation').glob('S*/split*/*_retuned.npy')))+len(list((O/'single/ablation').glob('S*/split*/*_fixed.npy')));assert records['single_saved_predictions']==900
 records['double_folds']=20;records['double_variants_per_arm']=17;records['double_saved_predictions']=len(list((O/'double/ablation').glob('D*/fold*/seen*/*_retuned.npy')))+len(list((O/'double/ablation').glob('D*/fold*/seen*/*_fixed.npy')));assert records['double_saved_predictions']==2040
 replay=[];checked=0
 for p in (O/'single/ablation').glob('S*/split*/PREDICTIONS_SEALED.json'):
  z=json.loads(p.read_text());replay.append(dict(task='single',source=str(p),full=z['full_replay_max_error'],uniform=z['uniform_replay_max_error']))
  for name,h in z['predictions'].items():assert sha(p.parent/name)==h;assert np.isfinite(np.load(p.parent/name,mmap_mode='r')).all();checked+=1
 for p in (O/'double/ablation').glob('D*/fold*/seen*/PREDICTIONS_SEALED.json'):
  z=json.loads(p.read_text());replay.append(dict(task='double',source=str(p),full=z['full_replay_error']))
  for name,h in z['files'].items():assert sha(p.parent/name)==h;assert np.isfinite(np.load(p.parent/name,mmap_mode='r')).all();checked+=1
 records['max_full_prediction_replay_error']=max(x['full'] for x in replay)
 for c,n in dict(k562=884,hepg2=867,rpe1=983,jurkat=978).items():
  r=O/'transfer'/c;files=list(r.glob('*_native.npy'));assert len(files)==20,(c,len(files));assert (r/'PRIORS_SEALED.json').exists() and (r/'IDENTITY_TEMPERATURE_SEALED.json').exists()
  seals=json.loads((r/'factorial_SEALED.json').read_text())['predictions']
  for name,h in seals.items():assert sha(r/(name+'_native.npy'))==h;checked+=1
  for p in r.glob('drop_*_SEALED.json'):
   z=json.loads(p.read_text());assert sha(r/(p.name.replace('_SEALED.json','_native.npy')))==z['sha256'];checked+=1
  z=json.loads((r/'IDENTITY_TEMPERATURE_SEALED.json').read_text());assert sha(r/'weighted_identity_retuned_native.npy')==z['prediction_sha256'];checked+=1
  for p in files:
   name=p.name[:-11];a=np.load(p,mmap_mode='r');assert a.shape==(n,6642) and np.isfinite(a).all();b=np.load(r/(name+'_adapted.npy'),mmap_mode='r');assert b.shape==a.shape and np.isfinite(b).all();f=pd.read_csv(r/(name+'_de_metrics.csv'));assert len(f)==n and not f.perturbation.duplicated().any()
 records['transfer_variants']=20;records['transfer_conditions_per_variant']=3712;records['transfer_genes']=6642;records['bootstrap']=2000
 assert checked==3020,checked;records['verified_new_prediction_hashes']=checked
 save(O/'reports/COMPLETION_QA.json',dict(status='PASS',**records,replay=replay,environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,torch=importlib.metadata.version('torch'),gpudge=importlib.metadata.version('gpudge'),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()),code_sha256={p.name:sha(p) for p in (O/'code').glob('*.py')},protocol_sha256=sha(O/'PROTOCOL.json')))
if __name__=='__main__':
 preserve()
 if '--complete' in sys.argv:complete()
 print('QA PASS',flush=True)
