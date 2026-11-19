import subprocess,os,time,json,concurrent.futures,shutil
from pathlib import Path
R=Path('@STUDY@/kipert_supplement_20261117');PY='@STUDY@/uc_cross_context_transfer/experiments/pie_uc_state_benchmark_20261106/env/bin/python';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',KIPERT_USE_GPU='1',KIPERT_GPU_CHUNK='6642')
def job(script,args):
 name=script+'_'+'_'.join(map(str,args));cmd=[PY,'-u',str(R/'code'/(script+'.py'))]+list(map(str,args))
 assert shutil.disk_usage(R).free>5*1024**3,'Less than 5GB disk free'
 with (R/'logs'/(name+'.log')).open('a') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 if p.returncode:raise RuntimeError(name+' failed: '+str(p.returncode))
 print('DONE',name,flush=True)
def batch(tasks,n):
 with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
  fut=[pool.submit(job,*t) for t in tasks]
  for f in fut:f.result()
def main():
 while not (R/'transfer/PRIORITY_COMPLETE.json').exists() or not (R/'single/rescore/COMPLETE.json').exists():
  if (R/'transfer/PRIORITY_FAILED.json').exists():raise RuntimeError('Priority comparison failed')
  time.sleep(5)
 # Replay full/uniform first; subsequent jobs are only scheduled if these guards pass.
 job('single_ablation',['S3',101]);job('double_ablation',['D1',0]);job('transfer_ablation',['spectrum']);job('transfer_ablation',['prior','k562']);job('transfer_ablation',['temperature','k562'])
 tasks=[('single_ablation',[d,s]) for d in ['S3','S4','S5','S6','S7'] for s in [101,202,303]]
 tasks += [('double_ablation',[d,f]) for d in ['D1','D2','D3','D4'] for f in range(5)]
 tasks += [('transfer_ablation',[stage,c]) for c in ['hepg2','rpe1','jurkat'] for stage in ['prior','temperature']]
 batch(tasks,6)
 batch([('transfer',['primary',c]) for c in ['k562','hepg2','rpe1','jurkat']],4)
 job('transfer',['de_all'])
 (R/'ALL_EXPERIMENTS_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',single_splits=15,double_folds=20,transfer_contexts=4),indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (R/'ABLATIONS_FAILED.json').write_text(json.dumps(dict(error=repr(e))));raise
