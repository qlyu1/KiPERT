import subprocess,concurrent.futures,json,sys,shutil
from pathlib import Path
R=Path('@STUDY@/kipert_prior_exploration_20261117');PY='@STUDY@/uc_cross_context_transfer/experiments/pie_uc_state_benchmark_20261106/env/bin/python'
def job(d,s):
 if (R/f'runs/{d}/split{s}/COMPLETE.json').exists():return
 assert shutil.disk_usage(R).free>7*1024**3
 with (R/f'logs/{d}_{s}.log').open('a') as f:p=subprocess.run([PY,'-u',str(R/'code/prior_exploration_run.py'),d,str(s)],stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{d}/{s} failed: see its log')
 print('DONE',d,s,flush=True)
def main():
 assert (R/'runs/S3/split101/COMPLETE.json').exists(),'Full pilot must pass before batch execution'
 tasks=[(d,s) for d in ['S3','S4','S5','S6','S7'] for s in [101,202,303]]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for f in [pool.submit(job,*t) for t in tasks]:f.result()
 (R/'EXPERIMENTS_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',splits=15,configurations=144,modes=2)))
if __name__=='__main__':
 try:main()
 except Exception as e:(R/'FAILED.json').write_text(json.dumps(dict(error=repr(e))));raise
