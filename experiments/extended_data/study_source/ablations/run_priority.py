import subprocess,os,time,json,concurrent.futures
from pathlib import Path
R=Path('@STUDY@/kipert_supplement_20261117');PY='@STUDY@/uc_cross_context_transfer/experiments/pie_uc_state_benchmark_20261106/env/bin/python';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',KIPERT_USE_GPU='1',KIPERT_GPU_CHUNK='6642')
def job(name,args):
 with (R/'logs'/(name+'.log')).open('a') as f:
  p=subprocess.run([PY,'-u',str(R/'code/transfer.py')]+args,stdout=f,stderr=subprocess.STDOUT,env=env)
 if p.returncode:raise RuntimeError(name+' failed: '+str(p.returncode))
def main():
 while not (R/'transfer/GPU_DE_PARITY.json').exists():time.sleep(2)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  fut=[pool.submit(job,'primary_'+c,['primary',c]) for c in ['k562','hepg2','rpe1','jurkat']]
  for f in fut:f.result()
 job('transfer_de',['de'])
 (R/'transfer/PRIORITY_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',factorial_arms=4,contexts=4,conditions_per_arm=3712,genes=6642),indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (R/'transfer/PRIORITY_FAILED.json').write_text(json.dumps(dict(error=repr(e))));raise
