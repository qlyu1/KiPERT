"""Verify prediction artifacts without opening any response matrices."""
from pathlib import Path
import hashlib,json,time,argparse
import numpy as np
import pandas as pd
R=Path('@STUDY@/uc_fig2_benchmark_20260911')
NEURAL=['biolord','scouter','txpert','gears','stamp','regformer']
METHODS=NEURAL+['UC','zero','mean','knn','ridge','full_krr','lowrank_krr','UC_without_depmap','UC_without_string','UC_without_genept']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def verify_one(base,run,result):
 def require(condition,message):
  if not condition:raise ValueError(str(run)+': '+message)
 require(result.get('status')=='PREDICTIONS_SEALED','not sealed')
 require(result.get('test_metrics_computed') is False,'test metric status missing or true')
 pred=run/'predictions_sealed.npz'
 require(sha(pred)==result.get('prediction_sha256'),'prediction checksum mismatch')
 q=pd.read_csv(base/'query/metadata.tsv',sep='\t',dtype=str)
 genes=pd.read_csv(base/'genes.tsv',sep='\t',dtype=str)
 require(not q.duplicated(['target','context']).any(),'duplicate query keys')
 require(not genes.gene_id.duplicated().any(),'duplicate gene identifiers')
 with np.load(pred,allow_pickle=False) as z:
  yp=z['prediction'];require(yp.shape==(len(q),len(genes)),'prediction shape mismatch')
  require(np.issubdtype(yp.dtype,np.number) and np.isrealobj(yp),'invalid prediction dtype')
  require(np.isfinite(yp).all(),'nonfinite predictions')
  for key in ['target','context']:
   require(z[key].shape==(len(q),),'query key shape mismatch')
   require(z[key].astype(str).tolist()==q[key].tolist(),key+' order mismatch')
  fb=z['native_fallback'];require(fb.shape==(len(q),) and fb.dtype==np.dtype(bool),'fallback flag shape or type mismatch')
  shape=list(yp.shape);fallback_count=int(fb.sum())
 weights=[p for p in [run/'model_final.pt',run/'model_final.pkl'] if p.exists()]
 require(len(weights)==1,'missing or ambiguous final weights')
 require(sha(weights[0])==result.get('weight_sha256'),'weight checksum mismatch')
 return {'shape':shape,'fallback_queries':fallback_count,'prediction_sha256':result['prediction_sha256'],'weight_sha256':result['weight_sha256']}
def audit(require_complete=False):
 rows=[];pending=[];errors=[]
 for d in ['S3','S4','S5','S6','S7']:
  for seed in [101,202,303]:
   base=R/'data/formal'/d/f'split{seed}'
   for method in METHODS:
    root=R/'runs/formal'/d/f'split{seed}';run=root/method if method in NEURAL else root/'uc_and_baselines'/method
    key={'dataset':d,'split':seed,'method':method}
    try:result=json.loads((run/'result.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):pending.append(key);continue
    try:rows.append({**key,**verify_one(base,run,result)})
    except Exception as e:errors.append({**key,'error':str(e)})
 report={'timestamp_unix':time.time(),'status':'PASS_COMPLETE' if len(rows)==240 and not errors else ('FAILED' if errors else 'PASS_AVAILABLE'),'verified_sets':len(rows),'required_sets':240,'pending':pending,'errors':errors,'verified':rows,'response_matrices_opened':False}
 out=R/'manifests/prediction_integrity.json';temp=out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(out)
 if errors or (require_complete and pending):raise RuntimeError('Prediction preflight failed: '+json.dumps({'errors':errors,'pending_count':len(pending)}))
 print(json.dumps({k:v for k,v in report.items() if k not in ['verified','pending']},indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');args=p.parse_args();audit(args.require_complete)
