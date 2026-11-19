from resources import R,digest
import json,time,os
from pathlib import Path

def check_gate(base,phase,method):
 r=json.loads((base/'READY.json').read_text());assert r['protocol']=='SEEN012_V1'
 assert digest(base/'fit/expression.npz')==r['fit_response_sha256'] and digest(base/'genes.tsv')==r['axis_sha256']
 if phase=='train':
  gate=json.loads(Path(os.environ['FIG3_GATE_SNAPSHOT']).read_text());assert gate['ready'] and method in gate['models']  and r['dataset'] in gate['datasets']

def seal(base,out,yp,q,method,selection,start,seed):
 import numpy as np
 assert yp.ndim==2 and len(yp)==len(q) and np.isfinite(yp).all();q=q.copy();q.to_csv(out/'query_order.tsv',sep='\t',index=False)
 np.savez_compressed(out/'predictions_sealed.npz',prediction=yp.astype(np.float32),target=q.target.to_numpy(dtype=str),context=q.context.to_numpy(dtype=str),native_fallback=np.zeros(len(q),bool))
 r={'status':'PREDICTIONS_SEALED','formal_training':True,'method':method,'selection':selection,'seed':seed,'queries':len(q),'genes':yp.shape[1],'prediction_sha256':digest(out/'predictions_sealed.npz'),'weight_sha256':digest(out/'model_final.pt'),'seconds':time.time()-start,'test_metrics_computed':False,'refit':False}
 (out/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
