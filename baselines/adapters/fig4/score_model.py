"""Frozen cell-output adapter and unmodified PIE scorer. No held-out responses feed inference."""
from pathlib import Path
import os,sys,json,time,hashlib,gc
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['WANDB_MODE']='disabled'
import numpy as np,pandas as pd,torch,h5py
from pie.data.preprocessed import PreprocessedDir
from pie.predict import PredictionBlock,Predictions,write_predictions_parquet
from pie.evaluate import scoring_inputs,metrics_table,granular_table
from pie.metrics import score
B=Path(__file__).resolve().parents[1];P=B.parent/'pie_four_context_reproduction_20261106';kind,fold=sys.argv[1:3];assert kind in ['UC','STATE','PMean','CMean','STACK'];torch.set_num_threads(4)
D=P/'downloads/official_data/preprocessed';meta=json.loads((D/'meta.json').read_text());genes=meta['genes'];gindex={g:i for i,g in enumerate(genes)};ci=np.load(D/'ctx_ids.npy');pi=np.load(D/'pert_ids.npy');pnames={i:g for g,i in meta['pert_to_id'].items()};cnames={i:c for c,i in meta['context_to_id'].items()};rowmap={(cnames[int(c)],pnames[int(p)]):i for i,(c,p) in enumerate(zip(ci,pi))}
settings=['unseen_ctx'];splits={s:json.loads((P/'downloads/official_splits/replogle_wdataset'/s/fold/'test.json').read_text()) for s in settings};queries=sorted({(c.split('.',1)[1],p) for d in splits.values() for c,ps in d.items() for p in ps});out=B/'runs/evaluation'/kind/fold;out.mkdir(parents=True,exist_ok=True)
allblocks={}
for ctx in sorted({c for c,p in queries}):
 z=np.load(out/(ctx+'_adapter.npz'));allblocks[ctx]={k:z[k] for k in ['perts','delta_p','lfc','p_de']}
# Held-out response arrays are read only after outputs and DE summaries have been frozen.
d=PreprocessedDir.open(D)
for setting in settings:
 dest=out/setting;dest.mkdir(exist_ok=True);pairs=sorted((c.split('.',1)[1],p) for c,ps in splits[setting].items() for p in ps);rows=np.array([rowmap[x] for x in pairs],np.int64);fields={k:[] for k in ['delta_p','lfc','p_de']}
 for c,p in pairs:
  block=allblocks[c];i=int(np.flatnonzero(block['perts']==p)[0])
  for k in fields:fields[k].append(block[k][i])
 b=PredictionBlock(dataset='replogle',genes=genes,contexts=[c for c,p in pairs],perts=[p for c,p in pairs],row_index=rows,p_de=np.stack(fields['p_de']),lfc_pred=np.stack(fields['lfc']),delta_p_pred=np.stack(fields['delta_p']));write_predictions_parquet(Predictions([b]),dest/'predictions.parquet');result=score([scoring_inputs(b,d,.05)]);metrics_table(result).to_csv(dest/'metrics.csv',index=False);granular_table(result).to_csv(dest/'granular.csv',index=False);(dest/'DONE.json').write_text(json.dumps({'model':kind,'fold':fold,'setting':setting,'pairs':len(pairs),'genes':len(genes),'metrics':result.context_mean},indent=2));print('SCORED',kind,fold,setting,result.context_mean,flush=True)
(out/'COMPLETE.json').write_text(json.dumps({'status':'complete','model':kind,'fold':fold,'settings':settings},indent=2))
