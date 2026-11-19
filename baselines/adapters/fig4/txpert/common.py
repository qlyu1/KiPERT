"""PIE adapter: exact public TxPert class, public GAT and graph implementation."""
from pathlib import Path
import os, sys, json, ast, copy, hashlib
T=Path(__file__).resolve().parent
B=T.parent; P=B.parent/'pie_four_context_reproduction_20261106'
sys.path[:0]=[str(T/'python_deps'), str(T/'code')]
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict,Any
from gspp.data.graphmodule import GSPGraph
from gspp.models.pert_models.basic_gnn import GNN
import yaml
torch.set_num_threads(4)
CFG=yaml.safe_load((T/'code/configs/config-x-cell-gat.yaml').read_text())
META=json.loads((P/'downloads/official_data/preprocessed/meta.json').read_text())
GENES=np.array(META['genes']); CTX=['k562','hepg2','rpe1','jurkat']
SPL=P/'downloads/official_splits/replogle_wdataset/unseen_ctx'
def sha(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def savej(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def splits(fold,part):
 return {c.split('.',1)[1]:ps for c,ps in json.loads((SPL/fold/(part+'.json')).read_text()).items()}
def model(device):
 # Execute only the unmodified class AST. Unused VAE/Exphormer imports otherwise
 # force unrelated timm/torch_scatter dependencies; the selected GAT is imported normally.
 path=T/'code/gspp/models/txpert.py';tree=ast.parse(path.read_text())
 node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='TxPert')
 env=dict(torch=torch,nn=nn,F=F,Dict=Dict,Any=Any,GSPGraph=GSPGraph,CONTROL_LABEL='ctrl',BASAL_STATE_MODEL_DICT={},PERT_MODEL_DICT={'gnn':GNN})
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
 vocab=json.loads((T/'vocab.json').read_text())
 graph=object.__new__(GSPGraph);graph.graph_dict={'graph1':torch.load(T/'graph.pt',weights_only=True)}
 args=copy.deepcopy(CFG['model']);args.pop('model_type');args.pop('slow_benchmark')
 args['cntr_model_args']=args.pop('cntr_model');args['pert_model_args']=args.pop('pert_model')
 m=env['TxPert'](input_dim=len(GENES),output_dim=len(GENES),adata_output_dim=len(GENES),graph=graph,device=device,**args).to(device)
 return m,vocab
def predict(m,idxs):
 """Exact no-basal inference, caching the full GNN once per evaluation."""
 m.eval()
 with torch.inference_mode():
  z=m.pert_model(m.edge_index,m.edge_weight,0.0)
  out=[]
  for lo in range(0,len(idxs),128):
   a=idxs[lo:lo+128];emb=torch.stack([z[p].sum(0) for p in a])
   out.append(m.decoder(emb).cpu().numpy())
 return np.concatenate(out)
