from pathlib import Path
import os,sys,json,time
os.environ['WANDB_MODE']='disabled';os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch,h5py
from omegaconf import OmegaConf
from cell_load.data_modules import PerturbationDataModule
from cell_load.dataset._perturbation import PerturbationDataset
B=Path(__file__).resolve().parents[1];P=B.parent/'pie_four_context_reproduction_20261106';fold=sys.argv[1];cfg=OmegaConf.load(B/'manifests'/f'state_{fold}.yaml');torch.set_num_threads(4)
assert int(cfg.training.max_steps)==50000 and float(cfg.training.lr)==1e-4, 'PIE paper STATE training budget and LR required'
while not (B/'manifests/raw_preparation_complete.json').exists():time.sleep(15)
# Read-only shared-memory copies preserve exact float32 data and cell-load sampling.
shm=Path('/dev/shm/pie_uc_state_20261106');shm.mkdir(exist_ok=True)
import fcntl
with open(shm/'prepare.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for ctx in ['hepg2','jurkat','k562','rpe1']:
  dest=shm/(ctx+'.npy')
  if dest.exists():continue
  with h5py.File(B/'data/normalized'/f'{ctx}.h5ad') as h:
   x=h['X'];tmp=dest.with_suffix('.partial.npy');m=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=x.shape)
   for lo in range(0,len(m),1024):m[lo:lo+1024]=x[lo:lo+1024]
   m.flush();ii=np.unique(np.linspace(0,len(m)-1,257,dtype=int));np.testing.assert_array_equal(m[ii],x[ii]);del m;tmp.rename(dest)
cache={}
def cached_obsm(self,indices,key):
 assert key=='X_hvg';ctx=Path(self.h5_path).stem
 if ctx not in cache:cache[ctx]=np.load(shm/(ctx+'.npy'),mmap_mode='r')
 return torch.from_numpy(np.array(cache[ctx][indices],dtype=np.float32))
PerturbationDataset._fetch_obsm_expression_batch=cached_obsm
def cached_single(self,idx,key):return cached_obsm(self,np.array([idx]),key)[0]
PerturbationDataset.fetch_obsm_expression=cached_single
setup=PerturbationDataModule.setup
expected={}
for split in ['train','val']:
 d=json.loads((P/'downloads/official_splits/replogle_wdataset/unseen_ctx'/fold/(split+'.json')).read_text());expected[split]={(c.split('.',1)[1],p) for c,ps in d.items() for p in ps}
expected['test']=set()
for setting in ['unseen_ctx','unseen_ctx_pert','unseen_pert']:
 d=json.loads((P/'downloads/official_splits/replogle_wdataset'/setting/fold/'test.json').read_text());expected['test']|={(c.split('.',1)[1],p) for c,ps in d.items() for p in ps}
def audited_setup(self,stage=None):
 setup(self,stage)
 if getattr(self,'pie_audited',False):return
 report={}
 for split in ['train','val','test']:
  pairs=set();n=0;ctrl=0
  for subset in getattr(self,split+'_datasets'):
   ds=subset.dataset;ix=np.asarray(subset.indices,dtype=np.int64);md=ds.metadata_cache;pc=np.asarray(md.pert_categories)[md.pert_codes[ix]];cc=np.asarray(md.cell_type_categories)[md.cell_type_codes[ix]];pairs|={(str(c),str(p)) for c,p in zip(cc,pc) if str(p)!='non-targeting'};n+=len(ix);ctrl+=int(np.sum(pc=='non-targeting'))
  assert pairs==expected[split],(fold,split,'missing',list(expected[split]-pairs)[:8],'extra',list(pairs-expected[split])[:8]);report[split]={'pairs':len(pairs),'cells_including_controls':n,'controls':ctrl,'exact_PIE_pair_equality':True}
 dims=self.get_var_dims();assert dims['input_dim']==dims['hvg_dim']==dims['output_dim']==6642,dims
 genes=json.loads((P/'downloads/official_data/preprocessed/meta.json').read_text())['genes'];assert list(dims['gene_names'])==genes
 (B/'reports'/f'state_split_audit_{fold}.json').write_text(json.dumps(report,indent=2));self.pie_audited=True;print('PIE_SPLIT_AUDIT_PASS',fold,report,flush=True)
PerturbationDataModule.setup=audited_setup
if '--audit-only' in sys.argv:
 from cell_load.utils.modules import get_datamodule
 dm=get_datamodule(cfg.data.name,OmegaConf.to_container(cfg.data.kwargs),batch_size=int(cfg.training.batch_size),cell_sentence_len=int(cfg.model.kwargs.cell_set_len));dm.setup('fit');raise SystemExit(0)
from state._cli._tx._train import run_tx_train
run_tx_train(cfg)
(B/'runs/state'/fold/'TRAINING_COMPLETE.json').write_text(json.dumps({'fold':fold,'max_steps':int(cfg.training.max_steps),'status':'complete'}))
