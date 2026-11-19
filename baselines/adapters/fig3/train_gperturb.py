"""Official GPerturb Gaussian architecture/loss with frozen Figure 3 splits."""
from pathlib import Path
import argparse, sys, json, time, importlib.util
R = Path('@STUDY@/uc_seen012_benchmark_20261005')
sys.path.insert(0, str(R/'src'))
p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True); p.add_argument('--split', type=int, required=True)
p.add_argument('--protocol', choices=['seen0','seen1','seen2'], default='seen0')
p.add_argument('--seed', type=int, default=17); p.add_argument('--phase', choices=['smoke','train'], default='train')
a = p.parse_args(); method = 'gperturb'
B = R/'data/formal'/a.dataset/a.protocol/f'fold{a.split}'
O = R/'runs'/('formal' if a.phase=='train' else 'integration')/a.dataset/a.protocol/f'fold{a.split}'/f'seed{a.seed}'/method
from isolation import confine
confine(B,O)
from worker_common import check_gate, seal
from resources import digest
from p0_support import finish_inner
from common_seen import get_force,balanced_order
check_gate(B,a.phase,method)
import numpy as np, pandas as pd, torch, gc
from scipy import sparse
source = R/'assets/replacement_repos/GPerturb/GPerturb/GPerturb_model.py'
spec = importlib.util.spec_from_file_location('official_gperturb',source)
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
torch.set_num_threads(int(__import__('os').environ.get('FIG3_TORCH_THREADS','4'))); torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.benchmark=False
def seed():
 np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
seed(); start=time.time()
xt=sparse.load_npz(B/'fit/expression.npz').tocsr(); mt=pd.read_csv(B/'fit/metadata.tsv',sep='\t',dtype=str)
xc=sparse.load_npz(B/'controls_fit/expression.npz').tocsr(); mc=pd.read_csv(B/'controls_fit/metadata.tsv',sep='\t',dtype=str)
gm=pd.read_csv(B/'target_prior_mapping.tsv',sep='\t',dtype=str)
targets=sorted(gm.target.unique()); ti={g:i for i,g in enumerate(targets)}
contexts=sorted(mc.context.unique()); ci={c:i for i,c in enumerate(contexts)}
def design(labels):
 z=np.zeros((len(labels),len(targets)),np.float32)
 for i,t in enumerate(labels):
  if t=='CTRL':continue
  for g in t.split('+'):z[i,ti[g]]=1
 return torch.from_numpy(z)
def context_design(labels):
 return torch.eye(len(contexts))[torch.tensor([ci[c] for c in labels])]
x=sparse.vstack([xt,xc],format='csr')
cond=design(list(mt.target)+['CTRL']*len(mc)); cell=context_design(list(mt.context)+list(mc.context))
force=get_force(B,method,a.seed)
cfg={'model':'GPerturb-Gaussian','official_source_sha256':digest(source),'worker_sha256':digest(__file__),
 'axis_sha256':digest(B/'genes.tsv'),'fit_response_sha256':digest(B/'fit/expression.npz'),
 'seed':a.seed,'hidden_node':700,'hidden_layers':[4,4],'tau':1.,'batch_size':100,
 'learning_rates':[.001,.001/3,.003],'max_epochs':250,'early_stop_patience':20,
 'gp_hyperparameters':[1.,.1,1.,.1,1.,.1],'extra_reg':.01,'training_cells':x.shape[0],
 'input':'binary multigene design; context one-hot; no post-perturbation cell covariates',
 'inference':'native Gaussian_estimates mean formula; no observed response argument',
 'scope':'P0/P1 combinatorial extrapolation; P2 unseen input columns are untrained, exploratory only',
 'test_scoring':False,'force':force}
(O/'RUN_LOCK.json').write_text(json.dumps(cfg,indent=2)); (O/'worker_source.py').write_bytes(Path(__file__).read_bytes())
def emit(z):
 z={'elapsed_seconds':time.time()-start,**z}
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(z)+'\n')
 print(json.dumps(z),flush=True)
def model():
 seed()
 return native.GPerturb_Gaussian(len(targets),xt.shape[1],len(contexts),x.shape[0],700,4,4,torch.tensor(1.,device='cuda')).cuda()
def step(net,opt,ix):
 opt.zero_grad(set_to_none=True)
 loss=net.normal_loss(torch.from_numpy(x[ix].toarray()).cuda(),cond[ix].cuda(),cell[ix].cuda(),1.,.1,1.,.1,1.,.1)
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite official GPerturb loss')
 loss.backward(); opt.step(); return float(loss.detach())
@torch.no_grad()
def predict(net,labels,ctx):
 net.eval(); ans=[]
 for i in range(0,len(labels),100):
  out=net(design(labels[i:i+100]).cuda(),context_design(ctx[i:i+100]).cuda())
  ans.append((out[6]+torch.sigmoid(out[4])*out[0]).cpu().numpy())
 result=np.concatenate(ans); assert np.isfinite(result).all(); return np.maximum(result,0)
def save(net,path):
 torch.save({k:v.detach().cpu().clone() for k,v in net.state_dict().items()},path)
if a.phase=='smoke':
 net=model(); opt=torch.optim.Adam(net.parameters(),lr=.001)
 ix=np.flatnonzero(mt.target.str.contains('+',regex=False).to_numpy())
 assert len(ix)>0; torch.cuda.reset_peak_memory_stats()
 row=mt.iloc[ix[0]]; losses=[]
 for j in range(12):
  if j==2:torch.cuda.synchronize(); t0=time.time()
  losses.append(step(net,opt,np.resize(ix,1200)[j*100:(j+1)*100]))
 torch.cuda.synchronize(); seconds=(time.time()-t0)/10
 assert all(v.grad is None or torch.isfinite(v.grad).all() for v in net.parameters())
 y=predict(net,[row.target,'+'.join(row.target.split('+')[::-1]),row.target.split('+')[0]],[row.context]*3)
 assert np.allclose(y[0],y[1],atol=1e-6); assert not np.allclose(y[0],y[2],atol=1e-6)
 for denied in [B/'evaluation_only/expression.npz']:
  try:
   with denied.open('rb') as f:f.read(1)
   raise AssertionError('test response readable')
  except PermissionError:pass
 torch.cuda.synchronize(); epoch_start=time.time(); order=np.random.RandomState(a.seed).permutation(x.shape[0]); net.train()
 for lo in range(0,len(order),100):step(net,opt,order[lo:lo+100])
 torch.cuda.synchronize(); full_epoch=time.time()-epoch_start
 r={'status':'PASS','formal_training':False,'full_training_epoch_seconds':full_epoch,'timing_context':'shared GPU with existing Scouter worker; no validation/checkpoint time included','native_loss':losses[-1],'seconds':time.time()-start,
 'seconds_per_100_cell_step':seconds,'projected_full_epoch_seconds':seconds*((x.shape[0]+99)//100),
 'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'test_response_denied':True,'pair_swap_max_error':float(abs(y[0]-y[1]).max()),'output_genes':xt.shape[1]}
 (O/'result.json').write_text(json.dumps(r,indent=2)); emit(r); sys.exit(0)
if force is None:
 xv=sparse.load_npz(B/'validation/expression.npz').tocsr(); mv=pd.read_csv(B/'validation/metadata.tsv',sep='\t',dtype=str)
 groups=list(mv.groupby(['target','context'],sort=True).indices.items())
 vt=[k[0] for k,ix in groups]; vc=[k[1] for k,ix in groups]
 vy=np.stack([np.asarray(xv[ix].mean(0)).ravel() for k,ix in groups])
 def score(net):
  err=((predict(net,vt,vc).astype(np.float64)-vy)**2).mean(1)
  return float(pd.DataFrame({'target':vt,'e':err}).groupby('target').e.mean().mean())
else:
 def score(net):return float(force['cv_mse'])
from training_resume import TrainingResume
resume=TrainingResume(O,interval=5)
selection=resume.selections()
for n,lr in enumerate([force['lr']] if force else cfg['learning_rates']):
 if resume.skip(n):continue
 net=model(); opt=torch.optim.Adam(net.parameters(),lr=lr); rng=np.random.RandomState(a.seed)
 best=float('inf'); be=0; stale=0; path=O/f'candidate_{n}_best.pt'
 loaded=resume.restore(n,net,opt,rng)
 first_epoch=1; finished=False
 if loaded:
  first_epoch=loaded['epoch']+1;best=loaded['values']['best'];be=loaded['values']['be'];stale=loaded['values']['stale'];finished=loaded['values']['finished']
 for epoch in ([] if finished else range(first_epoch,(force['epochs'] if force else cfg['max_epochs'])+1)):
  net.train(); order=balanced_order(mt,B,100,rng,control_meta=mc); loss=[]; t0=time.time()
  bad=False
  for i in range(0,len(order),100):
   try:loss.append(step(net,opt,order[i:i+100]))
   except FloatingPointError:
    bad=True;emit({'phase':'CANDIDATE_NONFINITE','candidate':n,'epoch':epoch});break
  if bad:
   if force:raise FloatingPointError('Nonfinite forced common configuration')
   break
  val=score(net); improved=bool(force) or val<best-1e-9
  if improved:best=val; be=epoch; stale=0; save(net,path)
  else:stale+=1
  emit({'phase':'P0_REFIT' if force else 'SELECTION','candidate':n,'lr':lr,'epoch':epoch,'epoch_seconds':time.time()-t0,'training_loss':float(np.mean(loss)),'validation_macro_mse':val,'improved':improved})
  finished=(not force and stale>=cfg['early_stop_patience']) or epoch==(force['epochs'] if force else cfg['max_epochs'])
  resume.save(n,epoch,net,opt,rng,{'best':best,'be':be,'stale':stale,'finished':finished},selection,force=finished)
  if finished:break
 if be>0:selection.append({'candidate':n,'lr':lr,'best_epoch':be,'best_validation_macro_mse':best})
 del net,opt; gc.collect(); torch.cuda.empty_cache()
if a.protocol=='P0':finish_inner(B,O,selection,method,a.seed)
best=min(selection,key=lambda z:(z['best_validation_macro_mse'],z['candidate']))
(O/'selection.json').write_text(json.dumps({'candidates':selection,'chosen':best},indent=2))
net=model(); net.load_state_dict(torch.load(O/f"candidate_{best['candidate']}_best.pt",map_location='cpu',weights_only=True)); save(net,O/'model_final.pt')
q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str)
seal(B,O,predict(net,list(q.target),list(q.context)),q,'GPerturb-Gaussian',best,start,a.seed)
if force:
 r=json.loads((O/'result.json').read_text());r['refit']=True;r['refit_scope']='P0 all legal singles';(O/'result.json').write_text(json.dumps(r,indent=2))
