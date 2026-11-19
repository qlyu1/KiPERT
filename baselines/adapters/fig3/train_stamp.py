"""Official STAMP three-task network/loss with label-free expression reconstruction."""
from pathlib import Path
import argparse, sys, json, time, importlib.util
R=Path('@STUDY@/uc_seen012_benchmark_20261005');sys.path.insert(0,str(R/'src'))
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split',type=int,required=True)
p.add_argument('--protocol',choices=['seen0','seen1','seen2'],default='seen0');p.add_argument('--seed',type=int,default=17)
p.add_argument('--phase',choices=['smoke','train'],default='train');a=p.parse_args();method='stamp'
B=R/'data/formal'/a.dataset/a.protocol/f'fold{a.split}'
O=R/'runs'/('formal' if a.phase=='train' else 'integration')/a.dataset/a.protocol/f'fold{a.split}'/f'seed{a.seed}'/method
from isolation import confine
confine(B,O)
from worker_common import check_gate,seal
from resources import digest
from p0_support import finish_inner
from common_seen import get_force,balanced_order
check_gate(B,a.phase,method)
import numpy as np,pandas as pd,torch,gc
from scipy import sparse,stats
source=R/'assets/replacement_repos/STAMP/stamp/Modules.py'
s=importlib.util.spec_from_file_location('official_stamp_modules',source);native=importlib.util.module_from_spec(s);s.loader.exec_module(native)
torch.set_num_threads(int(__import__('os').environ.get('FIG3_TORCH_THREADS','4')));torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.benchmark=False
def seed():
 np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
seed();start=time.time()
xt=sparse.load_npz(B/'fit/expression.npz').tocsr();mt=pd.read_csv(B/'fit/metadata.tsv',sep='\t',dtype=str)
xc=sparse.load_npz(B/'controls_fit/expression.npz').tocsr();mc=pd.read_csv(B/'controls_fit/metadata.tsv',sep='\t',dtype=str)
genes=pd.read_csv(B/'genes.tsv',sep='\t');mapping=pd.read_csv(B/'target_prior_mapping.tsv',sep='\t')
bankfile=R/'assets/stamp_static_gene_embeddings.pt';bank=torch.load(bankfile,map_location='cpu',weights_only=True)
emb=bank[torch.tensor(genes.scgpt_token.to_numpy(),dtype=torch.long)].cuda()
target_emb={r.target:bank[int(r.scgpt_token)] for r in mapping.itertuples()}
def encode(labels):
 return torch.stack([torch.stack([target_emb[g] for g in t.split('+')]).mean(0) for t in labels]).cuda()
def moments(x):
 x=x.astype(np.float64);n=x.shape[0];mu=np.asarray(x.mean(0)).ravel()
 var=np.maximum(0,np.asarray(x.power(2).mean(0)).ravel()-mu**2)*n/max(n-1,1)
 return mu,var,n
control={c:moments(xc[ix]) for c,ix in mc.groupby('context',sort=True).indices.items()}
rows=list(mt.groupby(['target','context'],sort=True).indices.items());labels=[k[0] for k,ix in rows]
inputs=encode(labels);levels=[[],[],[]];row_meta=[]
for (t,c),ix in rows:
 mu,var,n=moments(xt[ix]);cm,cv,cn=control[c]
 pv=stats.ttest_ind_from_stats(mu,np.sqrt(var),n,cm,np.sqrt(cv),cn,equal_var=False).pvalue
 pv=np.nan_to_num(pv,nan=1.);order=np.argsort(pv);adj=np.empty(len(pv));adj[order]=np.minimum(1,np.minimum.accumulate((pv[order]*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1])
 fc=(np.expm1(mu+.1)+1e-9)/(np.expm1(cm+.1)+1e-9)
 levels[0].append((adj<=.01).astype(np.float32));levels[1].append((fc>=1).astype(np.float32));levels[2].append(fc.astype(np.float32))
 row_meta.append({'target':t,'context':c,'cells':n,'de_genes':int((adj<=.01).sum())})
level=[torch.from_numpy(np.stack(z)).cuda() for z in levels];assert all(torch.isfinite(z).all() for z in level)
force=get_force(B,method,a.seed)
cfg={'model':'STAMP','variant':'author scGPT static gene vocabulary + STAMP; no transformer training or cell encoding',
 'official_source_sha256':digest(source),'worker_sha256':digest(__file__),'embedding_sha256':digest(bankfile),
 'axis_sha256':digest(B/'genes.tsv'),'fit_response_sha256':digest(B/'fit/expression.npz'),'seed':a.seed,
 'learning_rates':[.0001,.0001/3,.0003],'max_epochs':200,'early_stop_patience':20,'batch_size':64,
 'optimizer':'AdamW default weight_decay=0.01','native_loss_weights':[.1,.5,.4],
 'native_architecture':'First 512-1024-2048-G dropout0.9; Second 1024-128-64-32; Third32-16-8-1 exp',
 'training_unit':'target/context pseudobulk; all legal training cells contribute, context-matched C_fit',
 'training_labels':'Welch t-test BH adjusted p<=.01; fold-change of expm1(mean(log1p)+.1), Scanpy formula',
 'pair_encoding':'native arithmetic mean of two static token embeddings',
 'expression_adapter':'predicted DEG p>=0.5 chooses predicted fold-change, else fold-change1; invert training transform with matched C_fit; preserve raw reconstructed values, common scoring projects separately',
 'checkpoint_selection':'shared full-axis validation macro MSE, one joint checkpoint; no oracle DEG masks',
 'test_scoring':False,'force':force,'training_rows':len(rows)}
(O/'RUN_LOCK.json').write_text(json.dumps(cfg,indent=2));(O/'worker_source.py').write_bytes(Path(__file__).read_bytes())
(O/'training_label_summary.json').write_text(json.dumps(row_meta,indent=2))
def emit(z):
 z={'elapsed_seconds':time.time()-start,**z}
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(z)+'\n')
 print(json.dumps(z),flush=True)
def model():
 seed();return native.TaskCombineLayer_multi_task(emb.shape[1],xt.shape[1],gene_embedding_notebook=emb).cuda()
def step(net,opt,ix):
 if not level[0][ix].any():return None
 opt.zero_grad(set_to_none=True);losses=net(inputs[ix],*(z[ix] for z in level))
 loss=.1*losses[0]+.5*losses[1]+.4*losses[2]
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite official STAMP loss')
 loss.backward();opt.step();return float(loss.detach())
@torch.no_grad()
def raw(net,ts):
 net.eval();ins=encode(ts);p1=net.first_level_layer(ins)
 p2,mask,hid=net.second_level_layer(torch.ones_like(p1),ins)
 p3,_=net.third_level_layer(hid,mask)
 return p1,p2.squeeze(-1),p3.squeeze(-1)
@torch.no_grad()
def predict(net,ts,cs,return_levels=False):
 unique=sorted(set(ts));cache={}
 for i in range(0,len(unique),32):
  sub=unique[i:i+32];out=raw(net,sub)
  for j,t in enumerate(sub):cache[t]=[z[j].cpu().numpy() for z in out]
 preds=[]
 for t,c in zip(ts,cs):
  p1,p2,fc=cache[t];fc=np.where(p1>=.5,fc.astype(np.float64),1.)
  cm=control[c][0];y=np.log1p(np.maximum(0,fc*(np.expm1(cm+.1)+1e-9)-1e-9))-.1
  preds.append(y)
 ans=np.maximum(np.stack(preds).astype(np.float32),0);assert np.isfinite(ans).all()
 return (ans,cache) if return_levels else ans
def save(net,path):torch.save({k:v.detach().cpu().clone() for k,v in net.state_dict().items()},path)
if a.phase=='smoke':
 net=model();opt=torch.optim.AdamW(net.parameters(),lr=.0001);torch.cuda.reset_peak_memory_stats()
 ix=np.flatnonzero(np.array(['+' in t for t in labels]) & np.array([r['de_genes']>0 for r in row_meta]));assert len(ix)>0
 losses=[]
 for j in range(12):
  if j==2:torch.cuda.synchronize();t0=time.time()
  losses.append(step(net,opt,np.resize(ix,64)))
 torch.cuda.synchronize();seconds=(time.time()-t0)/10
 assert all(v.grad is None or torch.isfinite(v.grad).all() for v in net.parameters())
 t=labels[ix[0]];ts=[t,'+'.join(t.split('+')[::-1]),t.split('+')[0]]
 vals=[z.cpu().numpy() for z in raw(net,ts)];assert all(np.allclose(z[0],z[1],atol=1e-6) for z in vals)
 assert any(not np.allclose(z[0],z[2],atol=1e-6) for z in vals)
 pred=predict(net,ts,[row_meta[ix[0]]['context']]*3)
 # The official second/third heads never use the supplied mask to form their predictions.
 with torch.no_grad():
  ins=encode(ts);one=net.second_level_layer(torch.ones((3,xt.shape[1]),device='cuda'),ins)
  zero=net.second_level_layer(torch.zeros((3,xt.shape[1]),device='cuda'),ins)
  assert torch.equal(one[0],zero[0]) and torch.equal(one[2],zero[2])
 try:
  with (B/'evaluation_only/expression.npz').open('rb') as f:f.read(1)
  raise AssertionError('test response readable')
 except PermissionError:pass
 torch.cuda.synchronize(); epoch_start=time.time(); order=np.random.RandomState(a.seed).permutation(len(rows)); net.train()
 for lo in range(0,len(order),64):step(net,opt,order[lo:lo+64])
 torch.cuda.synchronize(); full_epoch=time.time()-epoch_start
 r={'status':'PASS','formal_training':False,'full_training_epoch_seconds':full_epoch,'timing_context':'shared GPU with existing Scouter worker; no validation/checkpoint time included','seconds':time.time()-start,'native_loss':losses[-1],
 'seconds_per_64_condition_step':seconds,'projected_full_epoch_seconds':seconds*((len(rows)+63)//64),
 'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'output_genes':xt.shape[1],
 'test_response_denied':True,'oracle_mask_invariance':True,'pair_swap_max_error':float(abs(pred[0]-pred[1]).max())}
 (O/'result.json').write_text(json.dumps(r,indent=2));emit(r);sys.exit(0)
if force is None:
 xv=sparse.load_npz(B/'validation/expression.npz').tocsr();mv=pd.read_csv(B/'validation/metadata.tsv',sep='\t',dtype=str)
 groups=list(mv.groupby(['target','context'],sort=True).indices.items());vt=[k[0] for k,ix in groups];vc=[k[1] for k,ix in groups]
 vy=np.stack([np.asarray(xv[ix].mean(0)).ravel() for k,ix in groups])
 def score(net):
  err=((predict(net,vt,vc).astype(np.float64)-vy)**2).mean(1)
  return float(pd.DataFrame({'target':vt,'e':err}).groupby('target').e.mean().mean())
else:
 def score(net):return float(force['cv_mse'])
from training_resume import TrainingResume
resume=TrainingResume(O,interval=25)
selection=resume.selections()
for n,lr in enumerate([force['lr']] if force else cfg['learning_rates']):
 if resume.skip(n):continue
 net=model();opt=torch.optim.AdamW(net.parameters(),lr=lr);rng=np.random.RandomState(a.seed)
 best=float('inf');be=0;stale=0;path=O/f'candidate_{n}_best.pt'
 loaded=resume.restore(n,net,opt,rng)
 first_epoch=1; finished=False
 if loaded:
  first_epoch=loaded['epoch']+1;best=loaded['values']['best'];be=loaded['values']['be'];stale=loaded['values']['stale'];finished=loaded['values']['finished']
 for epoch in ([] if finished else range(first_epoch,(force['epochs'] if force else cfg['max_epochs'])+1)):
  net.train();order=balanced_order(pd.DataFrame(row_meta),B,64,rng,pseudobulk=True);loss=[];t0=time.time()
  for i in range(0,len(order),64):
   z=step(net,opt,order[i:i+64])
   if z is not None:loss.append(z)
  if not loss:raise RuntimeError('No training DEG supports native STAMP loss; do not fabricate supervision')
  val=score(net);improved=bool(force) or val<best-1e-9
  if improved:best=val;be=epoch;stale=0;save(net,path)
  else:stale+=1
  emit({'phase':'P0_REFIT' if force else 'SELECTION','candidate':n,'lr':lr,'epoch':epoch,'epoch_seconds':time.time()-t0,'training_loss':float(np.mean(loss)),'validation_macro_mse':val,'improved':improved})
  finished=(not force and stale>=cfg['early_stop_patience']) or epoch==(force['epochs'] if force else cfg['max_epochs'])
  resume.save(n,epoch,net,opt,rng,{'best':best,'be':be,'stale':stale,'finished':finished},selection,force=finished)
  if finished:break
 selection.append({'candidate':n,'lr':lr,'best_epoch':be,'best_validation_macro_mse':best});del net,opt;gc.collect();torch.cuda.empty_cache()
if a.protocol=='P0':finish_inner(B,O,selection,method,a.seed)
best=min(selection,key=lambda z:(z['best_validation_macro_mse'],z['candidate']));(O/'selection.json').write_text(json.dumps({'candidates':selection,'chosen':best},indent=2))
net=model();net.load_state_dict(torch.load(O/f"candidate_{best['candidate']}_best.pt",map_location='cpu',weights_only=True));save(net,O/'model_final.pt')
q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str);yp,levels=predict(net,list(q.target),list(q.context),True)
np.savez_compressed(O/'subtask_predictions_sealed.npz',target=np.array(sorted(levels)),de_probability=np.stack([levels[t][0] for t in sorted(levels)]),up_probability=np.stack([levels[t][1] for t in sorted(levels)]),fold_change=np.stack([levels[t][2] for t in sorted(levels)]))
seal(B,O,yp,q,'STAMP-static-scGPT',best,start,a.seed)
if force:
 r=json.loads((O/'result.json').read_text());r['refit']=True;r['refit_scope']='P0 all legal singles';(O/'result.json').write_text(json.dumps(r,indent=2))
