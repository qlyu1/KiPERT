"""Official GEARS network and loss; fixed controls-only graph and explicit legal cells."""
from pathlib import Path
import argparse,json,time,sys,ast,importlib.util,types
from resources import R,digest
from isolation import confine
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--protocol',required=True);p.add_argument('--split',type=int,required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--phase',default='train');a=p.parse_args()
B=R/f'data/formal/{a.dataset}/{a.protocol}/fold{a.split}';O=R/f'runs/formal/{a.dataset}/{a.protocol}/fold{a.split}/seed{a.seed}/gears';confine(B,O)
from worker_common import check_gate,seal
check_gate(B,a.phase,'gears')
import numpy as np,pandas as pd,torch,gc
from scipy import sparse
from common_seen import get_force,balanced_order
from training_resume import TrainingResume
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
source=R/'assets/GEARS/gears/model.py';spec=importlib.util.spec_from_file_location('official_gears_network',source);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
# Extract the exact official loss function, avoiding unrelated optional analysis imports.
utilsrc=R/'assets/GEARS/gears/utils.py';tree=ast.parse(utilsrc.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='loss_fct');ns=dict(np=np,torch=torch);exec(compile(ast.Module(body=[node],type_ignores=[]),str(utilsrc),'exec'),ns);loss_fct=ns['loss_fct']
start=time.time();force=get_force(B,'gears',a.seed)
def seed():np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
def load(role):return sparse.load_npz(B/role/'expression.npz').tocsr(),pd.read_csv(B/role/'metadata.tsv',sep='\t',dtype=str)
xt,mt=load('fit');xc,mc=load('controls_fit');targets=sorted(pd.read_csv(B/'target_prior_mapping.tsv',sep='\t').target);ti={g:i for i,g in enumerate(targets)};G=xt.shape[1]
ctrlrows=mc.groupby('context').indices;ctrl={c:np.asarray(xc[ix].mean(0)).ravel() for c,ix in ctrlrows.items()};ctrl_t={c:torch.tensor(v,device='cuda') for c,v in ctrl.items()}
mask={t:np.flatnonzero(np.asarray(xt[ix].sum(0)).ravel()>0) for t,ix in mt.groupby('target').indices.items()}
graph=dict(np.load(R/'assets/gears_graphs'/f'{a.dataset}.npz'))
model_args=dict(num_genes=G,num_perts=len(targets),hidden_size=64,uncertainty=False,num_go_gnn_layers=1,num_gene_gnn_layers=1,decoder_hidden_size=16,no_perturb=False,device='cuda',**{k:torch.from_numpy(v) for k,v in graph.items()})
CFG=dict(method='GEARS-official',seed=a.seed,batch_size=32,learning_rates=[.001,.001/3,.003],max_epochs=40,patience=5,hidden_size=64,direction_lambda=.1,loss='official loss_fct gamma2',sampling='condition-balanced fixed128-step maximum virtual epoch',graph='controls-only coexpression + public GO top20',network_sha256=digest(source),loss_source_sha256=digest(utilsrc),graph_sha256=digest(R/'assets/gears_graphs'/f'{a.dataset}.npz'),worker_sha256=digest(__file__),force=force)
(O/'RUN_LOCK.json').write_text(json.dumps(CFG,indent=2));(O/'worker_source.py').write_bytes(Path(__file__).read_bytes())
def event(**z):
 z=dict(seconds=time.time()-start,**z)
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(z)+'\n')
 print(json.dumps(z),flush=True)
def model():seed();return native.GEARS_Model(model_args).cuda()
def forward(net,labels,controls):
 n=len(labels);data=types.SimpleNamespace(x=controls.reshape(-1,1),pert_idx=[[ti[g] for g in t.split('+')] for t in labels],batch=torch.arange(n,device='cuda').repeat_interleave(G));return net(data)
def step(net,opt,ix,rng):
 rr=mt.iloc[ix];cr=np.array([rng.choice(ctrlrows[c]) for c in rr.context]);truth=torch.from_numpy(xt[ix].toarray()).cuda();yp=forward(net,list(rr.target),torch.from_numpy(xc[cr].toarray()).cuda());loss=0.
 for c,ii in rr.reset_index(drop=True).groupby('context').indices.items():loss=loss+len(ii)/len(ix)*loss_fct(yp[ii],truth[ii],rr.iloc[ii].target.to_numpy(),ctrl=ctrl_t[c],direction_lambda=.1,dict_filter=mask)
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite GEARS loss')
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_value_(net.parameters(),1.);opt.step();return float(loss.detach())
@torch.no_grad()
def predict(net,rows):
 net.eval();ans=[]
 for lo in range(0,len(rows),32):
  sub=rows[lo:lo+32];cc=torch.from_numpy(np.stack([ctrl[c] for t,c in sub])).cuda();ans.append(forward(net,[t for t,c in sub],cc).cpu().numpy())
 return np.maximum(np.concatenate(ans),0)
xv,mv=load('validation');vg=list(mv.groupby(['target','context'],sort=True).indices.items());vr=[k for k,ix in vg];vy=np.stack([np.asarray(xv[ix].mean(0)).ravel() for k,ix in vg])
def score(net):
 err=np.mean((predict(net,vr).astype(float)-vy)**2,axis=1);return float(pd.DataFrame(dict(target=[t for t,c in vr],err=err)).groupby('target').err.mean().mean())
def save(net,p):torch.save({k:v.detach().cpu().clone() for k,v in net.state_dict().items()},p)
resume=TrainingResume(O);selection=resume.selections()
for candidate,lr in enumerate([force['lr']] if force else CFG['learning_rates']):
 if resume.skip(candidate):continue
 net=model();opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-4);sched=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.5);rng=np.random.RandomState(a.seed);best=float('inf');be=0;stale=0;first=1;finished=False
 loaded=resume.restore(candidate,net,opt,rng,sched)
 if loaded:first=loaded['epoch']+1;best=loaded['values']['best'];be=loaded['values']['be'];stale=loaded['values']['stale'];finished=loaded['values']['finished']
 for ep in ([] if finished else range(first,(force['epochs'] if force else 40)+1)):
  net.train();order=balanced_order(mt,B,32,rng);loss=[]
  for lo in range(0,len(order),32):loss.append(step(net,opt,order[lo:lo+32],rng))
  val=force['cv_mse'] if force else score(net);sched.step()
  if force or val<best-1e-9:best=val;be=ep;stale=0;save(net,O/f'candidate_{candidate}_best.pt')
  else:stale+=1
  event(candidate=candidate,epoch=ep,training_loss=float(np.mean(loss)),validation_macro_mse=val)
  finished=(not force and stale>=5) or ep==(force['epochs'] if force else 40)
  resume.save(candidate,ep,net,opt,rng,dict(best=best,be=be,stale=stale,finished=finished),selection,scheduler=sched,force=finished)
  if finished:break
 selection.append(dict(candidate=candidate,lr=lr,best_epoch=be,best_validation_macro_mse=best));del net,opt,sched;gc.collect();torch.cuda.empty_cache()
best=min(selection,key=lambda z:(z['best_validation_macro_mse'],z['candidate']));(O/'selection.json').write_text(json.dumps(dict(chosen=best,candidates=selection),indent=2))
net=model();net.load_state_dict(torch.load(O/f"candidate_{best['candidate']}_best.pt",map_location='cpu',weights_only=True));save(net,O/'model_final.pt');q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str);seal(B,O,predict(net,list(zip(q.target,q.context))),q,'GEARS-official',best,start,a.seed)
