"""Native public STRING+GO TxPert Exphormer training, validation-only selection."""
from pathlib import Path
import argparse,sys,json,time,hashlib,copy
R=Path('@STUDY@/uc_seen012_benchmark_20261005');sys.path.insert(0,str(R/'src'))
from isolation import confine
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split',type=int,required=True);p.add_argument('--protocol',choices=['seen0','seen1','seen2'],default='seen0');p.add_argument('--seed',type=int,default=17);p.add_argument('--phase',choices=['smoke','train'],default='train');a=p.parse_args();SEED=a.seed;B=R/'data/formal'/a.dataset/a.protocol/f'fold{a.split}';O=R/'runs'/('formal' if a.phase=='train' else 'integration')/a.dataset/a.protocol/f'fold{a.split}'/f'seed{SEED}'/'txpert';confine(B,O)
(O/'worker_source.py').write_bytes(Path(__file__).read_bytes())
from worker_common import check_gate
check_gate(B,a.phase,'txpert')
import numpy as np,pandas as pd,torch,random,gc
from scipy import sparse
from omegaconf import OmegaConf
sys.path.insert(0,str(R/'assets/txpert_compat'))
from gspp.data.graphmodule import GSPGraph
from gspp.models.txpert import TxPert
from resources import Resolver
from common_seen import get_force,balanced_order
force=get_force(B,'txpert',SEED)
start=time.time();torch.set_num_threads(2);device='cuda';torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def event(**z):
 z={'seconds':time.time()-start,**z}
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(z)+'\n')
 print(json.dumps(z),flush=True)
def load(role):return sparse.load_npz(B/role/'expression.npz').tocsr(),pd.read_csv(B/role/'metadata.tsv',sep='\t',dtype=str)
xt,mt=load('fit');xc,mc=load('controls_fit');q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str);mapping=pd.read_csv(B/'target_prior_mapping.tsv',sep='\t',dtype=str).fillna('');targets=sorted(mapping.target);targetid={t:i for i,t in enumerate(targets)};res=Resolver();aliases={n:i for t,i in targetid.items() for n in res.candidates(t)};geneids=pd.read_csv(B/'genes.tsv',sep='\t',dtype=str).gene_id;gene2id={n:i for i,t in enumerate(geneids) for n in res.candidates(t)};ctrl={c:np.asarray(xc[ix].mean(0)).ravel() for c,ix in mc.groupby('context',sort=True).indices.items()};G=xt.shape[1]
cfg=OmegaConf.to_container(OmegaConf.load(R/'repos/TxPert/configs/config-exphormer-mg.yaml'),resolve=False);graph=GSPGraph(pert2id=aliases,gene2id=gene2id,graph_cfg=copy.deepcopy(cfg['graph']['graph_cfg']),cache_dir=str(R/'repos/TxPert/data/graphs'))
# GSPGraph counts dictionary keys. Alias keys map to the same stable node index; correct node count without altering edges.
for k,(ei,ew,n) in graph.graph_dict.items():
 assert int(ei.max())<len(targets);graph.graph_dict[k]=(ei,ew,len(targets))
CFG={'method':'TxPert-public-STRING-GO','native_config':'config-exphormer-mg.yaml','pretrained_target_checkpoint':False,'seed':SEED,'batch_size':64,'learning_rates':[.001,.001/3,.003],'max_epochs':100,'patience':10,'optimizer':'Adam','weight_decay':0.,'scheduler':'none','control':'context C_fit mean, author avg_cntr=true','gene_axis_sha256':sha(B/'genes.tsv'),'data_manifest_sha256':sha(B/'READY.json'),'native_source_sha256':sha(R/'repos/TxPert/gspp/models/txpert.py'),'go_graph_sha256':sha(R/'repos/TxPert/data/graphs/go/go_top_50.csv'),'string_graph_sha256':sha(R/'repos/TxPert/data/graphs/string/v11.5.parquet'),'worker_sha256':sha(__file__),'architecture':cfg['model'],'refit':'none; preserve validation-selected training checkpoint','test_scoring':False};(O/'RUN_LOCK.json').write_text(json.dumps(CFG,indent=2))
def seed():random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def model(lr):
 seed();c=copy.deepcopy(cfg['model']);c.pop('model_type');c.pop('slow_benchmark');cn=c.pop('cntr_model');pn=c.pop('pert_model');m=TxPert(G,G,G,graph,cn,pn,device=device,**c).to(device);return m,torch.optim.Adam(m.parameters(),lr=lr)
def batches(order):
 z=[order[i:i+64] for i in range(0,len(order),64)]
 if len(z)>1 and len(z[-1])==1:z[-2]=np.r_[z[-2],z[-1]];z.pop()
 return z
def forward(m,targets_,contexts):
 xx=torch.from_numpy(np.stack([ctrl[c] for c in contexts])).to(device);ids=[torch.tensor([targetid[g] for g in t.split('+')],dtype=torch.long,device=device) for t in targets_];return m(xx,ids,None)
def step(m,opt,x,meta,ix):
 m.train();z=meta.iloc[ix];yp,mu,lv=forward(m,z.target,z.context);loss,_=m.loss(yp,mu,lv,torch.from_numpy(x[ix].toarray()).to(device));assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();return float(loss.detach())
@torch.no_grad()
def predict(m,rows):
 m.eval();pp=[]
 for lo in range(0,len(rows),64):
  rr=rows[lo:lo+64];yp,_,_=forward(m,[t for t,c in rr],[c for t,c in rr]);pp.append(yp.cpu().numpy())
 return np.maximum(np.concatenate(pp),0)
def validation(m,x,meta):
 rows=[];truth=[]
 for (t,c),ix in meta.groupby(['target','context'],sort=True).indices.items():rows.append((t,c));truth.append(np.asarray(x[ix].mean(0)).ravel())
 yp=predict(m,rows);err=np.mean((yp.astype(float)-truth)**2,axis=1);by={}
 for (t,c),v in zip(rows,err):by.setdefault(t,[]).append(v)
 return float(np.mean([np.mean(v) for v in by.values()]))
def save(m,name):torch.save({k:v.detach().cpu().clone() for k,v in m.state_dict().items()},O/name)
if a.phase=='smoke':
 m,opt=model(.001)
 loss=step(m,opt,xt,mt,np.flatnonzero(mt.target.str.contains('+',regex=False).to_numpy())[:64])
 row=mt[mt.target.str.contains('+',regex=False)].iloc[0]
 pred=predict(m,[(row.target,row.context)])
 swap=predict(m,[('+'.join(row.target.split('+')[::-1]),row.context)])
 repeat=predict(m,[(row.target,row.context)])
 event(phase='SYMMETRY_PROBE',max_abs_error=float(np.max(np.abs(pred-swap))),finite=bool(np.isfinite(pred).all()),repeat_error=float(np.max(np.abs(pred-repeat))))
 assert np.isfinite(pred).all() and np.allclose(pred,swap,atol=1e-6)
 event(status='SMOKE_PASS',loss=loss,genes=G,formal_training=False)
 sys.exit(0)
xv,mv=load('validation');steps_per_epoch=len(batches(balanced_order(mt,B,64,np.random.RandomState(0))));candidates=[]
from training_resume import TrainingResume
resume=TrainingResume(O)
candidates=resume.selections()
for candidate,lr in enumerate([force['lr']] if force else CFG['learning_rates']):
 if resume.skip(candidate):continue
 m,opt=model(lr);rng=np.random.RandomState(SEED);best=float('inf');bestep=0;stale=0
 loaded=resume.restore(candidate,m,opt,rng)
 first_epoch=1;finished=False
 if loaded:
  first_epoch=loaded['epoch']+1;best=loaded['values']['best'];bestep=loaded['values']['bestep'];stale=loaded['values']['stale'];finished=loaded['values']['finished']
 for ep in ([] if finished else range(first_epoch,(force['epochs'] if force else 100)+1)):
  losses=[step(m,opt,xt,mt,ix) for ix in batches(balanced_order(mt,B,64,rng))];val=force['cv_mse'] if force else validation(m,xv,mv)
  if force or val<best-1e-9:best=val;bestep=ep;stale=0;save(m,f'candidate_{candidate}_best.pt')
  else:stale+=1
  event(phase='SELECTION',candidate=candidate,epoch=ep,validation_macro_mse=val,training_loss=float(np.mean(losses)))
  finished=(not force and stale>=10) or ep==(force['epochs'] if force else 100)
  resume.save(candidate,ep,m,opt,rng,{'best':best,'bestep':bestep,'stale':stale,'finished':finished},candidates,force=finished)
  if finished:break
 candidates.append({'candidate':candidate,'lr':lr,'best_epoch':bestep,'best_validation_macro_mse':best,'selected_updates':bestep*steps_per_epoch});del m,opt;gc.collect();torch.cuda.empty_cache()
best=min(candidates,key=lambda z:(z['best_validation_macro_mse'],z['best_epoch'],z['candidate']));(O/'selection.json').write_text(json.dumps({'candidates':candidates,'chosen':best},indent=2))
m,opt=model(best['lr']);m.load_state_dict(torch.load(O/f"candidate_{best['candidate']}_best.pt",map_location='cpu',weights_only=True));save(m,'model_final.pt')
yp=predict(m,list(zip(q.target,q.context)))
from worker_common import seal
seal(B,O,yp,q,'TxPert-public-STRING-GO',best,start,SEED)
