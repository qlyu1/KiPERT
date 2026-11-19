"""Native public STRING+GO TxPert Exphormer training, validation-only selection."""
from pathlib import Path
import argparse,sys,json,time,hashlib,copy
R=Path('@STUDY@/uc_fig2_benchmark_20260911');sys.path.insert(0,str(R/'src'))
from isolation import confine
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split',type=int,required=True);p.add_argument('--phase',choices=['smoke','train'],default='train');a=p.parse_args();B=R/'data/formal'/a.dataset/f'split{a.split}';O=R/'runs'/('formal' if a.phase=='train' else 'integration')/a.dataset/f'split{a.split}'/'txpert';confine(B,O)
import numpy as np,pandas as pd,torch,random,gc
from scipy import sparse
from omegaconf import OmegaConf
sys.path.insert(0,str(R/'repos/TxPert'))
from gspp.data.graphmodule import GSPGraph
from gspp.models.txpert import TxPert
from prepare_formal import Resolver
start=time.time();torch.set_num_threads(2);device='cuda';torch.backends.cuda.matmul.allow_tf32=True

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
CFG={'method':'TxPert-public-STRING-GO','native_config':'config-exphormer-mg.yaml','pretrained_target_checkpoint':False,'seed':42,'batch_size':64,'learning_rates':[.001,.001/3,.003],'max_epochs':100,'patience':10,'optimizer':'Adam','weight_decay':0.,'scheduler':'none','control':'context C_fit mean, author avg_cntr=true','gene_axis_sha256':sha(B/'genes.tsv'),'data_manifest_sha256':sha(B/'READY.json'),'native_source_sha256':sha(R/'repos/TxPert/gspp/models/txpert.py'),'go_graph_sha256':sha(R/'repos/TxPert/data/graphs/go/go_top_50.csv'),'string_graph_sha256':sha(R/'repos/TxPert/data/graphs/string/v11.5.parquet'),'worker_sha256':sha(__file__),'architecture':cfg['model'],'refit':'selected optimizer updates','test_scoring':False};(O/'RUN_LOCK.json').write_text(json.dumps(CFG,indent=2))
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def model(lr):
 seed();c=copy.deepcopy(cfg['model']);c.pop('model_type');c.pop('slow_benchmark');cn=c.pop('cntr_model');pn=c.pop('pert_model');m=TxPert(G,G,G,graph,cn,pn,device=device,**c).to(device);return m,torch.optim.Adam(m.parameters(),lr=lr)
def batches(order):
 z=[order[i:i+64] for i in range(0,len(order),64)]
 if len(z)>1 and len(z[-1])==1:z[-2]=np.r_[z[-2],z[-1]];z.pop()
 return z
def forward(m,targets_,contexts):
 xx=torch.from_numpy(np.stack([ctrl[c] for c in contexts])).to(device);ids=[torch.tensor([targetid[t]],dtype=torch.long,device=device) for t in targets_];return m(xx,ids,None)
def step(m,opt,x,meta,ix):
 m.train();z=meta.iloc[ix];yp,mu,lv=forward(m,z.target,z.context);loss,_=m.loss(yp,mu,lv,torch.from_numpy(x[ix].toarray()).to(device));assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();return float(loss.detach())
@torch.no_grad()
def predict(m,rows):
 m.eval();pp=[]
 for lo in range(0,len(rows),64):
  rr=rows[lo:lo+64];yp,_,_=forward(m,[t for t,c in rr],[c for t,c in rr]);pp.append(yp.cpu().numpy())
 return np.concatenate(pp)
def validation(m,x,meta):
 rows=[];truth=[]
 for (t,c),ix in meta.groupby(['target','context'],sort=True).indices.items():rows.append((t,c));truth.append(np.asarray(x[ix].mean(0)).ravel())
 yp=predict(m,rows);err=np.sqrt(np.mean((yp.astype(float)-truth)**2,axis=1));by={}
 for (t,c),v in zip(rows,err):by.setdefault(t,[]).append(v)
 return float(np.mean([np.mean(v) for v in by.values()]))
def save(m,name):torch.save({k:v.detach().cpu().clone() for k,v in m.state_dict().items()},O/name)
if a.phase=='smoke':
 m,opt=model(.001);loss=step(m,opt,xt,mt,np.arange(64));pred=predict(m,list(zip(mt.target[:2],mt.context[:2])));assert np.isfinite(pred).all();event(status='SMOKE_PASS',loss=loss,genes=G,formal_training=False);sys.exit(0)
xv,mv=load('validation');steps_per_epoch=len(batches(np.arange(len(mt))));candidates=[]
for candidate,lr in enumerate(CFG['learning_rates']):
 m,opt=model(lr);rng=np.random.RandomState(42);best=float('inf');bestep=0;stale=0
 for ep in range(1,101):
  losses=[step(m,opt,xt,mt,ix) for ix in batches(rng.permutation(len(mt)))];val=validation(m,xv,mv)
  if val<best-1e-9:best=val;bestep=ep;stale=0;save(m,f'candidate_{candidate}_best.pt')
  else:stale+=1
  event(phase='SELECTION',candidate=candidate,epoch=ep,validation_macro_rmse=val,training_loss=float(np.mean(losses)))
  if stale>=10:break
 candidates.append({'candidate':candidate,'lr':lr,'best_epoch':bestep,'best_validation_macro_rmse':best,'selected_updates':bestep*steps_per_epoch});del m,opt;gc.collect();torch.cuda.empty_cache()
best=min(candidates,key=lambda z:(z['best_validation_macro_rmse'],z['candidate']));(O/'selection.json').write_text(json.dumps({'candidates':candidates,'chosen':best},indent=2));x=sparse.vstack([xt,xv],format='csr');meta=pd.concat([mt,mv],ignore_index=True);m,opt=model(best['lr']);rng=np.random.RandomState(42);done=0
while done<best['selected_updates']:
 for ix in batches(rng.permutation(len(meta))):
  if done>=best['selected_updates']:break
  loss=step(m,opt,x,meta,ix);done+=1
  if done%steps_per_epoch==0:event(phase='REFIT',updates=done)
save(m,'model_final.pt');yp=predict(m,list(zip(q.target,q.context)));assert np.isfinite(yp).all();np.savez_compressed(O/'predictions_sealed.npz',prediction=yp,target=q.target.to_numpy(dtype=str),context=q.context.to_numpy(dtype=str),native_fallback=np.zeros(len(q),bool));q.to_csv(O/'query_order.tsv',sep='\t',index=False);result={'status':'PREDICTIONS_SEALED','formal_training':True,'dataset':a.dataset,'split':a.split,'method':'TxPert-public-STRING-GO','selection':best,'refit_updates':done,'prediction_sha256':sha(O/'predictions_sealed.npz'),'weight_sha256':sha(O/'model_final.pt'),'seconds':time.time()-start,'test_metrics_computed':False};(O/'result.json').write_text(json.dumps(result,indent=2));event(**result)
