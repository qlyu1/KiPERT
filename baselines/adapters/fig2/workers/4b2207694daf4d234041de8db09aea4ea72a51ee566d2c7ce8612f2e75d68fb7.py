"""Author BiolordModule, GO-neighbor pseudobulk formulation; no unknown latent attributes."""
from pathlib import Path
import sys,argparse,json,time,hashlib,os
R=Path('@STUDY@/uc_fig2_benchmark_20260911');sys.path.insert(0,str(R/'src'))
from isolation import confine
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split',type=int,required=True);p.add_argument('--phase',default='train',choices=['train','smoke']);a=p.parse_args();B=R/'data/formal'/a.dataset/f'split{a.split}';O=R/'runs'/('formal' if a.phase=='train' else 'integration')/a.dataset/f'split{a.split}'/'biolord';confine(B,O)
(O/'worker_source.py').write_bytes(Path(__file__).read_bytes())
import numpy as np,pandas as pd,torch,random,copy
from scipy import sparse
from biolord._module import BiolordModule
from prepare_formal import Resolver
os.environ['WANDB_MODE']='disabled';torch.set_num_threads(2);device='cpu';start=time.time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def event(**z):
 z={'seconds':time.time()-start,**z}
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(z)+'\n')
 print(json.dumps(z),flush=True)
def load(role):
 x=sparse.load_npz(B/role/'expression.npz');m=pd.read_csv(B/role/'metadata.tsv',sep='\t',dtype=str);rows=[];ys=[]
 for (t,c),ix in m.groupby(['target','context'],sort=True).indices.items():rows.append((t,c));ys.append(np.asarray(x[ix].mean(0)).ravel())
 return rows,np.array(ys,np.float32)
tr,ytr=load('fit');cr,ycr=load('controls_fit');control={c:y for (_,c),y in zip(cr,ycr)};q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str);mapping=pd.read_csv(B/'target_prior_mapping.tsv',sep='\t',dtype=str).fillna('');res=Resolver();targets=mapping.target.tolist();allctx=sorted(control);ctxid={c:i for i,c in enumerate(allctx)}
gopath=R/'repos/TxPert/data/graphs/go/go_top_50.csv';g=pd.read_csv(gopath);g['sid']=g.source.map(res.resolve);g['tid']=g.target.map(res.resolve);g=g.dropna(subset=['sid','tid']);g=g.groupby('tid',sort=True,group_keys=True).apply(lambda z:z.nlargest(21,'importance'),include_groups=False).reset_index();g=g[g.tid.isin(targets)];neighbors=sorted(g.sid.unique());nid={t:i for i,t in enumerate(neighbors)};emb={t:np.zeros(len(neighbors),np.float32) for t in targets};emb['control']=np.zeros(len(neighbors),np.float32)
for z in g.itertuples():emb[z.tid][nid[z.sid]]=max(emb[z.tid][nid[z.sid]],float(z.importance))
assert len(neighbors)>0;G=ytr.shape[1]
CFG={'method':'biolord','native_module':'BiolordModule','formulation':'target/context pseudobulk means; GO top21 weighted neighbors; known context categorical attribute; unknown_attributes=False','seed':42,'device':device,'learning_rates':[.001,.001/3,.003],'max_epochs':1000,'patience':10,'batch_size':32,'ordered_latent':512,'latent':32,'context_latent':16,'attribute_width':64,'attribute_depth':6,'decoder_width':64,'decoder_depth':1,'attribute_dropout':.1,'attribute_activation':False,'decoder_activation':False,'reconstruction_penalty':1000,'decoder_wd':.01,'attribute_wd':4e-8,'cosine_T_max':45,'eta_min':1e-5,'refit':'selected optimizer updates','gene_axis_sha256':sha(B/'genes.tsv'),'data_manifest_sha256':sha(B/'READY.json'),'native_source_sha256':sha(R/'repos/biolord/src/biolord/_module.py'),'go_graph_sha256':sha(gopath),'worker_sha256':sha(__file__),'test_scoring':False}
(O/'RUN_LOCK.json').write_text(json.dumps(CFG,indent=2))
def model(lr):
 random.seed(42);np.random.seed(42);torch.manual_seed(42)
 m=BiolordModule(n_genes=G,n_samples=1,x_loc='X',ordered_attributes_map={'perturbation_neighbors':len(neighbors)},categorical_attributes_map={'context':ctxid},n_latent=32,n_latent_attribute_ordered=512,n_latent_attribute_categorical=16,gene_likelihood='normal',reconstruction_penalty=1000,unknown_attribute_penalty=10000,use_batch_norm=False,use_layer_norm=False,unknown_attributes=False,attribute_dropout_rate=.1,attribute_nn_width=64,attribute_nn_depth=6,attribute_nn_activation=False,decoder_width=64,decoder_depth=1,decoder_activation=False,seed=42).to(device)
 opt=torch.optim.Adam([{'params':m.ordered_networks.parameters(),'lr':lr,'weight_decay':4e-8},{'params':m.categorical_embeddings.parameters(),'lr':lr,'weight_decay':4e-8},{'params':m.decoder.parameters(),'lr':lr,'weight_decay':.01}]);return m,opt
# Inference input expression supplies shape only for the normal-likelihood, no-unknown-attribute model.
# Positive dummy values avoid log(0) in unused library computation, and cannot encode query responses.
def forward(m,rows):
 e=torch.from_numpy(np.stack([emb[t] for t,c in rows])).to(device);ci=torch.tensor([ctxid[c] for t,c in rows],device=device);dummy=torch.ones((len(rows),G),device=device);ii=m.inference(dummy,torch.zeros(len(rows),dtype=torch.long,device=device),{'context':ci},{'perturbation_neighbors':e});oo=m.generative(ii['latent'],ii['library']);return ii,oo

def bat(order):
 z=[order[i:i+32] for i in range(0,len(order),32)]
 if len(z)>1 and len(z[-1])==1:z[-2]=np.r_[z[-2],z[-1]];z.pop()
 return z

def step(m,opt,rows,y,ix):
 m.train();ii,oo=forward(m,[rows[i] for i in ix]);losses=m.loss({'X':torch.from_numpy(y[ix]).to(device)},ii,oo);loss=sum(losses.values());assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();return float(loss.detach())
@torch.no_grad()
def predict(m,rows):
 m.eval();out=[]
 for lo in range(0,len(rows),32):
  rr=rows[lo:lo+32];singleton=len(rr)==1
  if singleton:rr=rr+rr
  _,oo=forward(m,rr);out.append(oo['means'].cpu().numpy()[:1 if singleton else len(rr)])
 return np.concatenate(out)
def score(m,rows,y):
 yp=predict(m,rows);errors=np.sqrt(np.mean((yp.astype(float)-y)**2,axis=1));by={}
 for (t,c),v in zip(rows,errors):by.setdefault(t,[]).append(v)
 return float(np.mean([np.mean(v) for v in by.values()]))
def save(m,name):torch.save({k:v.detach().cpu().clone() for k,v in m.state_dict().items()},O/name)
# Include C_fit pseudobulk as the native control training condition, never C_check.
trainrows=tr+cr;trainy=np.concatenate([ytr,ycr]);steps_per_epoch=len(bat(np.arange(len(trainrows))))
if a.phase=='smoke':
 m,opt=model(.001);v=step(m,opt,trainrows,trainy,np.arange(min(32,len(trainrows))));pred=predict(m,tr[:2]);assert np.isfinite(pred).all();event(status='SMOKE_PASS',loss=v,genes=G,formal_training=False);sys.exit(0)
va,yva=load('validation');candidates=[]
for candidate,lr in enumerate(CFG['learning_rates']):
 m,opt=model(lr);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=45,eta_min=1e-5);rng=np.random.RandomState(42);best=float('inf');bestep=0;stale=0
 for ep in range(1,1001):
  losses=[step(m,opt,trainrows,trainy,ix) for ix in bat(rng.permutation(len(trainrows)))];val=score(m,va,yva);sch.step()
  if val<best-1e-9:best=val;bestep=ep;stale=0;save(m,f'candidate_{candidate}_best.pt')
  else:stale+=1
  event(phase='SELECTION',candidate=candidate,epoch=ep,validation_macro_rmse=val,training_loss=float(np.mean(losses)))
  if stale>=10:break
 candidates.append({'candidate':candidate,'lr':lr,'best_epoch':bestep,'best_validation_macro_rmse':best,'selected_updates':bestep*steps_per_epoch})
best=min(candidates,key=lambda z:(z['best_validation_macro_rmse'],z['candidate']));(O/'selection.json').write_text(json.dumps({'candidates':candidates,'chosen':best},indent=2));rows=tr+va+cr;y=np.concatenate([ytr,yva,ycr]);m,opt=model(best['lr']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=45,eta_min=1e-5);rng=np.random.RandomState(42);done=0
while done<best['selected_updates']:
 for ix in bat(rng.permutation(len(rows))):
  if done>=best['selected_updates']:break
  loss=step(m,opt,rows,y,ix);done+=1
  if done%steps_per_epoch==0:sch.step()
save(m,'model_final.pt');queries=list(zip(q.target,q.context));yp=predict(m,queries);assert np.isfinite(yp).all();np.savez_compressed(O/'predictions_sealed.npz',prediction=yp,target=q.target.to_numpy(dtype=str),context=q.context.to_numpy(dtype=str),native_fallback=np.array([not emb[t].any() for t,c in queries]));q.to_csv(O/'query_order.tsv',sep='\t',index=False);result={'status':'PREDICTIONS_SEALED','formal_training':True,'dataset':a.dataset,'split':a.split,'method':'biolord','selection':best,'refit_updates':done,'queries':len(q),'genes':G,'prediction_sha256':sha(O/'predictions_sealed.npz'),'weight_sha256':sha(O/'model_final.pt'),'seconds':time.time()-start,'test_metrics_computed':False};(O/'result.json').write_text(json.dumps(result,indent=2));event(**result)
