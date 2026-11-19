"""Fixed official GEARS / RegFormer-GEARS networks under response-file isolation."""
from pathlib import Path
import sys,argparse,json,time,hashlib,importlib.util,random,gc,ast,os
R=Path('@STUDY@/uc_fig2_benchmark_20260911');sys.path.insert(0,str(R/'src'))
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split',type=int,required=True);p.add_argument('--method',choices=['gears','regformer'],required=True);p.add_argument('--phase',choices=['smoke','train'],default='train');a=p.parse_args()
from execution_control import assert_can_start
if a.phase=='train':assert_can_start(a.method)
from isolation import confine
B=R/'data/formal'/a.dataset/f'split{a.split}';O=R/'runs'/('formal' if a.phase=='train' else 'integration')/a.dataset/f'split{a.split}'/a.method;confine(B,O)
(O/'worker_source.py').write_bytes(Path(__file__).read_bytes())
import numpy as np,pandas as pd,torch
from scipy import sparse
from torch_geometric.data import Data
from prepare_formal import Resolver
from scf_graphs import coexpression
source=R/'repos'/('GEARS/gears' if a.method=='gears' else 'RegFormer/regformer/repo/gears')
spec=importlib.util.spec_from_file_location('official_gears_model',source/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
node=next(n for n in ast.parse((source/'utils.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='loss_fct');space={'torch':torch,'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source/'utils.py'),'exec'),space);loss_fct=space['loss_fct']
start=time.time();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.benchmark=False
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def event(**z):
 z={'seconds':time.time()-start,**z}
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(z)+'\n')
 print(json.dumps(z),flush=True)
def load(role):return sparse.load_npz(B/role/'expression.npz').tocsr(),pd.read_csv(B/role/'metadata.tsv',sep='\t',dtype=str)
xt,mt=load('fit');xc,mc=load('controls_fit');ga=pd.read_csv(B/'genes.tsv',sep='\t',dtype=str);G=len(ga);q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str);mapping=pd.read_csv(B/'target_prior_mapping.tsv',sep='\t',dtype=str).fillna('');res=Resolver()
# Author's perturbation GO graph includes all annotated prior genes, not only observed intervention labels.
godf=pd.read_csv(R/'repos/TxPert/data/graphs/go/go_top_50.csv');godf['source']=godf.source.map(res.resolve);godf['target']=godf.target.map(res.resolve);godf=godf.dropna(subset=['source','target']);targets=sorted(set(godf.source)|set(godf.target)|set(mapping.target));tid={t:i for i,t in enumerate(targets)}
godf=godf.sort_values(['target','importance','source'],ascending=[True,False,True],kind='stable').groupby('target',sort=True).head(21);go=torch.tensor(np.array([godf.source.map(tid),godf.target.map(tid)]),dtype=torch.long);gow=torch.tensor(godf.importance.to_numpy(),dtype=torch.float32)
ctrl={c:np.asarray(xc[ix].mean(0)).ravel() for c,ix in mc.groupby('context',sort=True).indices.items()};groups={c:ix for c,ix in mc.groupby('context',sort=True).indices.items()}
base_lr=.001 if a.method=='gears' else .0001
CFG={'method':'GEARS' if a.method=='gears' else 'RegFormer-GEARS','seed':42,'batch_size':32,'hidden_size':64,'learning_rates':[base_lr,base_lr/3,base_lr*3],'max_epochs':20,'patience':5,'scheduler_gamma':.5,'weight_decay':5e-4,'gradient_clip_value':1.,'direction_lambda':.1,'loss':'unchanged official loss_fct: fourth power plus sign penalty','coexpression':'fit plus C_fit only; refit fit+validation+C_fit; abs Pearson top21 >.4','go_graph':'public GO Jaccard prior, top21, all annotated nodes','prediction_control':'context C_fit mean; exact affine-input inference in eval','native_network_sha256':sha(source/'model.py'),'native_loss_sha256':sha(source/'utils.py'),'worker_sha256':sha(__file__),'data_manifest_sha256':sha(B/'READY.json'),'go_sha256':sha(R/'repos/TxPert/data/graphs/go/go_top_50.csv'),'test_scoring':False,'output_genes':G,'perturbation_graph_nodes':len(targets),'audited_batch_graph_fix':'same coexpression SGConv for every repeated gene-identity block; shared exact computation, fixes unoffset native batch edges'}
pretrained=None
if a.method=='regformer':
 bank=np.load(R/'assets/regformer/gene_embeddings.npz');bank_vectors=bank['embedding'];vocab=json.loads((R/'assets/regformer/vocab.json').read_text());torch.manual_seed(42);pretrained=torch.nn.Embedding(G,bank_vectors.shape[1]).weight.detach();covered=[]
 for i,z in enumerate(ga.itertuples()):
  key=next((k for k in [z.symbol,z.gene_id]+res.candidates(z.gene_id) if k in vocab),None)
  if key is not None:pretrained[i]=torch.from_numpy(bank_vectors[vocab[key]]);covered.append(i)
 CFG.update(pretrained='official 10k checkpoint encoder incl LayerNorm; fine-tuned as official code',pretrained_sha256=sha(R/'assets/regformer/gene_embeddings.npz'),pretrained_coverage=len(covered)/G,missing_embedding='seed42 native nn.Embedding initialization; no gene/target dropped',pretraining_overlap='not fully excluded; externally pretrained stratum')
(O/'RUN_LOCK.json').write_text(json.dumps(CFG,indent=2))
def graph_for(x,label):
 path=O/(label+'_coexpression.npz')
 if path.exists():z=np.load(path);return torch.from_numpy(z['edges']),torch.from_numpy(z['weights'])
 e,w=coexpression(sparse.vstack([x,xc],format='csr'),np.arange(G));np.savez_compressed(path,edges=e.numpy(),weights=w.numpy());event(phase='GRAPH_READY',label=label,edges=e.shape[1]);return e,w

class SharedCoexpression(torch.nn.Module):
 def __init__(self,layer):super().__init__();self.layer=layer
 def forward(self,x,edge,weight):
  # Gene identity features are exactly repeated in the native forward. Apply the same
  # graph to every sample; original code passed unoffset edges to the repeated matrix.
  assert x.shape[0]%G==0
  return self.layer(x[:G],edge,weight).repeat(x.shape[0]//G,1)

def model(lr,e,w):
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 args=dict(num_genes=G,num_perts=len(targets),hidden_size=64,uncertainty=False,num_go_gnn_layers=1,num_gene_gnn_layers=1,decoder_hidden_size=16,no_perturb=False,G_coexpress=e,G_coexpress_weight=w,G_go=go,G_go_weight=gow,device='cuda')
 if a.method=='regformer':args.update(use_pretrained=True,layer_size=pretrained.shape[1],gene_emb_weight=pretrained.clone(),gene2ids=None,gene_names=ga.symbol.tolist(),pretrain_freeze=False)
 m=module.GEARS_Model(args);m.layers_emb_pos=torch.nn.ModuleList([SharedCoexpression(z) for z in m.layers_emb_pos]);m=m.cuda();opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=5e-4);return m,opt

def parts(x,meta):
 labels=meta.target.to_numpy();mask={t:np.flatnonzero(np.asarray(x[np.flatnonzero(labels==t)].sum(0)).ravel()>0) for t in set(labels)};assert all(len(z) for z in mask.values());rng=np.random.RandomState(42);pairs=np.empty(len(meta),int)
 for c,ix in meta.groupby('context',sort=True).indices.items():pool=np.resize(groups[c],len(ix));rng.shuffle(pool);pairs[ix]=pool
 return mask,pairs

def batches(order):
 z=[order[i:i+32] for i in range(0,len(order),32)]
 if len(z)>1 and len(z[-1])==1:z[-2]=np.r_[z[-2],z[-1]];z.pop()
 return z

def forward(m,x,ts):
 xx=torch.from_numpy(np.asarray(x,np.float32)).cuda();d=Data(x=xx.reshape(-1,1),pert_idx=[[tid[t]] for t in ts],batch=torch.arange(len(ts),device='cuda').repeat_interleave(G));return m(d)

def step(m,opt,x,meta,ps,ix):
 m.train();opt.zero_grad(set_to_none=True);mask,pairs=ps;z=meta.iloc[ix];yp=forward(m,xc[pairs[ix]].toarray(),z.target.tolist());truth=torch.from_numpy(x[ix].toarray()).cuda();loss=0
 for c in sorted(z.context.unique()):
  jj=np.flatnonzero(z.context.to_numpy()==c);loss=loss+loss_fct(yp[jj],truth[jj],z.target.iloc[jj].to_numpy(),ctrl=torch.tensor(ctrl[c],device='cuda'),direction_lambda=.1,dict_filter=mask)*len(jj)/len(ix)
 assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_value_(m.parameters(),1.);opt.step();return float(loss.detach())

@torch.no_grad()
def predict(m,rows):
 m.eval();out=[]
 for lo in range(0,len(rows),32):
  rr=rows[lo:lo+32];out.append(forward(m,np.stack([ctrl[c] for t,c in rr]),[t for t,c in rr]).cpu().numpy())
 result=np.concatenate(out);assert np.isfinite(result).all();return result

def validation(m,x,meta):
 rows=[];ys=[]
 for (t,c),ix in meta.groupby(['target','context'],sort=True).indices.items():rows.append((t,c));ys.append(np.asarray(x[ix].mean(0)).ravel())
 yp=predict(m,rows);err=np.sqrt(np.mean((yp.astype(float)-ys)**2,axis=1));by={}
 for (t,c),v in zip(rows,err):by.setdefault(t,[]).append(float(v))
 return float(np.mean([np.mean(v) for v in by.values()]))
def save(m,name):
 tmp=O/(name+'.tmp');torch.save({k:v.detach().cpu().clone() for k,v in m.state_dict().items()},tmp);tmp.replace(O/name)
def checkpoint(m,opt,sch,rng,**state):
 tmp=O/'resume_checkpoint.pt.tmp';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),state=state),tmp);tmp.replace(O/'resume_checkpoint.pt')
e,w=graph_for(xt,'train');ps=parts(xt,mt)
if a.phase=='smoke':
 m,opt=model(base_lr,e,w);loss=step(m,opt,xt,mt,ps,np.arange(32));rows=[(mt.target.iloc[0],mt.context.iloc[0])];yp=predict(m,rows)
 # Verify the official network's control dependence is exactly additive in evaluation.
 with torch.no_grad():
  m.eval();c=rows[0][1];ii=groups[c][:8];allp=forward(m,xc[ii].toarray(),[rows[0][0]]*len(ii)).cpu().numpy();diff=np.max(np.abs(allp-xc[ii].toarray()-(yp[0]-ctrl[c])))
 assert diff<1e-4;event(status='SMOKE_PASS',loss=loss,genes=G,affine_error=float(diff),gpu_peak_bytes=torch.cuda.max_memory_allocated(),formal_training=False);(O/'result.json').write_text(json.dumps({'status':'PASS','loss':loss,'affine_error':float(diff)}));sys.exit(0)
xv,mv=load('validation');spe=len(batches(np.arange(len(mt))));candidates=[]
for candidate,lr in enumerate(CFG['learning_rates']):
 m,opt=model(lr,e,w);sch=torch.optim.lr_scheduler.StepLR(opt,1,.5);rng=np.random.RandomState(42);best=float('inf');bestep=0;stale=0
 for ep in range(1,21):
  losses=[]
  for j,ix in enumerate(batches(rng.permutation(len(mt)))):
   losses.append(step(m,opt,xt,mt,ps,ix))
   if j%100==0:event(phase='TRAIN',candidate=candidate,epoch=ep,update=j,total=spe)
  val=validation(m,xv,mv);sch.step()
  if val<best-1e-9:best=val;bestep=ep;stale=0;save(m,f'candidate_{candidate}_best.pt')
  else:stale+=1
  checkpoint(m,opt,sch,rng,phase='selection',candidate=candidate,epoch=ep,best=best,best_epoch=bestep,stale=stale,candidates=candidates)
  event(phase='SELECTION',candidate=candidate,epoch=ep,validation_macro_rmse=val,training_loss=float(np.mean(losses)))
  if stale>=5:break
 candidates.append({'candidate':candidate,'lr':lr,'best_epoch':bestep,'best_validation_macro_rmse':best,'selected_updates':bestep*spe});del m,opt,sch;gc.collect();torch.cuda.empty_cache()
best=min(candidates,key=lambda z:(z['best_validation_macro_rmse'],z['candidate']));(O/'selection.json').write_text(json.dumps({'candidates':candidates,'chosen':best},indent=2));x=sparse.vstack([xt,xv],format='csr');meta=pd.concat([mt,mv],ignore_index=True);ps=parts(x,meta);e,w=graph_for(x,'refit');m,opt=model(best['lr'],e,w);sch=torch.optim.lr_scheduler.StepLR(opt,1,.5);rng=np.random.RandomState(42);done=0
while done<best['selected_updates']:
 for ix in batches(rng.permutation(len(meta))):
  if done>=best['selected_updates']:break
  step(m,opt,x,meta,ps,ix);done+=1
  if done%spe==0:sch.step();checkpoint(m,opt,sch,rng,phase='refit',updates=done,selection=best);event(phase='REFIT',updates=done)
save(m,'model_final.pt');yp=predict(m,list(zip(q.target,q.context)));np.savez_compressed(O/'predictions_sealed.npz',prediction=yp,target=q.target.to_numpy(dtype=str),context=q.context.to_numpy(dtype=str),native_fallback=np.zeros(len(q),bool));q.to_csv(O/'query_order.tsv',sep='\t',index=False);result={'status':'PREDICTIONS_SEALED','formal_training':True,'dataset':a.dataset,'split':a.split,'method':CFG['method'],'selection':best,'refit_updates':done,'prediction_sha256':sha(O/'predictions_sealed.npz'),'weight_sha256':sha(O/'model_final.pt'),'seconds':time.time()-start,'test_metrics_computed':False};(O/'result.json').write_text(json.dumps(result,indent=2));event(**result)
