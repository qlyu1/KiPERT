"""Published STAMP with official Geneformer V1 static token embeddings; strict held-out targets."""
from pathlib import Path
import sys,argparse,json,time,hashlib,importlib.util,random,gc,ast
R=Path('@STUDY@/uc_fig2_benchmark_20260911');sys.path.insert(0,str(R/'src'));p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split',type=int,required=True);p.add_argument('--phase',choices=['smoke','train'],default='train');a=p.parse_args()
from execution_control import assert_can_start
if a.phase=='train':assert_can_start('stamp')
from isolation import confine
B=R/'data/formal'/a.dataset/f'split{a.split}';O=R/'runs'/('formal' if a.phase=='train' else 'integration')/a.dataset/f'split{a.split}'/'stamp';confine(B,O);(O/'worker_source.py').write_bytes(Path(__file__).read_bytes())
import numpy as np,pandas as pd,torch
from scipy import sparse,stats
from prepare_formal import Resolver
source=R/'repos/STAMP/stamp/Modules.py';spec=importlib.util.spec_from_file_location('official_stamp_modules',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# Native masking has undefined 0/0 for batches containing no DEGs. Preserve every
# nonempty-batch value; define absent tasks' losses as zero without introducing labels.
original=module.TaskCombineLayer_multi_task.forward
fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TaskCombineLayer_multi_task')
fn=next(n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name=='forward');src=ast.unparse(fn).replace('torch.sum(new_weights)','torch.sum(new_weights).clamp_min(1)').replace('/ torch.sum(mask)','/ torch.sum(mask).clamp_min(1)');space={'torch':torch,'nn':torch.nn};exec(src,space);module.TaskCombineLayer_multi_task.forward=space['forward']
start=time.time();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.benchmark=False

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def event(**z):
 z={'seconds':time.time()-start,**z}
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(z)+'\n')
 print(json.dumps(z),flush=True)
def load(role):return sparse.load_npz(B/role/'expression.npz').tocsr(),pd.read_csv(B/role/'metadata.tsv',sep='\t',dtype=str)
xt,mt=load('fit');xc,mc=load('controls_fit');ga=pd.read_csv(B/'genes.tsv',sep='\t',dtype=str);G=len(ga);q=pd.read_csv(B/'query/metadata.tsv',sep='\t',dtype=str);mapping=pd.read_csv(B/'target_prior_mapping.tsv',sep='\t',dtype=str);res=Resolver();bank=np.load(R/'assets/stamp/geneformer_v1_embeddings.npz');vocab=json.loads((R/'assets/stamp/geneformer_vocab.json').read_text());vec=bank['embedding'];E=vec.shape[1];missing=set()
def embedding(g):
 key=next((k for k in [g]+res.candidates(g) if k in vocab),None)
 if key is not None:return vec[vocab[key]]
 missing.add(g);seed=int(hashlib.sha256(('STAMP_missing_v1:'+g).encode()).hexdigest()[:8],16);return np.random.RandomState(seed).normal(0,.02,E).astype(np.float32)
out_emb=torch.tensor(np.stack([embedding(g) for g in ga.gene_id]),device='cuda');t_emb={t:embedding(t) for t in sorted(mapping.target)};contexts=sorted(mt.context.unique());ci={c:str(i) for i,c in enumerate(contexts)}
ctrl={c:np.asarray(xc[ix].mean(0)).ravel().astype(np.float64) for c,ix in mc.groupby('context',sort=True).indices.items()};cstats={}
for c,ix in mc.groupby('context',sort=True).indices.items():
 x=xc[ix].astype(np.float64);mu=np.asarray(x.mean(0)).ravel();var=(np.asarray(x.power(2).mean(0)).ravel()-mu**2)*len(ix)/(len(ix)-1);cstats[c]=(mu,np.maximum(var,0),len(ix))
CFG={'method':'STAMP-GeneformerV1','published_variant':'Geneformer+STAMP, Supplementary Note6','prior_dimension':E,'prior_sha256':sha(R/'assets/stamp/geneformer_v1_embeddings.npz'),'prior_coverage_output':float(np.mean([g not in missing for g in ga.gene_id])),'missing_targets':sorted(set(mapping.target)&missing),'missing_prior_policy':'fixed per-gene hash normal(0,.02) matching source initializer; no axis/target deletion','seed':42,'learning_rates':[.0001,.0001/3,.0003],'batch_size':64,'max_epochs':200,'patience':20,'optimizer':'AdamW default weight_decay=.01','scheduler':'none','loss_weights':[.1,.5,.4],'training_unit':'target/context pseudobulk; native network separately fitted per context','labels':'fit-only Welch t-test BH<=.01, sign of mean change, fold ratio expm1(mean_log+.1); refit recomputes using train+validation','inference':'predicted DEG probability>=.5 gates predicted fold ratio; inverse transform log1p(expm1(C_fit_mean+.1)*FC)-.1; clip expression at0; direction is auxiliary supervisor','test_labels_in_forward':False,'native_network_sha256':sha(source),'worker_sha256':sha(__file__),'data_manifest_sha256':sha(B/'READY.json'),'output_genes':G,'test_scoring':False,'pretraining_overlap':'not fully excluded; external-pretraining stratum','numerical_fix':'only empty-task denominator guard; finite nonempty cases unchanged'};(O/'RUN_LOCK.json').write_text(json.dumps(CFG,indent=2))

def labels(x,meta,label):
 output={};record=[]
 for c in contexts:
  rows=[];yy=[];mask=[];direction=[];inputs=[];mu0,v0,n0=cstats[c]
  for (t,ctx),ix in meta.groupby(['target','context'],sort=True).indices.items():
   if ctx!=c:continue
   z=x[ix].astype(np.float64);n=len(ix);mu=np.asarray(z.mean(0)).ravel();v=np.maximum((np.asarray(z.power(2).mean(0)).ravel()-mu**2)*n/(n-1),0)
   stat,pval=stats.ttest_ind_from_stats(mu,np.sqrt(v),n,mu0,np.sqrt(v0),n0,equal_var=False);pval=np.nan_to_num(pval,nan=1.);padj=stats.false_discovery_control(pval,method='bh');de=(padj<=.01).astype(np.float32);fc=(np.expm1(mu+.1)+1e-9)/(np.expm1(mu0+.1)+1e-9)
   assert np.isfinite(fc).all();inputs.append(t_emb[t]);rows.append((t,c));yy.append(fc.astype(np.float32));mask.append(de);direction.append((fc>=1).astype(np.float32));record.append({'target':t,'context':c,'n_fit':n,'deg_count':int(de.sum())})
  output[c]=dict(rows=rows,inputs=np.stack(inputs),level1=np.stack(mask),level2=np.stack(direction),level3=np.stack(yy))
 pd.DataFrame(record).to_csv(O/f'{label}_label_summary.tsv',sep='\t',index=False);event(phase='LABELS_READY',role=label,conditions=len(record));return output

def model(lr):
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 m=torch.nn.ModuleDict({ci[c]:module.TaskCombineLayer_multi_task(in_features=E,out_features=G,hid1_features_2=128,hid2_features_2=64,hid3_features_2=32,in_feature_3=32,hid1_features_3=16,hid2_features_3=8,gene_embedding_notebook=out_emb) for c in contexts}).cuda();return m,torch.optim.AdamW(m.parameters(),lr=lr)
def batches(data,rng):
 out=[]
 for c in contexts:
  order=rng.permutation(len(data[c]['rows']));out.extend((c,order[i:i+64]) for i in range(0,len(order),64))
 return out

def step(m,opt,data,batch):
 c,ix=batch;d=data[c];net=m[ci[c]];net.train();z=[torch.from_numpy(d[k][ix]).cuda() for k in ['inputs','level1','level2','level3']];losses=net(*z);loss=.1*losses[0]+.5*losses[1]+.4*losses[2];assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();return float(loss.detach())

@torch.no_grad()
def predict(m,rows):
 m.eval();out=np.empty((len(rows),G),np.float32)
 for c in contexts:
  jj=[i for i,(t,ctx) in enumerate(rows) if ctx==c];net=m[ci[c]]
  for lo in range(0,len(jj),32):
   ix=jj[lo:lo+32];x=torch.tensor(np.stack([t_emb[rows[i][0]] for i in ix]),device='cuda');prob=net.first_level_layer(x);dummy=torch.zeros((len(ix),G),device='cuda');d,mask,h=net.second_level_layer(dummy,x);fc,_=net.third_level_layer(h,mask);fold=fc.squeeze(-1).cpu().numpy().astype(np.float64);estimate=np.log1p(np.expm1(ctrl[c]+.1)*fold)-.1;out[ix]=np.where(prob.cpu().numpy()>=.5,np.maximum(estimate,0),ctrl[c]).astype(np.float32)
 assert np.isfinite(out).all();return out

def validation(m,x,meta):
 rows=[];ys=[]
 for (t,c),ix in meta.groupby(['target','context'],sort=True).indices.items():rows.append((t,c));ys.append(np.asarray(x[ix].mean(0)).ravel())
 yp=predict(m,rows);err=np.sqrt(np.mean((yp.astype(float)-ys)**2,axis=1));by={}
 for (t,c),v in zip(rows,err):by.setdefault(t,[]).append(float(v))
 return float(np.mean([np.mean(v) for v in by.values()]))
def save(m,name):
 tmp=O/(name+'.tmp');torch.save({k:v.detach().cpu().clone() for k,v in m.state_dict().items()},tmp);tmp.replace(O/name)
def checkpoint(m,opt,rng,**state):
 tmp=O/'resume_checkpoint.pt.tmp';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),state=state),tmp);tmp.replace(O/'resume_checkpoint.pt')
data=labels(xt,mt,'fit');spe=len(batches(data,np.random.RandomState(42)))
if a.phase=='smoke':
 m,opt=model(.0001);b=batches(data,np.random.RandomState(42))[0];loss=step(m,opt,data,b);yp=predict(m,[data[b[0]]['rows'][0]]);net=m[ci[b[0]]].eval();d=data[b[0]];ix=b[1][:2];zs=[torch.from_numpy(d[k][ix]).cuda() for k in ['inputs','level1','level2','level3']]
 if zs[1].sum()>0:
  with torch.no_grad():
   old=original(net,*zs);new=net(*zs)
   assert all(torch.equal(old[i],new[i]) for i in range(3))
 # Labels are irrelevant to component predictions; only the returned loss masks depend on them.
 with torch.no_grad():
  z1=net.second_level_layer(torch.zeros_like(zs[1]),zs[0]);z2=net.second_level_layer(torch.ones_like(zs[1]),zs[0]);assert torch.equal(z1[0],z2[0]);f1=net.third_level_layer(z1[2],z1[1])[0];f2=net.third_level_layer(z2[2],z2[1])[0];assert torch.equal(f1,f2)
 event(status='SMOKE_PASS',loss=loss,genes=G,label_independent_inference=True,gpu_peak_bytes=torch.cuda.max_memory_allocated(),formal_training=False);(O/'result.json').write_text(json.dumps({'status':'PASS','loss':loss,'label_independent_inference':True}));sys.exit(0)
xv,mv=load('validation');candidates=[]
for candidate,lr in enumerate(CFG['learning_rates']):
 m,opt=model(lr);rng=np.random.RandomState(42);best=float('inf');bestep=0;stale=0
 for ep in range(1,201):
  losses=[step(m,opt,data,b) for b in batches(data,rng)];val=validation(m,xv,mv)
  if val<best-1e-9:best=val;bestep=ep;stale=0;save(m,f'candidate_{candidate}_best.pt')
  else:stale+=1
  checkpoint(m,opt,rng,phase='selection',candidate=candidate,epoch=ep,best=best,best_epoch=bestep,stale=stale,candidates=candidates);event(phase='SELECTION',candidate=candidate,epoch=ep,validation_macro_rmse=val,training_loss=float(np.mean(losses)))
  if stale>=20:break
 candidates.append({'candidate':candidate,'lr':lr,'best_epoch':bestep,'best_validation_macro_rmse':best,'selected_updates':bestep*spe});del m,opt;gc.collect();torch.cuda.empty_cache()
best=min(candidates,key=lambda z:(z['best_validation_macro_rmse'],z['candidate']));(O/'selection.json').write_text(json.dumps({'candidates':candidates,'chosen':best},indent=2));x=sparse.vstack([xt,xv],format='csr');meta=pd.concat([mt,mv],ignore_index=True);data=labels(x,meta,'refit');m,opt=model(best['lr']);rng=np.random.RandomState(42);done=0
while done<best['selected_updates']:
 for b in batches(data,rng):
  if done>=best['selected_updates']:break
  step(m,opt,data,b);done+=1
  if done%spe==0:checkpoint(m,opt,rng,phase='refit',updates=done,selection=best);event(phase='REFIT',updates=done)
save(m,'model_final.pt');yp=predict(m,list(zip(q.target,q.context)));np.savez_compressed(O/'predictions_sealed.npz',prediction=yp,target=q.target.to_numpy(dtype=str),context=q.context.to_numpy(dtype=str),native_fallback=np.array([t in missing for t in q.target],bool));q.to_csv(O/'query_order.tsv',sep='\t',index=False);result={'status':'PREDICTIONS_SEALED','formal_training':True,'dataset':a.dataset,'split':a.split,'method':CFG['method'],'selection':best,'refit_updates':done,'prediction_sha256':sha(O/'predictions_sealed.npz'),'weight_sha256':sha(O/'model_final.pt'),'seconds':time.time()-start,'test_metrics_computed':False};(O/'result.json').write_text(json.dumps(result,indent=2));event(**result)
