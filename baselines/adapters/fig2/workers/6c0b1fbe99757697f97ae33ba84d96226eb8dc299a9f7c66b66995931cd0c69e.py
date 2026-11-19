"""Official Scouter network/loss, shared frozen data, validation selection, sealed predictions."""
from pathlib import Path
import sys,argparse,json,hashlib,time,importlib.util,math
R=Path('@STUDY@/uc_fig2_benchmark_20260911');sys.path.insert(0,str(R/'src'))
from isolation import confine
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split',type=int,required=True);p.add_argument('--phase',choices=['smoke','train'],default='train');args=p.parse_args()
BASE=R/'data/formal'/args.dataset/('split'+str(args.split));OUT=R/'runs'/('formal' if args.phase=='train' else 'integration')/args.dataset/('split'+str(args.split))/'scouter'
confine(BASE,OUT)
(OUT/'worker_source.py').write_bytes(Path(__file__).read_bytes())
import os,pickle,gc,copy
import numpy as np
import pandas as pd
from scipy import sparse
import torch
CFG={'model':'Scouter','implementation':'official network and gears_loss','prior':'GenePT-v1','seed':42,'learning_rates':[.001,.001/3,.003],'batch_size':256,'max_epochs':40,'early_stop_patience':5,'scheduler_gamma':.9,'loss_gamma':0,'loss_lambda':.5,'encoder':[2048,512],'encoder_out':64,'decoder':[2048],'dropout':0.,'prediction_control_cells_per_context':128,'validation':'macro target mean of context RMSE on common full gene axis','refit':'train+validation, exactly selected optimizer update count','tf32':True,'test_scoring':False}
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
modelmod=load_module('official_scouter_model',R/'repos/scouter/scouter/_model.py');lossmod=load_module('official_scouter_utils',R/'repos/scouter/scouter/_utils.py')
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def seed():np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
seed();start=time.time()
ready=json.loads((BASE/'READY.json').read_text());assert ready['protocol']=='FIG2_FORMAL_V1_1';assert digest(BASE/'fit/expression.npz')==ready['fit_response_sha256'];assert digest(BASE/'genes.tsv')==ready['axis_sha256']
xt=sparse.load_npz(BASE/'fit/expression.npz').tocsr();mt=pd.read_csv(BASE/'fit/metadata.tsv',sep='\t',dtype=str);xc=sparse.load_npz(BASE/'controls_fit/expression.npz').tocsr();mc=pd.read_csv(BASE/'controls_fit/metadata.tsv',sep='\t',dtype=str);gm=pd.read_csv(BASE/'target_prior_mapping.tsv',sep='\t',dtype=str).fillna('')
control_rows={c:np.flatnonzero(mc.context.to_numpy()==c) for c in sorted(mc.context.unique())};control_mean={c:np.asarray(xc[ix].mean(0)).ravel() for c,ix in control_rows.items()};control_prediction={c:ix[:min(128,len(ix))] for c,ix in control_rows.items()}
with (R/'assets/priors/GPT_3_5_gene_embeddings.pickle').open('rb') as f:bank=pickle.load(f)
keys={z.target:z.genept_v1_key for z in gm.itertuples() if z.genept_v1_key};targets=sorted(keys);emb=np.stack([np.asarray(bank[keys[t]],np.float32) for t in targets]);del bank;gc.collect();assert np.isfinite(emb).all();emb=np.vstack([np.zeros((1,emb.shape[1]),np.float32),emb]);target_index={t:i+1 for i,t in enumerate(targets)};assert set(mt.target)<=set(target_index)
CFG.update(prior_sha256=digest(R/'assets/priors/GPT_3_5_gene_embeddings.pickle'),official_model_sha256=digest(R/'repos/scouter/scouter/_model.py'),official_loss_sha256=digest(R/'repos/scouter/scouter/_utils.py'),dataset_manifest_sha256=digest(BASE/'READY.json'),worker_sha256=digest(__file__),training_targets=len(set(mt.target)),training_cells=len(mt),output_genes=xt.shape[1])
(OUT/'RUN_LOCK.json').write_text(json.dumps(CFG,indent=2));events=OUT/'events.jsonl'
def emit(z):
 z={'elapsed_seconds':time.time()-start,**z}
 with events.open('a') as f:f.write(json.dumps(z)+'\n')
 print(json.dumps(z),flush=True)
def model():
 seed();return modelmod.ScouterModel(xt.shape[1],torch.from_numpy(emb),(2048,512),64,(2048,),True,False,0.).cuda()
def batches(order):
 out=[order[i:i+256] for i in range(0,len(order),256)]
 if len(out)>1 and len(out[-1])==1:out[-2]=np.r_[out[-2],out[-1]];out.pop()
 return out
def training_parts(x,m):
 labels=m.target.to_numpy();contexts=m.context.to_numpy();indices=np.array([[target_index[t],0] for t in labels],dtype=np.int64);mask={t:np.flatnonzero(np.asarray(x[np.flatnonzero(labels==t)].sum(0)).ravel()>0) for t in sorted(set(labels))};assert all(len(z)>0 for z in mask.values())
 rng=np.random.RandomState(42);pairs=np.empty(len(m),dtype=np.int64)
 for c in sorted(set(contexts)):
  ix=np.flatnonzero(contexts==c);ci=np.resize(control_rows[c],len(ix));rng.shuffle(ci);pairs[ix]=ci
 return labels,indices,mask,pairs

def train_steps(net,opt,x,parts,order,remaining=None):
 labels,idx,mask,pairs=parts;total=0.;n=0
 for ix in batches(order):
  if remaining is not None and n>=remaining:break
  ctrl=torch.from_numpy(xc[pairs[ix]].toarray()).cuda();truth=torch.from_numpy(x[ix].toarray()).cuda();pert=torch.from_numpy(idx[ix]).cuda();opt.zero_grad(set_to_none=True);pred=net(pert,ctrl);loss=lossmod.gears_loss(pred,truth,ctrl,labels[ix],mask,0,.5)
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite native training loss')
  loss.backward();opt.step();total+=float(loss.detach());n+=1
 return total/max(n,1),n
@torch.no_grad()
def predict(net,t,c):
 net.eval();ix=control_prediction[c];ctrl=torch.from_numpy(xc[ix].toarray()).cuda();ids=torch.tensor([[target_index[t],0]]*len(ix),device='cuda');result=net(ids,ctrl).mean(0).float().cpu().numpy();assert np.isfinite(result).all();return result

def groups(x,m):
 return [(t,c,np.asarray(x[ix].mean(0)).ravel()) for (t,c),ix in m.groupby(['target','context'],sort=True).indices.items()]
def validation(net,truth):
 per={}
 for t,c,y in truth:
  pred=predict(net,t,c);per.setdefault(t,[]).append(float(np.sqrt(np.mean((pred.astype(np.float64)-y)**2))))
 return float(np.mean([np.mean(v) for v in per.values()]))
def save_weights(net,path):
 state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};tmp=Path(str(path)+'.tmp');torch.save(state,tmp);tmp.replace(path)
parts=training_parts(xt,mt)
if args.phase=='smoke':
 net=model();opt=torch.optim.Adam(net.parameters(),lr=.001);torch.cuda.reset_peak_memory_stats();loss,steps=train_steps(net,opt,xt,parts,np.arange(len(mt)),remaining=2)
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters());emit({'phase':'SMOKE_PASS','loss':loss,'steps':steps,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'formal_training':False});(OUT/'result.json').write_text(json.dumps({'status':'PASS','steps':steps,'seconds':time.time()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'output_genes':xt.shape[1]},indent=2));sys.exit(0)
xv=sparse.load_npz(BASE/'validation/expression.npz').tocsr();mv=pd.read_csv(BASE/'validation/metadata.tsv',sep='\t',dtype=str);valtruth=groups(xv,mv);selection=[];steps_per_epoch=len(batches(np.arange(len(mt))))
for c,lr in enumerate(CFG['learning_rates']):
 net=model();opt=torch.optim.Adam(net.parameters(),lr=lr);sched=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9);rng=np.random.RandomState(42);best=float('inf');bestepoch=0;stale=0;path=OUT/('candidate_'+str(c)+'_best.pt')
 for epoch in range(1,41):
  net.train();order=rng.permutation(len(mt));loss,n=train_steps(net,opt,xt,parts,order);score=validation(net,valtruth);sched.step()
  improved=score<best-1e-9
  if improved:best=score;bestepoch=epoch;stale=0;save_weights(net,path)
  else:stale+=1
  emit({'phase':'SELECTION','candidate':c,'lr':lr,'epoch':epoch,'training_loss':loss,'validation_macro_rmse':score,'improved':improved})
  if stale>=5:break
 selection.append({'candidate':c,'lr':lr,'best_epoch':bestepoch,'best_validation_macro_rmse':best,'selected_updates':bestepoch*steps_per_epoch});del net,opt,sched;gc.collect();torch.cuda.empty_cache()
best=min(selection,key=lambda z:(z['best_validation_macro_rmse'],z['candidate']));(OUT/'selection.json').write_text(json.dumps({'candidates':selection,'chosen':best},indent=2));emit({'phase':'REFIT_START',**best})
xfit=sparse.vstack([xt,xv],format='csr');mfit=pd.concat([mt,mv],ignore_index=True);refparts=training_parts(xfit,mfit);net=model();opt=torch.optim.Adam(net.parameters(),lr=best['lr']);rng=np.random.RandomState(42);done=0;next_decay=steps_per_epoch
# Update-based schedule preserves the selected optimization budget after adding validation labels.
while done<best['selected_updates']:
 order=rng.permutation(len(mfit));net.train()
 for ix in batches(order):
  if done>=best['selected_updates']:break
  loss,n=train_steps(net,opt,xfit,refparts,ix,remaining=1);done+=n
  if done==next_decay:
   for pg in opt.param_groups:pg['lr']*=.9
   next_decay+=steps_per_epoch;emit({'phase':'REFIT','updates':done,'training_loss':loss})
save_weights(net,OUT/'model_final.pt');q=pd.read_csv(BASE/'query/metadata.tsv',sep='\t',dtype=str);traintruth=groups(xfit,mfit);mean_delta={}
for t,c,y in traintruth:mean_delta.setdefault(c,[]).append(y-control_mean[c])
global_delta=np.mean([z for vv in mean_delta.values() for z in vv],axis=0);mean_delta={c:np.mean(v,axis=0) for c,v in mean_delta.items()};predictions=[];fallback=[]
for row in q.itertuples():
 missing=row.target not in target_index;pred=control_mean[row.context]+mean_delta.get(row.context,global_delta) if missing else predict(net,row.target,row.context)
 predictions.append(pred);fallback.append(missing)
predictions=np.asarray(predictions,np.float32);assert np.isfinite(predictions).all();np.savez_compressed(OUT/'predictions_sealed.npz',prediction=predictions,target=q.target.to_numpy(dtype=str),context=q.context.to_numpy(dtype=str),native_fallback=np.asarray(fallback));q.to_csv(OUT/'query_order.tsv',sep='\t',index=False)
result={'status':'PREDICTIONS_SEALED','formal_training':True,'method':'Scouter','dataset':args.dataset,'split':args.split,'queries':len(q),'fallback_queries':sum(fallback),'n_genes':predictions.shape[1],'selection':best,'refit_updates':done,'weight_sha256':digest(OUT/'model_final.pt'),'prediction_sha256':digest(OUT/'predictions_sealed.npz'),'run_lock_sha256':digest(OUT/'RUN_LOCK.json'),'seconds':time.time()-start,'test_truth_access':'denied by kernel Landlock','test_metrics_computed':False}
(OUT/'result.json').write_text(json.dumps(result,indent=2));emit(result)
