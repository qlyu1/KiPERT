from common import *
import random,time,math,argparse
ap=argparse.ArgumentParser();ap.add_argument('fold',choices=CTX);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
fold=a.fold;out=T/'runs'/fold;out.mkdir(parents=True,exist_ok=True)
protocol=json.loads((T/'protocol.json').read_text());assert protocol['training']['max_updates']==50000
random.seed(1);np.random.seed(1);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
m,vocab=model('cuda');tr=splits(fold,'train');va=splits(fold,'val')
assert fold not in tr and fold not in va
sums={};counts={};valid=[];control_n=0
for ctx in CTX:
 z=np.load(T/'summaries'/f'{ctx}.npz');ix={str(p):i for i,p in enumerate(z['targets'])};control_n+=int(z['n_control'])
 if ctx not in tr:continue
 residual_means=z['residual_mean'];cell_counts=z['counts']
 assert not set(tr[ctx])&set(va.get(ctx,[]))
 for p in tr[ctx]:
  j=ix[p];n=int(cell_counts[j]);sums[p]=sums.get(p,0)+residual_means[j].astype(float)*n;counts[p]=counts.get(p,0)+n
 for p in va.get(ctx,[]):valid.append((ctx,p,residual_means[ix[p]]))
names=sorted(sums)+['ctrl'];counts['ctrl']=control_n
targets=np.stack([sums[p]/counts[p] if p!='ctrl' else np.zeros(len(GENES)) for p in names]).astype(np.float32)
prob=np.array([counts[p] for p in names],float);prob/=prob.sum();idxs=[[vocab[p],-1] if p!='ctrl' else [-1] for p in names]
V=np.stack([x[2] for x in valid]).astype(float);Vc=V-V.mean(1,keepdims=True);vn=np.linalg.norm(Vc,axis=1)
vidx=[[vocab[p],-1] for c,p,v in valid];targets=torch.from_numpy(targets).cuda();zero=torch.zeros(64,len(GENES),device='cuda')
opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=0);rng=np.random.default_rng(1)
def validate():
 pred=predict(m,vidx).astype(float);pc=pred-pred.mean(1,keepdims=True);den=vn*np.linalg.norm(pc,axis=1)
 pear=np.divide((pc*Vc).sum(1),den,out=np.full(len(pred),np.nan),where=den>0)
 by={c:float(np.nanmean([pear[i] for i,x in enumerate(valid) if x[0]==c])) for c in va}
 return float(np.mean(list(by.values()))),by,float(np.mean((pred-V)**2))
if a.smoke:
 m.eval();q=idxs[:12]
 with torch.inference_mode():native=m(torch.zeros(12,len(GENES),device='cuda'),q,None)[0].cpu().numpy()
 fast=predict(m,q);np.testing.assert_allclose(native,fast,rtol=2e-5,atol=2e-6)
 start=time.time();losses=[]
 for step in range(6):
  m.train();j=rng.choice(len(names),64,p=prob);pred=m(zero,[idxs[i] for i in j],None)[0];loss=m.loss(pred,None,None,targets[j])[0];opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
 torch.cuda.synchronize();elapsed=time.time()-start;val=validate()
 savej(T/f'SMOKE_{fold}.json',dict(status='passed',losses=losses,seconds_per_update=elapsed/6,validation_score=val[0],fast_inference_max_error=float(np.max(np.abs(native-fast))),gpu_peak_bytes=torch.cuda.max_memory_allocated(),graph_edges=m.edge_index.shape[1],training_conditions=len(names),validation_conditions=len(valid),training_cells=int(sum(counts.values()))))
 print('SMOKE',json.loads((T/f'SMOKE_{fold}.json').read_text()),flush=True);sys.exit()
assert (T/f'SMOKE_k562.json').exists()
resume=Path('/dev/shm/txpert_'+fold+'_resume.pt');best=-float('inf');beststep=0;step0=0
if resume.exists():
 r=torch.load(resume,map_location='cpu',weights_only=False);assert r['protocol_sha256']==sha(T/'protocol.json');m.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);rng.bit_generator.state=r['rng'];torch.set_rng_state(r['torch_rng']);torch.cuda.set_rng_state(r['cuda_rng']);step0=r['step'];best=r['best'];beststep=r['beststep']
start=time.time();lastloss=0
for step in range(step0+1,50001):
 frac=(step-1)/3125;lr=.0003*(.1+.9*frac) if step<=3125 else .0003*.5*(1+math.cos(math.pi*(step-3125)/(50000-3125)))
 for pg in opt.param_groups:pg['lr']=lr
 m.train();j=rng.choice(len(names),64,p=prob);pred=m(zero,[idxs[i] for i in j],None)[0];loss=m.loss(pred,None,None,targets[j])[0]
 assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();lastloss=float(loss.detach())
 if step%100==0:
  status=dict(fold=fold,stage='training',step=step,max_steps=50000,best_validation_pearson=best if np.isfinite(best) else None,best_step=beststep,loss=lastloss,lr=lr,seconds_per_update=(time.time()-start)/(step-step0),updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
  savej(out/'STATUS.json',status);print(json.dumps(status),flush=True)
 if step%500==0:
  score,by,mse=validate();rec=dict(step=step,pearson=score,by_context=by,mse=mse,lr=lr)
  with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  if score>best:
   best=score;beststep=step;tmp=out/'best.pt.tmp';torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},tmp);tmp.replace(out/'best.pt');savej(out/'best.json',rec)
  tmp=resume.with_suffix('.tmp');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),step=step,best=best,beststep=beststep,protocol_sha256=sha(T/'protocol.json')),tmp);tmp.replace(resume)
  if step>=10000 and step-beststep>=10000:break
m.load_state_dict(torch.load(out/'best.pt',weights_only=True,map_location='cpu'))
# Only test QUERY IDENTITIES are consulted to generate the frozen outputs.
te=splits(fold,'test');assert set(te)=={fold};qs=sorted(te[fold]);delta=predict(m,[[vocab[p],-1] for p in qs]);assert np.isfinite(delta).all()
np.save(out/'native_delta.npy',delta);np.savez(out/'axes.npz',targets=np.array(qs),genes=GENES)
savej(out/'PREDICTIONS_SEALED.json',dict(model='TxPert',fold=fold,step=step,selected_step=beststep,source_validation_pearson=best,checkpoint_sha256=sha(out/'best.pt'),prediction_sha256=sha(out/'native_delta.npy'),protocol_sha256=sha(T/'protocol.json'),queries=len(qs),genes=len(GENES),time=time.strftime('%Y-%m-%d %H:%M:%S %z')))
savej(out/'STATUS.json',dict(fold=fold,stage='predictions_sealed',step=step,selected_step=beststep))
print('TRAINING_AND_PREDICTIONS_COMPLETE',fold,step,beststep,best,flush=True)
