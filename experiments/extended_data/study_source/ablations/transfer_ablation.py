import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,gc,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from transfer import O,B,C,genes,save,sha
V=Path('@STUDY@/uc_cross_context_transfer/experiments/uc_state_prior_relearn_20261101/prepared_new');axes=np.load(V/'view_axes.npz');targets=axes['targets'];views=axes['views'];gindex={g:i for i,g in enumerate(targets)}
class Data:
 def __init__(self,c):
  self.c=c;self.r=B/'runs/uc'/c;self.sources=json.loads((self.r/'protocol.json').read_text())['folds'][c];self.meta={};self.y={};self.noise={};self.lookup={};self.rows={};self.scales=json.loads((self.r/'results'/c/'source_scales.json').read_text())
  for s in self.sources+[c]:
   p=self.r/('prepared' if s!=c else 'target_inputs')/s;z=np.load(p/'meta.npz');self.meta[s]={k:z[k] for k in z.files};self.lookup[s]={g:i for i,g in enumerate(z['targets'])}
   if s!=c:self.y[s]=np.load(p/'delta.npy',mmap_mode='r');self.noise[s]=np.load(p/'noise_mean.npy');self.rows[s]=np.array([gindex[g] for g in z['targets']])
 def geometry(self,pool,target):
  means=np.stack([self.meta[s]['control'] for s in sorted(pool)]);v=np.mean([self.meta[s]['control_var'] for s in sorted(pool)],0);v=np.maximum(v,.05*v.mean());dd=((means[:,None]-means[None,:])**2/v).mean(2);upper=dd[np.triu_indices(len(pool),1)];ell=max(float(np.median(upper)),1e-12) if len(upper) else 1.;dist=np.array([np.mean((self.meta[s]['control']-self.meta[target]['control'])**2/v) for s in pool]);return ell,dist
 def gate(self,pool,target,var,known,temp):
  ell,dist=self.geometry(pool,target);logg=np.zeros(len(pool)) if temp is None else -(dist-dist.min())/(temp*ell);lw=logg[:,None]-np.log(var+.05);lw[~known]=-np.inf;ok=known.any(0);w=np.zeros_like(lw);w[:,ok]=np.exp(lw[:,ok]-logsumexp(lw[:,ok],axis=0)[None,:]);return w
 def posterior(self,phi,rem):
  Q=np.column_stack([np.ones(len(targets)),phi]);post={}
  for s in self.sources:
   ix=self.rows[s];F=Q[ix];tau=rem[ix];D=tau+np.maximum(self.noise[s]/self.scales[s],1e-6);chol=cho_factor(np.eye(Q.shape[1])+F.T@(F/D[:,None]),lower=True,check_finite=False);H=cho_solve(chol,Q.T,check_finite=False).T;q=np.einsum('ij,ij->i',Q,H);rr=np.array([self.lookup[s].get(g,-1) for g in targets]);known=rr>=0;a=np.zeros(len(rr));a[known]=tau[rr[known]]/D[rr[known]];var=rem.copy();var[known]-=tau[rr[known]]**2/D[rr[known]];var+=(1-a)**2*q;assert var.min()>-1e-8;var=np.maximum(var,0)/(rem+np.einsum('ij,ij->i',Q,Q));post[s]=dict(var=var,known=known)
  return post
 def effects(self,s,perts,ot=False,target=None):
  rr=np.array([self.lookup[s].get(g,-1) for g in perts]);known=rr>=0;y=np.zeros((len(perts),len(genes)));y[known]=self.y[s][rr[known]]
  if ot:
   a,b=sorted([s,target]);key=a+'__'+b;u=np.load(self.r/'cache/exact_maps'/key/'basis.npz')['U'][:,:128];m=np.load(self.r/'cache/no_shrink_maps'/key/'map.npz')['ab' if s==a else 'ba'];y+=(y@u@(m-np.eye(128)))@u.T
  return y
def spectrum(name,drop=None):
 folder=O/'transfer/spectra';folder.mkdir(exist_ok=True);file=folder/(name+'.npz')
 if not file.exists():
  kernels=np.load(V/'view_kernels.npy',mmap_mode='r');ids=[i for i in range(12) if i!=drop];k=np.zeros((len(targets),len(targets)))
  for i in ids:k+=kernels[i]/len(ids)
  diagonal=np.diag(k).copy();ev,u=eigh(k,overwrite_a=True,check_finite=False,driver='evd');order=np.argsort(ev)[::-1][:512];phi=u[:,order]*np.sqrt(np.maximum(ev[order],0));rem=np.maximum(diagonal-np.einsum('ij,ij->i',phi,phi),0);np.savez_compressed(file,phi=phi,rem=rem,targets=targets,active=np.array(ids),weights=np.full(len(ids),1/len(ids)));print('SPECTRUM',name,flush=True)
 z=np.load(file);return z['phi'],z['rem']
def verify_full(d):
 z=np.load(d.r/'shared_gene_basis.npz');np.testing.assert_array_equal(z['targets'],targets);phi,rem=spectrum('full12');reference=z['phi']@z['phi'].T;err=float(np.max(abs(phi@phi.T-reference)));assert err<1e-7,err;assert np.max(abs(rem-z['diagonal_remainder']))<1e-7
 post=d.posterior(z['phi'],z['diagonal_remainder']);out=O/'transfer'/d.c;qa=np.load(out/'axes.npz');perts=qa['perts'];qi=np.array([gindex[p] for p in perts]);wz=np.load(d.r/'results'/d.c/'UC_observed_no_shrink_adaptive_weights.npz');ix=np.array([{g:i for i,g in enumerate(wz['targets'])}[p] for p in perts]);var=np.stack([post[s]['var'][qi] for s in d.sources]);known=np.stack([post[s]['known'][qi] for s in d.sources]);np.testing.assert_array_equal(wz['sources'],d.sources);ve=float(np.max(abs(var-wz['relative_posterior_variance'][:,ix])));assert ve<1e-9,ve;temp=json.loads((d.r/'results'/d.c/'SOURCE_SELECTION_FROZEN.json').read_text())['selected']['temperature'];w=d.gate(d.sources,d.c,var,known,temp);we=float(np.max(abs(w-qa['weights'])));assert we<1e-9,we
 save(out/'PRIOR_REPLAY_QA.json',dict(kernel_max_error=err,variance_max_error=ve,source_weight_max_error=we));return post,temp
def prior(c):
 d=Data(c);post,temp=verify_full(d);out=O/'transfer'/c;perts=np.load(out/'axes.npz')['perts'];qi=np.array([gindex[p] for p in perts]);mapped=[d.effects(s,perts,True,c) for s in d.sources]
 for j,v in enumerate(views):
  name='drop_'+str(v);path=out/(name+'_native.npy')
  if path.exists():continue
  phi,rem=spectrum(name,j);pp=d.posterior(phi,rem);var=np.stack([pp[s]['var'][qi] for s in d.sources]);known=np.stack([pp[s]['known'][qi] for s in d.sources]);w=d.gate(d.sources,c,var,known,temp);pred=sum(w[i,:,None]*a for i,a in enumerate(mapped));np.save(path,pred);np.savez_compressed(out/(name+'_weights.npz'),weights=w,variance=var,known=known);save(out/(name+'_SEALED.json'),dict(sha256=sha(path),temperature=temp,active_views=[str(x) for i,x in enumerate(views) if i!=j],target_responses_used=False));print('TRANSFER PRIOR',c,v,flush=True)
 save(out/'PRIORS_SEALED.json',dict(status='SEALED',ablations=12))
def score_source(delta,truth,held,perts):
 seed=20261101+int(hashlib.sha256(('specificity_panel_'+held).encode()).hexdigest()[:7],16);rng=np.random.default_rng(seed);n=min(256,len(perts));ix=rng.choice(len(perts),n,replace=False);nq=min(128,n);a=delta[ix[:nq]];b=truth[ix];r={}
 for name,dist in [('l1',cdist(a,b,'cityblock')),('l2',np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0))]:r['rank_'+name]=float(np.mean(1-np.argmax(np.argsort(dist,axis=1)==np.arange(nq)[:,None],axis=1)/n))
 return dict(nmse=float(np.mean((delta-truth)**2)/max(float(np.mean(truth*truth)),1e-20)),rank_score=(r['rank_l1']+r['rank_l2'])/2,**r)
def temperature(c):
 d=Data(c);post,temp=verify_full(d);out=O/'transfer'/c
 if (out/'IDENTITY_TEMPERATURE_SEALED.json').exists():return
 original=json.loads((d.r/'results'/c/'SOURCE_SELECTION_FROZEN.json').read_text());cv=original['selection_weights']['weights'];temps=json.loads((d.r/'protocol.json').read_text())['temperature_grid'];records=[dict(id='identity_'+str(t),temperature=t,cases=[]) for t in temps];replay=[]
 for held in d.sources:
  pool=[s for s in d.sources if s!=held];perts=d.meta[held]['targets'];qi=np.array([gindex[p] for p in perts]);var=np.stack([post[s]['var'][qi] for s in pool]);known=np.stack([post[s]['known'][qi] for s in pool]);effects=[d.effects(s,perts) for s in pool];truth=np.asarray(d.y[held],float)
  for r in records:
   w=d.gate(pool,held,var,known,r['temperature']);delta=sum(w[i,:,None]*a for i,a in enumerate(effects));r['cases'].append(dict(target=held,**score_source(delta,truth,held,perts)))
  mapped=[d.effects(s,perts,True,held) for s in pool];w=d.gate(pool,held,var,known,temp);rr=score_source(sum(w[i,:,None]*a for i,a in enumerate(mapped)),truth,held,perts);gold=next(x for x in original['selected']['cases'] if x['target']==held);err=max(abs(rr[k]-gold[k]) for k in rr);assert err<1e-8,(held,rr,gold);replay.append(dict(target=held,max_error=err))
 for r in records:r['weighted']={k:float(sum(cv[x['target']]*x[k] for x in r['cases'])) for k in ['nmse','rank_l1','rank_l2','rank_score']}
 base=next(r for r in records if r['temperature']==.25);cap=1.05*base['weighted']['nmse'];eligible=[r for r in records if r['weighted']['nmse']<=cap];chosen=min(eligible,key=lambda r:(-r['weighted']['rank_score'],r['weighted']['nmse'],-(1e6 if r['temperature'] is None else r['temperature']),r['id'])) if eligible else min(records,key=lambda r:(r['weighted']['nmse'],-r['weighted']['rank_score'],r['id']))
 perts=np.load(out/'axes.npz')['perts'];qi=np.array([gindex[p] for p in perts]);var=np.stack([post[s]['var'][qi] for s in d.sources]);known=np.stack([post[s]['known'][qi] for s in d.sources]);w=d.gate(d.sources,c,var,known,chosen['temperature']);pred=sum(w[i,:,None]*d.effects(s,perts) for i,s in enumerate(d.sources));path=out/'weighted_identity_retuned_native.npy';np.save(path,pred);save(out/'IDENTITY_TEMPERATURE_SEALED.json',dict(status='SEALED',selected=chosen,grid=records,original_ot_temperature=temp,source_validation_replay=replay,selection_weights=cv,nmse_cap=cap,target_responses_used=False,prediction_sha256=sha(path)));print('IDENTITY T',c,temp,'to',chosen['temperature'],flush=True)
if __name__=='__main__':
 if sys.argv[1]=='spectrum':
  spectrum('full12')
  for i,v in enumerate(views):spectrum('drop_'+str(v),i)
 elif sys.argv[1]=='prior':prior(sys.argv[2])
 elif sys.argv[1]=='temperature':temperature(sys.argv[2])
