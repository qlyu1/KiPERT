from common import *
from scipy.linalg import cho_factor,cho_solve
from ot_core import fit_gaussian_bures
from collections import OrderedDict

class Data:
 def __init__(self,fold):
  global CACHE,R
  self.fold=fold;self.sources=json.load(open(ROOT/'protocol.json'))['folds'][fold];assert fold not in self.sources and len(set(self.sources))==3;self.bio={s:s for s in self.sources};self.G=np.load(P/'output_genes.npy');CACHE=ROOT/'cache'/fold;R=ROOT/'results'/fold;R.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
  self.meta={};self.ctrl={};self.y={};self.noise={};self.lookup={};self.scales={};self.features={};self.experts={};self.geometries={};self.map_audit={};self.fit_audit={};self.map_cache=OrderedDict()
  for n in self.sources+[fold]:
   p=P/n if n in self.sources else ROOT/'target_inputs'/n;z=np.load(p/'meta.npz');self.meta[n]={k:z[k] for k in z.files};assert np.array_equal(z['genes'],self.G);self.lookup[n]={g:i for i,g in enumerate(z['targets'])};self.ctrl[n]=np.load(p/'controls.npy')
   if n in self.sources:
    self.y[n]=np.load(p/'delta.npy',mmap_mode='r');self.noise[n]=np.load(p/'noise_mean.npy');self.scales[n]=max(float(np.var(self.y[n].astype(float),axis=0).mean()),1e-12)
  z=np.load(P/'shared_gene_basis.npz');self.features[8]={k:z[k] for k in ['targets','phi','diagonal_remainder']};self.gindex={g:i for i,g in enumerate(z['targets'])};self.rows={n:np.array([self.gindex[g] for g in self.meta[n]['targets']]) for n in self.sources};save(R/'source_scales.json',self.scales)
 def fit(self,prior,n):
  key=(prior,n)
  if key in self.experts:return self.experts[key]
  z=self.features[prior];ii=self.rows[n];F=np.column_stack([np.ones(len(ii)),z['phi'][ii]]);tau=.25+z['diagonal_remainder'][ii];D=tau+np.maximum(self.noise[n]/self.scales[n],1e-6);V=F/D[:,None];matrix=np.eye(F.shape[1])+F.T@V;chol=cho_factor(matrix,lower=True,check_finite=False)
  p=CACHE/'experts'/str(prior)/n/'beta.npy';p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():
   beta=np.lib.format.open_memmap(p,mode='w+',dtype=np.float64,shape=(F.shape[1],len(self.G)))
   for lo in range(0,len(self.G),256):beta[:,lo:lo+256]=cho_solve(chol,V.T@np.asarray(self.y[n][:,lo:lo+256],float),check_finite=False)
   beta.flush();del beta
  beta=np.load(p,mmap_mode='r');jj=np.unique(np.linspace(0,len(self.G)-1,8,dtype=int));rhs=V.T@np.asarray(self.y[n][:,jj],float);res=float(np.linalg.norm(matrix@beta[:,jj]-rhs)/max(np.linalg.norm(rhs),1e-20));assert res<1e-10
  ex=dict(beta=beta,chol=chol,D=D,tau=tau);self.experts[key]=ex;self.fit_audit[str(prior)+'/'+n]=dict(n_targets=len(ii),n_output_genes=len(self.G),normal_equation_residual=res,source_only=n,status='PASS');save(R/'expert_fit_audit.json',self.fit_audit);log(stage='expert_fitted',prior=prior,source=n,n_targets=len(ii),n_genes=len(self.G));return ex
 def predict(self,prior,n,q):
  ex=self.fit(prior,n);z=self.features[prior];qi=np.array([self.gindex[g] for g in q]);F=np.column_stack([np.ones(len(q)),z['phi'][qi]]);mu=F@ex['beta'];rr=np.array([self.lookup[n].get(g,-1) for g in q]);known=rr>=0;tau=.25+z['diagonal_remainder'][qi];a=np.zeros(len(q));a[known]=ex['tau'][rr[known]]/ex['D'][rr[known]];mu[known]+=a[known,None]*(np.asarray(self.y[n][rr[known]],float)-mu[known]);H=cho_solve(ex['chol'],F.T,check_finite=False).T;var=tau.copy();var[known]-=ex['tau'][rr[known]]**2/ex['D'][rr[known]];var+=(1-a)**2*(F*H).sum(1);assert var.min()>-1e-8
  return mu,np.maximum(var,0)/(tau+(F*F).sum(1)),known
 def geometry(self,pool,target):
  key=(tuple(pool),target)
  if key in self.geometries:return self.geometries[key]
  bios=sorted({self.bio[s] for s in pool});means=np.stack([np.mean([self.meta[s]['control'] for s in pool if self.bio[s]==b],0) for b in bios]);v=np.mean([np.mean([self.meta[s]['control_var'] for s in pool if self.bio[s]==b],0) for b in bios],0);v=np.maximum(v,.05*v.mean());dd=((means[:,None]-means[None,:])**2/v).mean(2);upper=dd[np.triu_indices(len(bios),1)];ell=max(float(np.median(upper)),1e-12) if len(upper) else 1.;dt=np.array([np.mean((self.meta[s]['control']-self.meta[target]['control'])**2/v) for s in pool]);self.geometries[key]=(ell,dt);return ell,dt
 def gate(self,pool,target,variance,known,recipe):
  ell,dist=self.geometry(pool,target);temp=recipe['temperature'];g=np.ones(len(pool)) if temp is None else np.exp(-(dist-dist.min())/(temp*ell));eligible=known.copy() if recipe['measured_only'] else np.ones_like(known,bool);empty=~eligible.any(0);eligible[:,empty]=True;w=np.zeros_like(variance)
  for i,s in enumerate(pool):
   count=eligible[[j for j,n in enumerate(pool) if self.bio[n]==self.bio[s]]].sum(0);w[i]=g[i]*eligible[i]/np.maximum(count,1)
  if recipe['precision']:w/=variance+.05
  assert (w.sum(0)>0).all();return w/w.sum(0)
 def pair_map(self,source,target):
  a,b=sorted([source,target]);key=a+'__'+b;p=CACHE/'maps'/(key+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
  if key in self.map_cache:
   z=self.map_cache.pop(key);self.map_cache[key]=z;return z['U'],z['ab'] if source==a else z['ba']
  if not p.exists():
   xa=np.asarray(self.ctrl[a],float);xb=np.asarray(self.ctrl[b],float);xa-=xa.mean(0);xb-=xb.mean(0);M=np.concatenate([xa/np.sqrt(2*len(xa)),xb/np.sqrt(2*len(xb))]);rank=min(32,len(M)-2,len(self.G));width=min(rank+8,len(M),len(self.G));rng=np.random.default_rng(seed_for(key));Q=np.linalg.qr(M@rng.normal(size=(len(self.G),width)),mode='reduced')[0];Z=np.linalg.qr(M.T@Q,mode='reduced')[0];Q=np.linalg.qr(M@Z,mode='reduced')[0];_,sv,vt=np.linalg.svd(Q.T@M,full_matrices=False);rank=min(rank,int((sv>max(sv[0]*1e-10,1e-12)).sum()));U=vt[:rank].T;np.testing.assert_allclose(U.T@U,np.eye(rank),atol=1e-10);fa=fit_gaussian_bures(xa@U,xb@U,shrinkage=.1);fb=fit_gaussian_bures(xb@U,xa@U,shrinkage=.1);error=max(fa['alignment_error_relative'],fb['alignment_error_relative']);assert error<1e-8
   retained=float(np.sum((M@U)**2)/max(float(np.sum(M*M)),1e-20));np.savez_compressed(p,U=U,ab=fa['A'],ba=fb['A'],alignment_error=error,retained_control_variance=retained);log(stage='control_pair_OT_ready',pair=key,rank=rank,error=error)
  f=np.load(p);z={k:f[k] for k in f.files};self.map_cache[key]=z
  if len(self.map_cache)>32:self.map_cache.popitem(last=False)
  U=z['U'];A=z['ab'] if source==a else z['ba'];self.map_audit[key]=dict(control_contexts=[a,b],rank=U.shape[1],alignment_error=float(z['alignment_error']),retained_control_variance=float(z['retained_control_variance']),perturbation_responses_used=False);return U,A
 def correction(self,source,target,pred):
  U,A=self.pair_map(source,target);return (pred@U@(A-np.eye(len(A))))@U.T

def mix(pred,w):return np.einsum('sqg,sq->qg',pred,w,optimize=True)
