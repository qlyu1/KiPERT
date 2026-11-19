from engine import *
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.optimize import minimize

class ViewBank:
 def __init__(self):
  z=np.load(ROOT/'prepared_new/view_axes.npz');self.targets=z['targets'];self.ensembl=z['ensembl'];self.views=z['views'];self.presence=z['presence'];self.kernels=np.load(ROOT/'prepared_new/view_kernels.npy',mmap_mode='r');self.stats={};self.N=len(self.targets)
  self.uniform=np.full(12,1/12);self.family=np.array([1/30 if str(v).startswith('GO_') else 1/10 for v in self.views]);assert abs(self.family.sum()-1)<1e-12
 def alignment(self,d,source):
  if source in self.stats:return self.stats[source]
  assert source in d.sources and source!=d.fold
  ix=d.rows[source];Y=np.asarray(d.y[source],float);Y-=Y.mean(0);R=Y@Y.T/Y.shape[1];np.fill_diagonal(R,0);norm=np.linalg.norm(R);assert norm>0;R/=norm;flat=np.empty((12,len(ix)**2))
  for j in range(12):
   K=np.asarray(self.kernels[j][np.ix_(ix,ix)],float);K=K-K.mean(0)[None,:]-K.mean(1)[:,None]+K.mean();np.fill_diagonal(K,0);n=np.linalg.norm(K);flat[j]=(K/max(n,1e-20)).ravel()
  G=flat@flat.T;h=flat@R.ravel();self.stats[source]=(G,h);del flat,R,Y;gc.collect();return G,h
 def weights(self,d,pool,rule):
  assert set(pool).issubset(d.sources) and d.fold not in pool
  if rule=='equal_views':return self.uniform.copy(),dict(rule=rule,weight_fit_sources=[])
  if rule=='equal_resources':return self.family.copy(),dict(rule=rule,weight_fit_sources=[])
  lam=float(rule.split('_')[1]);pairs=[self.alignment(d,s) for s in pool];G=np.mean([p[0] for p in pairs],0);h=np.mean([p[1] for p in pairs],0);u=self.family
  def fun(w):return float(w@G@w-2*h@w+lam*np.sum((w-u)**2))
  def jac(w):return 2*(G@w-h+lam*(w-u))
  opt=minimize(fun,u,jac=jac,method='SLSQP',bounds=[(0,1)]*12,constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(12))],options=dict(ftol=1e-11,maxiter=1000))
  assert opt.success,opt.message;w=np.maximum(opt.x,0);w/=w.sum();assert fun(w)<=fun(u)+1e-8
  return w,dict(rule=rule,lambda_value=lam,weight_fit_sources=list(pool),objective=fun(w),uniform_resource_objective=fun(u),iterations=int(opt.nit),alignment_h=h.tolist())
 def spectrum(self,weights):
  key=hashlib.sha256(np.round(weights,14).tobytes()).hexdigest()[:20];folder=ROOT/'cache/spectra';folder.mkdir(parents=True,exist_ok=True);path=folder/(key+'.npz')
  if path.exists():
   z=np.load(path);np.testing.assert_allclose(z['weights'],weights,atol=1e-13,rtol=0);return z['phi'],z['eigenvalues'],z['diagonal'],key
  t=time.time();K=np.zeros((self.N,self.N));diagonal=np.zeros(self.N)
  for w,k in zip(weights,self.kernels):
   if w:K+=w*k
  diagonal=np.diag(K).copy();ev,U=eigh(K,overwrite_a=True,check_finite=False,driver='evd');assert ev.min()>-1e-7,float(ev.min());ix=np.argsort(ev)[::-1];ev=np.maximum(ev[ix],0);U=U[:,ix];keep=ev>max(ev[0]*1e-12,1e-12);ev=ev[keep];phi=U[:,keep]*np.sqrt(ev)
  sample=np.unique(np.linspace(0,self.N-1,32,dtype=int));expected=sum(w*np.asarray(k[np.ix_(sample,sample)]) for w,k in zip(weights,self.kernels));np.testing.assert_allclose(phi[sample]@phi[sample].T,expected,atol=1e-8,rtol=1e-8)
  np.savez_compressed(path,phi=phi,eigenvalues=ev,diagonal=diagonal,weights=weights);log(stage='SPECTRUM_READY',key=key,numerical_rank=len(ev),seconds=time.time()-t,full_kernel_sample_parity='PASS');return phi,ev,diagonal,key

def source_predict(d,source,queries,phi,rem,force=None,save_dir=None):
 ix=d.rows[source];qi=np.array([d.gindex[g] for g in queries]);F=np.column_stack([np.ones(len(ix)),phi[ix]]);Q=np.column_stack([np.ones(len(qi)),phi[qi]]);tau=.25+rem[ix];D=tau+np.maximum(d.noise[source]/d.scales[source],1e-6);Y=np.asarray(d.y[source],float)
 kind=force or ('primal' if F.shape[1]<=len(F) else 'dual')
 if kind=='primal':
  V=F/D[:,None];M=np.eye(F.shape[1])+F.T@V;chol=cho_factor(M,lower=True,check_finite=False);coef=cho_solve(chol,V.T@Y,check_finite=False);mu=Q@coef;H=cho_solve(chol,Q.T,check_finite=False).T;q=np.einsum('ij,ij->i',Q,H)
  sample=np.unique(np.linspace(0,Y.shape[1]-1,8,dtype=int));rhs=V.T@Y[:,sample];res=float(np.linalg.norm(M@coef[:,sample]-rhs)/max(np.linalg.norm(rhs),1e-20))
 else:
  M=F@F.T;M.flat[::len(M)+1]+=D;chol=cho_factor(M,lower=True,check_finite=False);coef=cho_solve(chol,Y,check_finite=False);cross=Q@F.T;mu=cross@coef;H=cho_solve(chol,cross.T,check_finite=False).T;q=np.einsum('ij,ij->i',Q,Q)-np.einsum('ij,ij->i',cross,H)
  sample=np.unique(np.linspace(0,Y.shape[1]-1,8,dtype=int));res=float(np.linalg.norm(M@coef[:,sample]-Y[:,sample])/max(np.linalg.norm(Y[:,sample]),1e-20))
 assert res<1e-8,(source,kind,res);assert q.min()>-1e-7; q=np.maximum(q,0)
 rr=np.array([d.lookup[source].get(g,-1) for g in queries]);known=rr>=0;a=np.zeros(len(queries));a[known]=tau[rr[known]]/D[rr[known]];mu[known]+=a[known,None]*(Y[rr[known]]-mu[known]);tq=.25+rem[qi];variance=tq.copy();variance[known]-=tau[rr[known]]**2/D[rr[known]];variance+=(1-a)**2*q;variance=np.maximum(variance,0)/(tq+np.einsum('ij,ij->i',Q,Q))
 if save_dir is not None:
  save_dir.mkdir(parents=True,exist_ok=True);np.save(save_dir/'coefficients.npy',coef);np.savez_compressed(save_dir/'fit_arrays.npz',source_rows=ix,D=D,tau=tau);save(save_dir/'fit.json',dict(source=source,solver=kind,dimension=phi.shape[1],coefficient_shape=coef.shape,normal_equation_residual=res,independent_source_fit=True))
 return mu,variance,known,dict(source=source,solver=kind,rank=phi.shape[1],fit_residual=res)

def predict(d,pool,target,queries,recipe,phi=None,rem=None,model_dir=None):
 pred=[];var=[];known=[];audit=[]
 for source in pool:
  if phi is None:mu,v,k=d.predict(12,source,queries);fit=dict(source=source,solver='legacy')
  else:mu,v,k,fit=source_predict(d,source,queries,phi,rem,save_dir=model_dir/source if model_dir else None)
  rr=np.array([d.lookup[source].get(g,-1) for g in queries]);mu[k]+=recipe['rho']*(np.asarray(d.y[source][rr[k]],float)-mu[k]);pred.append(mu+d.correction(source,target,mu));var.append(v);known.append(k);audit.append(fit)
 w=d.gate(pool,target,np.stack(var),np.stack(known),recipe['gate']);return mix(np.stack(pred),w),np.stack(known),audit

def verify_engine(d):
 source=d.sources[0];q=d.meta[source]['targets'][:32];z=d.features[12];expected=d.predict(12,source,q)
 for solver in ['primal','dual']:
  result=source_predict(d,source,q,z['phi'],z['diagonal_remainder'],force=solver)
  np.testing.assert_allclose(result[0],expected[0],atol=1e-8,rtol=1e-8);np.testing.assert_allclose(result[1],expected[1],atol=1e-9,rtol=1e-8);np.testing.assert_array_equal(result[2],expected[2])
 log(stage='SOLVER_PARITY_PASS',fold=d.fold,source=source,solvers=['primal','dual'])
