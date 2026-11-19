from engine import *
from scipy.linalg import eigh,cho_factor,cho_solve
from ot_core import fit_gaussian_bures
from representation import source_predict as reference_predict

def aid(a):return format(a,'g').replace('.','p')
def point_id(a,rank):return 'UC_Jstyle_a'+aid(a)+'_d'+str(rank)
LEGACY='UC_Jstyle_legacy32_a0p25'

class GPBank:
 def __init__(self,d):
  self.d=d;z=np.load(ROOT/'shared_gene_basis.npz');self.phi=z['phi'];self.rem=z['diagonal_remainder'];self.targets=z['targets'];np.testing.assert_array_equal(self.targets,d.features[12]['targets']);assert self.phi.shape==(2023,512);self.Q=np.column_stack([np.ones(len(self.targets)),self.phi]);self.basis_hash=sha(ROOT/'shared_gene_basis.npz');self.parity=[]
 def model_path(self,source,identity):return ROOT/'cache/gp'/('a'+aid(identity))/source
 def posterior(self,source,identity):
  d=self.d;assert source in d.sources and source!=d.fold;path=self.model_path(source,identity);path.mkdir(parents=True,exist_ok=True);ix=d.rows[source];F=np.column_stack([np.ones(len(ix)),self.phi[ix]]);tau=identity+self.rem[ix];D=tau+np.maximum(d.noise[source]/d.scales[source],1e-6)
  fingerprints={name:sha(P/source/name) for name in ['delta.npy','noise_mean.npy','meta.npz']}
  if (path/'fit.json').exists():
   meta=json.load(open(path/'fit.json'));assert meta['source_sha256']==fingerprints and meta['basis_sha256']==self.basis_hash and meta['identity']==identity;z=np.load(path/'fit_arrays.npz');np.testing.assert_array_equal(z['D'],D);coef=np.load(path/'coefficients.npy',mmap_mode='r');chol=(z['chol'],True)
  else:
   V=F/D[:,None];M=np.eye(F.shape[1])+F.T@V;chol=cho_factor(M,lower=True,check_finite=False);coef=cho_solve(chol,V.T@np.asarray(d.y[source],float),check_finite=False);jj=np.unique(np.linspace(0,len(d.G)-1,8,dtype=int));rhs=V.T@np.asarray(d.y[source][:,jj],float);res=float(np.linalg.norm(M@coef[:,jj]-rhs)/max(np.linalg.norm(rhs),1e-20));assert res<1e-8
   np.save(path/'coefficients.npy',coef);np.savez_compressed(path/'fit_arrays.npz',chol=chol[0],D=D,tau=tau,source_rows=ix);save(path/'fit.json',dict(source=source,identity=identity,source_scale=d.scales[source],basis_sha256=self.basis_hash,source_sha256=fingerprints,normal_equation_residual=res,independent_source_fit=True))
  mu=self.Q@coef;H=cho_solve(chol,self.Q.T,check_finite=False).T;q=np.einsum('ij,ij->i',self.Q,H);rr=np.array([d.lookup[source].get(g,-1) for g in self.targets]);known=rr>=0;a=np.zeros(len(rr));a[known]=tau[rr[known]]/D[rr[known]];Y=np.asarray(d.y[source],float);mu[known]+=a[known,None]*(Y[rr[known]]-mu[known]);tq=identity+self.rem;var=tq.copy();var[known]-=tau[rr[known]]**2/D[rr[known]];var+=(1-a)**2*q;assert var.min()>-1e-8;var=np.maximum(var,0)/(tq+np.einsum('ij,ij->i',self.Q,self.Q))
  if identity==.25:
   sample=np.r_[np.flatnonzero(known)[:16],np.flatnonzero(~known)[:16]];ref=reference_predict(d,source,self.targets[sample],self.phi,self.rem);np.testing.assert_allclose(mu[sample],ref[0],atol=1e-9,rtol=1e-9);np.testing.assert_allclose(var[sample],ref[1],atol=1e-10,rtol=1e-9);self.parity.append(dict(source=source,identity=identity,n_queries=len(sample),status='PASS'))
  mu[known]=Y[rr[known]]
  return dict(mu=mu,var=var,known=known,model_path=str(path))
 def all_sources(self,identity):return {s:self.posterior(s,identity) for s in self.d.sources}

class ExactMaps:
 def __init__(self,d,dimensions):self.d=d;self.dimensions=dimensions;self.cache={};self.audit={}
 def pair(self,source,target):
  a,b=sorted([source,target]);key=a+'__'+b
  if key in self.cache:return self.cache[key]
  p=ROOT/'cache/exact_maps'/key;p.mkdir(parents=True,exist_ok=True);fingerprints={s:sha((P/s if s in self.d.sources else ROOT/'target_inputs'/s)/'controls.npy') for s in [a,b]}
  if not (p/'READY.json').exists():
   xa=np.array(self.d.ctrl[a],dtype=float,copy=True);xb=np.array(self.d.ctrl[b],dtype=float,copy=True);xa-=xa.mean(0);xb-=xb.mean(0);M=np.concatenate([xa/np.sqrt(2*len(xa)),xb/np.sqrt(2*len(xb))]);total=float(np.einsum('ij,ij->',M,M));gram=M@M.T;maximum=max(self.dimensions);ev,V=eigh(gram,subset_by_index=[len(M)-maximum,len(M)-1],check_finite=False,driver='evr');ev=ev[::-1];V=V[:,::-1];assert ev[-1]>max(ev[0]*1e-12,1e-12);U=(M.T@V)/np.sqrt(ev);np.testing.assert_allclose(U.T@U,np.eye(maximum),atol=1e-9,rtol=1e-9);eigen_res=float(np.linalg.norm(gram@V-V*ev)/np.linalg.norm(gram));assert eigen_res<1e-8;np.savez_compressed(p/'basis.npz',U=U,eigenvalues=ev,total_control_energy=total)
   records=[]
   for rank in self.dimensions:
    if rank==0:continue
    fa=fit_gaussian_bures(xa@U[:,:rank],xb@U[:,:rank],shrinkage=.1);fb=fit_gaussian_bures(xb@U[:,:rank],xa@U[:,:rank],shrinkage=.1);err=max(fa['alignment_error_relative'],fb['alignment_error_relative']);assert err<1e-8;np.savez_compressed(p/('d'+str(rank)+'.npz'),ab=fa['A'],ba=fb['A']);records.append(dict(pair=key,dimension=rank,retained_control_variance=float(ev[:rank].sum()/total),covariance_alignment_error=err))
   save(p/'READY.json',dict(control_sha256=fingerprints,records=records,exact_pca_eigen_residual=eigen_res,perturbation_responses_used=False));log(stage='EXACT_MAPS_READY',pair=key,records=records)
  meta=json.load(open(p/'READY.json'));assert meta['control_sha256']==fingerprints;z=np.load(p/'basis.npz');maps={r:np.load(p/('d'+str(r)+'.npz')) for r in self.dimensions if r};obj=dict(U=z['U'],maps=maps,path=str(p));self.cache[key]=obj;self.audit[key]=meta;return obj
 def transformed(self,source,target,mu,rank,projection=None):
  if rank==0:return mu
  a,b=sorted([source,target]);pair=self.pair(source,target);U=pair['U'][:,:rank];A=pair['maps'][rank]['ab' if source==a else 'ba'];projected=mu@U if projection is None else projection[:,:rank];return mu+(projected@(A-np.eye(rank)))@U.T
 def projections(self,pool,target,parts):return {s:parts[s]['mu']@self.pair(s,target)['U'] for s in pool}

def subset(d,post,pool,queries):
 ix=np.array([d.gindex[g] for g in queries]);return {s:{k:post[s][k][ix] for k in ['mu','var','known']} for s in pool}
def response(d,maps,parts,pool,target,rank,projections=None,legacy=False):
 var=np.stack([parts[s]['var'] for s in pool]);known=np.stack([parts[s]['known'] for s in pool]);w=d.gate(pool,target,var,known,dict(temperature=.25,precision=True,measured_only=False));delta=np.zeros_like(parts[pool[0]]['mu'])
 for i,s in enumerate(pool):
  mu=parts[s]['mu'];mapped=mu+d.correction(s,target,mu) if legacy else maps.transformed(s,target,mu,rank,None if projections is None else projections[s]);delta+=w[i,:,None]*mapped
 return delta,w,known
