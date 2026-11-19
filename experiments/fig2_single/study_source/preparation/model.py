from common import *
from scipy import sparse
from scipy.linalg import eigh,solve
import sys
sys.path.insert(0,str(R/'assets/uc'))
from mean_model import response_subspace

class Bank:
 def __init__(self,ds):
  self.path=ROOT/'priors'/ds;self.targets=np.load(self.path/'targets.npy');self.index={g:i for i,g in enumerate(self.targets)};self.raw={};self.presence={};self.sizes={}
  for v in VIEWS:
   if v in ['DepMap','GenePT','STRING','HPA']:
    z=np.load(self.path/(v+'.npz'));self.raw[v]=z['x'].astype(float);self.presence[v]=z['present']
   else:self.raw[v]=sparse.load_npz(self.path/(v+'.npz'));self.sizes[v]=np.load(self.path/(v+'_metadata.npz'))['public_set_sizes']
 def kernel(self,train,drop=None):
  ix=np.array([self.index[g] for g in train]);assert len(set(ix))==len(ix)
  views=[v for v in VIEWS if v!=drop];K=np.zeros((len(self.targets),len(self.targets)));coverage=np.zeros(len(self.targets),int);transforms={};audit=[]
  for v in views:
   raw=self.raw[v]
   if not sparse.issparse(raw):
    present=self.presence[v];use=ix[present[ix]];z=np.zeros(raw.shape)
    if len(use):
     mean=raw[use].mean(0);scale=np.maximum(raw[use].std(0),1e-6);z[present]=(raw[present]-mean)/scale
    else:mean=np.zeros(raw.shape[1]);scale=np.ones(raw.shape[1])
    transforms[v+'_mean']=mean;transforms[v+'_scale']=scale
    norm=np.linalg.norm(z,axis=1);z=np.divide(z,norm[:,None],out=np.zeros_like(z),where=norm[:,None]>1e-12);kv=z@z.T
   else:
    freq=np.asarray(raw[ix].sum(0)).ravel() if v in ['Reactome','GO_BP','CORUM','CollecTRI'] else self.sizes[v]
    weight=1/np.sqrt(np.maximum(freq,1));transforms[v+'_column_weight']=weight;z=raw.multiply(weight).tocsr();norm=np.sqrt(np.asarray(z.multiply(z).sum(1)).ravel());z=sparse.diags(np.divide(1,norm,out=np.zeros_like(norm),where=norm>1e-12))@z;kv=(z@z.T).toarray()
   covered=norm>1e-12;coverage+=covered;K+=kv/len(views);audit.append(dict(view=v,fit_targets=len(ix),fit_covered=int(covered[ix].sum()),all_covered=int(covered.sum())))
  ev,u=eigh(K[np.ix_(ix,ix)]);assert ev.min()>-1e-8
  order=np.argsort(ev)[::-1];keep=order[ev[order]>max(ev.max()*1e-12,1e-12)][:512];phi=K[:,ix]@(u[:,keep]/np.sqrt(ev[keep]));rem=np.diag(K)-np.sum(phi*phi,axis=1);assert rem.min()>-1e-7;rem=np.maximum(rem,0)
  recovered=phi[ix]@phi[ix].T+np.diag(rem[ix]);np.testing.assert_allclose(recovered,K[np.ix_(ix,ix)],atol=1e-8,rtol=1e-8)
  return dict(ix=ix,phi=phi,rem=rem,coverage=coverage,transforms=transforms,audit=audit,effective_rank=len(keep),training_targets=np.array(train),kernel_fit_sha256=hashlib.sha256(K[np.ix_(ix,ix)].tobytes()).hexdigest())

def fit(bank,rows,config,features):
 targets=[r[0] for r in rows];Y=np.stack([r[1] for r in rows]);degs=[r[2] for r in rows];ix=features['ix'];assert np.array_equal(features['training_targets'],targets)
 if config['response']=='deg_subspace':Y,sub=response_subspace(Y,targets,degs)
 else:sub=dict(kind='identity')
 f=features['phi'][ix];K=f@f.T+np.diag(features['rem'][ix]);h=np.ones((len(ix),1));A=np.block([[K+config['alpha']*np.eye(len(ix)),h],[h.T,np.zeros((1,1))]]);rhs=np.vstack([Y,np.zeros((1,Y.shape[1]))]);solution=solve(A,rhs,assume_a='sym');res=np.linalg.norm(A@solution-rhs)/max(np.linalg.norm(rhs),1e-20);assert res<1e-8
 return dict(coef=solution[:-1],intercept=solution[-1],features=features,fit_residual=float(res),config=config,subspace_rank=int(sub.get('rank',0)),training_mean=np.stack([r[1] for r in rows]).mean(0))

def predict(bank,state,queries,control):
 feat=state['features'];qi=np.array([bank.index[g] for g in queries]);assert not set(qi)&set(feat['ix'])
 cross=feat['phi'][qi]@feat['phi'][feat['ix']].T;delta=cross@state['coef']+state['intercept'];missing=feat['coverage'][qi]==0;delta[missing]=state['training_mean'];out=np.maximum(control+delta,0).astype(np.float32);assert np.isfinite(out).all();return out,missing

def numerical_checks():
 rng=np.random.default_rng(20261105);f=rng.normal(size=(19,7));y=rng.normal(size=(19,11));q=rng.normal(size=(5,7));alpha=1.;k=f@f.T;h=np.ones((19,1));a=np.block([[k+alpha*np.eye(19),h],[h.T,np.zeros((1,1))]]);s=solve(a,np.vstack([y,np.zeros((1,11))]),assume_a='sym');fc=f-f.mean(0);yc=y-y.mean(0);beta=solve(fc.T@fc+alpha*np.eye(7),fc.T@yc,assume_a='pos');np.testing.assert_allclose(q@f.T@s[:-1]+s[-1],(q-f.mean(0))@beta+y.mean(0),atol=1e-10);return 'PASS'
