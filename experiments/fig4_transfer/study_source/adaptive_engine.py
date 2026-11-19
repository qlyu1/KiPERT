from grid_engine import *
from scipy.special import logsumexp

def cid(impute,precision,shrink):return 'I%d_V%d_C%d'%(impute,precision,shrink>0)
def cfgs():return [dict(id='baseline',impute=True,precision=True,shrinkage=.1),dict(id='no_imputation',impute=False,precision=True,shrinkage=.1),dict(id='no_variance_weight',impute=True,precision=False,shrinkage=.1),dict(id='no_shrinkage',impute=True,precision=True,shrinkage=0.)]
BASE='baseline';MAIN='adaptive_unclipped'

def stable_gate(d,pool,target,variance,known,temperature,impute,precision):
 ell,dist=d.geometry(pool,target);logg=np.zeros(len(pool)) if temperature is None else -(dist-dist.min())/(temperature*ell)
 eligible=np.ones_like(known,bool) if impute else known.copy();supported=eligible.any(0);logw=np.broadcast_to(logg[:,None],variance.shape).copy()
 for i,s in enumerate(pool):
  count=eligible[[j for j,t in enumerate(pool) if d.bio[t]==d.bio[s]]].sum(0);logw[i]-=np.log(np.maximum(count,1))
 if precision:logw-=np.log(variance+.05)
 logw[~eligible]=-np.inf;w=np.zeros_like(logw)
 w[:,supported]=np.exp(logw[:,supported]-logsumexp(logw[:,supported],axis=0)[None,:])
 np.testing.assert_allclose(w[:,supported].sum(0),1,atol=1e-12);assert not w[~eligible].any();assert not w[:,~supported].any()
 return w

class Maps:
 def __init__(self,d):self.d=d;self.exact=ExactMaps(d,[0,8,16,32,64,128,256]);self.audit={};self.zero={}
 def pair(self,source,target,shrink):
  pair=self.exact.pair(source,target);a,b=sorted([source,target]);key=a+'__'+b;direction='ab' if source==a else 'ba'
  if shrink==.1:return pair['U'][:,:128],pair['maps'][128][direction]
  assert shrink==0
  if key not in self.zero:
   p=ROOT/'cache/no_shrink_maps'/key;p.mkdir(parents=True,exist_ok=True)
   if not (p/'READY.json').exists():
    U=pair['U'][:,:128];xa=np.array(self.d.ctrl[a],dtype=float,copy=True);xb=np.array(self.d.ctrl[b],dtype=float,copy=True);xa-=xa.mean(0);xb-=xb.mean(0);xa=xa@U;xb=xb@U;fa=fit_gaussian_bures(xa,xb,shrinkage=0.);fb=fit_gaussian_bures(xb,xa,shrinkage=0.);err=max(fa['alignment_error_relative'],fb['alignment_error_relative']);assert err<1e-8
    ev_a=np.linalg.eigvalsh(fa['raw_cov_X']);ev_b=np.linalg.eigvalsh(fa['raw_cov_Y']);np.savez_compressed(p/'map.npz',ab=fa['A'],ba=fb['A'])
    save(p/'READY.json',dict(pair=key,shrinkage=0.,basis_sha256=sha(Path(pair['path'])/'basis.npz'),controls_sha256=self.exact.audit[key]['control_sha256'],covariance_alignment_error=err,floor_activated_a=int((ev_a<fa['source_covariance_floor']).sum()),floor_activated_b=int((ev_b<fa['target_covariance_floor']).sum()),min_eigenvalue_a=float(ev_a[0]),min_eigenvalue_b=float(ev_b[0])))
   meta=json.load(open(p/'READY.json'));assert meta['basis_sha256']==sha(Path(pair['path'])/'basis.npz');assert meta['controls_sha256']==self.exact.audit[key]['control_sha256'];self.audit[key]=meta;z=np.load(p/'map.npz');self.zero[key]={k:z[k] for k in ['ab','ba']}
  return pair['U'][:,:128],self.zero[key][direction]
 def responses(self,pool,target,parts,shrink):
  mapped=[]
  for s in pool:
   U,A=self.pair(s,target,shrink);mu=parts[s]['mu'];mapped.append(mu+(mu@U@(A-np.eye(128)))@U.T)
  return np.stack(mapped)

def source_score(delta,truth,panel):
 nmse=float(np.mean((delta-truth)**2)/max(float(np.mean(truth*truth)),1e-20));ix=panel['indices'];n=panel['nquery'];score=rank_scores(delta[ix[:n]],truth[ix],np.arange(n));return dict(nmse=nmse,rank_l1=score['l1'],rank_l2=score['l2'],rank_score=(score['l1']+score['l2'])/2)

def validation_weights(d):
 ell,dist=d.geometry(d.sources,d.fold);z=-(dist-dist.min())/ell;q=np.exp(z-logsumexp(z))
 return dict(sources=d.sources,distances=dist.tolist(),distance_scale=ell,weights={s:float(q[i]) for i,s in enumerate(d.sources)},effective_contexts=float(1/(q@q)),outer_target_responses_used=False)

def aggregate_weighted(cases,weights):
 keys=['nmse','rank_l1','rank_l2','rank_score'];return {k:float(sum(weights[c['target']]*c[k] for c in cases)) for k in keys}

def choose(records,cap):
 eligible=[r for r in records if r['weighted']['nmse']<=cap]
 def key(r):return (-r['weighted']['rank_score'],r['weighted']['nmse'],-(1e6 if r['temperature'] is None else r['temperature']),r['id'])
 win=min(eligible,key=key) if eligible else min(records,key=lambda r:(r['weighted']['nmse'],-r['weighted']['rank_score'],r['id']))
 return dict(**win,constraint_feasible=bool(eligible),nmse_cap=cap)


def assemble_prediction(d,maps,parts,pool,target,config):
 var=np.stack([parts[s]['var'] for s in pool]);known=np.stack([parts[s]['known'] for s in pool]);w=stable_gate(d,pool,target,var,known,config['temperature'],config['impute'],config['precision']);mapped=maps.responses(pool,target,parts,config['shrinkage']);mean=d.meta[target]['control']+mix(mapped,w)
 if config['output_clip'] is not None:mean=np.clip(mean,*config['output_clip'])
 return mean,w,known,var
