from models import Data,mix
from common import *
from scipy.spatial.distance import cdist

OLD=ROOT.parent/'uc_state_official_20261101'

class Dataset(Data):
 def __init__(self,fold):
  super().__init__(fold)
  z=np.load(P/'shared_gene_basis_12.npz');np.testing.assert_array_equal(z['targets'],self.features[8]['targets']);self.features[12]={k:z[k] for k in ['targets','phi','diagonal_remainder']}
  self.common_response={s:np.asarray(self.y[s],float).mean(0) for s in self.sources}

def recipes(legacy_gate):
 gates=[dict(temperature=None,measured_only=True,precision=False),dict(temperature=1.,measured_only=True,precision=False),legacy_gate]
 return [dict(rho=r,common_scale=c,specific_scale=g,gate=gate,gate_index=i,alpha_common=0.,alpha_specific=0.) for r,c,g,i in itertools.product([0.,.5,1.],[.5,1.],[.5,.75,1.,1.5,2.],range(3)) for gate in [gates[i]]]

def panel_for(target,queries):
 rng=np.random.default_rng(seed_for('specificity_panel_'+target));n=min(256,len(queries));indices=rng.choice(len(queries),n,replace=False);nquery=min(128,n)
 return dict(indices=indices,nquery=nquery,candidates=np.asarray(queries)[indices].tolist(),queries=np.asarray(queries)[indices[:nquery]].tolist())

def euclidean_matrix(a,b):
 return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0.)

def rank_scores(a,b,correct):
 out={}
 for name,dist in [('l1',cdist(a,b,'cityblock')),('l2',euclidean_matrix(a,b))]:
  ranks=np.argmax(np.argsort(dist,axis=1)==np.asarray(correct)[:,None],axis=1);out[name]=float(np.mean(1-ranks/len(b)))
 return out

def score_source(delta,truth,control,panel):
 delta=np.clip(delta+control,0,14)-control;nmse=float(np.mean((delta-truth)**2)/max(float(np.mean(truth*truth)),1e-20));ix=panel['indices'];n=panel['nquery'];scores=rank_scores(delta[ix[:n]],truth[ix],np.arange(n));return dict(nmse=nmse,rank_l1=scores['l1'],rank_l2=scores['l2'],rank_score=(scores['l1']+scores['l2'])/2)

def raw_components(d,pool,target,queries,prior):
 q=np.asarray(queries);pred=[];innovation=[];variance=[];known=[];common=[];mapped_pred=[];mapped_innovation=[];mapped_common=[];audit=[]
 for s in pool:
  mu,v,k=d.predict(prior,s,q);rr=np.array([d.lookup[s].get(g,-1) for g in q]);inn=np.zeros_like(mu);inn[k]=np.asarray(d.y[s][rr[k]],float)-mu[k];c=d.common_response[s]
  pred.append(mu);innovation.append(inn);variance.append(v);known.append(k);common.append(c);mapped_pred.append(d.correction(s,target,mu));mapped_innovation.append(d.correction(s,target,inn));mapped_common.append(d.correction(s,target,c[None])[0])
  ex=d.fit(prior,s);a=ex['tau']/ex['D'];audit.append(dict(source=s,n_observed=int(k.sum()),posterior_residual_retention_median=float(np.median(a)),posterior_residual_retention_q10=float(np.quantile(a,.1)),posterior_residual_retention_q90=float(np.quantile(a,.9)),observed_source_centered_energy=float(np.var(d.y[s],axis=0).mean()),gp_centered_energy=float(np.var(mu,axis=0).mean())))
 return dict(pool=pool,target=target,pred=np.stack(pred),innovation=np.stack(innovation),variance=np.stack(variance),known=np.stack(known),common=np.stack(common),mapped_pred=np.stack(mapped_pred),mapped_innovation=np.stack(mapped_innovation),mapped_common=np.stack(mapped_common),audit=audit)

def weighted_components(d,raw,recipe):
 w=d.gate(raw['pool'],raw['target'],raw['variance'],raw['known'],recipe['gate']);c=np.einsum('sg,sq->qg',raw['common'],w,optimize=True);dc=np.einsum('sg,sq->qg',raw['mapped_common'],w,optimize=True);p=mix(raw['pred']+recipe['rho']*raw['innovation'],w);dp=mix(raw['mapped_pred']+recipe['rho']*raw['mapped_innovation'],w)
 return dict(common=c,specific=p-c,mapped_common=dc,mapped_specific=dp-dc,unmeasured_weight_mean=float(np.sum(w*(~raw['known']),0).mean()))

def assemble(comp,recipe):
 return recipe['common_scale']*(comp['common']+recipe['alpha_common']*comp['mapped_common'])+recipe['specific_scale']*(comp['specific']+recipe['alpha_specific']*comp['mapped_specific'])

def aggregate(cases):
 return {k:float(np.mean([r[k] for r in cases])) for k in ['nmse','rank_l1','rank_l2','rank_score']}

def choose(records,point,cap):
 eligible=records if point=='specificity' else [r for r in records if r['aggregate']['nmse']<=cap]
 if eligible:
  selected=min(eligible,key=lambda r:(-r['aggregate']['rank_score'],r['aggregate']['nmse'],r['recipe']['rho'],r['id']));feasible=True
 else:selected=min(records,key=lambda r:(r['aggregate']['nmse'],-r['aggregate']['rank_score'],r['id']));feasible=False
 return dict(**selected,operating_point=point,nmse_cap=cap,constraint_feasible=feasible)
