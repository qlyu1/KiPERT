"""V3 shared single/double research model. Fits use legal training responses only."""
import hashlib
import numpy as np
from pair_research import pairhead

CORE = ('top100_mse','top100_pcc','axis_mse','axis_pcc','axis_spearman','top100_sign','ap100','specific_mrr')
LOWER = {'top100_mse','axis_mse'}
FAMILIES = (
 dict(name='JOINT_SHARED',node_mode='hybrid',lambda_pair=0.,lambda_int=0.,pair_intercept=False,response='full'),
 dict(name='JOINT_CALIBRATED',node_mode='hybrid',lambda_pair=1.,lambda_int=0.,pair_intercept=True,response='full'),
 dict(name='JOINT_TENSOR',node_mode='hybrid',lambda_pair=1.,lambda_int=.5,pair_intercept=True,response='full'),
 dict(name='JOINT_DEG_SUBSPACE',node_mode='hybrid',lambda_pair=1.,lambda_int=.5,pair_intercept=True,response='deg_subspace'),
 dict(name='JOINT_LINEAR_CALIBRATED',node_mode='linear',lambda_pair=1.,lambda_int=0.,pair_intercept=True,response='full'),
)
CONFIGS = tuple(dict(f,alpha=a,id=f["name"]+"_A"+str(int(a))) for f in FAMILIES for a in (1.,10.))

def balanced_folds(conditions,task):
    cond=[pairhead.canon(c) for c in conditions]
    if len(set(cond))!=len(cond): raise ValueError('Duplicate condition')
    expected=1 if task=='single' else 2
    if any(len(pairhead.parts(c))!=expected for c in cond):raise ValueError('Wrong validation task')
    if task=='single':
        if len(cond)<10: raise ValueError('Unified protocol requires at least 10 outer-fit singles')
        nfold=5 if len(cond)<12 else 3
    else:
        if len(cond)<3:raise ValueError('At least three legal pairs needed for edge CV')
        nfold=3
    order=sorted(range(len(cond)),key=lambda i:hashlib.blake2b((cond[i]+'|UCV3_FROZEN').encode(),digest_size=16).hexdigest())
    out=np.empty(len(cond),int)
    for j,i in enumerate(order):out[i]=j%nfold
    return out

def training_weights(conditions):
    arity=np.array([len(pairhead.parts(c)) for c in conditions])
    w=np.ones(len(arity))
    for k in np.unique(arity):w[arity==k]=len(arity)/(len(np.unique(arity))*sum(arity==k))
    return arity,w

def response_subspace(y,conditions,deg_indices):
    """A DEG-weighted low-rank response component plus a 25% full-rank tail.
    Preserve task intercepts and 25% of the orthogonal residual. All weights and
    directions derive from current fit rows. Full-rank column rescaling is avoided.
    """
    y=np.asarray(y,float);n,g=y.shape
    if len(deg_indices)!=n:raise ValueError('DEG list length mismatch')
    arity,w=training_weights(conditions)
    h=np.column_stack([arity,arity==2]).astype(float) if np.any(arity==2) else arity[:,None].astype(float)
    intercept=np.linalg.solve(h.T@(w[:,None]*h),h.T@(w[:,None]*y))
    residual=y-h@intercept
    frequency=np.zeros(g)
    for i,ix in enumerate(deg_indices):
        ix=np.asarray(ix,int)
        if ix.ndim!=1 or len(set(ix.tolist()))!=len(ix) or np.any(ix<0) or np.any(ix>=g) or len(ix)>100:
            raise ValueError('DEG indices must be unique valid original-Top100 intersections')
        frequency[ix]+=w[i]
    frequency/=w.sum()
    if frequency.max()<=0:raise ValueError('No legal training DEG intersection')
    gene_weight=1.+4.*frequency/frequency.max()
    root_weight=np.sqrt(gene_weight)
    a=np.sqrt(w)[:,None]*residual*root_weight[None,:]
    eigen,u=np.linalg.eigh(a@a.T)
    order=np.argsort(-eigen);eigen=eigen[order];u=u[:,order]
    active=np.flatnonzero(eigen>max(float(eigen[0]),1.)*1e-12)
    rank=min(16,max(0,(n-h.shape[1])//2),len(active))
    if rank==0: basis=np.empty((0,g))
    else: basis=(u[:,:rank].T@a)/np.sqrt(eigen[:rank,None])
    projected=((residual*root_weight)@basis.T)@basis/root_weight
    target=h@intercept+.75*projected+.25*residual
    return target,dict(rank=rank,tail_fraction=.25,gene_weight=gene_weight,basis=basis,
        training_frequency=frequency,arity_intercept=intercept,
        original_response_sha256=pairhead.array_hash(y),
        transformed_response_sha256=pairhead.array_hash(target),
        weighted_residual_energy_retained=float(eigen[:rank].sum()/max(eigen.clip(0).sum(),1e-12)),
        interpretation='DEG-weighted reduced response space with 25% residual tail; not full-rank weighted ridge and not a learned DEG oracle.')

def fit(bank,conditions,delta,control_mean,deg_indices,config,forbidden_genes=()):
    config=dict(config);conditions=[pairhead.canon(c) for c in conditions]
    y=np.asarray(delta,float);ctrl=np.asarray(control_mean,float)
    if ctrl.shape!=(y.shape[1],) or not np.isfinite(ctrl).all():raise ValueError('Invalid control axis')
    if config['response']=='deg_subspace':target,response=response_subspace(y,conditions,deg_indices)
    elif config['response']=='full':target=y;response=dict(kind='identity')
    else:raise ValueError('Unknown response structure')
    state=pairhead.fit_joint(bank,conditions,target,alpha=config['alpha'],
        lambda_pair=config['lambda_pair'],lambda_int=config['lambda_int'],
        node_mode=config['node_mode'],pair_intercept=config['pair_intercept'],
        balance_arities=True,forbidden_genes=forbidden_genes)
    return dict(format='UC_V3_MEAN_CANDIDATE_1',config=config,core=state,
        control_mean=ctrl,response_transform=response,train_conditions=conditions,
        response_sha256=pairhead.array_hash(y))
def predict(state,conditions,nonnegative=True,return_parts=False):
    result=pairhead.predict_joint(state['core'],conditions,return_parts=return_parts)
    if return_parts:raw,parts=result
    else:raw=result
    out=np.maximum(raw+state['control_mean'],0.)-state['control_mean'] if nonnegative else raw
    if return_parts:
        parts=dict(parts,nonnegative_projection=out-raw)
        return out,parts
    return out

def correlation(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    a=a-a.mean();b=b-b.mean()
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom>1e-15 else 0.
def ranks(a):
    order=np.argsort(a,kind='stable');s=np.asarray(a)[order];r=np.empty(len(a),float)
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];ends=np.r_[starts[1:],len(a)]
    for lo,hi in zip(starts,ends):r[order[lo:hi]]=(lo+hi-1)/2.+1
    return r
def average_precision(labels,scores):
    labels=np.asarray(labels,bool);scores=np.asarray(scores,float)
    if not labels.any():return np.nan
    order=np.argsort(-scores,kind='stable');s=scores[order];l=labels[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1])+1,len(s)]
    hits=np.cumsum(l)[ends-1];recall=hits/labels.sum();precision=hits/ends
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))
def metric_values(pred,truth,deg_indices,common,compute_retrieval=True):
    p=np.asarray(pred,float);y=np.asarray(truth,float);mu=np.asarray(common,float)
    if p.shape!=y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Invalid evaluation arrays')
    if mu.ndim==1:mu=np.broadcast_to(mu,p.shape)
    if mu.shape!=p.shape or len(deg_indices)!=len(p):raise ValueError('Axis mismatch')
    rows=[]
    for a,b,ix in zip(p,y,deg_indices):
        ix=np.asarray(ix,int);ap=np.nan;tm=np.nan;tc=np.nan;sg=np.nan
        if len(ix):
            if np.any(ix<0) or np.any(ix>=p.shape[1]) or len(set(ix.tolist()))!=len(ix):raise ValueError('Invalid DEG axis')
            tm=float(np.mean((a[ix]-b[ix])**2));tc=correlation(a[ix],b[ix])
            valid=abs(b[ix])>1e-8
            if valid.any():sg=float(np.mean((np.where(abs(a[ix])>1e-8,np.sign(a[ix]),0.)==np.sign(b[ix]))[valid]))
            label=np.zeros(p.shape[1],bool);label[ix]=True;ap=average_precision(label,abs(a))
        rows.append(dict(top100_mse=tm,top100_pcc=tc,axis_mse=float(np.mean((a-b)**2)),
            axis_pcc=correlation(a,b),axis_spearman=correlation(ranks(a),ranks(b)),top100_sign=sg,ap100=ap))
    for row in rows:row['specific_mrr']=np.nan
    if len(p)>=2 and compute_retrieval:
        a=p-mu;b=y-mu;an=np.linalg.norm(a,axis=1);bn=np.linalg.norm(b,axis=1)
        den=an[:,None]*bn[None,:];sim=np.divide(a@b.T,den,out=np.zeros((len(p),len(p))),where=den>1e-15)
        for i,row in enumerate(rows):
            score=sim[i] if an[i]>1e-15 and bn[i]>1e-15 else np.zeros(len(p))
            v=score[i];better=int(sum(score>v+1e-12));ties=int(sum(abs(score-v)<=1e-12))
            row['specific_mrr']=float(np.mean(1./np.arange(better+1,better+ties+1)))
    macro={k:float(np.mean([r[k] for r in rows if np.isfinite(r[k])])) if any(np.isfinite(r[k]) for r in rows) else np.nan for k in CORE}
    return macro,rows
