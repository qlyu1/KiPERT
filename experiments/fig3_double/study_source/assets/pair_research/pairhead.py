"""Pair-only correction heads with one shared first-order single/double interface.
New research candidates, not a relabeling of the original PD method.
Only NumPy is needed. No response files, query truth, or excluded models imported.
"""
import hashlib
import numpy as np

CONFIGS = (
 dict(name='PAIR_MV_DIRECT_SUM', target='direct', tensor_weight=0., alpha=10.),
 dict(name='PAIR_MV_RESIDUAL_SUM', target='residual', tensor_weight=0., alpha=10.),
 dict(name='PAIR_MV_RESIDUAL_SUM_TENSOR', target='residual', tensor_weight=.5, alpha=10.),
)
VIEWS = (('xdepmap','depmap_present',.5),('xstring','string_present',.25),('xgenept','genept_present',.25))
def parts(c):
 p=tuple(sorted(x.strip() for x in str(c).split('+') if x.strip() not in ('','ctrl','control')))
 if len(p) not in (1,2) or len(set(p))!=len(p):raise ValueError('Expected one or two distinct gene components')
 return p
def canon(c):return '+'.join(parts(c))
def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def indices(components,conditions):
 index={g:i for i,g in enumerate(components)}
 if len(index)!=len(components):raise ValueError('Duplicate feature-bank component')
 try:return [tuple(index[g] for g in parts(c)) for c in conditions]
 except KeyError as exc:raise ValueError('Component missing from explicit feature bank: '+str(exc)) from exc
def component_sum(components,conditions,prediction):
 pairs=indices(components,conditions);p=np.asarray(prediction,float)
 if p.ndim!=2 or p.shape[0]!=len(components) or not np.isfinite(p).all():raise ValueError('Bad component prediction matrix')
 return np.stack([p[list(ix)].sum(0) for ix in pairs])
def fit_node_map(bank,train_pair_conditions):
 components=np.asarray(bank['components']).astype(str)
 pp=indices(components,train_pair_conditions)
 fit=np.asarray(sorted({g for ix in pp for g in ix}),int)
 features=[];receipts=[];weight_sum=0.
 for key,maskkey,weight in VIEWS:
  if key not in bank:continue
  raw=np.asarray(bank[key],float)
  present=np.asarray(bank.get(maskkey,np.ones(len(components),bool)),bool)
  if raw.ndim!=2 or raw.shape[0]!=len(components) or present.shape!=(len(components),):raise ValueError(key+' malformed')
  if not np.isfinite(raw[present]).all():raise ValueError(key+' present values must be finite')
  good=fit[present[fit]]
  if len(good)==0:
   receipts.append(dict(view=key,enabled=False,reason='no covered training component'));continue
  mean=raw[good].mean(0);scale=raw[good].std(0);scale=np.where(scale>1e-8,scale,1.)
  z=np.zeros(raw.shape,float);z[present]=(raw[present]-mean)/scale
  norm=np.linalg.norm(z,axis=1,keepdims=True);z=np.divide(z,norm,out=np.zeros_like(z),where=norm>1e-12)
  # Mask is explicit, has fixed small amplitude, and never substitutes an invented gene embedding.
  z=np.column_stack([z,.1*(~present).astype(float)])
  features.append(np.sqrt(weight)*z);weight_sum+=weight
  receipts.append(dict(view=key,enabled=True,weight=weight,train_component_indices=good,
                       mean=mean,scale=scale,present=present))
 if not features:raise ValueError('No view with covered legal training components')
 features=np.concatenate(features,axis=1)/np.sqrt(weight_sum)
 return components,features,receipts,fit

def pair_kernel(node,query_pairs,train_pairs,kind):
 qa=np.asarray([x[0] for x in query_pairs],int);qb=np.asarray([x[1] for x in query_pairs],int)
 ta=np.asarray([x[0] for x in train_pairs],int);tb=np.asarray([x[1] for x in train_pairs],int)
 ac=node[np.ix_(qa,ta)];ad=node[np.ix_(qa,tb)]
 bc=node[np.ix_(qb,ta)];bd=node[np.ix_(qb,tb)]
 if kind=='sum':return ac+ad+bc+bd
 if kind=='tensor':return .5*(ac*bd+ad*bc)
 raise ValueError(kind)
def centered_train(k):
 col=k.mean(0);grand=float(k.mean());kc=k-col[None,:]-col[:,None]+grand
 scale=max(float(np.trace(kc)/len(kc)),1e-12)
 return kc/scale,dict(train_column_mean=col,grand_mean=grand,scale=scale)
def centered_query(k,normalizer):
 return (k-k.mean(1,keepdims=True)-normalizer['train_column_mean'][None,:]+normalizer['grand_mean'])/normalizer['scale']

def fit(bank,pair_conditions,pair_delta,component_prediction,config=None,measured_single_genes=(),forbidden_genes=()):
 config=dict(CONFIGS[2] if config is None else config)
 if config['target'] not in ('direct','residual') or not 0<=config['tensor_weight']<=1 or config['alpha']<=0:raise ValueError('Invalid config')
 conditions=[canon(c) for c in pair_conditions]
 if len(set(conditions))!=len(conditions) or any(len(parts(c))!=2 for c in conditions):raise ValueError('Unique double training conditions required')
 if set(forbidden_genes)&{g for c in conditions for g in parts(c)}:raise ValueError('Held gene appears in fitted double')
 components=np.asarray(bank['components']).astype(str);pred=np.asarray(component_prediction,float);y=np.asarray(pair_delta,float)
 if y.ndim!=2 or y.shape[0]!=len(conditions) or pred.shape!=(len(components),y.shape[1]) or not np.isfinite(y).all() or not np.isfinite(pred).all():raise ValueError('Bad training arrays')
 measured=set(measured_single_genes)
 if measured&set(forbidden_genes):raise ValueError('Held gene in measured-single set')
 seen=[sum(g in measured for g in parts(c)) for c in conditions]
 state=dict(config=config,components=components,train_pair_conditions=conditions,
            measured_single_genes=sorted(measured),train_seen_single_count=seen,n_genes=y.shape[1],
            base_component_sha256=array_hash(pred),response_sha256=array_hash(y),
            residual_interpretation='Pure measured GI target only for Seen2 when caller component predictions equal the measured single deltas; otherwise total correction includes completion error.',
            enabled=len(conditions)>=2)
 if len(conditions)<2:return state
 components,features,maps,feature_fit=fit_node_map(bank,conditions);node=features@features.T
 pp=indices(components,conditions);normalizers={};ks={}
 for kind in ['sum','tensor']:
  ks[kind],normalizers[kind]=centered_train(pair_kernel(node,pp,pp,kind))
 w=config['tensor_weight'];kernel=(1-w)*ks['sum']+w*ks['tensor']
 target=y-component_sum(components,conditions,pred) if config['target']=='residual' else y
 intercept=target.mean(0);coef=np.linalg.solve(kernel+config['alpha']*np.eye(len(y)),target-intercept)
 state.update(node_features=features,feature_maps=maps,feature_fit_component_indices=feature_fit,
              train_pair_indices=pp,normalizers=normalizers,intercept=intercept,coef=coef)
 return state
def predict(state,conditions,component_prediction,verify_same_base=True):
 conditions=[canon(c) for c in conditions];pred=np.asarray(component_prediction,float)
 if verify_same_base and array_hash(pred)!=state['base_component_sha256']:raise ValueError('Shared first-order component predictions changed since pair fitting')
 out=component_sum(state['components'],conditions,pred)
 where=[i for i,c in enumerate(conditions) if len(parts(c))==2]
 if state['enabled'] and where:
  qq=indices(state['components'],[conditions[i] for i in where]);node=state['node_features']@state['node_features'].T
  ks={kind:centered_query(pair_kernel(node,qq,state['train_pair_indices'],kind),state['normalizers'][kind]) for kind in ['sum','tensor']}
  w=state['config']['tensor_weight'];yp=((1-w)*ks['sum']+w*ks['tensor'])@state['coef']+state['intercept']
  if state['config']['target']=='direct':out[where]=yp
  else:out[where]+=yp
 if not np.isfinite(out).all():raise AssertionError('Nonfinite prediction')
 measured=set(state['measured_single_genes'])
 return out,[dict(condition=c,cardinality=len(parts(c)),seen_single_count=sum(g in measured for g in parts(c)),
                  pair_head_used=bool(state['enabled'] and len(parts(c))==2),target_type=state['config']['target']) for c in conditions]
def gene_fold_partition(train_conditions,held_genes):
 held=set(held_genes);fit=[];valid=[]
 for c in train_conditions:
  (valid if set(parts(c))&held else fit).append(canon(c))
 return fit,valid

def string_rbf_kernel(bank,components,fit_indices):
 """STRING RBF with fit-only standardization and bandwidth; missing rows/columns are zero."""
 n=len(components);out=np.zeros((n,n),float)
 if 'xstring' not in bank:return out,dict(enabled=False,reason='STRING view absent')
 raw=np.asarray(bank['xstring'],float)
 present=np.asarray(bank.get('string_present',np.ones(n,bool)),bool)
 if raw.ndim!=2 or raw.shape[0]!=n or present.shape!=(n,):raise ValueError('Malformed STRING view')
 if not np.isfinite(raw[present]).all():raise ValueError('Present STRING descriptors must be finite')
 fit=np.asarray(fit_indices,int);good=fit[present[fit]]
 if len(good)==0:return out,dict(enabled=False,reason='No present legal training STRING component',present=present,fit_component_indices=good)
 mean=raw[good].mean(0);scale=raw[good].std(0);scale=np.where(scale>1e-8,scale,1.)
 valid=np.flatnonzero(present);z=(raw[valid]-mean)/scale
 zfit=(raw[good]-mean)/scale
 fit_d2=np.maximum(np.sum(zfit*zfit,axis=1)[:,None]+np.sum(zfit*zfit,axis=1)[None,:]-2*zfit@zfit.T,0.)
 upper=fit_d2[np.triu_indices(len(good),1)];positive=upper[upper>1e-12]
 bandwidth_squared=float(np.median(positive)) if len(positive) else 1.
 d2=np.maximum(np.sum(z*z,axis=1)[:,None]+np.sum(z*z,axis=1)[None,:]-2*z@z.T,0.)
 # Compute only present-present entries: a missing zero row never masquerades as a real RBF vector.
 out[np.ix_(valid,valid)]=np.exp(-d2/(2.*bandwidth_squared))
 return out,dict(enabled=True,mean=mean,scale=scale,present=present,fit_component_indices=good,
  bandwidth_squared=bandwidth_squared,bandwidth_rule='median positive off-diagonal squared Euclidean distances among standardized present legal fit components; fallback1 when no positive distances',
  formula='exp(-squared_distance/(2*bandwidth_squared)); zero for either missing component')
def node_kernel(bank,fit_conditions,forbidden_genes=(),node_mode='linear'):
 """Return a PSD gene kernel with preprocessing fitted on legal training components only.
 hybrid is exactly 0.5 existing multiview linear kernel + 0.5 masked STRING RBF.
 """
 if node_mode not in ('linear','hybrid'):raise ValueError('node_mode must be linear or hybrid')
 used={g for c in fit_conditions for g in parts(c)}
 if used&set(forbidden_genes):raise ValueError('Forbidden held gene in node-kernel fitting components')
 components,features,maps,fit_indices=fit_node_map(bank,fit_conditions)
 linear=features@features.T;rbf_state=None
 if node_mode=='hybrid':
  rbf,rbf_state=string_rbf_kernel(bank,components,fit_indices);kernel=.5*linear+.5*rbf
 else:kernel=linear
 return dict(components=components,kernel=kernel,features=features,maps=maps,
             fit_component_indices=fit_indices,node_mode=node_mode,string_rbf_state=rbf_state)
def condition_kernel(node,components,query_conditions,train_conditions,lambda_pair=1.,lambda_int=.5,return_parts=False):
 if lambda_pair<0 or lambda_int<0:raise ValueError('Kernel weights must be nonnegative')
 qq=indices(components,query_conditions);tt=indices(components,train_conditions)
 bq=np.zeros((len(qq),len(components)));bt=np.zeros((len(tt),len(components)))
 for i,ix in enumerate(qq):bq[i,list(ix)]=1.
 for i,ix in enumerate(tt):bt[i,list(ix)]=1.
 shared=bq@node@bt.T
 qp=np.asarray([len(ix)==2 for ix in qq]);tp=np.asarray([len(ix)==2 for ix in tt])
 pair_sum=np.zeros(shared.shape);interaction=np.zeros(shared.shape)
 if qp.any() and tp.any():
  pair_sum[np.ix_(qp,tp)]=shared[np.ix_(qp,tp)]
  interaction[np.ix_(qp,tp)]=pair_kernel(node,[qq[i] for i in np.flatnonzero(qp)],[tt[i] for i in np.flatnonzero(tp)],'tensor')
 total=shared+lambda_pair*pair_sum+lambda_int*interaction
 return (total,shared,lambda_pair*pair_sum,lambda_int*interaction) if return_parts else total
def fit_joint(bank,train_conditions,delta,alpha=10.,lambda_pair=1.,lambda_int=.5,balance_arities=True,forbidden_genes=(),node_mode='linear',pair_intercept=True):
 """One simultaneous fit: legal double labels can update the shared single effect.
 Objective: sum_i weight_i ||delta_i-f(C_i)-H_i b||^2 + alpha*RKHS_norm^2.
 Intercepts H=[cardinality,pair indicator] allow a shared gene offset and a pair-only offset.
 With singles-only data pair-only terms vanish and inference is additive with 2*single offset.
 """
 conditions=[canon(c) for c in train_conditions]
 if len(set(conditions))!=len(conditions):raise ValueError('Duplicate training condition')
 if alpha<=0:raise ValueError('Positive ridge alpha required')
 y=np.asarray(delta,float)
 if y.ndim!=2 or len(y)!=len(conditions) or not np.isfinite(y).all():raise ValueError('Bad response array')
 arity=np.asarray([len(parts(c)) for c in conditions])
 if sum(arity==1)<2:raise ValueError('Joint fit requires at least two legal single conditions')
 nk=node_kernel(bank,conditions,forbidden_genes,node_mode=node_mode)
 kernel=condition_kernel(nk['kernel'],nk['components'],conditions,conditions,lambda_pair,lambda_int)
 pair_present=bool(pair_intercept and np.any(arity==2))
 h=np.column_stack([arity,arity==2]).astype(float) if pair_present else arity[:,None].astype(float)
 weight=np.ones(len(y))
 if balance_arities:
  for a in np.unique(arity):weight[arity==a]=len(y)/(len(np.unique(arity))*sum(arity==a))
 a=kernel+alpha*np.diag(1./weight)
 # Block solve enforces H.T@coef=0 for exactly unpenalized arity intercepts.
 block=np.block([[a,h],[h.T,np.zeros((h.shape[1],h.shape[1]))]])
 solution=np.linalg.solve(block,np.vstack([y,np.zeros((h.shape[1],y.shape[1]))]))
 return dict(kind='JOINT_PD_CALIBRATION_TENSOR',components=nk['components'],node_features=nk['features'],node_kernel=nk['kernel'],string_rbf_state=nk['string_rbf_state'],
  feature_maps=nk['maps'],feature_fit_component_indices=nk['fit_component_indices'],train_conditions=conditions,
  coef=solution[:len(y)],intercepts=solution[len(y):],pair_intercept_fitted=pair_present,
  config=dict(alpha=float(alpha),lambda_pair=float(lambda_pair),lambda_int=float(lambda_int),balance_arities=bool(balance_arities),node_mode=node_mode,pair_intercept=bool(pair_intercept)),
  training_weights=weight,response_sha256=array_hash(y),n_genes=y.shape[1],
  interpretation='Shared first order plus pair-only affine calibration plus symmetric tensor correction; the RKHS decomposition is regularized, not a uniquely identifiable causal GI decomposition.')
def predict_joint(state,conditions,return_parts=False):
 conditions=[canon(c) for c in conditions];c=state['config']
 if 'node_kernel' in state:node=state['node_kernel']
 elif c.get('node_mode','linear')=='linear':node=state['node_features']@state['node_features'].T
 else:raise ValueError('Hybrid checkpoint missing its actual node kernel')
 _,shared,calibration,interaction=condition_kernel(node,state['components'],conditions,state['train_conditions'],
  c['lambda_pair'],c['lambda_int'],return_parts=True)
 arity=np.asarray([len(parts(x)) for x in conditions])
 f=shared@state['coef']+arity[:,None]*state['intercepts'][0]
 p=calibration@state['coef']
 if state['pair_intercept_fitted']:p+=(arity==2)[:,None]*state['intercepts'][1]
 g=interaction@state['coef'];out=f+p+g
 if not np.isfinite(out).all():raise AssertionError('Nonfinite joint prediction')
 return (out,dict(shared=f,pair_calibration=p,symmetric_interaction=g)) if return_parts else out
