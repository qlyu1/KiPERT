import numpy as np
from scipy import sparse

def view_kernels(bank,train):
 ix=np.array([bank.index[g] for g in train]);kernels=[];coverage=np.zeros(len(bank.targets),int)
 for v,raw in bank.raw.items():
  if not sparse.issparse(raw):
   present=bank.presence[v];use=ix[present[ix]];z=np.zeros(raw.shape)
   if len(use):z[present]=(raw[present]-raw[use].mean(0))/np.maximum(raw[use].std(0),1e-6)
   norm=np.linalg.norm(z,axis=1);z=np.divide(z,norm[:,None],out=np.zeros_like(z),where=norm[:,None]>1e-12);kv=z@z.T
  else:
   freq=np.asarray(raw[ix].sum(0)).ravel() if v in ['Reactome','GO_BP','CORUM','CollecTRI'] else bank.sizes[v]
   z=raw.multiply(1/np.sqrt(np.maximum(freq,1))).tocsr();norm=np.sqrt(np.asarray(z.multiply(z).sum(1)).ravel());z=sparse.diags(np.divide(1,norm,out=np.zeros_like(norm),where=norm>1e-12))@z;kv=(z@z.T).toarray()
  kernels.append(kv);coverage+=norm>1e-12
 return np.stack(kernels),ix,coverage,list(bank.raw)

def center(k):return k-k.mean(0)[None,:]-k.mean(1)[:,None]+k.mean()

def build(bank,rows,queries,keep):
 train=[r[0] for r in rows];assert not set(train)&set(queries)
 views,ix,coverage,names=view_kernels(bank,train);qi=np.array([bank.index[g] for g in queries]);Y=np.stack([r[1] for r in rows]);b=Y.mean(0);R=Y-b
 z=R[:,keep];norm=np.linalg.norm(z,axis=1,keepdims=True);unit=np.divide(z,norm,out=np.zeros_like(z),where=norm>1e-12);target=center(unit@unit.T);mask=~np.eye(len(rows),dtype=bool);tv=target[mask];alignment=[]
 for kv in views:
  a=center(kv[np.ix_(ix,ix)])[mask];den=np.linalg.norm(a)*np.linalg.norm(tv);alignment.append(max(0,float(a@tv/den)) if den>1e-12 else 0.)
 align=np.array(alignment);weights=.5*np.ones(12)/12+.5*(align/align.sum() if align.sum()>1e-12 else np.ones(12)/12)
 kernels={}
 for name,w in [('uniform',np.ones(12)/12),('aligned',weights)]:
  k=np.einsum('v,vij->ij',w,views);kt=k[np.ix_(ix,ix)];cross=k[np.ix_(qi,ix)];qc=cross-kt.mean(0)[None,:]-cross.mean(1)[:,None]+kt.mean();ev,u=np.linalg.eigh(center(kt));ev=np.maximum(ev,0)
  den=np.sqrt(np.maximum(np.diag(k)[qi,None]*np.diag(k)[None,ix],1e-24));sim=np.clip(cross/den,-1,1)
  kernels[name]=(ev,u,qc,sim)
 norms=np.linalg.norm(R,axis=1);median=np.median(norms[norms>1e-12]) if np.any(norms>1e-12) else 0.;direction=np.divide(R,norms[:,None],out=np.zeros_like(R),where=norms[:,None]>1e-12)*median;direction-=direction.mean(0)
 return dict(Y=Y,R=R,b=b,direction=direction,kernels=kernels,missing=coverage[qi]==0,weights=weights,alignment=align,view_names=names,raw_queries=queries,train=train)

def predict_candidate(state,ctrl,cfg):
 ev,u,qc,sim=state['kernels'][cfg['kernel']]
 if cfg['head']=='neighbors':
  k=min(cfg['k'],sim.shape[1]);order=np.argsort(-sim,axis=1,kind='stable')[:,:k];v=np.take_along_axis(sim,order,axis=1);w=np.exp((v-v.max(1,keepdims=True))/.1);w/=w.sum(1,keepdims=True);res=np.einsum('qk,qkg->qg',w,state['R'][order])
 else:
  w=(qc@u/(ev+cfg['alpha']))@u.T;res=w@(state['direction'] if cfg['head']=='direction' else state['R']);res*=cfg.get('gain',1.)
 delta=state['b']+res;delta[state['missing']]=state['b'];return np.maximum(ctrl+delta,0).astype(np.float32).astype(float)

def configurations():
 out=[]
 for kernel in ['uniform','aligned']:
  for head in ['raw','direction']:
   for alpha in [.01,.1,1.]:
    for gain in ([1.,2.] if head=='direction' else [1.]):
     out.append(dict(id=f'{kernel}_{head}_a{alpha:g}_g{gain:g}',family=f'{kernel}_{head}',kernel=kernel,head=head,alpha=alpha,gain=gain))
 for k in [3,10,30]:out.append(dict(id=f'aligned_neighbors_k{k}',family='aligned_neighbors',kernel='aligned',head='neighbors',k=k))
 return out
