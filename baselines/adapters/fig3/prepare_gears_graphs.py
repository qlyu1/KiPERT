from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
from scipy import sparse
from resources import R,Resolver,digest
res=Resolver();go=pd.read_csv(R/'repos/TxPert/data/graphs/go/go_top_50.csv');go['source_id']=go.source.map(res.resolve);go['target_id']=go.target.map(res.resolve)
dest=R/'assets/gears_graphs';dest.mkdir(exist_ok=True)
for d in ['D1','D2','D3','D4']:
 base=R/f'data/formal/{d}/seen0/fold0';targets=sorted(pd.read_csv(base/'target_prior_mapping.tsv',sep='\t').target);ti={g:i for i,g in enumerate(targets)}
 gg=go[go.source_id.isin(ti)&go.target_id.isin(ti)].sort_values('importance',ascending=False).groupby('target_id',sort=True).head(20)
 edges={(ti[a],ti[b]):float(w) for a,b,w in zip(gg.source_id,gg.target_id,gg.importance)}
 for i in range(len(targets)):edges[i,i]=1.
 ge=np.asarray(list(edges),dtype=np.int64).T;gw=np.asarray(list(edges.values()),np.float32)
 x=sparse.load_npz(base/'controls_fit/expression.npz').astype(np.float32);mu=np.asarray(x.mean(0)).ravel();cov=(x.T@x).toarray()/x.shape[0]-mu[:,None]*mu[None,:]
 sd=np.sqrt(np.maximum(np.diag(cov),1e-12));cov/=sd[:,None];cov/=sd[None,:];np.fill_diagonal(cov,1.)
 ce=[];cw=[]
 for j in range(cov.shape[0]):
  ix=np.argsort(-cov[j],kind='stable')[:20];ix=ix[cov[j,ix]>.4]
  for i in ix:ce.append([int(i),j]);cw.append(float(cov[j,i]))
 np.savez_compressed(dest/f'{d}.npz',G_go=ge,G_go_weight=gw,G_coexpress=np.asarray(ce,np.int64).T,G_coexpress_weight=np.asarray(cw,np.float32))
 (dest/f'{d}.json').write_text(json.dumps(dict(dataset=d,source='public GO top50 table restricted to target vocabulary, top20 incoming edges; coexpression from C_fit controls only',go_sha256=digest(R/'repos/TxPert/data/graphs/go/go_top_50.csv'),controls_sha256=digest(base/'controls_fit/expression.npz'),graph_sha256=digest(dest/f'{d}.npz'),go_edges=len(gw),coexpression_edges=len(cw),target_nodes=len(targets),isolated_go_nodes=[g for g,i in ti.items() if not any(a!=b and (a==i or b==i) for a,b in edges)]),indent=2))
 print(d,len(gw),len(cw),flush=True)
(dest/'READY.json').write_text(json.dumps(dict(status='READY',datasets=['D1','D2','D3','D4'])))
