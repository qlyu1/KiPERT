"""Response summaries preserve the cell-weighted MSE objective of no-basal TxPert."""
from common import *
import pandas as pd,h5py,time
def column(g):
 if isinstance(g,h5py.Group):
  a=g['categories'];v=a.asstr()[:] if a.dtype.kind in 'OS' else a[:];return v[g['codes'][:]]
 return g.asstr()[:] if g.dtype.kind in 'OS' else g[:]
graphfile=T/'code/data/graphs/string/v11.5.parquet'
tree=json.loads((T/'github_tree.json').read_text())
entry=next(x for x in tree['tree'] if x['path']=='data/graphs/string/v11.5.parquet')
blob=graphfile.read_bytes();gh=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest();assert gh==entry['sha'];del blob
original=pd.read_csv(T/'code/data/gears_gene_set.csv',index_col=0)['0'].to_dict()
vocab={str(v):int(k) for k,v in original.items()};missing=sorted(set(META['pert_to_id'])-set(vocab))
for p in missing:vocab[p]=len(vocab)
vocab['ctrl']=-1;savej(T/'vocab.json',vocab)
g=GSPGraph(vocab,{g:i for i,g in enumerate(GENES)},copy.deepcopy(CFG['graph']['graph_cfg']),str(T/'code/data/graphs'))
torch.save(next(iter(g.graph_dict.values())),T/'graph.pt')
(T/'summaries').mkdir(exist_ok=True)
checks=[]
for ctx in CTX:
 stats=B/'data/statistics'/ctx;z=np.load(stats/'meta.npz');cm=np.load(stats/'cells_metadata.npz');np.testing.assert_array_equal(z['genes'],GENES)
 with h5py.File(B/'data/normalized'/f'{ctx}.h5ad') as h:
  batches=column(h['obs/gem_group']).astype(str);labels=column(h['obs/gene']).astype(str)
  np.testing.assert_array_equal(column(h['var/gene_name']),GENES)
 np.testing.assert_array_equal(cm['labels'],labels)
 controls=np.load(stats/'controls_all.npy',mmap_mode='r');cb=batches[cm['control_rows']]
 assert np.all(labels[cm['control_rows']]=='non-targeting')
 bc={b:np.asarray(controls[cb==b],dtype=np.float64).mean(0) for b in np.unique(cb)}
 means=np.load(stats/'mean.npy',mmap_mode='r');res=np.empty(means.shape,np.float32)
 for j,p in enumerate(z['targets']):
  bs,ns=np.unique(batches[labels==p],return_counts=True);assert ns.sum()==z['n'][j]
  assert set(bs)<=set(bc),(ctx,p,set(bs)-set(bc))
  res[j]=means[j].astype(float)-sum(bc[b]*n for b,n in zip(bs,ns))/ns.sum()
 np.savez_compressed(T/'summaries'/f'{ctx}.npz',targets=z['targets'],residual_mean=res,counts=z['n'],n_control=z['n_control'])
 # Verify conditional-mean reduction on actual individual cells, including all genes.
 # Residual variance is independent of the prediction. This checks objective AND gradient.
 p=str(z['targets'][0]);rows=np.flatnonzero(labels==p);X=np.load('/dev/shm/pie_uc_state_20261106/'+ctx+'.npy',mmap_mode='r')
 r=np.asarray(X[rows],dtype=float)-np.stack([bc[b] for b in batches[rows]])
 mu=r.mean(0);np.testing.assert_allclose(mu,res[0],atol=2e-5,rtol=1e-4)
 pred=np.random.default_rng(1).normal(0,.1,len(GENES));lhs=np.mean((r-pred)**2);rhs=np.mean((mu-pred)**2)+np.mean((r-mu)**2)
 err=abs(lhs-rhs);gerr=float(np.max(np.abs((2*(pred-r)/len(GENES)).mean(0)-2*(pred-mu)/len(GENES))))
 assert err<1e-10 and gerr<1e-10
 checks.append(dict(context=ctx,targets=len(res),controls=int(z['n_control']),batches=len(bc),mse_identity_error=err,gradient_identity_error=gerr))
 print('PREPARED',ctx,checks[-1],flush=True)
audit={}
for fold in CTX:
 tr,va,te=[splits(fold,p) for p in ['train','val','test']]
 assert fold not in tr and fold not in va and set(te)=={fold}
 for c,ps in tr.items():assert not set(ps)&set(va.get(c,[]))
 audit[fold]={p:{c:len(ps) for c,ps in d.items()} for p,d in zip(['train','val','test'],[tr,va,te])}
protocol={'model':'TxPert public cross-cell GAT; local PIE-protocol adaptation','repo':'https://github.com/valence-labs/TxPert','commit':'08d82eea86746b044cf7531f4ec8c5f60e1cb73f','graph_blob_sha1':gh,'graph_sha256':sha(graphfile),'original_graph_vocabulary':len(original),'added_input_nodes':missing,'control_index':-1,'graph_edges':int(next(iter(g.graph_dict.values()))[0].shape[1]),'genes':len(GENES),'seed':1,'training':{'optimizer':'AdamW','lr':.0003,'weight_decay':0,'batch_size':64,'max_updates':50000,'warmup_updates':3125,'schedule':'linear 0.1 to 1 warmup then cosine to zero','validation_every':500,'selection':'maximum source-validation mean PearsonDelta, across conditions then equal source contexts','early_stopping_patience_evaluations':20,'minimum_updates':10000,'targets':'conditional residual means; sample perturbations proportional to number of authorized training cells, including all controls','loss_equivalence':'For no-basal additive model with MSE, E||prediction-r||²=||prediction-E[r|pert]||²+constant. Conditional label-batch distribution and decoder BatchNorm/Dropout retained; lower gradient variance; same expected objective. Updates are not claimed to replicate the author training trajectory.','control_matching':'mean control within context and gem_group; control training target zero; all four backgrounds control-only available','checkpoint_weights':'random initialization; no public learned weights reused'},'evaluation':'same frozen additive-to-500-heldout-controls adapter and PIE/gpudge settings as UC/PMean; all 6642 genes and exact existing folds','public_checkpoint_exclusion':'only K562 released, different source train/validation split and 3352-output genes; overlaps PIE source validation labels','split_audit':audit,'summary_checks':checks,'source_hashes':{n:sha(T/'code'/n) for n in ['gspp/models/txpert.py','gspp/models/pert_models/basic_gnn.py','gspp/data/graphmodule.py','configs/config-x-cell-gat.yaml']},'created_at':time.strftime('%Y-%m-%d %H:%M:%S %z')}
savej(T/'protocol.json',protocol)
print('PREPARATION_COMPLETE',flush=True)
