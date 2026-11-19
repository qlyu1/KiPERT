from pathlib import Path
import sys,json,pickle,numpy as np,pandas as pd,h5py,argparse,hashlib
R=Path('@STUDY@/uc_fig2_benchmark_20260911');sys.path.insert(0,str(R/'src'));from prepare_formal import Resolver
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args();B=R/'data/formal'/a.dataset/'split101';O=R/'assets/prior_banks';O.mkdir(exist_ok=True);res=Resolver();m=pd.read_csv(B/'target_prior_mapping.tsv',sep='\t',dtype=str);targets=m.target.tolist();bank={'components':np.array(targets,dtype=str)};receipt={'dataset':a.dataset,'target_count':len(targets),'data_response_used':False}
for name,file in [('depmap','depmap_crispr_gene_effect_processed.pkl'),('genept','GenePT_gene_embedding_ada_text.pickle')]:
 with (R/'assets/priors'/file).open('rb') as f:d=pickle.load(f)
 dim=len(next(iter(d.values())));x=np.zeros((len(targets),dim),np.float32);present=np.zeros(len(targets),bool)
 for i,t in enumerate(targets):
  key=res.map_prior(set(d),t,m.symbol.iloc[i]);
  if key is not None:x[i]=np.asarray(d[key],np.float32);present[i]=True
 bank['x'+name]=x;bank[name+'_present']=present;receipt[name+'_covered']=int(present.sum())
info=pd.read_csv(R/'assets/priors/9606.protein.info.v12.0.txt.gz',sep='\t');idcol=info.columns[0];id2gene={z[0]:res.resolve(z[1]) for z in info[[idcol,'preferred_name']].itertuples(index=False,name=None)}
with h5py.File(R/'assets/priors/9606.protein.network.embeddings.v12.0.h5') as f:proteins=f['proteins'].asstr()[:];emb=f['embeddings'][:]
lookup={}
for i,pid in enumerate(proteins):
 gid=id2gene.get(pid)
 if gid:lookup.setdefault(gid,[]).append(i)
x=np.zeros((len(targets),emb.shape[1]),np.float32);present=np.zeros(len(targets),bool)
for i,t in enumerate(targets):
 if t in lookup:x[i]=emb[lookup[t]].mean(0);present[i]=True
bank['xstring']=x;bank['string_present']=present;receipt['string_covered']=int(present.sum());receipt['multiple_proteins']='arithmetic mean static embeddings per stable gene';assert all(np.isfinite(z).all() for k,z in bank.items() if k.startswith('x'));np.savez_compressed(O/(a.dataset+'.npz'),**bank);receipt['sha256']=hashlib.sha256((O/(a.dataset+'.npz')).read_bytes()).hexdigest();(O/(a.dataset+'.json')).write_text(json.dumps(receipt,indent=2));print(receipt)
