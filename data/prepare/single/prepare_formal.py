"""Frozen target splits and train-only expression axes for Figure 2.
All raw responses are read only by this preparation program. Training receives separate files.
"""
from pathlib import Path
import argparse,json,hashlib,pickle,gc,re,time
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import sys
R=Path('@STUDY@/uc_fig2_benchmark_20260911');sys.path.insert(0,str(R/'src'))
from benchmark_contract import split_targets,stable_order
FILES={'S1':'ReplogleWeissman2022_K562_gwps.h5ad','S2':'ReplogleWeissman2022_rpe1.h5ad','S3':'TianKampmann2021_CRISPRi.h5ad','S4':'TianKampmann2021_CRISPRa.h5ad','S5':'AdamsonWeissman2016_GSM2406681_10X010.h5ad','S6':'NormanWeissman2019_filtered.h5ad','S7':'FrangiehIzar2021_RNA.h5ad'}
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
 return h.hexdigest()
class Resolver:
 def __init__(self):
  h=pd.read_csv(R/'assets/priors/hgnc_complete_set.txt',sep='\t',dtype=str).fillna('');self.byid={};self.current={};self.alias={}
  for row in h.to_dict('records'):
   gid=row['ensembl_gene_id'].split('.')[0]
   if not gid.startswith('ENSG'):continue
   names=[row['symbol']]+row['prev_symbol'].split('|')+row['alias_symbol'].split('|');names=[n for n in dict.fromkeys(names) if n]
   self.byid[gid]=names;self.current[row['symbol']]=gid
   for name in names:self.alias.setdefault(name,set()).add(gid)
 def resolve(self,name):
  if name in self.current:return self.current[name]
  ids=self.alias.get(name,set());return next(iter(ids)) if len(ids)==1 else None
 def candidates(self,gid,old=''):
  return list(dict.fromkeys(self.byid.get(gid,[])+([old] if old else [])))
 def map_prior(self,keys,gid,old=''):
  for n in self.candidates(gid,old):
   if n in keys and (self.resolve(n) in (None,gid)):return n
  return None

def annotate(a,d,res):
 o=a.obs.copy();o.index=o.index.astype(str);assert not o.index.duplicated().any();o['cell_id']=o.index
 pert=o.perturbation.astype(str).str.strip().to_numpy();single=np.zeros(len(o),bool);ctrl=np.zeros(len(o),bool);symbols=np.array(['']*len(o),dtype=object)
 if d in ('S3','S4'):
  single=o.nperts.to_numpy()==1;ctrl=(pert=='control')&(o.nperts.to_numpy()==0);symbols=pert.copy();o['context']=o.batch.astype(str)
 elif d=='S6':
  single=o.nperts.to_numpy()==1;ctrl=(pert=='control')&(o.nperts.to_numpy()==0);symbols=pert.copy();o['context']='gemgroup='+o.gemgroup.astype(str)
 elif d=='S7':
  guide=o.guide_id.astype(object).fillna('').astype(str);single=(o.MOI.to_numpy()==1)&(o.nperts.to_numpy()==1)&~guide.str.contains(';').to_numpy()
  ctrl=(o.MOI.to_numpy()==1)&(o.nperts.to_numpy()==0)&guide.str.startswith('NO_SITE_').to_numpy();symbols=pert.copy();o['context']='stimulus='+o.perturbation_2.astype(str)
  expected=guide.str.replace(r'_\d+$','',regex=True).to_numpy();single&=(expected==pert)
 elif d=='S5':
  # Only enabled after original-source control/construct audit has been saved.
  gate=R/'manifests/S5_source_adapter_verified.json';assert gate.exists(),'Adamson source annotation gate incomplete'
  controls=json.loads(gate.read_text())['control_constructs'];ctrl=np.isin(pert,controls)
  for i,p in enumerate(pert):
   m=re.fullmatch(r'(.+)_p[A-Z]+\d+',p)
   if m and not ctrl[i]:symbols[i]=m.group(1);single[i]=True
  o['context']='GSM2406681_10X010'
 elif d in ('S1','S2'):
  assert 'gene' in o or 'perturbation' in o
  symbols=pert.copy();ctrl=(pert=='control')&(o.nperts.to_numpy()==0);single=o.nperts.to_numpy()==1
  # Field semantics must be recorded before enabling each Replogle adapter.
  gate=R/'manifests'/(d+'_source_adapter_verified.json');assert gate.exists(),'Replogle source annotation gate incomplete'
  fields=json.loads(gate.read_text())['context_fields'];assert fields
  o['context']=o[fields].astype(str).agg('|'.join,axis=1)
 else:raise ValueError(d)
 raw_ids=a.var['ensembl_id' if 'ensembl_id' in a.var else 'ensemble_id'].astype(str).map(lambda v: re.sub(r'^(ENSG[0-9]+)\.[0-9]+$',r'\1',v))
 raw_symbols=a.var['gene_symbol'].astype(str).to_numpy() if 'gene_symbol' in a.var else a.var_names.astype(str).to_numpy()
 original_map={s:g for s,g in zip(raw_symbols,raw_ids)}
 symbol_lookup={p:(res.resolve(p) or original_map.get(p)) for p in set(symbols) if p not in ('','control','nan')}
 target=np.array([symbol_lookup.get(p,'') or '' for p in symbols],dtype=object);single&=target!='';target[ctrl]='control'
 o['target']=target;o['symbol']=symbols;o.loc[ctrl,'symbol']='control';o['raw_total']=0.
 if 'guide_id' not in o:o['guide_id']=pert
 return o,single,ctrl,raw_ids.to_numpy(),raw_symbols

def main(d,seeds):
 start=time.time();raw=R/'data/raw'/FILES[d];receipt=R/'manifests'/(FILES[d]+'.schema.json');schema=json.loads(receipt.read_text());assert schema['status']=='SCHEMA_CANDIDATE_PASS'
 resolver=Resolver();a=ad.read_h5ad(raw);o,single,ctrl,gids,symbols=annotate(a,d,resolver)
 x=a.X.tocsr().astype(np.float32,copy=False);a.X=None;del a;gc.collect();x.eliminate_zeros()
 assert len(set(gids))==len(gids),'duplicate stable gene axis requires explicit aggregation'
 for s in range(0,len(x.data),1000000):
  values=x.data[s:s+1000000];assert np.isfinite(values).all() and (values>=0).all() and np.allclose(values,np.rint(values),rtol=0,atol=1e-5)
 totals=np.asarray(x.sum(1)).ravel();detected=np.diff(x.indptr);mt=np.array([s.startswith('MT-') for s in symbols]);mito=np.asarray(x[:,mt].sum(1)).ravel()/np.maximum(totals,1)
 qc=(totals>0)&(detected>=200)&(mito<=.2);o['raw_total']=totals
 control_rows={};conditions=[];counts=[]
 contexts=o.context.to_numpy();targets=o.target.to_numpy()
 for context in sorted(o.context.unique()):
  ci=np.flatnonzero(qc&ctrl&(contexts==context))
  if len(ci)<50:continue
  ci=sorted(ci,key=lambda i:hashlib.sha256(('fig2-controls-'+d+'-'+context+'\0'+o.iloc[i].cell_id).encode()).digest())
  n_check=max(1,int(.2*len(ci)+.5));control_rows[context]={'fit':ci[n_check:],'check':ci[:n_check]}
 valid=np.flatnonzero(qc&single&o.context.isin(control_rows).to_numpy())
 for (target,context),ix in o.iloc[valid].groupby(['target','context'],sort=True).indices.items():
  ix=valid[ix];counts.append({'target':target,'context':context,'n':len(ix),'eligible':len(ix)>=50})
  if len(ix)>=50:conditions.append((target,context,ix))
 eligible=sorted(set(t for t,c,ix in conditions));assert len(eligible)>=30
 with (R/'assets/priors/GPT_3_5_gene_embeddings.pickle').open('rb') as f:emb=pickle.load(f)
 keys=set(emb);del emb;gc.collect()
 vocab=json.loads((R/'assets/scgpt/vocab.json').read_text());scf=pd.read_csv(R/'repos/scFoundation/OS_scRNA_gene_index.19264.tsv',sep='\t')
 scf_id={resolver.resolve(n):int(i) for n,i in zip(scf.gene_name,scf['index']) if resolver.resolve(n)}
 scgpt_id={resolver.resolve(n):int(i) for n,i in vocab.items() if resolver.resolve(n)}
 coverage=[]
 for t in eligible:
  old=o.loc[o.target==t,'symbol'].iloc[0];prior=resolver.map_prior(keys,t,old);token=scgpt_id.get(t)
  coverage.append({'target':t,'symbol':old,'genept_v1_key':prior,'scgpt_token':token,'common_supported':prior is not None and token is not None})
 supported={z['target'] for z in coverage if z['common_supported']};assert len(supported)>=30
 cfit=sum([v['fit'] for v in control_rows.values()],[]);ccheck=sum([v['check'] for v in control_rows.values()],[])
 # Normalize each cell from its full original measured library, in bounded blocks.
 for lo in range(0,x.shape[0],2048):
  hi=min(x.shape[0],lo+2048);a0,b0=x.indptr[lo],x.indptr[hi];scale=np.repeat(10000/np.maximum(totals[lo:hi],1),np.diff(x.indptr[lo:hi+1]));x.data[a0:b0]*=scale
 np.log1p(x.data,out=x.data)
 registry=[]
 for seed in seeds:
  base=R/'data/formal'/d/('split'+str(seed));base.mkdir(parents=True,exist_ok=True)
  if (base/'READY.json').exists():print('already_prepared',d,seed,flush=True);continue
  splits=split_targets(eligible,seed);sets={k:set(v) for k,v in splits.items()};rows={'fit':[],'validation':[],'evaluation_only':[]};all_test=[];all_validation=[]
  for target,context,ix in conditions:
   role='fit' if target in sets['train'] else 'validation' if target in sets['validation'] else 'evaluation_only'
   if role=='evaluation_only':all_test.extend(ix.tolist())
   if role=='validation':all_validation.extend(ix.tolist())
   if target not in supported:continue
   if role=='fit':
    ix=sorted(ix,key=lambda i:hashlib.sha256(('fig2-cells-'+d+'-'+target+'-'+context+'\0'+o.iloc[i].cell_id).encode()).digest())[:512]
   rows[role].extend(map(int,ix))
  assert len(set(o.iloc[rows['validation']].target))>=5 and len(set(o.iloc[rows['evaluation_only']].target))>=5
  fit_indices=rows['fit']+cfit
  detected_train=np.asarray((x[fit_indices]>0).sum(0)).ravel();axis=[i for i,g in enumerate(gids) if detected_train[i]>=10 and g in scf_id and g in scgpt_id];axis.sort(key=lambda i:gids[i]);assert len(axis)>5000
  genes=pd.DataFrame({'gene_id':gids[axis],'symbol':symbols[axis],'scf_index':[scf_id[gids[i]] for i in axis],'scgpt_token':[scgpt_id[gids[i]] for i in axis]});genes.to_csv(base/'genes.tsv',sep='\t',index=False)
  nativegenes=pd.DataFrame({'gene_id':gids,'symbol':symbols,'scf_index':[scf_id.get(g,-1) for g in gids],'scgpt_token':[scgpt_id.get(g,-1) for g in gids]});nativegenes.to_csv(base/'genes_all.tsv',sep='\t',index=False)
  # Full evaluation cohort retained. Common-support mask is metadata, not a score-dependent filter.
  rows['evaluation_only']=all_test
  rows.update(controls_fit=cfit,controls_check=ccheck)
  for role,ix in rows.items():
   dest=base/role;dest.mkdir(exist_ok=True);sparse.save_npz(dest/'expression.npz',x[ix][:,axis]);meta=o.iloc[ix][['cell_id','symbol','target','context','guide_id','raw_total']].copy();meta['common_supported']=meta.target.isin(supported);meta.to_csv(dest/'metadata.tsv',sep='\t',index=False)
   if role=='controls_fit':sparse.save_npz(dest/'all_genes.npz',x[ix])
  query=base/'query';query.mkdir(exist_ok=True);q=o.iloc[all_test][['target','symbol','context']].drop_duplicates().sort_values(['target','context']);q['common_supported']=q.target.isin(supported);q.to_csv(query/'metadata.tsv',sep='\t',index=False)
  cov=pd.DataFrame(coverage);cov.to_csv(base/'target_prior_mapping.tsv',sep='\t',index=False)
  manifest={'dataset':d,'seed':seed,'source_file':FILES[d],'raw_sha256':schema['sha256'],'raw_shape':list(x.shape),'qc_cells':int(qc.sum()),'eligible_targets':len(eligible),'common_supported_targets':len(supported),'full_splits':splits,'actual_training_targets':sorted(set(o.iloc[rows['fit']].target)),'actual_validation_targets':sorted(set(o.iloc[rows['validation']].target)),'n_cells':{k:len(v) for k,v in rows.items()},'n_genes':len(axis),'axis_sha256':digest(base/'genes.tsv'),'fit_metadata_sha256':digest(base/'fit/metadata.tsv'),'fit_response_sha256':digest(base/'fit/expression.npz'),'validation_sha256':digest(base/'validation/expression.npz'),'control_sha256':digest(base/'controls_fit/expression.npz'),'query_metadata_sha256':digest(query/'metadata.tsv'),'hgnc_sha256':digest(R/'assets/priors/hgnc_complete_set.txt'),'protocol':'FIG2_FORMAL_V1_1','count_validation':'all stored values checked finite, nonnegative, integer before normalization','native_control_axis':'all measured genes; input library sizes are from control cells only','common_training_rule':'static GenePT-v1 and scGPT vocabulary support intersection shared by every primary method; full test cohort retained with explicit common-support mask','query_truth_loaded_by_training':False}
  (base/'READY.json').write_text(json.dumps(manifest,indent=2));registry.append(manifest);print(json.dumps({'prepared':d,'seed':seed,'genes':len(axis),'train_targets':len(manifest['actual_training_targets']),'validation_targets':len(manifest['actual_validation_targets']),'full_test_targets':len(splits['test']),'cells':manifest['n_cells']}),flush=True)
 (R/'manifests'/(d+'_formal_preparation.json')).write_text(json.dumps({'datasets':registry,'conditions':counts,'seconds':time.time()-start},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=FILES);p.add_argument('--seeds',nargs='+',type=int,default=[101,202,303]);a=p.parse_args();main(a.dataset,a.seeds)
