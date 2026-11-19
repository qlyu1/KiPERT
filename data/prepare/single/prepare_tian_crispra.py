"""Dataset-specific preprocessing; raw data are only opened by this preparation stage."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from benchmark_contract import split_targets, stable_order
ROOT=Path('@STUDY@/uc_fig2_benchmark_20260911')
ALIASES={'ATP5C1':'ATP5F1C','SMEK1':'PPP4R3A'}
SOURCES={'ATP5C1':'https://www.ncbi.nlm.nih.gov/gene/509/','SMEK1':'https://www.ncbi.nlm.nih.gov/gene/55671/'}
def main():
    raw=ROOT/'data/raw/TianKampmann2021_CRISPRa.h5ad'
    schema=json.loads((ROOT/'manifests/TianKampmann2021_CRISPRa.h5ad.schema.json').read_text())
    assert schema['status']=='SCHEMA_CANDIDATE_PASS'
    a=ad.read_h5ad(raw);x=a.X.tocsr().astype(np.float32)
    assert np.isfinite(x.data).all() and (x.data>=0).all() and np.allclose(x.data,np.rint(x.data),rtol=0,atol=1e-5)
    assert not a.obs_names.duplicated().any() and not a.var.ensembl_id.duplicated().any()
    o=a.obs.copy();o['cell_id']=o.index.astype(str);o['symbol']=o.perturbation.astype(str).replace(ALIASES);o['context']=o.batch.astype(str)
    symbol_to_id=a.var.ensembl_id.astype(str).to_dict()
    assert set(o.symbol)-{'control'} <= set(symbol_to_id)
    o['target']=o.symbol.map(symbol_to_id).fillna('control')
    assert (o.loc[o.symbol=='control','nperts']==0).all() and (o.loc[o.symbol!='control','nperts']==1).all()
    total=np.asarray(x.sum(1)).ravel();ng=np.asarray((x>0).sum(1)).ravel()
    mito=np.asarray(x[:,a.var_names.str.startswith('MT-')].sum(1)).ravel()/np.maximum(total,1)
    qc=(total>0)&(ng>=200)&(mito<=.2)
    control_rows={};eligible=[]
    for context in sorted(o.context.unique()):
        ci=np.flatnonzero(qc&(o.symbol.to_numpy()=='control')&(o.context.to_numpy()==context))
        if len(ci)<50:continue
        idx={str(o.iloc[i].cell_id):int(i) for i in ci};order=stable_order(idx,'fig2-controls-S4-'+context)
        n_check=max(1,int(.2*len(order)+.5));control_rows[context]={'check':[idx[c] for c in order[:n_check]],'fit':[idx[c] for c in order[n_check:]]}
        for target in sorted(set(o.target)-{'control'}):
            ix=np.flatnonzero(qc&(o.target.to_numpy()==target)&(o.context.to_numpy()==context))
            if len(ix)>=50:eligible.append((target,context,ix))
    targets=sorted(set(t for t,c,ix in eligible));all_splits={str(s):split_targets(targets,s) for s in (101,202,303)}
    split=all_splits['101'];train_rows=[];validation_rows=[];test_rows=[]
    for target,context,ix in eligible:
        destination='train' if target in split['train'] else 'validation' if target in split['validation'] else 'test'
        if destination=='train':
            lookup={str(o.iloc[i].cell_id):int(i) for i in ix}
            ix=[lookup[c] for c in stable_order(lookup,'fig2-cells-S4-'+target+'-'+context)[:512]]
        {'train':train_rows,'validation':validation_rows,'test':test_rows}[destination].extend(map(int,ix))
    cfit=sum([z['fit'] for z in control_rows.values()],[])
    expressed=np.asarray((x[train_rows+cfit]>0).sum(0)).ravel()>=10
    scf=set(pd.read_csv(ROOT/'repos/scFoundation/OS_scRNA_gene_index.19264.tsv',sep='\t').gene_name.replace(ALIASES))
    scgpt=set(ALIASES.get(g,g) for g in json.loads((ROOT/'assets/scgpt/vocab.json').read_text()))
    gene_indices=[i for i,g in enumerate(a.var_names) if expressed[i] and g in scf and g in scgpt]
    gene_indices.sort(key=lambda i:str(a.var.iloc[i].ensembl_id))
    assert len(gene_indices)>5000
    # Library size comes from each original full-axis cell, before gene subsetting.
    norm=sparse.diags(10000/np.maximum(total,1)).dot(x).tocsr();norm.data=np.log1p(norm.data)
    norm=norm[:,gene_indices].tocsr().astype(np.float32)
    base=ROOT/'data/preflight/S4/split101';base.mkdir(parents=True,exist_ok=True)
    for role,rows in [('fit',train_rows),('validation',validation_rows),('evaluation_only',test_rows),('controls_fit',cfit),('controls_check',sum([z['check'] for z in control_rows.values()],[]))]:
        d=base/role;d.mkdir(exist_ok=True)
        sparse.save_npz(d/'expression.npz',norm[rows])
        o.iloc[rows][['cell_id','symbol','target','context','guide_id']].to_csv(d/'metadata.tsv',sep='\t',index=False)
    gene_frame=pd.DataFrame({'gene_id':a.var.iloc[gene_indices].ensembl_id.to_numpy(),'symbol':a.var_names[gene_indices]})
    gene_frame.to_csv(base/'genes.tsv',sep='\t',index=False)
    condition_table=[{'target':t,'symbol':o.iloc[int(ix[0])].symbol,'context':c,'n_cells':len(ix)} for t,c,ix in eligible]
    out={'dataset':'S4','raw_shape':list(a.shape),'raw_sha256':schema['sha256'],'all_count_values_valid':True,'qc_cells':int(qc.sum()),'eligible_targets':len(targets),'eligible_conditions':condition_table,'splits':all_splits,'prepared_split':101,'split_counts':{k:len(v) for k,v in split.items()},'prepared_cells':{'fit':len(train_rows),'validation':len(validation_rows),'evaluation_only':len(test_rows),'controls_fit':len(cfit)},'output_genes':len(gene_indices),'output_axis_sha256':hashlib.sha256(gene_frame.to_csv(index=False).encode()).hexdigest(),'aliases':ALIASES,'alias_sources':SOURCES,'source_control_definition':'harmonized perturbation=control and nperts=0; original study harmonization history review remains required','status':'REAL_DATA_PREFLIGHT_PREPARED','formal_run_lock':False,'scope':'Dataset-specific preflight, not full seven-dataset certification; OS-level training file-access isolation not yet established'}
    (base/'preparation.json').write_text(json.dumps(out,indent=2));(ROOT/'manifests/S4_preflight.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ('eligible_conditions','splits')},indent=2))
if __name__=='__main__':main()
