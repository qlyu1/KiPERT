from common import *
import h5py,shutil
from scipy.sparse import csr_matrix

def aggregate(file):
 with h5py.File(file,'r') as h:
  labels=column(h['obs/gene']);groups,inverse=np.unique(labels,return_inverse=True);counts=np.bincount(inverse);s=np.zeros((len(groups),h['X'].shape[1]));ss=s.copy();minimum=1e10;maximum=-1e10
  for lo in range(0,len(labels),2048):
   hi=min(len(labels),lo+2048);a=np.asarray(h['X'][lo:hi],float);assert np.isfinite(a).all();minimum=min(minimum,float(a.min()));maximum=max(maximum,float(a.max()));m=csr_matrix((np.ones(hi-lo),(inverse[lo:hi],np.arange(hi-lo))),shape=(len(groups),hi-lo));s+=m@a;ss+=m@(a*a)
  return dict(targets=groups,mean=s/counts[:,None],var=np.maximum(ss-s*s/counts[:,None],0)/np.maximum(counts[:,None]-1,1),n=counts,minimum=minimum,maximum=maximum,variables=column(h['var'][h['var'].attrs['_index']]))

def main(target,models,kinds):
 SP=ROOT.parent/'uc_state_official_20261101/prepared';assert (SP/'SOURCE_PREPARED.json').exists();E=ROOT/'evaluation_only';E.mkdir(exist_ok=True);ann=np.load(SP/'source_annotations.npz');contexts=ann['contexts'];sourcefile=ROOT/'official/State-Replogle-Filtered/replogle_concat.h5ad';audits={}
 with h5py.File(sourcefile,'r') as source:
  cells=column(source['obs'][source['obs'].attrs['_index']])
  for target in [target]:
   out=E/target;out.mkdir(exist_ok=True);common=None;ctxrows=np.flatnonzero(contexts==target);localcells=cells[ctxrows];order=np.argsort(localcells);sortedcells=localcells[order];assert len(np.unique(localcells))==len(localcells)
   for model in models:
    np.testing.assert_array_equal(model_gene_names(model,target),ann['genes'])
    for kind in kinds:
     file=official(model,target,kind);receipt=Path(str(file.resolve())+'.receipt.json');assert receipt.exists();z=aggregate(file);np.savez_compressed(out/(model+'_'+kind+'_bulk.npz'),**z);assert z['mean'].shape[1]==2000;assert np.array_equal(z['variables'],np.arange(2000).astype(str));audit=dict(n_cells=int(z['n'].sum()),n_targets=len(z['targets'])-1,minimum=float(z['minimum']),maximum=float(z['maximum']),file_sha256=json.load(open(receipt))['actual_sha256'])
     if common is None:common=z
     np.testing.assert_array_equal(z['targets'],common['targets']);np.testing.assert_array_equal(z['n'],common['n'])
     if kind=='real':
      np.testing.assert_allclose(z['mean'],common['mean'],atol=1e-6,rtol=1e-6);audit['truth_bulk_max_abs_difference_to_HVG']=float(np.max(np.abs(z['mean']-common['mean'])))
      with h5py.File(file,'r') as h:
       sample=np.unique(np.linspace(0,h['X'].shape[0]-1,128,dtype=int));bars=h['obs/pert_cell_barcode'].asstr()[sample];assert all(x.startswith('cell_') and x[5:].isdigit() for x in bars);sr=np.array([int(x[5:]) for x in bars]);assert (sr>=0).all() and (sr<len(contexts)).all();np.testing.assert_array_equal(contexts[sr],target);np.testing.assert_array_equal(ann['labels'][sr],column(h['obs/gene'])[sample]);audit['cell_id_mapping']='cell_<zero-based global source row>';sort=np.argsort(sr);back=np.argsort(sort);ref=source['obsm/X_hvg'][sr[sort],:][back];actual=h['X'][sample,:];err=float(np.max(np.abs(np.clip(ref,0,14)-actual)));assert err<1e-6,(model,target,err);audit['source_cell_numeric_match_max_abs_error']=err;audit['matched_cells']=len(sample)
     audits[target+'/'+model+'/'+kind]=audit;log(stage='official_file_aggregated',target=target,model=model,kind=kind,**audit)
   assert common is not None;mask=common['targets']!='non-targeting';q=common['targets'][mask];ci=np.flatnonzero(~mask)[0];dest=ROOT/'target_inputs'/target;dest.mkdir(parents=True,exist_ok=True);source_meta=np.load(P/target/'meta.npz');assert set(q)<=set(source_meta['targets']);metadata=dict(targets=q,genes=source_meta['genes'],control=common['mean'][ci],control_var=common['var'][ci],n=common['n'][mask],n_control=common['n'][ci])
   if (dest/'meta.npz').exists():
    old=np.load(dest/'meta.npz')
    for k,v in metadata.items():np.testing.assert_array_equal(old[k],v)
   else:
    np.savez_compressed(dest/'meta.npz',**metadata);shutil.copy2(P/target/'controls.npy',dest/'controls.npy')

   # Controls in the common truth and source file should represent the same target control distribution.
   assert int(common['n'][ci])==int(source_meta['n_control']);np.testing.assert_allclose(common['mean'][ci],source_meta['control'],atol=1e-6,rtol=1e-6)
   np.savez_compressed(out/'common_truth.npz',targets=q,genes=source_meta['genes'],mean=common['mean'][mask],delta=common['mean'][mask]-common['mean'][ci],control=common['mean'][ci],n=common['n'][mask]);np.save(dest/'official_variables.npy',common['variables'])
 save(R/(target+'_OFFICIAL_AXIS_AUDIT_'+('_'.join(models))+'_'+('_'.join(kinds))+'.json'),dict(status='PASS',files=audits,official_gene_names_identical_in_all_models=True,source_hvg_mapping=json.load(open(SP/'SOURCE_PREPARED.json'))['hvg_full_matrix_max_abs_error'],truth_rule='Identical real pseudobulks verified for HVG and SE; HVG selected deterministically',output_space='X, 2000 genes, identical ordered numerical columns with named HVG mapping verified'))
 log(stage='OFFICIAL_INPUTS_READY',files=len(audits))
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('fold',choices=['k562','hepg2']);ap.add_argument('--first',action='store_true');a=ap.parse_args();main(a.fold,['HVG'] if a.first else ['HVG','SE'],['real'] if a.first else ['real','pred'])
