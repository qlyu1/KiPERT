from pathlib import Path
import json,shutil,hashlib,time
import numpy as np
B=Path(__file__).resolve().parents[1];P=B.parent/'pie_four_context_reproduction_20261106';O=B.parent/'uc_state_four_context_zeroshot_20261102';D=P/'downloads/official_data/preprocessed';m=json.loads((D/'meta.json').read_text());delta=np.load(D/'delta_p.npy',mmap_mode='r')
while not (B/'manifests/raw_preparation_complete.json').exists():time.sleep(15)
for fold in m['context_to_id']:
 root=B/'runs/uc'/fold;root.mkdir(parents=True,exist_ok=True);shutil.copytree(O/'src',root/'src',dirs_exist_ok=True);p=root/'prepared';p.mkdir(exist_ok=True);protocol=json.loads((O/'protocol.json').read_text());protocol.update(design='PIE_EXACT_TRAIN_PAIRS_6642_GENES',target_evaluation='All PIE test rows; 6642 ordered genes; frozen published PIE scorer',historical_folds=[],new_folds=list(m['context_to_id']),data_source=str(D),input_alignment=str(B/'reports/raw_panel_control_alignment.json'));(root/'protocol.json').write_text(json.dumps(protocol,indent=2));np.save(p/'output_genes.npy',np.array(m['genes']))
 for src,dst in [(O/'shared_gene_basis.npz',root/'shared_gene_basis.npz'),(O/'prepared/shared_gene_basis.npz',p/'shared_gene_basis.npz'),(O/'prepared/shared_gene_basis_12.npz',p/'shared_gene_basis_12.npz')]:
  if not dst.exists():dst.symlink_to(src)
 train=json.loads((P/'downloads/official_splits/replogle_wdataset/unseen_ctx'/fold/'train.json').read_text());audit=[]
 for ctx in protocol['folds'][fold]+[fold]:
  stats=B/'data/statistics'/ctx;z=np.load(stats/'meta.npz');meta={k:z[k] for k in z.files};allowed=train.get('replogle.'+ctx,[]) if ctx!=fold else list(z['targets']);idx=np.array([i for i,t in enumerate(z['targets']) if t in set(allowed)]);assert set(z['targets'][idx])==set(allowed);dst=p/ctx if ctx!=fold else root/'target_inputs'/ctx;dst.mkdir(parents=True,exist_ok=True)
  for k in ['targets','n','ref_rows']:meta[k]=meta[k][idx]
  np.savez_compressed(dst/'meta.npz',**meta)
  if not (dst/'controls.npy').exists():(dst/'controls.npy').symlink_to(stats/'controls_ot.npy')
  if ctx!=fold:
   np.save(dst/'delta.npy',delta[meta['ref_rows']]);np.save(dst/'noise_mean.npy',np.load(stats/'noise_mean.npy')[idx])
  audit.append({'context':ctx,'role':'target_controls_and_query_names_only' if ctx==fold else 'PIE_train_only','n_pairs':len(idx),'targets':list(meta['targets'])})
 (root/'input_audit.json').write_text(json.dumps(audit,indent=2));print('UC_READY',fold,flush=True)
(B/'manifests/uc_preparation_complete.json').write_text(json.dumps({'status':'complete','source_point_means':'exact PIE published train-only delta_p','noise':'same filtered raw cells','controls':'512 deterministic cells per context; means from PIE'}))
