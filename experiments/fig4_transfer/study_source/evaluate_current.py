from current_engine import *
from safe_metrics import evaluate_arrays,OLD,METRICS
from evaluate_new import writecsv

def main():
 verify_seals();rows=[];details={};parent=ROOT.parent/'uc_state_adaptive_temperature_ablation_20261101'
 for fold in ['rpe1','jurkat']:
  r=ROOT/'results'/fold;ax=np.load(r/'axes.npz');truth=np.load(OLD/'evaluation_only'/fold/'common_truth.npz');np.testing.assert_array_equal(ax['targets'],truth['targets']);np.testing.assert_array_equal(ax['genes'],truth['genes']);np.testing.assert_array_equal(ax['control'],truth['control']);mask=ax['source_coverage'].any(0)
  for method in [BASE,MAIN]:
   p=np.load(r/(method+'_mean.npy'));m,diag,parity=evaluate_arrays(truth['mean'],p,ax['control'],ax['control']);np.savez_compressed(r/(method+'_metrics.npz'),targets=ax['targets'],**m);save(r/(method+'_metric_parity.json'),parity)
   for group,keep in [('all',np.ones(len(mask),bool)),('source_supported',mask)]:
    item=dict(target_context=fold,method=method,group=group,n_targets=int(keep.sum()))
    for k,v in m.items():
     ok=keep&np.isfinite(v);item[k]=float(v[ok].mean()) if ok.any() else None;item[k+'_n_finite']=int(ok.sum())
    rows.append(item)
   del p;gc.collect()
  details[fold]=dict(selection=json.load(open(r/'SOURCE_SELECTION_FROZEN.json')),diagnostics=json.load(open(r/'prediction_diagnostics.json')),replay=json.load(open(r/'REPLAY_CHECKS.json')),no_shrink_maps=json.load(open(r/'no_shrink_maps.json')));log(stage='EVALUATED',fold=fold,summary=rows[-2])
 h=json.load(open(ROOT/'HISTORICAL_HASHES.json'));assert all(sha(n)==v for n,v in h.items());writecsv(ROOT/'delivery/comparison.csv',rows);save(ROOT/'COMPLETE.json',dict(status='COMPLETE',utc=now(),folds=details,historical_files_unchanged=len(h)));log(stage='COMPLETE')
if __name__=='__main__':main()
