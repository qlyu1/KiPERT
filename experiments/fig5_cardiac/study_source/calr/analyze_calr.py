"""Read-only extraction of a fixed published Calr response from sealed predictions.

No fits, alternative genes, alternative endpoints, or target discovery.
Writes only within the new known_response directory.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import hashlib, json, argparse
import numpy as np
import pandas as pd
from scipy import sparse

OLD = Path(os.environ.get('KIPERT_CAI_D14_DIR', '@STUDY@/uc_cross_context_transfer/experiments/cai_d14_application_20261112'))
CACHE = Path('@STUDY@/uc_cross_context_transfer/experiments/cai_expression_programs_20261111/diagnostics_20261111/prepared')
EXPECTED = ['Tpm1','Tnnt2','Ryr2','Ttn','Ctnna3','Myl4','Myh6','Actc1']
METHODS = {'KiPERT_OT':'KiPERT_OT_Cai_prior_adaptation', 'source_mean':'equal_source_delta_mean', 'confidence_noOT':'same_weight_no_OT'}
SEED = 2026111327
BOOT = 2000

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def rank_interval(values, ref):
    values = np.asarray(values)
    return int(1+np.sum(values > ref)), int(np.sum(values >= ref))

def main(out):
    result = out/'results'; result.mkdir(parents=True, exist_ok=True)
    protocol = dict(status='FIXED_BEFORE_EXTRACTION', reference='Calr only; published positive reference',
                    genes=EXPECTED, thresholds=[5,10], methods=list(METHODS), seed=SEED, bootstrap_draws=BOOT,
                    task='Known response preservation in sealed source-seen D14 leave-batch-out predictions',
                    no_new_model=True, no_other_target_nominations=True,
                    scale='Original published normalized expression delta versus same-batch NT_73; cardiac score is fixed eight-gene arithmetic mean.',
                    cell_intervals='Percentile 95% cell bootstrap; sample target cells and NT cells independently within each batch, jointly over all eight genes; module interval averages gene values within the same draw.',
                    inference='Cell intervals describe conditional measurement precision. No gene-as-replicate or animal-level test; eight genes are a fixed correlated panel. Prediction inclusion in measurement CI is descriptive, not predictive calibration.',
                    scope='Retrospective recovery of an already known response, not unseen-target validation, new discovery, fate/yield prediction, or demonstration that OT outperforms source means.')
    dump(out/'protocol.json', protocol)
    prep, res = OLD/'prepared', OLD/'results_transfer'
    seal = json.load(open(res/'PREDICTIONS_SEALED.json'))
    assert seal['status'] == 'SEALED'
    programs = json.load(open(prep/'programs.json'))
    assert programs['cardiac']['used'] == EXPECTED
    axis = np.load(prep/'axis.npz')
    targets, genes, batches = [axis[k].astype(str) for k in ['targets','genes','batches']]
    ti = int(np.flatnonzero(targets == 'Calr')[0]); gi = np.array([np.flatnonzero(genes == g)[0] for g in EXPECTED])
    assert list(batches) == ['2w1','2w2','2w3','2w4']
    cache_genes = np.load(CACHE/'genes.npy').astype(str)
    np.testing.assert_array_equal(genes, cache_genes)
    meta = pd.read_csv(CACHE/'cell_metadata.csv')
    assert len(meta) == 4347 and meta.perturb_captured.all() and not meta.predicted_doublet.any()
    assert meta.is_cm.dtype == bool
    np.testing.assert_array_equal(meta.is_cm.to_numpy(), meta.celltype.eq('c12_Tnnt2').to_numpy())
    X = sparse.load_npz(CACHE/'cells.npz')
    assert X.shape == (len(meta), len(genes))
    wanted = meta.perturb_identity.isin(['Calr','NT_73']).to_numpy()
    cells = X[np.flatnonzero(wanted)][:,gi].toarray().astype(float)
    sub = meta.loc[wanted].reset_index(drop=True)
    del X
    truth, lo, hi, module_lo, module_hi = [], [], [], [], []
    counts, annotations = [], []
    rng = np.random.default_rng(SEED)
    for j,batch in enumerate(batches):
        isbatch = sub.sample_batch.eq(batch)
        a = cells[isbatch & sub.perturb_identity.eq('Calr')]
        c = cells[isbatch & sub.perturb_identity.eq('NT_73')]
        assert len(a) == [14,13,12,13][j] and len(c) == [27,59,26,30][j]
        delta = a.mean(0)-c.mean(0)
        stored = np.load(prep/f'batch_{j}/delta.npy', mmap_mode='r')[ti,gi]
        np.testing.assert_allclose(delta, stored, rtol=1e-12, atol=1e-12)
        draws = a[rng.integers(0,len(a),size=(BOOT,len(a)))].mean(1) - c[rng.integers(0,len(c),size=(BOOT,len(c)))].mean(1)
        lower,upper = np.quantile(draws,[.025,.975],axis=0)
        ml,mh = np.quantile(draws.mean(1),[.025,.975])
        truth.append(delta);lo.append(lower);hi.append(upper);module_lo.append(float(ml));module_hi.append(float(mh));counts.append((len(a),len(c)))
        cm_a = sub.loc[isbatch & sub.perturb_identity.eq('Calr'),'is_cm'].astype(bool)
        cm_c = sub.loc[isbatch & sub.perturb_identity.eq('NT_73'),'is_cm'].astype(bool)
        annotations.append(dict(batch=str(batch),target='Calr',target_n=len(a),nt_n=len(c),annotated_c12_Tnnt2_target_n=int(cm_a.sum()),annotated_c12_Tnnt2_nt_n=int(cm_c.sum()),annotated_target_fraction=float(cm_a.mean()),annotated_nt_fraction=float(cm_c.mean()),auxiliary_only=True))
    truth,lo,hi = map(np.asarray,[truth,lo,hi])
    observed = truth.mean(1)
    audit = {'prediction_seal_sha256':sha(res/'PREDICTIONS_SEALED.json'), 'source_prediction_seal':seal,
             'axis_sha256':sha(prep/'axis.npz'), 'fixed_programs_sha256':sha(prep/'programs.json'),
             'cache_meta_sha256':sha(CACHE/'cell_metadata.csv'), 'calr_cell_truth_matches_prepared_truth':True,
             'verified_used_prediction_hashes':{}}
    measurement = []
    for j,b in enumerate(batches):
        for k,g in enumerate(EXPECTED):
            measurement.append(dict(target='Calr',batch=str(b),gene=g,target_n=counts[j][0],nt_n=counts[j][1],observed=float(truth[j,k]),cell_sampling_ci_low=float(lo[j,k]),cell_sampling_ci_high=float(hi[j,k]),measurement_direction='positive_supported' if lo[j,k]>0 else 'negative_supported' if hi[j,k]<0 else 'uncertain'))
    pd.DataFrame(measurement).to_csv(result/'Calr_gene_measurement.csv',index=False)
    pd.DataFrame(annotations).to_csv(result/'Calr_annotation_auxiliary.csv',index=False)
    modules, summary, all_genes = [], [], []
    pred_by_threshold = {}
    for threshold in [5,10]:
        folder = res/f'n{threshold}'
        vp = folder/'valid.npy'; assert sha(vp) == seal['prediction_sha256'][str(vp.relative_to(res))]
        valid = np.load(vp).astype(bool); assert valid[ti].all()
        pred_by_threshold[threshold] = {}
        wide = [dict(threshold=threshold,**r) for r in measurement]
        for method, name in METHODS.items():
            path = folder/(name+'.npy'); digest = sha(path)
            assert digest == seal['prediction_sha256'][str(path.relative_to(res))]
            audit['verified_used_prediction_hashes'][str(path)] = digest
            a = np.load(path,mmap_mode='r')
            pred = np.asarray(a[ti][:,gi]).copy()
            assert pred.shape == (4,8) and np.isfinite(pred).all()
            all_module = np.asarray(a[:,:,gi]).mean(2)
            pred_by_threshold[threshold][method] = pred
            for j,b in enumerate(batches):
                # Use the same reduction for the reference and ranked array;
                # independent floating-point sums can otherwise rank Calr above itself.
                score = float(all_module[ti,j]);rank_min,rank_max = rank_interval(all_module[valid[:,j],j],score)
                assert 1 <= rank_min <= rank_max <= int(valid[:,j].sum())
                errs = pred[j]-truth[j]
                supported = (lo[j]>0)|(hi[j]<0)
                matches = np.sign(pred[j])==np.sign(truth[j])
                modules.append(dict(threshold=threshold,target='Calr',batch=str(b),method=method,target_n=counts[j][0],nt_n=counts[j][1],n_ranked_targets=int(valid[:,j].sum()),module_prediction=score,module_observed=float(observed[j]),module_cell_ci_low=module_lo[j],module_cell_ci_high=module_hi[j],module_direction_match=bool(np.sign(score)==np.sign(observed[j])),module_rank_min=rank_min,module_rank_max=rank_max,module_error=score-float(observed[j]),module_absolute_error=abs(score-float(observed[j])),module_squared_error=(score-float(observed[j]))**2,eight_gene_direction_matches=int(matches.sum()),eight_gene_mae=float(np.abs(errs).mean()),eight_gene_rmse=float(np.sqrt(np.mean(errs**2))),measurement_supported_gene_n=int(supported.sum()),measurement_supported_gene_direction_matches=int(matches[supported].sum()),predictions_within_measurement_interval=int(((pred[j]>=lo[j])&(pred[j]<=hi[j])).sum())))
                for k in range(8):
                    row = wide[j*8+k]
                    row[method+'_prediction'] = float(pred[j,k])
                    row[method+'_error'] = float(errs[k])
                    row[method+'_abs_error'] = float(abs(errs[k]))
                    row[method+'_direction_match'] = bool(matches[k])
                    row[method+'_within_measurement_ci'] = bool(lo[j,k] <= pred[j,k] <= hi[j,k])
            errors = pred-truth
            supported = (lo>0)|(hi<0)
            summary.append(dict(threshold=threshold,method=method,n_held_batches=4,n_fixed_genes=8,n_gene_batch_entries=32,module_mse=float(np.mean((pred.mean(1)-observed)**2)),module_mae=float(np.mean(abs(pred.mean(1)-observed))),eight_gene_batch_mse=float(np.mean(errors**2)),eight_gene_batch_mae=float(np.mean(abs(errors))),gene_batch_direction_matches=int(np.sum(np.sign(pred)==np.sign(truth))),measurement_supported_entries=int(supported.sum()),supported_direction_matches=int(np.sum((np.sign(pred)==np.sign(truth))[supported])),predictions_within_measurement_interval=int(np.sum((pred>=lo)&(pred<=hi)))))
        for row in wide:
            row['KiPERT_abs_error_minus_source_mean'] = row['KiPERT_OT_abs_error']-row['source_mean_abs_error']
            row['KiPERT_abs_error_minus_noOT'] = row['KiPERT_OT_abs_error']-row['confidence_noOT_abs_error']
        assert len(wide) == 32
        pd.DataFrame(wide).to_csv(result/f'Calr_four_batches_eight_genes_n{threshold}.csv',index=False)
        all_genes.extend(wide)
    audit['Calr_predictions_n5_equal_n10'] = {m:bool(np.array_equal(pred_by_threshold[5][m],pred_by_threshold[10][m])) for m in METHODS}
    pd.DataFrame(modules).to_csv(result/'Calr_module_by_batch.csv',index=False)
    pd.DataFrame(summary).to_csv(result/'Calr_summary.csv',index=False)
    pd.DataFrame(all_genes).to_csv(result/'Calr_all_thresholds_gene_predictions.csv',index=False)
    dump(result/'integrity.json',audit)
    dump(result/'COMPLETE.json',dict(status='COMPLETE',target='Calr',fixed_genes=EXPECTED,thresholds=[5,10],n_gene_rows_per_threshold=32,methods=list(METHODS),old_predictions_modified=False,script_sha256=sha(__file__)))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(pd.DataFrame(modules).query('threshold==5').to_string(index=False),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
