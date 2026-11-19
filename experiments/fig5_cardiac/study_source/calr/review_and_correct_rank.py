"""Independent small-table verification and floating-point rank correction.

Use the previously sealed full query export's single reduction for reference and
all candidates. Never changes frozen predictions. All extracted Calr scores must
match the full query export within floating point tolerance before correction.
"""
import csv,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT.parents[1]/'d14_application/evaluation'

def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

queries=read(EVAL/'query_results.csv');modules=read(ROOT/'results/Calr_module_by_batch.csv')
changes=[]
for r in modules:
    q=[q for q in queries if all(q[k]==r[k] for k in ['threshold','method','batch']) and q['program']=='cardiac']
    focal=[q for q in q if q['target']=='Calr'];assert len(focal)==1
    score=float(focal[0]['prediction']);obs=float(focal[0]['observed'])
    assert math.isclose(score,float(r['module_prediction']),rel_tol=1e-12,abs_tol=1e-12)
    assert math.isclose(obs,float(r['module_observed']),rel_tol=1e-12,abs_tol=1e-12)
    ranks=(1+sum(float(q['prediction'])>score for q in q),sum(float(q['prediction'])>=score for q in q))
    assert 1<=ranks[0]<=ranks[1]<=len(q)
    if (int(r['module_rank_min']),int(r['module_rank_max']))!=ranks:
        changes.append(dict(threshold=r['threshold'],method=r['method'],batch=r['batch'],old=[int(r['module_rank_min']),int(r['module_rank_max'])],new=list(ranks)))
    r['module_rank_min'],r['module_rank_max']=ranks
    assert int(r['n_ranked_targets'])==len(q)
write(ROOT/'results/Calr_module_by_batch.csv',modules)

for threshold in [5,10]:
    genes=read(ROOT/f'results/Calr_four_batches_eight_genes_n{threshold}.csv')
    assert len(genes)==32 and len({(r['batch'],r['gene']) for r in genes})==32
    for method in ['KiPERT_OT','source_mean','confidence_noOT']:
        for b in ['2w1','2w2','2w3','2w4']:
            rows=[r for r in genes if r['batch']==b];assert len(rows)==8
            m=[r for r in modules if int(r['threshold'])==threshold and r['method']==method and r['batch']==b][0]
            errors=[float(r[method+'_prediction'])-float(r['observed']) for r in rows]
            matches=sum((float(r[method+'_prediction'])>0)-(float(r[method+'_prediction'])<0)==(float(r['observed'])>0)-(float(r['observed'])<0) for r in rows)
            assert matches==int(m['eight_gene_direction_matches'])
            assert math.isclose(statistics.mean(abs(e) for e in errors),float(m['eight_gene_mae']),rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(statistics.mean(float(r[method+'_prediction']) for r in rows),float(m['module_prediction']),rel_tol=1e-12,abs_tol=1e-12)
            for r,e in zip(rows,errors):
                assert math.isclose(e,float(r[method+'_error']),rel_tol=1e-12,abs_tol=1e-12)
                assert float(r['cell_sampling_ci_low'])<=float(r['cell_sampling_ci_high'])

record=dict(status='PASS',changes=changes,reason='Reference and comparison independently summed eight genes caused roundoff self-ranking; correct against one shared reduction from original complete query export.',
            prediction_and_outcome_values_changed=False,reference_query_export_sha256=sha(EVAL/'query_results.csv'),
            corrected_module_table_sha256=sha(ROOT/'results/Calr_module_by_batch.csv'),script_sha256=sha(Path(__file__)),
            verification='All 24 module predictions/truth match original query export to1e-12; all64 gene table rows, errors, means and directions independently checked; no selected genes or new endpoints.')
(ROOT/'review/independent_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
