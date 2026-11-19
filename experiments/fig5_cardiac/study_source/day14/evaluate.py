"""Evaluate sealed D14 predictions within held-out batches, keeping target clusters."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1];PREP=ROOT/'prepared';RES=ROOT/'results_transfer';OUT=ROOT/'evaluation';OUT.mkdir(exist_ok=True)
SEED=2026111231;BOOT=2000
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def dump(p,x):Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def rho(a,b):
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return np.nan
    return float(spearmanr(a,b).statistic)
def ci(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return np.quantile(a,[.025,.975]) if len(a) else [np.nan,np.nan]
def macro_rho(a,b,valid,take):
    vals=[]
    for j in range(valid.shape[1]):
        ix=take[valid[take,j]];v=rho(a[ix,j],b[ix,j])
        if np.isfinite(v):vals.append(v)
    return np.mean(vals) if vals else np.nan
def targetmean(x,valid):
    return np.divide(np.where(valid,x,0).sum(1),valid.sum(1),out=np.full(len(x),np.nan),where=valid.sum(1)>0)
def weighted_top_mean(observed,chosen,weights):
    vals=[]
    for j in range(chosen.shape[1]):
        w=weights*chosen[:,j]
        if w.sum()>0:vals.append(np.nansum(w*observed[:,j])/w.sum())
    return np.mean(vals) if vals else np.nan
def resampled_top_mean(predicted,observed,valid,take):
    vals=[]
    for j in range(valid.shape[1]):
        ix=take[valid[take,j]]
        if not len(ix):continue
        k=max(1,math.ceil(.2*len(ix)))
        selected=ix[np.argsort(-predicted[ix,j],kind='stable')[:k]]
        vals.append(np.mean(observed[selected,j]))
    return np.mean(vals) if vals else np.nan

assert (RES/'PREDICTIONS_SEALED.json').exists(),'All predictions must be frozen before labels are opened.'
seal=json.load(open(RES/'PREDICTIONS_SEALED.json'))
for path,digest in seal['prediction_sha256'].items():assert sha(RES/path)==digest,path
print('PREDICTION_HASHES_VERIFIED',flush=True)
axis=np.load(PREP/'axis.npz');targets=axis['targets'].astype(str);genes=axis['genes'].astype(str);batches=axis['batches'].astype(str);mask=axis['evaluation_gene_mask'].astype(bool)
N=len(targets);B=len(batches);G=len(genes)
programs=json.load(open(PREP/'programs.json'));pnames=list(programs);P=len(pnames)
cols={p:np.flatnonzero(np.isin(genes,programs[p]['used'])) for p in pnames}
truth=[np.load(PREP/f'batch_{j}/delta.npy',mmap_mode='r') for j in range(B)]
counts=np.column_stack([np.load(PREP/f'batch_{j}/n.npy') for j in range(B)])
observed=np.stack([np.column_stack([np.asarray(truth[j][:,cols[p]]).mean(1) for p in pnames]) for j in range(B)],axis=1)
print('OBSERVED_PROGRAMS_READY',observed.shape,flush=True)
obslo=np.full_like(observed,np.nan);obshi=np.full_like(observed,np.nan);cm=np.full((N,B),np.nan);cm_delta=np.full_like(cm,np.nan)
rng=np.random.default_rng(SEED);measurement=[]
for j in range(B):
    z=np.load(PREP/f'batch_{j}/evaluation_cells.npz');score=z['scores'];labels=z['target'].astype(str);cml=z['is_cm']
    assert list(z['program_names'])==pnames
    ctrl=score[labels=='NT_73'];cmctrl=cml[labels=='NT_73'].mean()
    bootctrl=ctrl[rng.integers(0,len(ctrl),size=(BOOT,len(ctrl)))].mean(1)
    for i,t in enumerate(targets):
        s=score[labels==t]
        if not len(s):continue
        cm[i,j]=cml[labels==t].mean();cm_delta[i,j]=cm[i,j]-cmctrl
        if len(s)>1:
            draws=s[rng.integers(0,len(s),size=(BOOT,len(s)))].mean(1)-bootctrl
            obslo[i,j],obshi[i,j]=np.quantile(draws,[.025,.975],axis=0)
        for p,name in enumerate(pnames):measurement.append(dict(target=t,batch=batches[j],program=name,n=len(s),nt_n=len(ctrl),observed=observed[i,j,p],cell_sampling_ci_low=obslo[i,j,p],cell_sampling_ci_high=obshi[i,j,p],observed_cm_fraction=cm[i,j],observed_cm_delta=cm_delta[i,j]))
    print('MEASUREMENT_INTERVALS_READY',j,flush=True)
pd.DataFrame(measurement).to_csv(OUT/'observed_programs.csv',index=False)

# File names below match the prediction manifest; no outcome-selected methods.
methods=['KiPERT_OT','source_mean','confidence_noOT','zero']
files={'KiPERT_OT':'KiPERT_OT_Cai_prior_adaptation','source_mean':'equal_source_delta_mean','confidence_noOT':'same_weight_no_OT','zero':'zero_control'}
queryrows=[];batchrows=[];summaryrows=[];contrastrows=[];selectionrows=[];candidate_rows=[];rankrows=[]
for threshold in [5,10]:
    D=RES/f'n{threshold}';valid=np.load(D/'valid.npy').astype(bool);idx=np.flatnonzero(valid.any(1));n=len(idx)
    assert np.array_equal(valid,(counts>=threshold)&((counts>=threshold).sum(1)[:,None]>=3))
    rng=np.random.default_rng(SEED+threshold);take=rng.choice(idx,size=(BOOT,n),replace=True)
    weights=np.stack([np.bincount(t,minlength=N) for t in take])
    predictions={};errors={};fullerrors={};chosen={}
    for method in methods:
        a=np.load(D/f'{files[method]}.npy',mmap_mode='r');assert a.shape==(N,B,G)
        assert np.isfinite(a[valid]).all()
        s=np.stack([np.asarray(a[:,:,cols[p]]).mean(2) for p in pnames],axis=2)
        predictions[method]=s;errors[method]=(s-observed)**2
        fullerrors[method]=np.full((N,B),np.nan)
        for j in range(B):
            ii=np.flatnonzero(valid[:,j]);fullerrors[method][ii,j]=((np.asarray(a[ii,j])[:,mask]-truth[j][ii][:,mask])**2).mean(1)
        chosen[method]=np.zeros((N,B),bool)
        if method!='zero':
            for j in range(B):
                ii=np.flatnonzero(valid[:,j]);k=max(1,math.ceil(.2*len(ii)));order=np.argsort(-s[ii,j,0],kind='stable');chosen[method][ii[order[:k]],j]=True
        for j in range(B):
            ii=np.flatnonzero(valid[:,j]);k=int(chosen[method][:,j].sum())
            for p,name in enumerate(pnames):
                batchrows.append(dict(threshold=threshold,method=method,batch=batches[j],program=name,n=len(ii),mse=errors[method][ii,j,p].mean(),rho=rho(s[ii,j,p],observed[ii,j,p]),direction=np.mean(np.sign(s[ii,j,p])==np.sign(observed[ii,j,p]))))
            rankrows.append(dict(threshold=threshold,method=method,batch=batches[j],n=len(ii),predicted_cardiac_vs_observed_cm_delta=rho(s[ii,j,0],cm_delta[ii,j]),measured_cardiac_vs_observed_cm_delta=rho(observed[ii,j,0],cm_delta[ii,j])))
            if k:
                cc=chosen[method][:,j]
                selectionrows.append(dict(threshold=threshold,method=method,batch=batches[j],n_candidates=len(ii),k=k,
                    selected_targets=';'.join(targets[cc]),observed_cardiac_mean=observed[cc,j,0].mean(),all_candidate_observed_cardiac_mean=observed[ii,j,0].mean(),
                    positive_fraction=np.mean(observed[cc,j,0]>0),positive_cell_sampling_ci_fraction=np.mean(obslo[cc,j,0]>0),
                    cm_fraction_mean=np.mean(cm[cc,j]),cm_delta_mean=np.mean(cm_delta[cc,j])))
            for i in ii:
                for p,name in enumerate(pnames):queryrows.append(dict(threshold=threshold,method=method,target=targets[i],batch=batches[j],program=name,n_query=int(counts[i,j]),prediction=s[i,j,p],observed=observed[i,j,p],cell_sampling_ci_low=obslo[i,j,p],cell_sampling_ci_high=obshi[i,j,p],squared_error=errors[method][i,j,p],cardiac_top20=bool(chosen[method][i,j]),observed_cm_fraction=cm[i,j],observed_cm_delta=cm_delta[i,j],expression_mse=fullerrors[method][i,j]))
        for p,name in enumerate(pnames):
            e=targetmean(errors[method][:,:,p],valid);mr=macro_rho(s[:,:,p],observed[:,:,p],valid,idx)
            rboot=[macro_rho(s[:,:,p],observed[:,:,p],valid,t) for t in take] if name=='cardiac' and method!='zero' else []
            lo,hi=ci(rboot)
            summaryrows.append(dict(threshold=threshold,method=method,program=name,n_targets=n,n_queries=int(valid.sum()),mse=float(np.mean(e[idx])),macro_batch_rho=mr,rho_ci_low=lo,rho_ci_high=hi,direction=float(np.mean(targetmean((np.sign(s[:,:,p])==np.sign(observed[:,:,p])).astype(float),valid)[idx])),full_expression_mse=np.mean(targetmean(fullerrors[method],valid)[idx])))
    # Model contribution to prediction, not a general benchmark leaderboard.
    for metric in ['cardiac','fibrosis','injury','expression']:
        p=pnames.index(metric) if metric!='expression' else None
        e=targetmean(fullerrors['KiPERT_OT'] if p is None else errors['KiPERT_OT'][:,:,p],valid)
        for base in ['source_mean','confidence_noOT','zero']:
            be=targetmean(fullerrors[base] if p is None else errors[base][:,:,p],valid)
            diff=be-e;bd=diff[take].mean(1);rel=1-e[take].mean(1)/be[take].mean(1);lo,hi=ci(bd);rl,rh=ci(rel)
            contrastrows.append(dict(threshold=threshold,metric=metric,baseline=base,n_targets=n,mse_reduction=diff[idx].mean(),ci_low=lo,ci_high=hi,relative_error_reduction=1-e[idx].mean()/be[idx].mean(),relative_ci_low=rl,relative_ci_high=rh,fraction_targets_improved=np.mean(diff[idx]>0)))
    sel_model=weighted_top_mean(observed[:,:,0],chosen['KiPERT_OT'],np.ones(N))
    for base in ['source_mean','confidence_noOT']:
        point=sel_model-weighted_top_mean(observed[:,:,0],chosen[base],np.ones(N))
        diffs=[resampled_top_mean(predictions['KiPERT_OT'][:,:,0],observed[:,:,0],valid,t)-resampled_top_mean(predictions[base][:,:,0],observed[:,:,0],valid,t) for t in take]
        lo,hi=ci(diffs)
        contrastrows.append(dict(threshold=threshold,metric='selected_cardiac_response',baseline=base,n_targets=n,mse_reduction=point,ci_low=lo,ci_high=hi))
    for i in idx:
        v=valid[i];sel=chosen['KiPERT_OT'][i]&(predictions['KiPERT_OT'][i,:,0]>0);ns=int(sel.sum())
        candidate_rows.append(dict(threshold=threshold,target=targets[i],eligible_query_batches=int(v.sum()),nominated_batches=ns,nominated_in=';'.join(batches[sel]),
            observed_positive_nominated_batches=int(np.sum(observed[i,sel,0]>0)),positive_sampling_ci_nominated_batches=int(np.sum(obslo[i,sel,0]>0)),
            all_nominated_observed_positive=bool(ns and np.all(observed[i,sel,0]>0)),repeated_nomination=bool(ns>=2),
            repeated_measurement_support=bool(ns>=2 and np.all(observed[i,sel,0]>0) and np.sum(obslo[i,sel,0]>0)>=2)))
    print('EVALUATED',threshold,n,int(valid.sum()),flush=True)

for name,rows in [('query_results',queryrows),('batch_results',batchrows),('program_summary',summaryrows),('paired_application_checks',contrastrows),('selected_candidates',selectionrows),('candidate_consistency',candidate_rows),('annotation_association',rankrows)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
S=pd.DataFrame(summaryrows);C=pd.DataFrame(contrastrows);BDF=pd.DataFrame(batchrows)
main=S.query('threshold==5 and method=="KiPERT_OT" and program=="cardiac"').iloc[0]
comp=C.query('threshold==5 and metric=="cardiac" and baseline=="source_mean"').iloc[0]
select=C.query('threshold==5 and metric=="selected_cardiac_response" and baseline=="source_mean"').iloc[0]
mb=BDF.query('threshold==5 and method=="KiPERT_OT" and program=="cardiac"')
gates=dict(coverage_sufficient=bool(main.n_targets>=20 and (mb.n>=10).sum()>=3),cardiac_prediction_added_value=bool(comp.ci_low>0),positive_ranking=bool(main.rho_ci_low>0),ranking_positive_in_at_least3_batches=bool((mb.rho>0).sum()>=3),candidate_selection_added_value=bool(select.ci_low>0))
dump(OUT/'decision.json',dict(status='COMPLETE',gates=gates,sufficient_for_model_specific_cardiac_application=all(gates.values()),
    interpretation='Frozen retrospective application assessment; no independent functional or animal-level validation. No conclusion of biological ineffectiveness follows from prediction failure.',
    bootstrap='Target clusters retaining all batches; fixed predictions, no model refitting or control resampling. Cell intervals are conditional measurement precision only.',
    candidates='Descriptive cross-batch nomination consistency only, not newly discovered or independently validated targets.',
    known_reference='Calr is a published reference, not a discovery.',
    scores_are_raw_gene_mean_changes=True,no_LOBO_prediction_averaging_before_correlation=True,original_prediction_seal=seal))
for path,digest in seal['prediction_sha256'].items():assert sha(RES/path)==digest,path
dump(OUT/'integrity.json',dict(predictions_unchanged=True,evaluation_code_sha256=sha(__file__),application_protocol_sha256=sha(ROOT/'protocol.json'),seed=SEED,bootstrap_draws=BOOT))
print(S.query('threshold==5 and program=="cardiac"').to_string(index=False),flush=True)
print(C.query('threshold==5 and metric in ["cardiac","selected_cardiac_response"]').to_string(index=False),flush=True)
print(json.dumps(gates),flush=True)
