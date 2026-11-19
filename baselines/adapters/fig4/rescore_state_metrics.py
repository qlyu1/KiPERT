"""Frozen PIE-cohort predictions, rescored with STATE/Cell-Eval definitions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS']:
    os.environ[k]='2'
import json, hashlib, warnings, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='An input array is constant')
B=Path('@STUDY@/uc_cross_context_transfer/experiments/pie_uc_state_benchmark_20261106')
P=Path('@STUDY@/uc_cross_context_transfer/experiments/pie_four_context_reproduction_20261106')
O=B/'state_metric_rescore_20261107';O.mkdir(exist_ok=True)
D=P/'downloads/official_data/preprocessed'
meta=json.loads((D/'meta.json').read_text()); genes=np.array(meta['genes'])
truth={k:np.load(D/(k+'.npy'),mmap_mode='r') for k in ['ctx_ids','pert_ids','delta_p','lfc_true','fdr','tested']}
assert truth['tested'].all()
lookup={(int(c),int(p)):i for i,(c,p) in enumerate(zip(truth['ctx_ids'],truth['pert_ids']))}
files={}; allframes=[]; summaries=[]; checks=[]
def freeze(path):
    path=Path(path)
    if str(path) not in files:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        files[str(path)]={'sha256':h.hexdigest(),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
for f in ['meta.json','ctx_ids.npy','pert_ids.npy','delta_p.npy','lfc_true.npy','fdr.npy','tested.npy']:freeze(D/f)
freeze(Path(__file__))
def corr(a,b):
    return float(spearmanr(a,b).statistic) if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else np.nan
def ranking(pred,real,perts):
    # Float64 distances, ordinal tie resolution by np.argsort, as Cell-Eval.
    pred=pred.astype('float64');real=real.astype('float64');n=len(perts)
    out={}
    geneidx={g:i for i,g in enumerate(genes)}
    def chunk(start):
        end=min(start+32,n); a=pred[start:end]
        l1=cdist(a,real,metric='cityblock');l2=cdist(a,real,metric='sqeuclidean')
        cos=cdist(a,real,metric='cosine');cos=np.nan_to_num(cos,nan=1.0)
        vals={}
        for label,dist in [('l1',l1),('l2',l2),('cosine',cos)]:
            for excluded in [False,True]:
                m=dist.copy()
                if excluded:
                    for j,i in enumerate(range(start,end)):
                        g=geneidx.get(perts[i])
                        if g is None:continue
                        if label=='l1':m[j]-=abs(pred[i,g]-real[:,g])
                        elif label=='l2':m[j]-=(pred[i,g]-real[:,g])**2
                        else:
                            dot=real@pred[i]-real[:,g]*pred[i,g]
                            den=np.sqrt(np.maximum(0,(pred[i]@pred[i]-pred[i,g]**2)*(np.sum(real*real,axis=1)-real[:,g]**2)))
                            m[j]=1-np.divide(dot,den,out=np.zeros_like(dot),where=den>0)
                order=np.argsort(m,axis=1)
                ranks=np.argmax(order==np.arange(start,end)[:,None],axis=1)
                vals[f'pds_{label}_'+('exclude_target' if excluded else 'all_genes')]=1-ranks/n
        return start,end,vals
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start,end,v in pool.map(chunk,range(0,n,32)):
            for k,a in v.items():
                if k not in out:out[k]=np.empty(n)
                out[k][start:end]=a
    return out

def oracle_check(perts,delta,rd,lfc,rlfc,q,rq,frame):
    import polars as pl
    from cell_eval._types._de import initialize_de_comparison
    from cell_eval.metrics._de import de_overlap_metric,DESpearmanLFC,DEDirectionMatch,DESpearmanSignificant,compute_pr_auc,compute_roc_auc
    from cell_eval.metrics._anndata import discrimination_score
    from types import SimpleNamespace
    ids=np.linspace(0,len(perts)-1,12,dtype=int);pp=perts[ids]
    def de(x,y):
        return pl.DataFrame({'target':np.repeat(pp,len(genes)),'feature':np.tile(genes,len(ids)),'log2_fold_change':x[ids].ravel(),'p_value':y[ids].ravel(),'fdr':y[ids].ravel()})
    pair=initialize_de_comparison(de(rlfc,rq),de(lfc,q))
    def check_dict(label,v):
        a=np.array([v.get(str(p),np.nan) for p in pp],dtype=float)
        b=frame.set_index('perturbation').loc[pp,label].to_numpy(dtype=float)
        assert np.array_equal(np.isnan(a),np.isnan(b)),(label,a,b)
        err=float(np.nanmax(abs(a-b))) if np.isfinite(a).any() else 0.
        assert err<1e-6,(label,err)
        return err
    errs={}
    for label,f in [('auprc',compute_pr_auc),('auroc',compute_roc_auc),('lfc_spearman_true_de',DESpearmanLFC()),('direction_true_de',DEDirectionMatch())]:
        errs[label]=check_dict(label,f(pair))
    for k in [None,50,100,200]:
        errs['overlap_'+str(k)]=check_dict('de_overlap_'+('N' if k is None else str(k)),de_overlap_metric(pair,k=k))
    class Bulk:
        def __init__(self,i):self.i=i
        def perturbation_effect(self,which,abs=False):return (rd if which=='real' else delta)[self.i].astype('float64')
    proxy=SimpleNamespace(perts=pp,genes=genes,iter_bulk_arrays=lambda embed_key=None:iter([Bulk(i) for i in ids]))
    small=ranking(delta[ids],rd[ids],pp)
    for metric in ['l1','l2','cosine']:
        vals=discrimination_score(proxy,metric=metric)
        err=float(max(abs(vals[str(p)]-small['pds_'+metric+'_exclude_target'][i]) for i,p in enumerate(pp)))
        assert err<1e-10,(metric,err)
        errs['pds_'+metric]=err
    return errs

for ctx in ['k562','hepg2','rpe1','jurkat']:
    split=P/'downloads/official_splits/replogle_wdataset/unseen_ctx'/ctx/'test.json';freeze(split)
    expected=json.loads(split.read_text())
    # Official split contains one held-out context.
    expected=set(expected['replogle.'+ctx])
    for model in ['PIE','STATE','UC','PMean','CMean']:
        t=time.time();print('START',ctx,model,flush=True)
        if model=='PIE':
            path=P/'runs'/ctx/'eval/unseen_ctx/predictions_best_auprc.parquet';freeze(path)
            tab=pq.read_table(path); dd=tab.to_pydict()
            md=json.loads(tab.schema.metadata[b'pie']);assert md['genes']['replogle']==list(genes)
            assert set(dd['context'])=={ctx}
            perts=np.array(dd['perturbation']);delta=np.array(dd['delta_p_pred'],dtype='float32');lfc=np.array(dd['lfc_pred'],dtype='float32');scores=np.array(dd['p_de'],dtype='float32');q=None;sig=scores>=.5;delta=np.maximum(delta.astype("float64"),-np.load(D/"ctrl_means.npy")[meta["context_to_id"][ctx]].astype("float64"))
        else:
            path=B/'runs/evaluation'/model/ctx/(ctx+'_adapter.npz');freeze(path)
            z=np.load(path);assert np.array_equal(z['genes'],genes)
            perts=z['perts'].astype(str);delta=z['delta_p'];lfc=z['lfc'].astype('float32');q=z['q_value'].astype('float32');sig=q<.05;scores=-np.log10(np.clip(q,1e-10,1))
        assert len(perts)==len(set(perts)) and set(perts)==expected,(ctx,model,len(perts),len(expected))
        order=np.argsort(perts);perts=perts[order];delta=delta[order];lfc=lfc[order];scores=scores[order];sig=sig[order]
        if q is not None:q=q[order]
        idx=[lookup[(meta['context_to_id'][ctx],meta['pert_to_id'][p])] for p in perts]
        rd=np.array(truth['delta_p'][idx]);rlfc=np.array(truth['lfc_true'][idx],dtype='float32');rq=np.array(truth['fdr'][idx]);rs=rq<.05
        assert all(np.isfinite(x).all() for x in [delta,lfc,scores,rd,rlfc,rq])
        rows=[]
        for i,p in enumerate(perts):
            a=delta[i].astype('float64');b=rd[i].astype('float64');a0=a-a.mean();b0=b-b.mean();den=np.linalg.norm(a0)*np.linalg.norm(b0)
            y=rs[i];sy=sig[i];ny=int(y.sum());npred=int(sy.sum())
            row={'model':model,'context':ctx,'perturbation':p,'n_genes':len(genes),'n_true_de':ny,'n_pred_de':npred,'pearson_delta':float(a0@b0/den) if den>0 else np.nan,'mse_delta':float(np.mean((a-b)**2)),'mae_delta':float(np.mean(abs(a-b))),'auprc':float(average_precision_score(y,scores[i])) if 0<ny<len(genes) else np.nan,'auroc':float(roc_auc_score(y,scores[i])) if 0<ny<len(genes) else np.nan,'lfc_spearman_true_de':corr(rlfc[i,y],lfc[i,y]),'direction_true_de':float(np.mean(np.sign(rlfc[i,y])==np.sign(lfc[i,y]))) if ny else np.nan}
            both=y&sy
            row['direction_overlap_de']=float(np.mean(np.sign(rlfc[i,both])==np.sign(lfc[i,both]))) if both.any() else np.nan
            tr=np.flatnonzero(y);pr=np.flatnonzero(sy)
            tr=tr[np.argsort(-abs(rlfc[i,tr]),kind='stable')];pr=pr[np.argsort(-abs(lfc[i,pr]),kind='stable')]
            for k in [None,50,100,200]:
                key='N' if k is None else str(k);ke=ny if k is None else min(k,ny)
                row['de_overlap_'+key]=len(set(tr[:ke])&set(pr[:ke]))/ke if ke and npred else 0.
                kp=npred if k is None else min(k,npred)
                row['de_precision_'+key]=len(set(tr[:kp])&set(pr[:kp]))/kp if kp and ny else 0.
            rows.append(row)
        frame=pd.DataFrame(rows)
        for k,v in ranking(delta,rd,perts).items():frame[k]=v
        frame['pds_l1_centered']=2*frame.pds_l1_exclude_target-1
        if model=='STATE':checks.append({'context':ctx,'model':model,'official_cell_eval_0.8.2_errors':oracle_check(perts,delta,rd,lfc,rlfc,q,rq,frame)})
        numerical=[c for c in frame.columns if c not in ['model','context','perturbation','n_genes']]
        summary={'model':model,'context':ctx,'n_conditions':len(frame),'n_genes':len(genes),'native_de_probability':model=='PIE'}
        for col in numerical:
            summary[col]=float(frame[col].mean());summary[col+'_valid_n']=int(frame[col].notna().sum())
        selected=frame.n_true_de>0
        summary['de_nsig_spearman']=corr(frame.loc[selected,'n_true_de'],frame.loc[selected,'n_pred_de'])
        summary['de_nsig_spearman_valid_n']=int(selected.sum())
        frame.to_csv(O/f'{model}_{ctx}_per_condition.csv',index=False)
        summaries.append(summary);allframes.append(frame)
        pd.DataFrame(summaries).to_csv(O/'per_context_metrics.csv',index=False)
        (O/'validation.json').write_text(json.dumps(checks,indent=2))
        print('DONE',ctx,model,round(time.time()-t,1),summary['pearson_delta'],summary['de_overlap_N'],flush=True)
per=pd.DataFrame(summaries);pd.concat(allframes).to_csv(O/'per_condition_metrics.csv',index=False)
cols=[c for c in per.select_dtypes(include='number').columns if not c.endswith('_valid_n') and c not in ['n_conditions','n_genes']]
mac=[]
for model in ['PIE','STATE','UC','PMean','CMean']:
    d=per[per.model==model];assert len(d)==4
    row={'model':model,'n_contexts':4,'n_conditions':int(d.n_conditions.sum()),'n_genes':6642}
    for c in cols:
        row[c]=float(d[c].mean());row[c+'_context_sd']=float(d[c].std(ddof=1));row[c+'_valid_contexts']=int(d[c].notna().sum())
    mac.append(row)
pd.DataFrame(mac).to_csv(O/'macro_metrics.csv',index=False)
for path,info in files.items():
    s=Path(path).stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns']),('input changed',path)
(O/'input_manifest.json').write_text(json.dumps(files,indent=2))
(O/'COMPLETE.json').write_text(json.dumps({'models':[m['model'] for m in mac],'n_contexts':4,'n_conditions_per_model':3712,'n_genes':6642,'training_performed':False,'source_protocol':'PIE unseen_ctx revision3','metric_implementation':'Cell-Eval 0.8.2; PIE native probability adaptation','input_files':len(files),'validation':'official API comparison, 12 conditions/context, STATE frozen DE arrays and 3 PDS distances'},indent=2))
print('COMPLETE',flush=True)
