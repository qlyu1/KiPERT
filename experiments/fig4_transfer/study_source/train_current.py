from current_engine import *
import argparse,resource

def main(fold):
 guard(fold);d=Dataset(fold);out=ROOT/'results'/fold;assert not (out/'PREDICTIONS_SEALED.json').exists();start=time.time();codes={p.name:sha(p) for p in (ROOT/'src').glob('*.py')};ph=sha(ROOT/'protocol.json');bank=ObservedBank(d);maps=Maps(d);post=bank.all_sources(0.);cv=validation_weights(d);temps=json.load(open(ROOT/'protocol.json'))['temperature_grid'];records=[dict(id=MAIN,temperature=t,cases=[]) for t in temps];design=[]
 for held in d.sources:
  pool=[s for s in d.sources if s!=held];q=d.meta[held]['targets'];parts=subset(d,post,pool,q);known=np.stack([parts[s]['known'] for s in pool]);var=np.stack([parts[s]['var'] for s in pool]);truth=np.asarray(d.y[held],float);panel=panel_for(held,q);mapped=maps.responses(pool,held,parts,0.);design.append(dict(held=held,pool=pool,n_queries=len(q),n_unsupported=int((~known.any(0)).sum()),panel_candidates=panel['candidates'],panel_queries=panel['queries']))
  for r in records:
   w=stable_gate(d,pool,held,var,known,r['temperature'],False,True);delta=mix(mapped,w);np.testing.assert_array_equal(delta[~known.any(0)],np.zeros((int((~known.any(0)).sum()),len(d.G))));r['cases'].append(dict(target=held,**source_score(delta,truth,panel)))
  del mapped,delta,parts,known,var,truth;gc.collect();log(stage='SOURCE_VALIDATION_READY',fold=fold,held=held)
 for r in records:r['weighted']=aggregate_weighted(r['cases'],cv['weights']);r['unweighted']=aggregate(r['cases'])
 baseline=next(r for r in records if r['temperature']==.25);cap=1.05*baseline['weighted']['nmse'];selected=choose(records,cap);methods={BASE:current_recipe(.25),MAIN:current_recipe(selected['temperature'])};save(out/'source_temperature_grid.json',records);save(out/'validation_design.json',design)
 frozen=dict(status='FROZEN',utc=now(),selection_weights=cv,selected=selected,baseline=baseline,nmse_cap=cap,methods=methods,primary_method=MAIN,outer_target_responses_used=False,protocol_sha256=ph,code_sha256=codes);save(out/'SOURCE_SELECTION_FROZEN.json',frozen);log(stage='TEMPERATURE_FROZEN',fold=fold,temperature=selected['temperature'],score=selected['weighted'],baseline=baseline['weighted'])
 q=d.meta[fold]['targets'];parts=subset(d,post,d.sources,q);predhash={};diags=[]
 for method,recipe in methods.items():
  pred,w,known,var=current_prediction(d,maps,parts,d.sources,fold,recipe['temperature']);f=out/(method+'_mean.npy');np.save(f,pred);predhash[f.name]=sha(f);np.savez_compressed(out/(method+'_weights.npz'),targets=q,sources=np.array(d.sources),weights=w,source_measured=known,relative_posterior_variance=var)
  save(out/(method+'_recipe.json'),dict(**recipe,identity_strength=0.,gene_dimension=512,prior_weight=1/12,ot_dimension=128,rho=1.,source_models={s:str(bank.model_path(s,0.)) for s in d.sources},shared_gene_basis=str(ROOT/'shared_gene_basis.npz'),basis_sha256=bank.basis_hash,source_selection=selected if method==MAIN else None))
  diags.append(dict(method=method,temperature=recipe['temperature'],unsupported_targets=int((~known.any(0)).sum()),unknown_source_weight=float(w[~known].sum()),raw_min=float(pred.min()),raw_max=float(pred.max()),negative_entries=int((pred<0).sum()),above14_entries=int((pred>14).sum()),mean_source_weights={s:float(w[i].mean()) for i,s in enumerate(d.sources)}));del pred;gc.collect()
 np.savez_compressed(out/'axes.npz',targets=q,genes=d.G,control=d.meta[fold]['control'],sources=np.array(d.sources),source_coverage=known);save(out/'prediction_diagnostics.json',diags);save(out/'no_shrink_maps.json',maps.audit)
 artifacts={str(ROOT/'shared_gene_basis.npz'):sha(ROOT/'shared_gene_basis.npz')}
 for s in d.sources:
  for n in ['fit.json','fit_arrays.npz','coefficients.npy']:
   p=bank.model_path(s,0.)/n;artifacts[str(p)]=sha(p)
 for pair in maps.exact.audit:
  p=ROOT/'cache/exact_maps'/pair/'basis.npz';artifacts[str(p)]=sha(p)
 for pair in maps.audit:
  for n in ['map.npz','READY.json']:
   p=ROOT/'cache/no_shrink_maps'/pair/n;artifacts[str(p)]=sha(p)
 for n,v in codes.items():assert sha(ROOT/'src'/n)==v
 assert sha(ROOT/'protocol.json')==ph
 save(out/'PREDICTIONS_SEALED.json',dict(status='SEALED',utc=now(),fold=fold,code_sha256=codes,protocol_sha256=ph,source_selection_sha256=sha(out/'SOURCE_SELECTION_FROZEN.json'),prediction_sha256=predhash,artifact_sha256=artifacts,elapsed_seconds=time.time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));log(stage='FOLD_SEALED',fold=fold)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('fold',choices=['k562','hepg2','rpe1','jurkat']);main(p.parse_args().fold)
