from models import *
import argparse

def cases_for(d):
 cases=[]
 for held in d.sources:
  pool=[s for s in d.sources if s!=held];seen=set().union(*(set(d.meta[s]['targets']) for s in pool));q=np.array([g for g in d.meta[held]['targets'] if g in seen]);assert len(q)>=5
  cases.append(dict(target=held,pool=pool,queries=q.tolist(),n=len(q),weight=1/len(d.sources)))
 return cases

def batch_moments(pred,correction,weights,truth):
 # Compute all candidate gates in query blocks; no sources x genes x gates
 # tensor is retained across validation datasets.
 C=len(weights);N,G=truth.shape;den=max(float(np.mean(truth*truth)),1e-12);values=np.zeros((C,6))
 for lo in range(0,N,16):
  hi=min(lo+16,N);w=np.transpose(weights[:,:,lo:hi],(2,1,0));mu=np.matmul(np.ascontiguousarray(np.transpose(pred[:,lo:hi],(1,2,0))),w);ot=np.matmul(np.ascontiguousarray(np.transpose(correction[:,lo:hi],(1,2,0))),w);y=truth[lo:hi,:,None]
  values[:,0]+=(mu*mu).sum((0,1));values[:,1]+=(mu*y).sum((0,1));values[:,2]+=(ot*ot).sum((0,1));values[:,3]+=(ot*y).sum((0,1));values[:,4]+=(mu*ot).sum((0,1));values[:,5]+=corr(mu,np.broadcast_to(y,mu.shape),axis=1).sum(0)
 values[:,:5]/=N*G*den;values[:,5]/=N
 return [dict(A0=float(a),B0=float(b),A1=float(c),B1=float(e),Across=float(f),pcc=float(g),zero_mse=den) for a,b,c,e,f,g in values]

def select(d,cases,out):
 records=[]
 for c in cases:
  q=np.array(c['queries']);pool=c['pool'];y=np.asarray(d.y[c['target']][[d.lookup[c['target']][g] for g in q]],float);shape=(len(pool),len(q),len(d.G));pred=np.empty(shape,np.float32);delta=np.empty(shape,np.float32);var=np.empty(shape[:2]);known=np.empty(shape[:2],bool)
  for i,s in enumerate(pool):
   mu,v,k=d.predict(8,s,q);pred[i]=mu;delta[i]=d.correction(s,c['target'],mu);var[i]=v;known[i]=k
  weights=np.stack([d.gate(pool,c['target'],var,known,g) for g in GATES]);mom=batch_moments(pred,delta,weights,y);record=dict(**c,moments=mom);save(out/'validation_cases'/(c['target']+'.json'),record);records.append(record);del pred,delta,weights,y;gc.collect();log(stage='source_validation_completed',outer=d.fold,held=c['target'],n=c['n'])
 aggregate=[]
 for gi,gate in enumerate(GATES):
  m={k:sum(r['weight']*r['moments'][gi][k] for r in records) for k in ['A0','B0','A1','B1','Across','pcc']}
  for alpha in ALPHAS:
   A=max(m['A0']+2*alpha*m['Across']+alpha*alpha*m['A1'],0);B=m['B0']+alpha*m['B1'];gamma=float(np.clip(B/max(A,1e-30),0,2));aggregate.append(dict(gate_index=gi,gate=gate,alpha=alpha,gamma=gamma,objective=1+gamma*gamma*A-2*gamma*B,raw_objective=1+A-2*B,moments=m))
 uc=min([r for r in aggregate if r['alpha']==0],key=lambda r:(round(r['objective'],12),r['gate_index']));ot=min([r for r in aggregate if r['gate_index']==uc['gate_index']],key=lambda r:(round(r['objective'],12),r['alpha']));assert ot['objective']<=uc['objective']+1e-12
 return dict(UC=uc,UC_OT=ot,candidates=aggregate,n_validation_cases=len(cases),n_validation_targets=sum(c['n'] for c in cases))

def main(fold):
 guard(fold);d=Data(fold);out=ROOT/'results'/fold;out.mkdir(exist_ok=True,parents=True);assert not (out/'PREDICTIONS_SEALED.json').exists();t0=time.time();cases=cases_for(d);save(out/'validation_design.json',cases);selection=select(d,cases,out);save(out/'source_selection.json',selection);save(out/'SOURCE_SELECTION_FROZEN.json',dict(utc=now(),sha256=sha(out/'source_selection.json'),held_out_perturbation_responses_read=False));q=d.meta[fold]['targets'];ctrl=d.meta[fold]['control'];shape=(len(q),len(d.G));names=['UC','UC_OT','source_measured_mean','zero_response'];dest={n:np.lib.format.open_memmap(out/(n+'_mean.npy'),mode='w+',dtype=np.float64,shape=shape) for n in names};coverage=np.zeros((len(d.sources),len(q)),bool);clipped={n:0 for n in names}
 for lo in range(0,len(q),64):
  hi=min(lo+64,len(q));qq=q[lo:hi];sh=(len(d.sources),len(qq),len(d.G));pred=np.empty(sh,np.float32);correction=np.empty(sh,np.float32);var=np.empty(sh[:2]);known=np.empty(sh[:2],bool);raw=np.zeros(sh,np.float32)
  for i,s in enumerate(d.sources):
   mu,v,k=d.predict(8,s,qq);pred[i]=mu;correction[i]=d.correction(s,fold,mu);var[i]=v;known[i]=k;rr=np.array([d.lookup[s].get(g,-1) for g in qq]);raw[i,k]=d.y[s][rr[k]]
  coverage[:,lo:hi]=known
  for method in ['UC','UC_OT']:
   rec=selection[method];w=d.gate(d.sources,fold,var,known,rec['gate']);delta=rec['gamma']*(mix(pred,w)+rec['alpha']*mix(correction,w));mu=ctrl+delta;clipped[method]+=int(((mu<0)|(mu>14)).sum());dest[method][lo:hi]=np.clip(mu,0,14)
  average=raw.sum(0)/np.maximum(known.sum(0),1)[:,None]+ctrl;clipped['source_measured_mean']+=int(((average<0)|(average>14)).sum());dest['source_measured_mean'][lo:hi]=np.clip(average,0,14);dest['zero_response'][lo:hi]=ctrl
 for m in names:dest[m].flush()
 del dest;np.savez_compressed(out/'axes.npz',targets=q,genes=d.G,control=ctrl,source_coverage=coverage,sources=np.array(d.sources));save(out/'map_audit.json',d.map_audit);files=[out/(m+'_mean.npy') for m in names];save(out/'PREDICTIONS_SEALED.json',dict(utc=now(),status='SEALED',outer_fold=fold,source_contexts=d.sources,source_independent_fits=d.fit_audit,shared_basis_sha256=sha(P/'shared_gene_basis.npz'),selection_sha256=sha(out/'source_selection.json'),protocol_sha256=sha(ROOT/'protocol.json'),prediction_sha256={f.name:sha(f) for f in files},source_hashes={s:{name:sha(P/s/name) for name in ['delta.npy','noise_mean.npy','meta.npz','controls.npy']} for s in d.sources},source_code_sha256={f.name:sha(f) for f in (ROOT/'src').glob('*.py')},held_out_responses_read=False,n_targets=len(q),n_source_supported=int(coverage.any(0).sum()),n_clipped_mean_values=clipped,elapsed_seconds=time.time()-t0));log(stage='PREDICTIONS_SEALED',fold=fold,UC=selection['UC'],UC_OT=selection['UC_OT'],seconds=time.time()-t0)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('fold',choices=['rpe1','jurkat']);main(ap.parse_args().fold)
