from adaptive_engine import *
MAIN='UC_observed_no_shrink_adaptive';BASE='UC_observed_no_shrink_fixed025'

class ObservedBank(GPBank):
 def posterior(self,source,identity):
  assert identity==0.;d=self.d;assert source in d.sources and source!=d.fold;path=self.model_path(source,identity)
  # Existing independently fitted source model supplies the covariance factor.
  if not (path/'fit.json').exists():
   unused=super().posterior(source,identity);del unused
  ix=d.rows[source];tau=self.rem[ix];D=tau+np.maximum(d.noise[source]/d.scales[source],1e-6);meta=json.load(open(path/'fit.json'));fingerprints={name:sha(P/source/name) for name in ['delta.npy','noise_mean.npy','meta.npz']};assert meta['source_sha256']==fingerprints and meta['basis_sha256']==self.basis_hash and meta['identity']==0.;z=np.load(path/'fit_arrays.npz');np.testing.assert_array_equal(z['D'],D);chol=(z['chol'],True)
  rr=np.array([d.lookup[source].get(g,-1) for g in self.targets]);known=rr>=0;mu=np.zeros((len(rr),len(d.G)));mu[known]=np.asarray(d.y[source][rr[known]],float)
  # No GP posterior mean is used or generated during cached-model prediction.
  H=cho_solve(chol,self.Q.T,check_finite=False).T;q=np.einsum('ij,ij->i',self.Q,H);a=np.zeros(len(rr));a[known]=tau[rr[known]]/D[rr[known]];var=self.rem.copy();var[known]-=tau[rr[known]]**2/D[rr[known]];var+=(1-a)**2*q;assert var.min()>-1e-8;var=np.maximum(var,0)/(self.rem+np.einsum('ij,ij->i',self.Q,self.Q))
  assert not mu[~known].any();return dict(mu=mu,var=var,known=known,model_path=str(path))

def current_recipe(temperature):return dict(impute=False,precision=True,shrinkage=0.,temperature=temperature,output_clip=None)

def current_prediction(d,maps,parts,pool,target,temperature):
 for s in pool:assert not parts[s]['mu'][~parts[s]['known']].any()
 pred,w,known,var=assemble_prediction(d,maps,parts,pool,target,current_recipe(temperature));assert not w[~known].any();np.testing.assert_array_equal(pred[~known.any(0)],np.broadcast_to(d.meta[target]['control'],(int((~known.any(0)).sum()),len(d.G))))
 assert np.isfinite(pred).all();return pred,w,known,var

def verify_seals():
 for f in ['k562','hepg2','rpe1','jurkat']:
  base=ROOT if f in ['k562','hepg2'] else ROOT.parent/'uc_state_observed_noshrink_adaptive_20261102'
  r=base/'results'/f;s=json.load(open(r/'PREDICTIONS_SEALED.json'));assert sha(base/'protocol.json')==s['protocol_sha256'];assert sha(r/'SOURCE_SELECTION_FROZEN.json')==s['source_selection_sha256']
  for n,v in s['code_sha256'].items():assert sha(base/'src'/n)==v,n
  for n,v in s['prediction_sha256'].items():assert sha(r/n)==v,n
  for n,v in s['artifact_sha256'].items():assert sha(n)==v,n
