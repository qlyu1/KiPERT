from current_engine import *
import argparse

def run(fold):
 guard(fold);d=Dataset(fold);a=ObservedBank(d);b=GPBank(d);post={};parity={}
 for s in d.sources:
  got=a.posterior(s,0.);ref=b.posterior(s,0.);np.testing.assert_array_equal(got['known'],ref['known']);np.testing.assert_array_equal(got['mu'][got['known']],ref['mu'][got['known']]);np.testing.assert_allclose(got['var'],ref['var'],atol=1e-12,rtol=0);assert not got['mu'][~got['known']].any();parity[s]=dict(variance_max_abs_error=float(abs(got['var']-ref['var']).max()),unknown_mean_values_all_zero=True);post[s]=got;del ref
 out=ROOT/'results'/fold;sel=json.load(open(out/'SOURCE_SELECTION_FROZEN.json'));q=d.meta[fold]['targets'];parts=subset(d,post,d.sources,q);p,w,k,v=current_prediction(d,Maps(d),parts,d.sources,fold,sel['selected']['temperature']);expected=np.load(out/(MAIN+'_mean.npy'));np.testing.assert_array_equal(p,expected);save(out/'REPLAY_CHECKS.json',dict(status='PASS',utc=now(),gp_variance_parity=parity,output_clip=None,mean_replay_max_abs_error=0.,unknown_weight_sum=float(w[~k].sum()),unsupported_response_exactly_zero=True));log(stage='REPLAY_PASS',fold=fold)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('fold',choices=['k562','hepg2','rpe1','jurkat']);run(ap.parse_args().fold)
