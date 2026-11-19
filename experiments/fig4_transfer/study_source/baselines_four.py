from current_engine import *
import argparse

def main(fold):
 guard(fold);d=Dataset(fold);q=d.meta[fold]['targets'];total=np.zeros((len(q),len(d.G)));count=np.zeros(len(q),int)
 for s in d.sources:
  ix=np.array([d.lookup[s].get(g,-1) for g in q]);known=ix>=0;total[known]+=d.y[s][ix[known]];count+=known
 delta=total/np.maximum(count[:,None],1);out=ROOT/'results'/fold
 paths={}
 for name,pred in [('source_measured_mean',d.meta[fold]['control']+delta),('zero_response',np.broadcast_to(d.meta[fold]['control'],delta.shape).copy())]:
  p=out/(name+'_mean.npy');np.save(p,pred);paths[name]=dict(path=str(p),sha256=sha(p))
 save(out/'BASELINES_SEALED.json',dict(utc=now(),fold=fold,predictions=paths,source_mean='Equal average of raw observed source responses for this query; unmeasured sources excluded; zero response when no source measured.',outer_target_responses_used=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('fold',choices=['k562','hepg2']);main(a.parse_args().fold)
