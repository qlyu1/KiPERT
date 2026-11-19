from current_engine import *
import argparse

def run(fold,output):
 guard(fold);r=ROOT/'results'/fold;sel=json.load(open(r/'SOURCE_SELECTION_FROZEN.json'));d=Dataset(fold);post=ObservedBank(d).all_sources(0.);q=d.meta[fold]['targets'];p,w,k,v=current_prediction(d,Maps(d),subset(d,post,d.sources,q),d.sources,fold,sel['selected']['temperature']);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);np.save(output,p);log(status='PREDICTED',fold=fold,temperature=sel['selected']['temperature'],imputation=False,covariance_shrinkage=0.,output_clip=None,output=str(output))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('fold',choices=['k562','hepg2','rpe1','jurkat']);ap.add_argument('--output',required=True);a=ap.parse_args();run(a.fold,a.output)
