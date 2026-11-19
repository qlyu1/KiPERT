from common import *
from scipy import sparse
import csv,gzip,shutil
import pandas as pd

E=R/'experiments/uc_10context_8prior_20261031/priors'
S=R/'experiments/uc_state_specificity_20261101/priors'
def main():
 assert not (ROOT/'PROTOCOL.json').exists(),'Protocol already frozen'
 inputs={};sets={};hgnc=Path('@STUDY@/uc_fig2_extension_20260930/assets/priors/hgnc_complete_set.txt');inputs[str(hgnc)]=sha(hgnc)
 current={};alias={};entrez={}
 for row in csv.DictReader(open(hgnc),delimiter='\t'):
  g=row['ensembl_gene_id'].split('.')[0]
  if not g.startswith('ENSG'):continue
  current[row['symbol']]=g
  if row['entrez_id']:entrez[row['entrez_id']]=g
  for n in [row['symbol']]+row['prev_symbol'].split('|')+row['alias_symbol'].split('|'):
   if n:alias.setdefault(n,set()).add(g)
 def resolve(s):
  if s.startswith('ENSG'):return s.split('.')[0]
  if s in current:return current[s]
  a=alias.get(s,set());return next(iter(a)) if len(a)==1 else None
 for name in ['Reactome','GO_BP','CORUM','CollecTRI']:
  p=E/(name+'_sets.json');inputs[str(p)]=sha(p);sets[name]={k:set(v) for k,v in json.load(open(p))['sets'].items()}
 sets.update(GO_MF={},GO_CC={},HPO={},WikiPathways={})
 p=E/'GO_human.gaf.gz';inputs[str(p)]=sha(p)
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('!'):continue
   a=line.rstrip().split('\t')
   if len(a)<15 or a[8] not in ['F','C'] or 'NOT' in a[3].split('|') or a[6]=='ND' or a[12]!='taxon:9606':continue
   g=resolve(a[2])
   if g:sets['GO_MF' if a[8]=='F' else 'GO_CC'].setdefault(a[4],set()).add(g)
 p=S/'HPO_genes_to_phenotype.txt';inputs[str(p)]=sha(p)
 with open(p) as f:
  for row in csv.DictReader((x for x in f if not x.startswith('#')),delimiter='\t'):
   g=resolve(row['gene_symbol']) or entrez.get(row.get('ncbi_gene_id',''))
   if g:sets['HPO'].setdefault(row['hpo_id'],set()).add(g)
 p=S/'WikiPathways.gmt';inputs[str(p)]=sha(p)
 for line in open(p):
  a=line.rstrip().split('\t');gs={entrez.get(x) or resolve(x) for x in a[2:]};gs.discard(None);sets['WikiPathways'][a[0]]=gs
 sets={n:{t:gs for t,gs in ss.items() if len(gs)>=2} for n,ss in sets.items()}
 p=E/'HPA_features.npz';inputs[str(p)]=sha(p);hpa=np.load(p);hi={g:i for i,g in enumerate(hpa['genes'])};coverage=[]
 for ds in DATASETS:
  dest=ROOT/'priors'/ds;dest.mkdir(parents=True,exist_ok=True);p=R/'assets/prior_banks'/(ds+'.npz');inputs[str(p)]=sha(p);bank=np.load(p);targets=bank['components'].astype(str);idx={g:i for i,g in enumerate(targets)};np.save(dest/'targets.npy',targets)
  for name,key in [('DepMap','depmap'),('GenePT','genept'),('STRING','string')]:
   x=bank['x'+key];mask=bank[key+'_present'];np.savez_compressed(dest/(name+'.npz'),x=x,present=mask);coverage.append(dict(dataset=ds,view=name,total=len(targets),covered=int(mask.sum())))
  for name,ss in sets.items():
   terms=sorted(ss);rr=[];cc=[]
   for j,t in enumerate(terms):
    for g in ss[t]:
     if g in idx:rr.append(idx[g]);cc.append(j)
   x=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(targets),len(terms)));sparse.save_npz(dest/(name+'.npz'),x)
   np.savez_compressed(dest/(name+'_metadata.npz'),terms=np.array(terms),public_set_sizes=np.array([len(ss[t]) for t in terms]));coverage.append(dict(dataset=ds,view=name,total=len(targets),covered=int((x.getnnz(1)>0).sum())))
  x=np.zeros((len(targets),len(hpa['tissues'])));mask=np.array([g in hi for g in targets])
  for i,g in enumerate(targets):
   if g in hi:x[i]=np.log1p(hpa['expression'][hi[g]])
  np.savez_compressed(dest/'HPA.npz',x=x,present=mask);coverage.append(dict(dataset=ds,view='HPA',total=len(targets),covered=int(mask.sum())))
 pd.DataFrame(coverage).to_csv(ROOT/'prior_coverage.tsv',sep='\t',index=False)
 old={}
 for ds in DATASETS:
  for seed in SEEDS:
   base=R/'data/formal'/ds/f'split{seed}'
   for n in ['READY.json','query/metadata.tsv','genes.tsv']:old[str(base/n)]=sha(base/n)
   for p in (R/'runs/formal'/ds/f'split{seed}').rglob('predictions_sealed.npz'):old[str(p)]=sha(p)
 for p in (R/'evaluation/formal').glob('*.tsv'):old[str(p)]=sha(p)
 for p in (R/'figures/formal').iterdir():
  if p.is_file():old[str(p)]=sha(p)
 save(ROOT/'ORIGINAL_FILES_SHA256.json',old)
 protocol=dict(version=VERSION,utc=now(),status='FROZEN_BEFORE_NEW_VALIDATION_AND_TEST',task='Same-context unseen single-target Figure 2; no cross-context responses',datasets=DATASETS,splits=SEEDS,source_model='equal12_512_identity0_observedOnly_noShrink_exactOT128_adaptiveT_unclipped',adaptation='Use 12 equal-weight public-prior views in the within-context single-condition joint kernel regression. No source-observed point copying, cross-context OT or source gating; those cannot predict a never-measured target in the original task.',views=VIEWS,weights='1/12 without missing-view renormalization; all-absent query uses training mean and is flagged',max_rank=512,rank_rule='Train-only eigenspace/Nystrom extension; numerical rank capped at 512, no invented dimensions. Diagonal residual retained.',transforms='Dense prior centering/scaling fitted on current fit targets only; first four membership priors inverse sqrt training-frequency; GO_MF/CC,HPO,WikiPathways inverse sqrt public set sizes as latest UC. Normalize present rows. Selection uses training only; final refit transforms use train+validation.',candidates=CONFIGS,head='Original singles-only UC joint KRR algebra with unpenalized intercept; optional original training-DEG response subspace; no double labels',selection='Validation target-macro context-averaged expression RMSE; lowest candidate index breaks ties',refit='Train+validation; frozen output gene axis and control set; no test responses',nonnegative_projection=True,nonnegative_reason='Retained from original Figure 2 within-context protocol; no upper clipping',ablations=['DepMap','STRING','GenePT'],ablation_rule='Remove each prior, reweight remaining 11 equally, keep full-model selected alpha and response transform; no ablation retuning',old_methods='All other model predictions are reused unchanged; old UC retained as UC_old',evaluation='Original Figure 2 evaluator, same common/full cohorts, axes, control reference and metrics; repeated splits aggregated within target before 10000 target-cluster bootstrap',old_new_primary_test='Paired RMSE difference on common cohort; Holm over five datasets; other metrics descriptive with paired CIs',public_prior_inputs_sha256=inputs,source_code_sha256={p.name:sha(p) for p in (ROOT/'src').glob('*.py')},test_outcomes_used_for_adaptation=False,known_limitation='Historical public benchmark, not a new blind/prospective test; version comparison bundles prior/kernel changes, not a pure architecture causal ablation')
 save(ROOT/'PROTOCOL.json',protocol);log(stage='PREPARED',version=VERSION,coverage_rows=len(coverage))
if __name__=='__main__':main()
