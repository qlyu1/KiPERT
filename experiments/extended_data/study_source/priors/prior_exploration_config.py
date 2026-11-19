"""Outcome-independent definitions for the next KiPERT prior exploration."""
import json
import numpy as np
from pathlib import Path
V=['DepMap','GenePT','STRING','Reactome','GO_BP','CORUM','CollecTRI','HPA','GO_MF','GO_CC','HPO','WikiPathways']
SEED=2026111702
def configs():
 rng=np.random.default_rng(SEED);sizes=[2,4,6,8,10];orders=[];used={m:set() for m in sizes}
 while len(orders)<2:
  order=rng.permutation(12);sets={m:[tuple(sorted(np.roll(order,-j)[:m])) for j in range(12)] for m in sizes}
  if any(len(set(sets[m]))!=12 or used[m]&set(sets[m]) for m in sizes):continue
  orders.append(order.tolist())
  for m in sizes:used[m].update(sets[m])
 out=[dict(name='full',family='full',active=list(range(12)),beta=.5),dict(name='centroid',family='centroid',active=[],beta=.5)]
 for m in sizes:
  for cycle,order in enumerate(orders):
   for j in range(12):out.append(dict(name=f'subset_m{m:02d}_c{cycle}_r{j:02d}',family='subset',active=sorted(np.roll(order,-j)[:m].tolist()),beta=.5,size=m,cycle=cycle,rotation=j))
 for name,ids in [('dense4',[0,1,2,7]),('annotation8',[3,4,5,6,8,9,10,11])]:out.append(dict(name=name,family='representation_group',active=ids,beta=.5))
 for family in ['joint_shuffle','independent_shuffle']:
  for j in range(10):out.append(dict(name=f'{family}_{j:02d}',family=family,active=list(range(12)),beta=.5,replicate=j))
 assert len(out)==144 and len({x['name'] for x in out})==144
 for m in sizes:
  a=[x for x in out if x['family']=='subset' and x['size']==m];assert len({tuple(x['active']) for x in a})==24;assert np.array_equal(np.bincount([v for x in a for v in x['active']],minlength=12),np.full(12,2*m))
 return out,orders
def protocol():
 cfg,orders=configs()
 return dict(date='2026-11-17',status='DEFINED_BEFORE_NEW_EXPLORATION_OUTCOMES',study='Retrospective exploratory prior analyses; not prospective preregistration',output_root='@STUDY@/kipert_prior_exploration_20261117',existing_root='@STUDY@/kipert_supplement_20261117',datasets=['S3','S4','S5','S6','S7'],splits=[101,202,303],views=V,configurations=cfg,subset_orders=orders,seed=SEED,questions=['Does true gene-prior correspondence add information beyond matched kernel structure and coverage?','How does performance vary with the number and composition of priors?','Do continuous feature views and binary annotation views offer complementary information?','How do prior coverage and proximity to training targets relate to prior benefit?'],selection=dict(grid=[.01,.1,1],method='Original validation mean L2/cosine PDS and original tie breaks; refit fit+validation; separate fixed original full alpha'),permutation=dict(joint='Common gene permutation within exact per-view coverage strata; each view kernel row/column permuted after original training-only feature preprocessing; preserves per-view spectrum, coverage and cross-view correspondence',independent='Independent gene permutations for each view within its coverage strata; preserves each view spectrum and per-gene coverage; disrupts correspondence between views',strata='Use coverage signatures over all fit/refit context feature kernels so the same gene permutation preserves coverage in both stages; public prior features only',mapping='Derangements within strata of size >=2; singleton strata unchanged; report movable target fractions',alignment='Recompute alignment from training responses after permutation',replicates=10,interpretation='Conditional negative controls, no randomization p value claimed'),subset=dict(sizes=[2,4,6,8,10],models_per_size=24,design='Two random cyclic orders; 24 unique subsets per size; every view appears 2*m times; nested within order',reuse_endpoints='Existing single-view models m=1, leave-one-out models m=11 and full m=12; existing three-prior combination shown separately',interpretation='Sampled balanced composition panel, not exhaustive subset search or optimal model selection'),groups=dict(dense4=['DepMap','GenePT','STRING','HPA'],annotation8=['Reactome','GO_BP','CORUM','CollecTRI','GO_MF','GO_CC','HPO','WikiPathways']),stratification=dict(coverage='Number of covered views: 0-6, 7-9, 10-12; retain empty/small strata in sample-count reporting',proximity='Within-dataset tertiles of mean nearest-training similarity under the uniform original kernel; bins based only on priors, not outcomes',interpretation='Descriptive effect heterogeneity; observational, not causal'),endpoints=['cosine_PDS','DEG_AUPRC','MSE','shared-response Pearson'],statistics=dict(bootstrap=2000,seed=2026111702,unit='Paired target clusters across repeated splits/contexts, equal dataset weights',subset_variability='Composition spread shown separately from target bootstrap uncertainty; subsets and permutation replicates are not biological replicates',inference='Pointwise intervals, no confirmatory p values; report all outcomes',no_promotion='No test-based default changes or best-subset promotion'),resources=dict(workers=6,threads_per_worker=1,expected_float32_prediction_GiB=16.1,minimum_disk_free_GiB=7),invariants=['Preserve prior results and original manuscript','No test outcomes used in fitting, alignment, permutation or alpha selection','Keep original common-support scoring mask and full target-bank exclusion axis','Same alpha budget for all kernel variants','Full-model replay before accepting results','Seal predictions before test scoring'])
if __name__=='__main__':
 import sys
 Path(sys.argv[1]).write_text(json.dumps(protocol(),ensure_ascii=False,indent=2))
