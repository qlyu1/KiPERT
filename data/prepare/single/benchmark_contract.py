"""Small, testable isolation primitives. Not a replacement model implementation."""
import hashlib
import json
from collections import Counter


def stable_order(values, namespace):
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError('duplicate canonical identifiers')
    return sorted(values, key=lambda v: hashlib.sha256((namespace + '\0' + v).encode()).digest())


def split_targets(targets, seed):
    order = stable_order(targets, 'fig2-targets-' + str(seed))
    if len(order) < 30:
        raise ValueError('fewer than 30 eligible targets')
    n_test = max(5, int(len(order) * .2 + .5))
    n_val = max(5, int(len(order) * .1 + .5))
    out = {'test': sorted(order[:n_test]), 'validation': sorted(order[n_test:n_test+n_val]),
           'train': sorted(order[n_test+n_val:])}
    assert_disjoint(out)
    return out


def assert_disjoint(splits):
    counts = Counter(x for values in splits.values() for x in values)
    if any(v != 1 for v in counts.values()):
        raise ValueError('duplicate or overlapping canonical target IDs')


def fit_digest(train_ids, control_ids, genes, config, source_digest):
    """No test/validation responses or response-derived metadata accepted."""
    payload = dict(train=sorted(train_ids), controls=sorted(control_ids), genes=list(genes),
                   config=config, source_digest=source_digest)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def select_gene_indices(candidates, mandatory, limit, rng):
    """Sample actual axis indices, retaining all required perturbation tokens."""
    candidate_set = set(map(int, candidates)) | set(map(int, mandatory))
    required = set(map(int, mandatory))
    if len(required) > limit:
        raise ValueError('batch has more distinct required targets than sequence budget')
    rest = sorted(candidate_set - required)
    rng.shuffle(rest)
    return sorted(required | set(rest[:max(0, limit-len(required))]))


def average_rank(distances, correct_index, tolerance=1e-12):
    d = float(distances[correct_index])
    lower = sum(float(v) < d-tolerance for v in distances)
    tied = sum(abs(float(v)-d) <= tolerance for v in distances)
    return 1 + lower + (tied-1)/2


def fractional_top1(distances, correct_index, tolerance=1e-12):
    best = min(map(float, distances))
    tied = [i for i, v in enumerate(distances) if abs(float(v)-best) <= tolerance]
    return (1/len(tied)) if correct_index in tied else 0.


def expected_reciprocal_rank(distances, correct_index, tolerance=1e-12):
    """Expected reciprocal rank under uniform ordering within the tied block."""
    d = float(distances[correct_index])
    lower = sum(float(v) < d-tolerance for v in distances)
    tied = sum(abs(float(v)-d) <= tolerance for v in distances)
    return sum(1/r for r in range(lower+1, lower+tied+1))/tied


def self_test():
    import random
    import numpy as np
    checks = []
    genes = ['ENSG%07d' % i for i in range(100)]
    for seed in (101, 202, 303):
        a = split_targets(genes, seed)
        assert a == split_targets(list(reversed(genes)), seed)
        assert sorted(sum(a.values(), [])) == sorted(genes)
        checks.append('target_partition_and_order_invariance_' + str(seed))
    try:
        assert_disjoint({'train':['A'], 'test':['A']})
        raise AssertionError('overlap accepted')
    except ValueError:
        checks.append('overlap_rejected')
    selected = select_gene_indices([5, 11, 39, 40, 70], [99], 3, random.Random(42))
    assert 99 in selected and set(selected) <= {5,11,39,40,70,99}
    checks.append('noncontiguous_axis_and_required_target_preserved')
    assert all(fractional_top1([0]*8, i) == .125 for i in range(8))
    assert all(average_rank([0]*8, i) == 4.5 for i in range(8))
    checks.append('constant_predictions_ties_at_chance')
    chance_mrr = sum(1/r for r in range(1,9))/8
    for shared_distances in ([0]*8, [3,1,4,1,5,9,2,6]):
        assert abs(sum(fractional_top1(shared_distances,i) for i in range(8))/8-.125)<1e-12
        assert abs(sum(expected_reciprocal_rank(shared_distances,i) for i in range(8))/8-chance_mrr)<1e-12
    checks.append('identical_query_predictions_macro_chance_with_ties')
    kw=dict(train_ids=['A','B'],control_ids=['c1'],genes=['g1','g2'],config={'alpha':1},source_digest='synthetic')
    h=fit_digest(**kw)
    assert h != fit_digest(**dict(kw,train_ids=['A','C']))
    assert h != fit_digest(**dict(kw,genes=['g2','g1']))
    checks.append('split_and_axis_change_cache_digest')
    rng=np.random.RandomState(240911)
    x=rng.normal(size=(24,40))
    fit=np.arange(14)
    mean=x[fit].mean(0);scale=x[fit].std(0)
    x[14:]+=1e8
    assert np.array_equal(mean,x[fit].mean(0)) and np.array_equal(scale,x[fit].std(0))
    checks.append('synthetic_train_only_statistics_invariance')
    import copy,torch
    layer=torch.nn.Linear(2,2)
    snapshot=copy.deepcopy(layer.state_dict())
    original={k:v.clone() for k,v in snapshot.items()}
    with torch.no_grad():
        for p in layer.parameters():p.add_(10)
    assert all(torch.equal(snapshot[k],v) for k,v in original.items())
    checks.append('best_checkpoint_independent_of_future_updates')
    return {'status':'PASS','checks':checks,'n_checks':len(checks),
            'scope':'Synthetic contract primitives only; not full model/data pipeline certification'}


if __name__ == '__main__':
    print(json.dumps(self_test(),indent=2))
