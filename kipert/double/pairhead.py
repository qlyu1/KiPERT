import numpy as np


def parts(c):
    p = tuple(
        sorted(
            x.strip()
            for x in str(c).split("+")
            if x.strip() not in ("", "ctrl", "control")
        )
    )
    if len(p) not in (1, 2) or len(set(p)) != len(p):
        raise ValueError("Expected one or two distinct gene components")
    return p


def indices(components, conditions):
    index = {g: i for i, g in enumerate(components)}
    if len(index) != len(components):
        raise ValueError("Duplicate feature-bank component")
    try:
        return [tuple(index[g] for g in parts(c)) for c in conditions]
    except KeyError as exc:
        raise ValueError(
            "Component missing from explicit feature bank: " + str(exc)
        ) from exc


def pair_kernel(node, query_pairs, train_pairs, kind):
    qa = np.asarray([x[0] for x in query_pairs], int)
    qb = np.asarray([x[1] for x in query_pairs], int)
    ta = np.asarray([x[0] for x in train_pairs], int)
    tb = np.asarray([x[1] for x in train_pairs], int)
    ac = node[np.ix_(qa, ta)]
    ad = node[np.ix_(qa, tb)]
    bc = node[np.ix_(qb, ta)]
    bd = node[np.ix_(qb, tb)]
    if kind == "sum":
        return ac + ad + bc + bd
    if kind == "tensor":
        return 0.5 * (ac * bd + ad * bc)
    raise ValueError(kind)


def condition_kernel(
    node,
    components,
    query_conditions,
    train_conditions,
    lambda_pair=1.0,
    lambda_int=0.5,
    return_parts=False,
):
    if lambda_pair < 0 or lambda_int < 0:
        raise ValueError("Kernel weights must be nonnegative")
    qq = indices(components, query_conditions)
    tt = indices(components, train_conditions)
    bq = np.zeros((len(qq), len(components)))
    bt = np.zeros((len(tt), len(components)))
    for i, ix in enumerate(qq):
        bq[i, list(ix)] = 1.0
    for i, ix in enumerate(tt):
        bt[i, list(ix)] = 1.0
    shared = bq @ node @ bt.T
    qp = np.asarray([len(ix) == 2 for ix in qq])
    tp = np.asarray([len(ix) == 2 for ix in tt])
    pair_sum = np.zeros(shared.shape)
    interaction = np.zeros(shared.shape)
    if qp.any() and tp.any():
        pair_sum[np.ix_(qp, tp)] = shared[np.ix_(qp, tp)]
        interaction[np.ix_(qp, tp)] = pair_kernel(
            node,
            [qq[i] for i in np.flatnonzero(qp)],
            [tt[i] for i in np.flatnonzero(tp)],
            "tensor",
        )
    total = shared + lambda_pair * pair_sum + lambda_int * interaction
    return (
        (total, shared, lambda_pair * pair_sum, lambda_int * interaction)
        if return_parts
        else total
    )
