"""UC v3-E: fixed evidence-preserving composition when no pair labels exist.
This explicitly retrospective follow-up retains the frozen v3 selected model.
No test responses, parameter search or refitting are exposed by this module.
"""
from pathlib import Path
import json,pickle,hashlib,sys
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
import mean_model
from pair_research import pairhead
POLICY="NO_PAIR_LABELS_PRESERVE_OBSERVED_SINGLE_V1"
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def predict_arrays(parent,train_conditions,training_delta,conditions,nonnegative=True):
    train=[pairhead.canon(c) for c in train_conditions]
    if train!=list(np.asarray(parent['train_conditions']).astype(str)):raise ValueError('Full legal train order differs')
    y=np.asarray(training_delta,np.float64);ctrl=np.asarray(parent['control_mean'],np.float64)
    if y.shape!=(len(train),len(ctrl)) or not np.isfinite(y).all():raise ValueError('Invalid legal training values')
    conditions=[pairhead.canon(c) for c in conditions]
    raw=mean_model.predict(parent,conditions,nonnegative=False)
    pair_count=sum(len(c.split('+'))==2 for c in train)
    known={c:y[i] for i,c in enumerate(train) if '+' not in c}
    modes=[];used=[]
    if pair_count:
        modes=['JOINT_V3_WITH_PAIR_SUPERVISION']*len(conditions)
        used=[[] for c in conditions]
    else:
        missing=sorted({g for c in conditions if any(x in known for x in c.split('+')) for g in c.split('+') if g not in known})
        predicted=mean_model.predict(parent,missing,nonnegative=False) if missing else np.empty((0,len(ctrl)))
        fallback=dict(zip(missing,predicted))
        for i,c in enumerate(conditions):
            genes=c.split('+');observed=[g for g in genes if g in known]
            if observed:
                # Apply observations before the expression nonnegativity projection.
                raw[i]=np.sum([known[g] if g in known else fallback[g] for g in genes],axis=0)
                modes.append('LEGAL_SINGLE_EVIDENCE_COMPOSITION')
            else:modes.append('FROZEN_V3_UNOBSERVED_COMPONENTS')
            used.append(observed)
    delta=np.maximum(ctrl+raw,0.)-ctrl if nonnegative else raw
    return delta,dict(policy=POLICY,n_legal_double_labels=pair_count,modes=modes,
        observed_single_components=used,raw_delta=raw)
def load_bundle(directory):
    directory=Path(directory);a=json.loads((directory/'MODEL.json').read_text())
    if a['policy']!=POLICY or a['source_sha256']!=sha(__file__):raise ValueError('Evidence policy/source differs')
    for name in ['parent_model','legal_single_input']:
        d=a[name]
        if sha(d['path'])!=d['sha256']:raise ValueError('Evidence artifact changed: '+name)
    frozen=R/'TRAINING_PROTOCOL_FROZEN.json';p=json.loads(frozen.read_text())
    if sha(frozen)!=a['root_training_protocol_sha256']:raise ValueError('Root training seal differs')
    for f,h in p['sources'].items():
        if sha(f)!=h:raise ValueError('Root numerical source differs')
    with open(a['parent_model']['path'],'rb') as f:parent=pickle.load(f)
    with np.load(a['legal_single_input']['path'],allow_pickle=False) as z:
        train=z['train_conditions'].astype(str);y=z['training_delta'].astype(np.float64)
        if not np.array_equal(z['genes'].astype(str),parent['genes']) or not np.array_equal(z['control_mean'],parent['control_mean']):
            raise ValueError('Evidence gene/control axis differs')
    if parent['source_seal_sha256']!=a['root_training_protocol_sha256']:raise ValueError('Parent source differs')
    return a,parent,train,y
def predict_bundle(directory,conditions,nonnegative=True):
    a,parent,train,y=load_bundle(directory)
    pred,meta=predict_arrays(parent,train,y,conditions,nonnegative)
    return pred,parent['control_mean'],parent['genes'],meta,a
