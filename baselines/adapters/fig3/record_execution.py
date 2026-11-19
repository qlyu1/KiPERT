"""Record implementation provenance and metadata, without reading test responses."""
from pathlib import Path
import json,hashlib,datetime,subprocess,importlib.metadata
R=Path('@STUDY@/uc_seen012_benchmark_20261005')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((R/'protocol/PROTOCOL.json').read_text())
z=dict(protocol='SEEN012_V1',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 implementation_notes={
 'seen1':'one fixed L-side singleton mask per split; no directional duplicate training',
 'projection':'all primary model/baseline predictions projected to nonnegative log-normalized expression',
 'common_selection':'seen0 seed17 selects the learning rate and epoch count. Other arms/seeds use that same configuration. Their logged selection validation MSE is the shared reference, not a separately measured validation score.',
 'training_budget':'128 optimizer steps maximum per virtual epoch for cell methods;16 for pseudobulk STAMP;condition-balanced replacement sampling, same base count across seen arms',
 'uc':'unmodified frozen core mean_model + pairhead; native mean_model.predict; evidence wrapper excluded',
 'scouter':'official architecture and gears_loss, condition-balanced sampler',
 'txpert':'public STRING+GO Exphormer-MG, same local compatibility adapter as previous experiment',
 'stamp':'official three-task network and loss, static scGPT embeddings, native predicted DEG mask',
 'gperturb':'official Gaussian multigene design; failed nonfinite tuning candidates recorded and successful validation checkpoints retained',
 'gears':'official network and loss; static coexpression graph from C_fit controls; public GO top20 incoming edges among target vocabulary; batch32, native gamma2 loss, scheduler gamma0.5',
 'evaluation':'sealed-prediction gate; held-out labels read only by final scorer; posthoc DEG100 restricted to shared5000 axis',
 'retention':'all seed17 final checkpoints retained; seed29/43 predictions/configs/hash/logs retained; their final weights and all temporary candidates deleted after sealing',
 'deferred_optional_analyses':'matched-cell-budget sensitivity and separately sealed singleton GI-residual analysis are not included in this execution'},
 source_sha256={p.name:sha(p) for p in (R/'src').glob('*.py')},
 uc_core_sha256={name:sha(R/'assets/uc'/name) for name in ['mean_model.py','pair_research/pairhead.py','train_mean_v3.py']},
 gears_commit=subprocess.check_output(['git','-C',str(R/'assets/GEARS'),'rev-parse','HEAD'],text=True).strip(),
 dependencies=sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
 hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'],text=True),
 protocol_sha256=sha(R/'protocol/PROTOCOL.json'))
(R/'manifests/IMPLEMENTATION.json').write_text(json.dumps(z,indent=2,ensure_ascii=False))
matrix=json.loads((R/'manifests/run_matrix.json').read_text())['jobs']
for j in matrix:
 p=R/f"runs/formal/{j['dataset']}/{j['arm']}/fold{j['fold']}/seed{j['seed']}/{j['method']}/result.json"
 if not p.exists():continue
 res=json.loads(p.read_text());res.update(protocol='SEEN012_V1',arm=j['arm'],refit=False,refit_scope='none; validation never added to fit',common_selection_source=f"{j['dataset']}_seen0_fold{j['fold']}_seed17_{j['method']}",selection_validation_mse_is_shared_reference=not(j['arm']=='seen0' and j['seed']==17));tmp=p.with_suffix('.provenance.tmp');tmp.write_text(json.dumps(res,indent=2));tmp.replace(p)
print('PROVENANCE_RECORDED',flush=True)
