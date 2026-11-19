from pathlib import Path
import json,yaml,toml,hashlib
B=Path(__file__).resolve().parents[1];P=B.parent/'pie_four_context_reproduction_20261106';O=B.parent/'uc_state_four_context_zeroshot_20261102';D=P/'downloads/official_data/preprocessed';m=json.loads((D/'meta.json').read_text())
import numpy as np
ci=np.load(D/'ctx_ids.npy');pi=np.load(D/'pert_ids.npy');pert={v:k for k,v in m['pert_to_id'].items()};allpairs={ctx:{pert[int(p)] for p in pi[ci==i]} for ctx,i in m['context_to_id'].items()}
for fold in m['context_to_id']:
 s=P/'downloads/official_splits/replogle_wdataset/unseen_ctx'/fold;train=json.loads((s/'train.json').read_text());val=json.loads((s/'val.json').read_text());cfg={'datasets':{'replogle':str(B/'data/normalized')},'training':{'replogle':'train'},'zeroshot':{'replogle.'+fold:'test'},'fewshot':{}}
 for ctx in m['context_to_id']:
  if ctx==fold:continue
  t=set(train.get('replogle.'+ctx,[]));v=set(val.get('replogle.'+ctx,[]));assert not t&v;test=allpairs[ctx]-t-v;assert t|v|test==allpairs[ctx];cfg['fewshot']['replogle.'+ctx]={'val':sorted(v),'test':sorted(test)}
 dest=B/'manifests'/f'state_{fold}.toml';dest.write_text(toml.dumps(cfg));c=yaml.safe_load((O/'official/ST-HVG-Replogle/zeroshot'/fold/'config.yaml').read_text());c['data']['kwargs'].update(toml_config_path=str(dest),num_workers=4);c.update(output_dir=str(B/'runs/state'),name=fold,use_wandb=False,overwrite=False);c['training']['use_mfu']=False;c['training'].pop('cumulative_flops_use_backward',None);c['training']['wandb_track']=False;c['training']['log_every_n_steps']=100;c['wandb']['local_wandb_dir']=str(B/'logs/wandb_disabled');(B/'manifests'/f'state_{fold}.yaml').write_text(yaml.safe_dump(c,sort_keys=False));print(fold,'max_steps',c['training']['max_steps'],'train',sum(map(len,train.values())),'val',sum(map(len,val.values())))
