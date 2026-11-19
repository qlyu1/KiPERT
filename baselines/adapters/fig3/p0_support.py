from resources import R
import json,sys

def get_force(base,method,seed):
 path=base/'p0_final_configs'/(method+'_'+str(seed)+'.json')
 if not path.exists():
  if base.name=='fold-1':raise RuntimeError('Final P0 refit requires completed single-only CV configuration')
  return None
 return json.loads(path.read_text())
def finish_inner(base,out,candidates,method,seed):
 if base.name=='fold-1':return
 result={'status':'INNER_SELECTION_COMPLETE','formal_training':True,'outer_predictions_generated':False,'method':method,'seed':seed,'candidates':candidates,'scope':'single-only inner validation; no double response read'}
 for p in out.glob('candidate_*'):
  if p.suffix in ['.pt','.pkl']:p.unlink()
 (out/'selection.json').write_text(json.dumps(result,indent=2));(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':'INNER_SELECTION_COMPLETE','method':method,'seed':seed}));sys.exit(0)
