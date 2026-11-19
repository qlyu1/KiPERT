from pathlib import Path
import requests,json,concurrent.futures
R=Path('@STUDY@/uc_fig2_benchmark_20260911')
def get(name,url):
 z=requests.get(url,timeout=45);z.raise_for_status();obj=z.json();(R/'manifests'/name).write_text(json.dumps(obj));return name,obj
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:
 f=[p.submit(get,'txpert_zenodo_15420279.json','https://zenodo.org/api/records/15420279'),p.submit(get,'gears_misc_tree.json','https://api.github.com/repos/yhr91/GEARS_misc/git/trees/main?recursive=1')]
 for task in concurrent.futures.as_completed(f):
  try:
   name,z=task.result();print(name,flush=True)
   if 'files' in z:print([(x['key'],x['size'],x['checksum'],x['links']['self']) for x in z['files']],flush=True)
   else:print([x['path'] for x in z.get('tree',[]) if 'adamson' in x['path'].lower() or 'preprocess' in x['path'].lower()],flush=True)
  except Exception as e:print(repr(e),flush=True)
