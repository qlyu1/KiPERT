import concurrent.futures,subprocess,pathlib,json,hashlib,time
r=pathlib.Path('@STUDY@/uc_fig2_benchmark_20260911')
def run(x):
 p=r/'data/raw'/x['key'];tmp=pathlib.Path(str(p)+'.part');url=x['links']['self'];t=time.time()
 log=open(r/'logs'/('download_'+x['key']+'.log'),'a')
 rc=subprocess.call(['curl','-L','--fail','--retry','5','--retry-delay','10','--connect-timeout','30','--speed-time','120','--speed-limit','1024','-C','-','-o',str(tmp),url],stdout=log,stderr=log)
 if rc: return {'file':x['key'],'status':'download_failed','rc':rc}
 h=hashlib.md5()
 with open(tmp,'rb') as fh:
  for b in iter(lambda:fh.read(8*1024*1024),b''):h.update(b)
 ok='md5:'+h.hexdigest()==x['checksum'] and tmp.stat().st_size==x['size']
 if ok:tmp.rename(p)
 res={'file':x['key'],'status':'verified' if ok else 'checksum_failed','seconds':time.time()-t,'md5':h.hexdigest()}
 (r/'manifests'/(x['key']+'.download.json')).write_text(json.dumps(res,indent=2));return res
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for z in ex.map(run,json.loads((r/'manifests/raw_downloads.json').read_text())):print(json.dumps(z),flush=True)
