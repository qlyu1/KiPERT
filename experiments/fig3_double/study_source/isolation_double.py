import ctypes,os,json,sys
from pathlib import Path
def confine(ds,fold,out):
 from kipert_double import R,E,BUNDLE,P
 READ=(1<<0)|(1<<2)|(1<<3);WRITE=sum(1<<i for i in (1,4,5,6,7,8,9,10,11,12))
 class RuleSet(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
 class PathRule(ctypes.Structure):_fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
 libc=ctypes.CDLL(None,use_errno=True)
 def call(n,*args):
  z=libc.syscall(n,*args)
  if z<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
  return z
 abi=call(444,ctypes.c_void_p(0),0,1);handled=READ|WRITE
 if abi>=2:handled|=1<<13
 if abi>=3:handled|=1<<14
 fd=call(444,ctypes.byref(RuleSet(handled)),ctypes.sizeof(RuleSet),0)
 roots=[Path(x) for x in ['/usr','/lib','/lib64','/etc','/proc','/sys']]+[Path(sys.executable).resolve().parents[1],(R/'envs/gpu').resolve(),(R/'assets/uc').resolve(),E/'src',E/'PROTOCOL.json',E/f'provenance/old_configs/{ds}_{fold}.json',P/ds]
 roots += [E/f'data/{ds}/{arm}/fold{fold}' for arm in ['seen0','seen1','seen2']]
 for p,bits in [(p,READ) for p in roots]+[(out,handled),(Path('/dev'),handled)]:
  if not p.exists():continue
  p=p.resolve();allowed=bits if p.is_dir() else bits&((1<<0)|(1<<1)|(1<<2)|(1<<14));pfd=os.open(p,os.O_PATH|os.O_CLOEXEC)
  try:call(445,fd,1,ctypes.byref(PathRule(allowed,pfd)),0)
  finally:os.close(pfd)
 assert libc.prctl(38,1,0,0,0)==0;call(446,fd,0);os.close(fd)
 denied=[]
 for p in [R/f'data/formal/{ds}/seen0/fold{fold}/evaluation_only/expression.npz',R/'reports/final/query_metrics.json']:
  try:p.open('rb').read(1);raise AssertionError('Forbidden test file readable')
  except PermissionError:denied.append(str(p))
 (out/'isolation.json').write_text(json.dumps(dict(kernel_enforced=True,abi=abi,denied=denied,scope='filesystem only'),indent=2))
