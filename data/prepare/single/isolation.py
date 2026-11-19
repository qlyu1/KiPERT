"""Linux Landlock filesystem confinement, inherited by model worker subprocesses."""
import ctypes,os,json,sys
from pathlib import Path
R=Path('@STUDY@/uc_fig2_benchmark_20260911')
READ=(1<<0)|(1<<2)|(1<<3)
WRITE=sum(1<<i for i in (1,4,5,6,7,8,9,10,11,12))
class RuleSet(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
class PathRule(ctypes.Structure):_fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
def confine(base,output,allow_validation=True):
 base=Path(base).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
 for name in ('tmp','mpl','cache','numba','cuda_cache'):(output/name).mkdir(exist_ok=True)
 os.environ.update(TMPDIR=str(output/'tmp'),TMP=str(output/'tmp'),TEMP=str(output/'tmp'),MPLCONFIGDIR=str(output/'mpl'),XDG_CACHE_HOME=str(output/'cache'),NUMBA_CACHE_DIR=str(output/'numba'),CUDA_CACHE_PATH=str(output/'cuda_cache'),CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled')
 import tempfile;tempfile.tempdir=str(output/'tmp')
 libc=ctypes.CDLL(None,use_errno=True)
 def call(number,*args):
  z=libc.syscall(number,*args)
  if z<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
  return z
 abi=call(444,ctypes.c_void_p(0),0,1)
 handled=READ|WRITE
 if abi>=2:handled|=1<<13
 if abi>=3:handled|=1<<14
 fd=call(444,ctypes.byref(RuleSet(handled)),ctypes.sizeof(RuleSet),0)
 readonly=[Path(p) for p in ('/usr','/lib','/lib64','/etc','/proc','/sys','/opt') if Path(p).exists()]
 readonly += [R/'envs',R/'repos',R/'assets',R/'src',Path(sys.executable).resolve().parents[1],R/'protocol']
 readonly += [base/'fit',base/'controls_fit',base/'query',base/'genes.tsv',base/'genes_all.tsv',base/'target_prior_mapping.tsv',base/'READY.json']
 if allow_validation:readonly.append(base/'validation')
 records=[]
 for path,bits in [(p,READ) for p in readonly]+[(output,handled),(Path('/dev'),handled),(Path('/proc/self/task'),READ|(1<<1))]:
  if not path.exists():continue
  allowed=bits if path.is_dir() else bits&((1<<0)|(1<<1)|(1<<2)|(1<<14))
  pfd=os.open(path,os.O_PATH|os.O_CLOEXEC)
  try:call(445,fd,1,ctypes.byref(PathRule(allowed,pfd)),0)
  finally:os.close(pfd)
  records.append({'path':str(path),'rights':allowed})
 if libc.prctl(38,1,0,0,0)!=0:raise RuntimeError('no_new_privs failed')
 call(446,fd,0);os.close(fd)
 rejected=[]
 for path in (base/'evaluation_only/expression.npz',R/'data/raw/TianKampmann2021_CRISPRa.h5ad'):
  try:
   with path.open('rb') as f:f.read(1)
   raise AssertionError('Landlock admitted protected response file')
  except PermissionError:rejected.append(str(path))
 receipt={'landlock_abi':abi,'kernel_enforced':True,'forbidden_read_checks':rejected,'allowlist':records,'pid':os.getpid(),'scope':'Filesystem reads/writes confined; no claim of network/IPC isolation'}
 (output/'isolation.json').write_text(json.dumps(receipt,indent=2));return receipt
