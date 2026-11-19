"""Kernel Landlock allowlist. No test truth or legacy responses readable by workers."""
import ctypes,os,json,sys
from pathlib import Path
from resources import R,OLD
READ=(1<<0)|(1<<2)|(1<<3);WRITE=sum(1<<i for i in (1,4,5,6,7,8,9,10,11,12))
class RuleSet(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
class PathRule(ctypes.Structure):_fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
def confine(base,output,allow_validation=True):
 base=Path(base).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
 gate_snapshot=output/'FORMAL_RUN_GATE.snapshot.json'
 gate_snapshot.write_bytes((R/'manifests/FORMAL_RUN_GATE.json').read_bytes())
 os.environ['FIG3_GATE_SNAPSHOT']=str(gate_snapshot)
 for n in ['tmp','mpl','cache','numba','cuda_cache']:(output/n).mkdir(exist_ok=True)
 os.environ.update(TMPDIR=str(output/'tmp'),MPLCONFIGDIR=str(output/'mpl'),XDG_CACHE_HOME=str(output/'cache'),NUMBA_CACHE_DIR=str(output/'numba'),CUDA_CACHE_PATH=str(output/'cuda_cache'),CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled')
 import tempfile;tempfile.tempdir=str(output/'tmp');libc=ctypes.CDLL(None,use_errno=True)
 def call(number,*args):
  z=libc.syscall(number,*args)
  if z<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
  return z
 abi=call(444,ctypes.c_void_p(0),0,1);handled=READ|WRITE
 if abi>=2:handled|=1<<13
 if abi>=3:handled|=1<<14
 fd=call(444,ctypes.byref(RuleSet(handled)),ctypes.sizeof(RuleSet),0)
 roots=[Path(p) for p in ['/usr','/lib','/lib64','/etc','/proc','/sys','/opt'] if Path(p).exists()]
 roots += [R/'envs',OLD/'envs',R/'repos',R/'assets',R/'assets/priors',R/'assets/scgpt',R/'assets/scfoundation.ckpt',R/'src',Path(sys.executable).resolve().parents[1],R/'protocol',R/'manifests/FORMAL_RUN_GATE.json']
 roots += [p.resolve() for p in (R/'assets').iterdir()]
 roots += [base/n for n in ['fit','controls_fit','query','genes.tsv','genes_all.tsv','target_prior_mapping.tsv','READY.json','split.json','p0_final_configs']]
 if allow_validation:roots.append(base/'validation')
 records=[]
 for p,bits in [(p,READ) for p in roots]+[(output,handled),(Path('/dev'),handled),(Path('/proc/self/task'),READ|(1<<1))]:
  if not p.exists():continue
  allowed=bits if p.is_dir() else bits&((1<<0)|(1<<1)|(1<<2)|(1<<14));pfd=os.open(p,os.O_PATH|os.O_CLOEXEC)
  try:call(445,fd,1,ctypes.byref(PathRule(allowed,pfd)),0)
  finally:os.close(pfd)
  records.append({'path':str(p),'rights':allowed})
 assert libc.prctl(38,1,0,0,0)==0;call(446,fd,0);os.close(fd)
 rejected=[]
 for p in [base/'evaluation_only/expression.npz',OLD/'data/raw/NormanWeissman2019_filtered.h5ad']:
  try:
   with p.open('rb') as f:f.read(1)
   raise AssertionError('protected truth readable')
  except PermissionError:rejected.append(str(p))
 result={'kernel_enforced':True,'landlock_abi':abi,'denied_paths':rejected,'allowlist':records,'scope':'filesystem only; network/IPC not isolated'}
 (output/'isolation.json').write_text(json.dumps(result,indent=2));return result
