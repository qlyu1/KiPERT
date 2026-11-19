from pathlib import Path
import h5py,numpy as np,json
B=Path(__file__).resolve().parents[1];report=[]
for c in ['hepg2','jurkat','k562','rpe1']:
 with h5py.File(B/'data/normalized'/f'{c}.h5ad','r+') as h:
  names=[]
  def find(name,obj):
   if isinstance(obj,h5py.Group) and obj.attrs.get('encoding-type')=='nullable-string-array':names.append(name)
  h.visititems(find)
  for name in names:
   g=h[name];assert not g['mask'][:].any();v=g['values'].asstr()[:];del h[name];ds=h.create_dataset(name,data=v.astype(object),dtype=h5py.string_dtype('utf-8'));ds.attrs['encoding-type']='string-array';ds.attrs['encoding-version']='0.2.0';assert np.array_equal(v,ds.asstr()[:])
  report.append({'context':c,'lossless_nullable_to_string_fields':names});print(c,names,flush=True)
(B/'reports/h5_string_compatibility.json').write_text(json.dumps(report,indent=2))
