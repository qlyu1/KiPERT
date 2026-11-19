from kipert_double import *
import gc,datetime
records=[]
for ds in ['D1','D2','D3','D4']:
 means={}
 for fold in range(5):
  for arm in ['seen0','seen1','seen2']:
   src=R/f'data/formal/{ds}/{arm}/fold{fold}';compact=BUNDLE/f'data/{ds}/{arm}/fold{fold}';out=E/f'data/{ds}/{arm}/fold{fold}';out.mkdir(parents=True,exist_ok=True)
   for name,role in [('train','fit'),('validation','validation'),('controls_fit','controls_fit')]:
    z=dict(np.load(compact/(name+'.npz')));key=str((src/role).resolve())
    if key not in means:
     x=sparse.load_npz(src/role/'expression.npz');m=pd.read_csv(src/role/'metadata.tsv',sep='\t',dtype=str);means[key]={(t,c):np.asarray(x[ix].mean(0)).ravel() for (t,c),ix in m.groupby(['target','context'],sort=True).indices.items()};del x,m;gc.collect()
    exact=np.stack([means[key][(str(t),str(c))] for t,c in zip(z['target'],z['context'])]);records.append(dict(dataset=ds,fold=fold,arm=arm,role=role,export_max_error=float(abs(exact-z['expression']).max()),source=key));z['expression']=exact;np.savez_compressed(out/(name+'.npz'),**z)
   for name in ['query.npz','genes.tsv']:
    import shutil
    shutil.copy2(compact/name,out/name)
  print('EXACT_CACHE',ds,fold,flush=True)
 del means;gc.collect()
save(E/'provenance/EXACT_MEANS.json',dict(status='READY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='Original scipy sparse float32 mean over grouped cells, identical train_uc.py load(role); training roles only',records=records))
