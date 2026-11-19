from pathlib import Path
import pandas as pd,re,json,hashlib
R=Path('@STUDY@/uc_seen012_benchmark_20261005');OLD=Path('@STUDY@/uc_fig2_benchmark_20260911')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8<<20),b''):h.update(z)
 return h.hexdigest()
class Resolver:
 def __init__(self):
  h=pd.read_csv(R/'assets/priors/hgnc_complete_set.txt',sep='\t',dtype=str).fillna('');self.byid={};self.current={};self.alias={};self.protein=set()
  for row in h.to_dict('records'):
   gid=row['ensembl_gene_id'].split('.')[0]
   if not gid.startswith('ENSG'):continue
   names=[row['symbol']]+row['prev_symbol'].split('|')+row['alias_symbol'].split('|');names=[n for n in dict.fromkeys(names) if n]
   self.byid[gid]=names;self.current[row['symbol']]=gid
   if row.get('locus_group')=='protein-coding gene':self.protein.add(gid)
   for name in names:self.alias.setdefault(name,set()).add(gid)
 def resolve(self,name):
  name=str(name).strip()
  if re.fullmatch(r'ENSG\d+(?:\.\d+)?',name):return name.split('.')[0]
  if name in self.current:return self.current[name]
  ids=self.alias.get(name,set());return next(iter(ids)) if len(ids)==1 else None
 def candidates(self,gid,old=''):return list(dict.fromkeys(self.byid.get(gid,[])+([old] if old else [])))
 def map_prior(self,keys,gid,old=''):
  for n in self.candidates(gid,old):
   if n in keys and self.resolve(n) in (None,gid):return n
  return None
