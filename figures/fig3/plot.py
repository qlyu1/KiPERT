"""Published chart forms applied to all sealed benchmark results (not original-paper data)."""
from pathlib import Path
import sys,json,string
import numpy as np,pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

O=Path(sys.argv[1]);O.mkdir(parents=True,exist_ok=True);D=Path(__file__).resolve().parents[2]/'source_data/fig3'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,'axes.linewidth':.6,'lines.linewidth':.9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
DS=['D1','D2','D3','D4'];DN=dict(zip(DS,['Norman','Replogle exp6','Wessels L4','Dixit K562']));ARMS=['seen0','seen1','seen2']
MODELS=['kipert','uc','gears','scouter','txpert','stamp','gperturb','same_prior_KRR','same_prior_linear','multi_hot_ridge','available_evidence_additive','matching_mean','double_train_mean','no_change','paper_perturbed_mean']
NAMES=['KiPERT','Old UC','GEARS','Scouter','TxPert','STAMP','GPerturb','Legacy KRR','Legacy linear','Multi-hot ridge','Avail.-evidence additive','Matching mean','Double-train mean','No change','Perturbed centroid']
KEEP=[i for i,m in enumerate(MODELS) if m not in ['uc','same_prior_KRR','same_prior_linear']]; ALL_MODELS=MODELS.copy();ALL_NAMES=NAMES.copy()
NM=dict(zip(MODELS,NAMES));palette=['#087F8C','#222222','#4477AA','#CC6677','#AA4499','#DDCC77','#332288','#6699CC','#88CCEE','#117733','#999933','#CCAA66','#999999','#666666','#44AA99'];C=dict(zip(MODELS,palette));MODELS=[ALL_MODELS[i] for i in KEEP];NAMES=[ALL_NAMES[i] for i in KEEP];NM={m:NM[m] for m in MODELS};C={m:C[m] for m in MODELS}

fold=pd.read_csv(D/'fold_metrics.csv')
def est(v):
 v=np.asarray(v);v=v[np.isfinite(v)];return (np.mean(v),stats.t.ppf(.975,len(v)-1)*np.std(v,ddof=1)/np.sqrt(len(v))) if len(v)>1 else (np.mean(v) if len(v) else np.nan,0)

def values(ds,arm,model,metric,level):return level.loc[(level.dataset==ds)&(level.arm==arm)&(level.model==model),metric].dropna().to_numpy()
from matplotlib.ticker import NullLocator
from matplotlib.patches import Rectangle
agg=pd.read_csv(D/'equal_dataset_fold_metrics.csv')
DROP=['multi_hot_ridge','available_evidence_additive','matching_mean','double_train_mean']
MODELS=[m for m in MODELS if m not in DROP];NAMES=[NM[m] for m in MODELS];agg=agg[agg.model.isin(MODELS)]
assert len(MODELS)==8
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,'font.size':5.5,'axes.labelsize':6,'axes.titlesize':6.5,'xtick.labelsize':5.5,'ytick.labelsize':5.5,'legend.fontsize':5.5,'axes.linewidth':.6,'lines.linewidth':.8,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2,'ytick.major.size':2})
W,H=183,250
fig=plt.figure(figsize=(W/25.4,H/25.4))
axes={};mapping=[];source_rows=[]
REMAP={'e':'a','f':'b','g':'c','a':'d','b':'e','j':'f','k':'g','l':'h','i':'i','m':'j','n':'k'}
POS={'a':(24,207,33,33),'b':(68,207,33,33),'c':(112,207,33,33),'d':(24,145,70,38),'e':(105,145,70,38),'f':(14,31,33,33),'g':(58,31,33,33),'h':(102,31,33,33),'j':(146,88,33,33),'i':(14,88,121,33),'k':(146,31,33,33)}
DISPLAY={'a':'a','b':'b','c':'c','d':'d','e':'e','i':'f','f':'g','g':None,'h':None,'j':'h','k':None}
FINAL={'a':'a','b':'b','c':'c','d':'d','e':'e','i':'f','f':'g','g':'g','h':'g','j':'h','k':'h'}
VIEW={'f':'Norman','g':'Replogle exp6','h':'Wessels L4','j':'KiPERT','k':'Additive'}
def panel(letter,x,y,w,h,title):
 letter=REMAP[letter];x,y,w,h=POS[letter]
 ax=fig.add_axes([x/W,y/H,w/W,h/H]);ax.set_label(letter)
 if DISPLAY[letter] is not None:ax.text(-5/w,1+2/h,DISPLAY[letter],transform=ax.transAxes,fontsize=8,weight='bold',ha='left',va='bottom')
 ax.set_title(title,pad=3);axes[letter]=ax;return ax
def addmap(letter,page,orig,description):mapping.append(dict(panel=letter,source_page_36page_atlas=page,source_panel=orig,description=description))
# a-d: retain both requested linear and logarithmic nMSE presentations.
for k,(letter,ds,islog) in enumerate([('a','D1',False),('b','D2',False)]):
 ax=panel(letter,12+k*43.75,211,35.75,26,DN[ds]+(' (log)' if islog else ''))
 for j,m in enumerate(MODELS):
  vals=[values(ds,a,m,'nmse20',fold) for a in ['seen2','seen1','seen0']];zz=[est(v) for v in vals]
  if islog:
   for i,v in enumerate(vals):assert (v>0).all();ax.scatter(np.full(len(v),i)+(j-(len(MODELS)-1)/2)*.015,v,s=2,color=C[m],alpha=.35,linewidths=0)
   ax.plot(range(3),[z[0] for z in zz],marker='o',ms=1.6,color=C[m],lw=.95 if m=='kipert' else .45)
  else:ax.errorbar(np.arange(3)+(j-(len(MODELS)-1)/2)*.07,[z[0] for z in zz],yerr=[z[1] for z in zz],fmt='o',ms=3,color=C[m],capsize=1.5,lw=.8,markeredgewidth=.8)
  for ai,v in enumerate(vals):
   for fi,value in enumerate(v):source_rows.append(dict(panel=letter,model=m,dataset=ds,arm=['seen2','seen1','seen0'][ai],fold=fi,metric='nmse20',value=value))
 ax.axhline(1,color='#999999',ls='--',lw=.45);ax.set_xticks(range(3),['seen 2','seen 1','seen 0']);ax.set_xlabel('');ax.set_xlim(-.4,2.4)
 if islog:ax.set_yscale('log');ax.set_yticks([.1,.3,1],['0.1','0.3','1']);ax.yaxis.set_minor_locator(NullLocator())
 if k in [0,2]:ax.set_ylabel('Top-20 nMSE',labelpad=2)
 addmap(letter,2 if islog else 1,'ab'[k%2],f'{DN[ds]} top20 nMSE'+(' log companion' if islog else ''))
handles=[Line2D([0],[0],marker='o',ms=2,color=C[m],lw=1 if m=='kipert' else .6,label=NM[m]) for m in MODELS]
fig.legend(handles=handles,loc='center',bbox_to_anchor=(166/W,223.5/H),ncol=1,frameon=False,handlelength=1.2,handletextpad=.5,columnspacing=1.5,labelspacing=.5)
# e-g: four datasets each receive one quarter of total weight within every split.
maxopp=max(est(g.opposite20)[0]+est(g.opposite20)[1] for _,g in agg.groupby(['arm','model']))
for ai,letter in enumerate('efg'):
 arm=f'seen{ai}';ax=panel(letter,31+ai*(148/3+10/3),151,(148-20)/3,31,f'seen {ai}')
 for j,m in enumerate(MODELS):
  v=agg[(agg.arm==arm)&(agg.model==m)].opposite20.to_numpy();mu,ci=est(v);ax.barh(j,mu,xerr=ci,color=C[m],height=.68,alpha=.9,error_kw={'elinewidth':.7,'capsize':1.2})
  for fi,value in enumerate(v):source_rows.append(dict(panel=letter,model=m,dataset='equal_weight_D1_D4',arm=arm,fold=fi,metric='opposite20',value=value))
 ax.set_yticks(range(len(MODELS)),NAMES if ai==0 else []);ax.invert_yaxis();ax.set_xlim(0,maxopp*1.06);ax.set_xlabel('Opposite-direction fraction',labelpad=2);ax.tick_params(axis='y',length=0,labelsize=5.5)
 addmap(letter,6+ai,'a–d merged',f'Four-dataset equal-weight opposite20, {arm}')
# i: same inverse-retrieval candidates and target macro aggregation as source page25a.
hits=pd.read_csv(D/'retrieval.csv.gz');hm=hits.groupby(['dataset','arm','model','target','k'],as_index=False).hit.mean()
ax=panel('i',130,109,49,26,'Norman | seen 0')
for j,m in enumerate(MODELS):
 s=hm[(hm.dataset=='D1')&(hm.arm=='seen0')&(hm.model==m)].groupby('k').hit.mean().reindex(range(1,9));ax.bar(np.arange(1,9)+(j-(len(MODELS)-1)/2)*.09,s,width=.082,color=C[m],alpha=.9 if m=='kipert' else .55)
 for kk,v in s.items():source_rows.append(dict(panel='i',model=m,dataset='D1',arm='seen0',metric='retrieval_hit',k=kk,value=v))
ax.set_xticks(range(1,9));ax.set_ylim(0,1.02);ax.set_xlabel('Top K predicted combinations',labelpad=2);ax.set_ylabel('Exact hit rate',labelpad=2);addmap('i',25,'a','Norman seen0 inverse combination retrieval')
fig.legend(handles=[Patch(facecolor=C[m],label=NM[m]) for m in MODELS],loc='center',bbox_to_anchor=(74.5/W,76/H),ncol=4,frameon=False,fontsize=5.5,handlelength=1,handletextpad=.4,labelspacing=.5)
# j-l: same three fold0 examples and top10 absolute true nonadditive residual genes.
case_rows=[]
for ai,(letter,ds) in enumerate(zip('jkl',['D1','D2','D3'])):
 ax=panel(letter,12+ai*(167+10)/3,65,(167-20)/3,26,DN[ds]+'\nseen 0');z=np.load(D/f'case_{ds}.npz');c=z['control'];g=np.argsort(-abs(z['truth']-z['single_a']-z['single_b']+c),kind='stable')[:10];x=np.arange(10);a=z['single_a'][g]-c[g];b=z['single_b'][g]-c[g];p=z['prediction'][(z['models']=='kipert')&(z['arms']=='seen0')].mean(0)-c;t=z['truth']-c
 ax.bar(x-.22,a,.34,color='#CCCCCC');ax.bar(x-.22,b,.34,bottom=a,color='#EEEEEE');ax.bar(x+.2,p[g],.34,color=C['kipert']);ax.plot(x,t[g],'k.',ms=2.5);ax.axhline(0,color='#999999',lw=.4);ax.set_xticks(x,z['symbol'][g],rotation=90,ha='right',rotation_mode='anchor');ax.tick_params(axis='x',length=1,labelsize=5.5)
 if ai==0:ax.set_ylabel('Expression change',labelpad=2)
 for gi in g:case_rows.append(dict(panel=letter,dataset=ds,gene=str(z['symbol'][gi]),single_a=float(z['single_a'][gi]-c[gi]),single_b=float(z['single_b'][gi]-c[gi]),KiPERT=float(p[gi]),observed=float(t[gi])))
 addmap(letter,22,'abc'[ai],f'{DN[ds]} same nonadditive expression case, seen0')
fig.legend(handles=[Patch(color='#CCCCCC',label='Single A'),Patch(color='#EEEEEE',label='Single B'),Patch(color=C['kipert'],label='KiPERT'),Line2D([0],[0],ls='',marker='.',ms=3,color='black',label='Observed')],loc='center',bbox_to_anchor=(74.5/W,6/H),ncol=4,frameon=False,handlelength=1.2,columnspacing=1.5)
# m-n: source page33a/c, both top-row scatters, before removing Balanced column.
thresholds=pd.read_csv(D/'interaction_thresholds.csv').set_index('dataset');lo,hi=thresholds.loc['D2',['lower','upper']];classes=['Additive','Other','Buffering','Synergy','Cryptic'];clcolors=['#DDDDDD','#767676','#BEAED4','#FDC086','#A00254']
def classify(ab,a,b,c,lo,hi):
 r=ab-a-b+c;add=a+b-c;same=(np.sign(a-c)==np.sign(b-c))&(np.sign(ab-c)==np.sign(a-c));out=np.zeros(ab.shape,dtype=int);out[(r<lo)|(r>hi)]=1
 out[same&(((add>c)&(r<lo))|((add<c)&(r>hi)))]=2
 out[same&(((add>c)&(r>hi))|((add<c)&(r<lo)))]=3
 out[same&(((add>c)&(r<lo)&(ab<c))|((add<c)&(r<lo)&(ab>c)))]=4
 return out
z=np.load(D/'GI_D2_fold0_seen2.npz');a=z['single_a'];b=z['single_b'];c=z['control'];y=z['truth'];tr=(y-a-b+c).ravel();tc=classify(y,a,b,c,lo,hi).ravel();scatter_rows=[]
for j,(letter,m) in enumerate([('m','kipert'),('n','available_evidence_additive')]):
 ax=panel(letter,12+j*89.5,12,77.5,25,'Replogle exp6\nseen 2 | '+('KiPERT' if m=='kipert' else 'Additive'))
 pred=z['prediction'][z['models']==m].mean(0);pr=(pred-a-b+c).ravel();sel=np.argsort(-abs(pr),kind='stable')[:min(500,len(pr))];ax.scatter(tr,pr,s=.6,color='#BBBBBB',alpha=.25,rasterized=True,linewidths=0);ax.scatter(tr[sel],pr[sel],s=1,c=np.array(clcolors)[tc[sel]],alpha=.7,rasterized=True,linewidths=0);lim=max(abs(tr).max(),abs(pr).max());ax.plot([-lim,lim],[-lim,lim],ls='--',color='#999999',lw=.4);ax.axvline(lo,color='#AAAAAA',lw=.35);ax.axvline(hi,color='#AAAAAA',lw=.35);ax.set_xlabel('Observed non-additive\nresidual',labelpad=2);ax.set_ylabel('Predicted residual',labelpad=2)
 flags=np.zeros(len(tr),dtype=bool);flags[sel]=True
 for q in range(len(tr)):scatter_rows.append(dict(panel=letter,model=m,index=q,observed=tr[q],predicted=pr[q],observed_class=classes[tc[q]],highlighted=bool(flags[q])))
 addmap(letter,33,'a' if j==0 else 'c',f'Replogle exp6 seen2 nonadditive residual scatter: {NM[m]}')
fig.legend(handles=[Patch(color=co,label=cl) for cl,co in zip(classes,clcolors)],loc='center',bbox_to_anchor=(161/W,13/H),ncol=2,frameon=False,handlelength=1.1,columnspacing=1.5)
fig.canvas.draw()
for letter,ax in axes.items():
 r=ax.get_window_extent()
 if letter not in ['d','e','i']:assert abs(r.width-r.height)<1e-7
# g and h are multi-axis panels; only the first axis carries the shared panel letter.
labels=[t.get_text() for ax in axes.values() for t in ax.texts]
assert sorted(labels)==list('abcdefgh'), labels

fig.savefig(O/'Figure3.pdf')
plt.close(fig)
