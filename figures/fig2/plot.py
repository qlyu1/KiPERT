"""Square plotting areas, zoomed PDS and two complete response landscapes."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy import stats

old=Path(__file__).resolve().parents[2]/'source_data/fig2'
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
for n in ['source_data','qa']:(out/n).mkdir(exist_ok=True)
M=['UC','biolord','scouter','txpert','gears','stamp','regformer']
N=['KiPERT','biolord','Scouter','TxPert-public','GEARS','STAMP–Geneformer','RegFormer–GEARS']
C=['#174F64','#D58B55','#569B80','#AC82A8','#648AC0','#AD9A4E','#70777F']
D=['S3','S4','S5','S6','S7'];DN=['Tian CRISPRi','Tian CRISPRa','Adamson','Norman singles','Frangieh'];SHORT=['Tian i','Tian a','Adamson','Norman','Frangieh']
u=pd.read_csv(old/'target_metrics.csv');f=pd.read_csv(old/'condition_metrics.csv');means=pd.read_csv(old/'ranks.csv');land=np.load(old/'landscapes.npz')
MET=['mse','pearson_delta','systema_cell_pearson','sp_mse100','sp_pcc100','auprc_dg','deg_precision100','deg_recall100','top100_direction','pds_query_cosine']
LAB=['All-gene\nMSE','Pearson\nΔ','Pearson\nΔ − shared','Top-100\nMSE','Top-100\nPCC Δ','DEG\nAUPRC','Precision\n@100','Recall\n@100','Sign@100','Cosine\nPDS']
assert {'sp_mse100','auprc_dg'}.issubset(MET)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],'font.size':6,'axes.labelsize':6,'xtick.labelsize':5.5,'ytick.labelsize':5.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.55,'legend.frameon':False})
W,H=183,200;fig=plt.figure(figsize=(W/25.4,H/25.4));axes=[];ids=[]
def square(x,y,side,name,polar=False):
 a=fig.add_axes([x/W,y/H,side/W,side/H],projection='polar' if polar else None);axes.append(a);ids.append(name);return a
def txt(x,y,text,**kw):return fig.text(x/W,y/H,text,**kw)
def tag(x,y,t):txt(x,y,t,fontsize=8,weight='bold')
tag(3,193,'a');tag(86,193,'b')
a=square(22,141,44,'a',True);a.set_theta_zero_location('N');a.set_theta_direction(-1);theta=np.linspace(0,2*np.pi,len(MET),endpoint=False)
for m,c in zip(M,C):
 v=means[means.method==m].set_index('metric').loc[MET,'mean_score'].to_numpy();a.plot(np.r_[theta,theta[0]],np.r_[v,v[0]],'o-',color=c,lw=1.1 if m=='UC' else .65,ms=2.2 if m=='UC' else 1.4,zorder=5 if m=='UC' else 2)
a.set_ylim(0,1);a.set_xticks(theta,LAB);a.tick_params(axis='x',pad=7,labelsize=5.5);a.set_yticks([0,1/3,2/3,1],['','','','']);a.grid(color='#CAD3D9',lw=.45);a.spines['polar'].set_color('#CAD3D9')
b=fig.add_axes([95/W,141/H,82/W,44/H]);axes.append(b);ids.append('b')
z=f.groupby(['dataset','method','split','target']).systema_cell_pearson.mean().reset_index().groupby(['dataset','method','split']).systema_cell_pearson.mean().reset_index();st=z.groupby(['dataset','method']).systema_cell_pearson.agg(['mean','std','count']).reset_index();st['ci']=stats.t.ppf(.975,st['count']-1)*st['std']/np.sqrt(st['count'])
for j,(m,c) in enumerate(zip(M,C)):
 t=st[st.method==m].set_index('dataset').loc[D];b.bar(np.arange(5)+(j-3)*.105,t['mean'],width=.095,yerr=t.ci,color=c,edgecolor='none',error_kw={'elinewidth':.45,'capsize':.8,'capthick':.45,'ecolor':'#33414B'},zorder=3)
b.set_xticks(range(5),SHORT,rotation=35,ha='right',rotation_mode='anchor');b.set_ylabel('Pearson Δ − shared');b.yaxis.set_major_locator(MaxNLocator(4));b.axhline(0,color='#AAB2B8',lw=.45)
st.replace({'method':dict(zip(M,N))}).to_csv(out/'source_data/b_errorbars.csv',index=False)

# PDS illustration and score matrices occupy an aligned 2 x 3 grid.
tag(3,118,'c');tag(26,118,'d')
c=fig.add_axes([2/W,77/H,20/W,35/H]);axes.append(c);ids.append('c');c.set_xlim(0,1);c.set_ylim(0,1);c.axis('off')
pal=['#AE4F45','#DCA28C','#F2E9E2','#99B6CD','#416C91']
patterns=[[0,1,2,3,4],[2,0,4,1,3],[4,3,2,1,0]]
def strip(x,y,pattern):
 for k,z in enumerate(pattern):
  c.add_patch(Rectangle((x+k*.067,y),.062,.07,facecolor=pal[z],edgecolor='none'))
strip(.20,.84,patterns[0])
c.annotate('',xy=(.367,.71),xytext=(.367,.82),arrowprops={'arrowstyle':'->','color':'#24313B','lw':.8})
c.text(.27,.67,'Observed',ha='center',fontsize=5.5)
c.text(.85,.67,'Distance',ha='center',fontsize=5.5)
for j,(y,pattern,w) in enumerate(zip([.51,.32,.13],patterns,[.045,.15,.25])):
 c.text(.10,y+.035,'ABC'[j],va='center',ha='center',fontsize=6,weight='bold' if j==0 else 'normal')
 strip(.20,y,pattern)
 c.add_patch(Rectangle((.65,y),w,.07,facecolor=C[0] if j==0 else '#B7C2C9',edgecolor='none'))
 if j==0:
  c.add_patch(Rectangle((.04,y-.035),.91,.14,facecolor='none',edgecolor=C[0],lw=.8))
  c.text(.84,y+.035,'#1',ha='center',va='center',fontsize=6,color=C[0],weight='bold')
pm=['pds_query_l2','pds_query_cosine'];ps=[]
norm=Normalize(.40,.70);cmap=plt.get_cmap('Blues')
for i,(metric,x) in enumerate(zip(pm,[44,106.5])):
 ax=fig.add_axes([x/W,77/H,59/W,35/H]);axes.append(ax);ids.append('d'+str(i));vals=np.zeros((7,5))
 for j,m in enumerate(M):
  for k,d in enumerate(D):
   g=u[(u.dataset==d)&(u.method==m)];v=g[metric].mean();se=g[metric].sem();vals[j,k]=v
   ps.append(dict(dataset=d,method=N[j],metric=metric,mean=v,sem=se))
 pim=ax.imshow(vals,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
 for r in range(7):
  for col in range(5):
   ax.text(col,r,f'{vals[r,col]:.2f}',ha='center',va='center',fontsize=6,color='white' if norm(vals[r,col])>.58 else '#172B3B')
 ax.set_xticks(range(5),DN,rotation=90,ha='right',rotation_mode='anchor');ax.tick_params(axis='x',length=0,pad=3,labelsize=5.5)
 labels=[n.replace('STAMP–Geneformer','STAMP–\nGeneformer').replace('RegFormer–GEARS','RegFormer–\nGEARS') for n in N]
 ax.set_yticks(range(7),labels if i==0 else ['']*7);ax.tick_params(axis='y',length=0,pad=2,labelsize=5.5)
 ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,7,1),minor=True);ax.grid(which='minor',color='white',lw=.5);ax.tick_params(which='minor',length=0)
 for sp in ax.spines.values():sp.set_visible(False)
 txt(x+29.5,115,['L2','Cosine'][i],ha='center',fontsize=6)
pd.DataFrame(ps).to_csv(out/'source_data/d_pds.csv',index=False)
means[means.metric.isin(MET)].replace({'method':dict(zip(M,N))}).to_csv(out/'source_data/a_radar_ranks.csv',index=False)
pca=fig.add_axes([169/W,77/H,2/W,35/H]);pcb=fig.colorbar(pim,cax=pca,orientation='vertical',ticks=[.4,.5,.6,.7]);pcb.set_label('Mean PDS',fontsize=6,labelpad=2);pcb.ax.tick_params(labelsize=5.5,length=1.5,pad=1)

heat=[]
for d,y,letter in [('S4',35,'e'),('S6',7,'f')]:
 tag(3,y+22,letter);txt(2.0,y+8.5,DN[D.index(d)],fontsize=5.5,ha='center',va='center',rotation=90,rotation_mode='anchor')
 for j,m in enumerate(['truth']+M):
  ax=square(5+20.5*j,y,17,letter+str(j));mat=land[d+'_'+m];assert mat.shape[0]==mat.shape[1]
  im=ax.imshow(mat,cmap='RdBu_r',vmin=-1,vmax=1,aspect='equal',interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
  name='Observed' if m=='truth' else N[M.index(m)];name=name.replace('STAMP–Geneformer','STAMP–\nGeneformer').replace('RegFormer–GEARS','RegFormer–\nGEARS')
  txt(13.5+20.5*j,y+18.5,name,ha='center',fontsize=5.5,color=C[0] if m=='UC' else '#24313B',weight='bold' if m=='UC' else 'normal')
  heat.extend(dict(dataset=d,model='Observed' if m=='truth' else N[M.index(m)],row_target=str(land[d+'_target'][r]),column_target=str(land[d+'_target'][q]),cosine=float(mat[r,q])) for r in range(len(mat)) for q in range(len(mat)))
 txt(9,y-3.5,'Perturbations × perturbations; n = '+str(len(land[d+'_target']))+'; split 101',fontsize=5.0)
ca=fig.add_axes([169/W,7/H,2/W,45/H]);cb=fig.colorbar(im,cax=ca,orientation='vertical',ticks=[-1,0,1]);cb.set_label('Response cosine similarity',fontsize=6,labelpad=2);cb.ax.tick_params(labelsize=5.5,length=1.5,pad=1)
pd.DataFrame(heat).to_csv(out/'source_data/e_f_response_matrices.csv',index=False)
fig.legend(handles=[Line2D([0],[0],color=col,marker='o',ms=2.4,lw=.8,label=n) for n,col in zip(N,C)],loc='upper center',bbox_to_anchor=(.5,126/H),ncol=7,fontsize=5.5,columnspacing=.9,handlelength=1.1,handletextpad=.4)
fig.canvas.draw()
from matplotlib.text import Text
text_inventory=[{'text':t.get_text(),'size_pt':t.get_fontsize(),'weight':t.get_fontweight()} for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
(out/'qa/text_inventory.json').write_text(json.dumps(text_inventory,indent=2,ensure_ascii=False))
sq=[]
for name,ax in zip(ids,axes):
 box=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted());diff=abs(box.width-box.height)*72;assert name in ['b','c','d0','d1'] or diff<.05,(name,diff);sq.append(dict(panel=name,width_mm=box.width*25.4,height_mm=box.height*25.4))
(out/'qa/square_geometry.json').write_text(json.dumps(sq,indent=2))
fig.savefig(out/'Figure2.pdf')
plt.close(fig)
