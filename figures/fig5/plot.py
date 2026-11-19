"""Figure 5: existing measured cardiac responses, from complete source tables.

Python-only vector figure at final print scale (180 x 170 mm). No model fits,
new tests, outcome selection, or averaged-prediction correlations are performed.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import Normalize

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/'source_data/fig5'
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(HERE/'qa'))

for name in ['Arial.ttf','Arial Bold.ttf','Arial Italic.ttf']:
    if (HERE/'fonts'/name).is_file():
        font_manager.fontManager.addfont(str(HERE/'fonts'/name))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans', 'Arial'],
 'font.size':6.5,'axes.labelsize':6.5,'axes.titlesize':7,'xtick.labelsize':5.5,'ytick.labelsize':5.5,
 'legend.fontsize':5.8,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
 'xtick.major.size':2.4,'ytick.major.size':2.4,'xtick.major.pad':2,'ytick.major.pad':2,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
KIPERT='#C96144';RAW='#B8B8B8';INK='#222222';OBS='#000000';LINE='#939CA3'
fig=plt.figure(figsize=(180/25.4,170/25.4))
rects={'a':[.089,.825,.878,.120],
       'b':[.089,.450,.350,.290], 'c':[.600,.450,.367,.290],
       'd':[.089,.105,.225,.240], 'e':[.407,.105,.283,.240], 'f':[.797,.105,.170,.240]}
axes={letter:fig.add_axes(rect) for letter,rect in rects.items()}
for letter,ax in axes.items():
    ax.set_gid(letter)
    ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-13,9),textcoords='offset points',
                fontsize=9,fontweight='bold',ha='left',va='bottom')

# a: Diagram defines two permitted input structures; cell shapes are schematic.
ax=axes['a'];ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
def box(x,y,w,h,edge,face):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.018',
                        linewidth=.7,edgecolor=edge,facecolor=face)
    ax.add_patch(patch)
def arrow(start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=6,linewidth=.7,color=LINE))
box(.012,.04,.151,.91,'#C7CFD4','#F4F6F7')
ax.text(.0875,.66,'In vivo cardiac\nreprogramming',ha='center',va='center',fontsize=6.2,fontweight='bold',linespacing=1.22)
ax.text(.0875,.29,'Perturb-seq\nDay 14',ha='center',va='center',fontsize=5.8,linespacing=1.22)
for y in [.75,.25]:
    arrow((.17,.5),(.215,y));arrow((.50,y),(.56,y));arrow((.735,y),(.795,y))
ax.text(.36,.75,'5 or 10 cells per perturbation\n+ other measured perturbations',ha='center',va='center',fontsize=6.2,linespacing=1.25)
ax.text(.36,.25,'Calr in training batches\n+ controls in all batches',ha='center',va='center',fontsize=6.2,linespacing=1.25)
box(.57,.60,.16,.30,KIPERT,'#FAEEE9');box(.57,.10,.16,.30,KIPERT,'#FAEEE9')
ax.text(.65,.75,'KiPERT',ha='center',va='center',fontsize=6.5,color=KIPERT,fontweight='bold')
ax.text(.65,.25,'KiPERT',ha='center',va='center',fontsize=6.5,color=KIPERT,fontweight='bold')
ax.text(.90,.75,'Held-out cells\n≥10 per perturbation',ha='center',va='center',fontsize=6.1,linespacing=1.25)
ax.text(.90,.25,'Calr response\nin test batch',ha='center',va='center',fontsize=6.1,linespacing=1.25)
ax.text(.36,1.08,'Inputs',ha='center',va='bottom',fontsize=6.3,color='#596872')
ax.text(.65,1.08,'Estimation',ha='center',va='bottom',fontsize=6.3,color='#596872')
ax.text(.90,1.08,'Evaluation',ha='center',va='bottom',fontsize=6.3,color='#596872')

# b: All target-level errors, first averaged over splits and eligible batches.
b=pd.read_csv(DATA/'panel_b_target_errors.csv')
assert b.groupby('budget').size().to_dict()=={5:33,10:24}
assert b.target.nunique()==33
assert np.all(b[['raw_cardiac_delta_mse','KiPERT_cardiac_delta_mse']].to_numpy()>0), 'Log axes require strictly positive MSE'
ax=axes['b'];lims=(.0018,.18)
ax.plot(lims,lims,'--',color=LINE,lw=.65,zorder=0)
for budget,marker in [(5,'o'),(10,'s')]:
    rows=b[b.budget==budget]
    ax.scatter(rows.KiPERT_cardiac_delta_mse,rows.raw_cardiac_delta_mse,
               s=15,marker=marker,facecolor=KIPERT if budget==5 else 'white',
               edgecolor=KIPERT,linewidth=.75,alpha=1,zorder=3,label=f'{budget} cells (n = {len(rows)})')
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(lims);ax.set_ylim(lims)
ticks=[.003,.01,.03,.1];ticklabels=['0.003','0.01','0.03','0.1']
ax.set_xticks(ticks,ticklabels);ax.set_yticks(ticks,ticklabels);ax.minorticks_off()
ax.set_xlabel('KiPERT cardiac MSE');ax.set_ylabel('Sample-mean cardiac MSE')
ax.legend(loc='upper left',handletextpad=.35,borderaxespad=.25,labelspacing=.5)

# c: Exact stored paired cluster intervals, including ten-cell uncertainty.
c=pd.read_csv(DATA/'panel_c_paired_reductions.csv').set_index(['budget','metric'])
ax=axes['c'];ax.axvline(0,color=LINE,lw=.65,ls='--',zorder=0)
order=[(5,'cardiac_delta_mse'),(5,'cardiac_absolute_mse'),(10,'cardiac_delta_mse'),(10,'cardiac_absolute_mse')]
ypositions=[3.3,2.5,1.2,.4]
for y,key in zip(ypositions,order):
    row=c.loc[key];value=row.relative_gain*100;low=row.relative_gain_ci_low*100;high=row.relative_gain_ci_high*100
    assert low<=value<=high
    ax.errorbar(value,y,xerr=[[value-low],[high-value]],fmt='o' if key[0]==5 else 's',
        color=KIPERT,markerfacecolor=KIPERT if key[0]==5 else 'white',markersize=3.8,
        markeredgewidth=.75,elinewidth=.8,capsize=2,capthick=.7,zorder=3)
    ax.text(73,y,f'{value:.1f}%',ha='right',va='center',fontsize=6.2,color=INK)
ax.set_yticks(ypositions,['5 cells\nExpression change','5 cells\nMean expression','10 cells\nExpression change','10 cells\nMean expression'])
ax.tick_params(axis='y',length=0,pad=5);ax.set_ylim(-.2,3.9);ax.set_xlim(-10,75)
ax.set_xticks([0,20,40,60]);ax.set_xlabel('MSE reduction versus sample mean (%)')
ax.spines['left'].set_visible(False)

# d: Complete source-observed landscape; known Calr highlighted in each pool.
d=pd.read_csv(DATA/'panel_d_predicted_landscape.csv');batches=['2w1','2w2','2w3','2w4']
assert len(d)==211 and not d.duplicated(['target','batch']).any()
assert d.groupby('batch').size().reindex(batches).tolist()==[55,48,54,54]
ax=axes['d'];groups=[d[d.batch==batch].sort_values('target') for batch in batches]
violins=ax.violinplot([x.prediction.to_numpy() for x in groups],positions=np.arange(4),
                     widths=.66,showmeans=False,showextrema=False,points=100,bw_method=.35)
for v in violins['bodies']:
    v.set_facecolor('#E7ECEF');v.set_edgecolor('#C5CDD2');v.set_linewidth(.55);v.set_alpha(1)
for i,rows in enumerate(groups):
    # Deterministic horizontal jitter represents category separation only.
    x=i+np.array([(int(hashlib.sha256(t.encode()).hexdigest()[:8],16)%1001/1000-.5)*.43 for t in rows.target])
    noncalr=rows.target.ne('Calr').to_numpy()
    ax.scatter(x[noncalr],rows.prediction.to_numpy()[noncalr],s=5,color='#87939C',alpha=.7,linewidths=0,zorder=2)
    calr=rows[rows.target.eq('Calr')];assert len(calr)==1 and calr.prediction.iloc[0]==rows.prediction.max()
    ax.scatter([i],calr.prediction,s=21,color=KIPERT,edgecolors='white',linewidths=.5,zorder=5)
ax.axhline(0,color='#D0D6DA',lw=.55,zorder=0)
ax.set_xlim(-.55,3.55);ax.set_ylim(-.35,1.25);ax.set_yticks([0,.4,.8,1.2])
ax.set_xticks(range(4),[f'{batch}\n(n = {len(rows)})' for batch,rows in zip(batches,groups)])
ax.set_xlabel('Test batch',labelpad=3)
ax.set_ylabel('Predicted expression change',labelpad=3)
ax.legend(handles=[Line2D([],[],marker='o',linestyle='none',color=KIPERT,markersize=3.8,label='Calr'),
                   Line2D([],[],marker='o',linestyle='none',color='#87939C',markersize=2.4,label='Other perturbations')],
          loc='upper right',fontsize=5.5,handletextpad=.4,borderaxespad=.15,labelspacing=.3)

# e: Same original normalized expression scale, all eight genes and four batches.
e=pd.read_csv(DATA/'panel_e_Calr_gene_responses.csv');genes=['Tpm1','Tnnt2','Ryr2','Ttn','Ctnna3','Myl4','Myh6','Actc1']
assert len(e)==32 and not e.duplicated(['gene','batch']).any()
lookup=e.set_index(['gene','batch'])
observed=np.array([[lookup.loc[(g,b),'observed'] for b in batches] for g in genes])
predicted=np.array([[lookup.loc[(g,b),'KiPERT_OT_prediction'] for b in batches] for g in genes])
supported=np.array([[(lookup.loc[(g,b),'cell_sampling_ci_low']>0 or lookup.loc[(g,b),'cell_sampling_ci_high']<0) for b in batches] for g in genes])
assert supported.sum()==18 and (np.sign(observed[supported])==np.sign(predicted[supported])).all()
assert np.all(observed[6]==0) and np.all(predicted[6]==0)
vmax=np.ceil(np.max(np.abs(np.r_[observed.ravel(),predicted.ravel()]))*10)/10
norm=Normalize(-vmax,vmax);ax=axes['e']
for left,values in [(0,observed),(5,predicted)]:
    im=ax.pcolormesh(np.arange(5)+left,np.arange(9),values,norm=norm,cmap='RdBu_r',
                    edgecolors='white',linewidth=.35,shading='flat',rasterized=False)
row,col=np.where(supported);ax.scatter(col+.5,row+.5,s=3.8,color=OBS,zorder=4)
ax.set_xlim(0,9);ax.set_ylim(8,0)
ax.set_yticks(np.arange(8)+.5,[g+'†' if g=='Myh6' else g for g in genes])
ax.set_xticks([.5,1.5,2.5,3.5,5.5,6.5,7.5,8.5],batches+batches)
ax.tick_params(length=0,pad=2,labelsize=5.5)
for spine in ax.spines.values():spine.set_visible(False)
ax.text(2,-.40,'Observed',ha='center',va='bottom',fontsize=6.5)
ax.text(7,-.40,'KiPERT',ha='center',va='bottom',fontsize=6.5,color=KIPERT,fontweight='bold')
cax=fig.add_axes([.451,.043,.195,.012])
cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-2,0,2])
cb.ax.tick_params(labelsize=5.5,length=2,pad=1);cb.outline.set_linewidth(.5)
cb.ax.set_title('Expression change',fontsize=5.5,pad=2)

# f: Measurement uncertainty belongs to observed means, not model predictions.
f=pd.read_csv(DATA/'panel_f_Calr_module.csv').set_index('batch').loc[batches]
assert len(f)==4
np.testing.assert_allclose(observed.mean(0),f.module_observed,rtol=0,atol=1e-12)
np.testing.assert_allclose(predicted.mean(0),f.module_prediction,rtol=0,atol=1e-12)
ax=axes['f'];ys=np.arange(3,-1,-1)
for y,(_,r) in zip(ys,f.iterrows()):
    ax.plot([r.module_observed,r.module_prediction],[y,y],color='#B7BFC4',lw=.7,zorder=1)
    ax.errorbar(r.module_observed,y,xerr=[[r.module_observed-r.module_cell_ci_low],[r.module_cell_ci_high-r.module_observed]],
                fmt='o',color=OBS,markersize=3,elinewidth=.65,capsize=2,capthick=.6,zorder=3)
    ax.scatter(r.module_prediction,y,s=17,marker='D',color=KIPERT,edgecolor='white',linewidth=.4,zorder=4)
ax.set_yticks(ys,batches);ax.set_ylim(-.5,4.4);ax.set_xlim(0,1.25);ax.set_xticks([0,.6,1.2])
ax.set_xlabel('Expression change',labelpad=3)
ax.legend(handles=[Line2D([],[],marker='o',linestyle='none',color=OBS,markersize=3,label='Observed'),
                   Line2D([],[],marker='D',linestyle='none',color=KIPERT,markersize=3,label='KiPERT')],
          loc='upper left',fontsize=5.5,handletextpad=.4,borderaxespad=.15,labelspacing=.3)


fig.savefig(OUT/'Figure5.pdf')
plt.close(fig)
