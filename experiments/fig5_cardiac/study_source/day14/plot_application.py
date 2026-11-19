"""All eligible D14 cardiac responses, evaluated separately in each held-out batch."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
assert (ROOT/'evaluation/decision.json').exists()
Q=pd.read_csv(ROOT/'evaluation/query_results.csv').query('threshold==5 and method=="KiPERT_OT" and program=="cardiac"').copy()
B=pd.read_csv(ROOT/'evaluation/batch_results.csv').query('threshold==5 and method=="KiPERT_OT" and program=="cardiac"')
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
fig,axs=plt.subplots(2,2,figsize=(175/25.4,158/25.4),sharex=True,sharey=True)
fig.subplots_adjust(left=.13,right=.97,bottom=.16,top=.92,wspace=.29,hspace=.35)
source=[]
for ax,batch,letter in zip(axs.ravel(),sorted(Q.batch.unique()),'abcd'):
 d=Q[Q.batch==batch].copy();n=len(d);x=(rankdata(d.observed)-1)/(n-1);y=(rankdata(d.prediction)-1)/(n-1)
 ax.plot([0,1],[0,1],ls='--',lw=.7,color='#bfc5c8',zorder=0)
 ref=d.target.eq('Calr').to_numpy()
 ax.scatter(x[~ref],y[~ref],s=15,alpha=.73,color='#527c8e',edgecolors='none',zorder=2)
 ax.scatter(x[ref],y[ref],s=70,marker='*',color='#bd653f',edgecolors='white',linewidths=.5,zorder=4)
 if ref.any():
  xx=float(x[ref][0]);yy=float(y[ref][0]);ax.annotate('Calr',(xx,yy),xytext=(-6 if xx>.8 else 6,-12 if yy>.8 else 6),textcoords='offset points',ha='right' if xx>.8 else 'left',fontsize=7,color='#a94e2c')
 r=B[B.batch==batch].iloc[0]
 ax.set_title(f'{batch} held out\nn = {n}; Spearman ρ = {r.rho:.2f}',loc='left',pad=10,fontsize=8)
 ax.text(-.18,1.11,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)
 ax.set_xlim(-.045,1.045);ax.set_ylim(-.045,1.045);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.set_aspect('equal')
 for (_,row),xx,yy in zip(d.iterrows(),x,y):source.append(dict(target=row.target,batch=batch,observed=row.observed,prediction=row.prediction,observed_percentile=xx,predicted_percentile=yy,n_query=row.n_query))
for ax in axs[1,:]:ax.set_xlabel('Observed cardiac-response percentile',labelpad=7)
for ax in axs[:,0]:ax.set_ylabel('Predicted percentile',labelpad=7)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#527c8e',label='Other targets',markersize=4),Line2D([],[],marker='*',ls='',color='#bd653f',label='Calr: published reference',markersize=8)],loc='lower center',bbox_to_anchor=(.53,.045),ncol=2,fontsize=7)
fig.text(.13,.012,'56 targets; 211 held-out target–batch predictions. No averaging across batches before ranking.',fontsize=6.5,color='#4d5960')
fig.canvas.draw()
# Check that every text bounding box stays inside the exported page.
renderer=fig.canvas.get_renderer();bounds=fig.bbox;issues=[]
for text in fig.findobj(mpl.text.Text):
 if not text.get_visible() or not text.get_text():continue
 box=text.get_window_extent(renderer)
 if box.x0<bounds.x0-1 or box.x1>bounds.x1+1 or box.y0<bounds.y0-1 or box.y1>bounds.y1+1:issues.append(text.get_text())
assert not issues,issues
for ext in ['pdf','svg']:fig.savefig(OUT/f'D14_cardiac_rank_recovery.{ext}',facecolor='white')
pd.DataFrame(source).to_csv(OUT/'D14_cardiac_rank_source.csv',index=False)
legend=('D14 cardiac-response recovery in four held-out sample batches. Each panel includes every eligible query in that batch. '
 'KiPERT uses measured same-target responses from at least two other D14 batches and only non-targeting controls from the held-out batch. '
 'Each represented source/query has at least five captured QC cells. Points show within-batch percentiles of the predicted and observed mean expression change over '
 'Tpm1, Tnnt2, Ryr2, Ttn, Ctnna3, Myl4, Myh6 and Actc1. The dashed line denotes identical ranks. '
 'Calr is highlighted as a prespecified published reference, not a newly discovered target. Batch-specific sample sizes count targets; '
 '211 target–batch predictions arise from 56 distinct targets and are not independent biological replicates. The program is a frozen historical exploratory readout. '
 'No batch-averaged LOBO predictions are correlated with pooled truth.\n')
(OUT/'D14_cardiac_rank_legend.txt').write_text(legend)
(OUT/'figure_checks.json').write_text(json.dumps({'all_text_inside_page':True,'all_eligible_queries_plotted':len(source)==211,'backend':'matplotlib','font_min_pt':6.5},indent=2))
print('FIGURE_COMPLETE',len(source),flush=True)
