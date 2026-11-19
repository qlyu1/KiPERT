"""Reformat the retained KiPERT panels from existing experimental results."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
W=Path(__file__).resolve().parents[2]/'experiments/extended_data/reference'
O=Path(sys.argv[1]);O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],
 'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
 'legend.fontsize':6.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'xtick.major.width':.65,'ytick.major.width':.65,'lines.linewidth':1.,'savefig.dpi':600})
BLUE='#287C95';RED='#C45D45';GRAY='#A8ADB0'
V=['DepMap','GenePT','STRING','Reactome','GO_BP','CORUM','CollecTRI','HPA','GO_MF','GO_CC','HPO','WikiPathways']
DS=['S3','S4','S5','S6','S7','macro']

def read(key):return pd.read_csv(W/f'{key}.csv')
def label(ax,letter):
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-28,13),textcoords='offset points',
                ha='left',va='bottom',fontsize=8,fontweight='bold',annotation_clip=False)

def grid(nrow=1,height=82):
    fig,arr=plt.subplots(nrow,2,figsize=(180/25.4,height/25.4),squeeze=False)
    fig.subplots_adjust(left=.18,right=.97,bottom=.23 if nrow==1 else .13,
                        top=.85 if nrow==1 else .91,wspace=.70,hspace=.80)
    axes=list(arr.ravel())
    for ax,letter in zip(axes,'abcd'):label(ax,letter)
    return fig,axes

def three(height=108):
    fig=plt.figure(figsize=(180/25.4,height/25.4))
    gs=fig.add_gridspec(2,2,left=.18,right=.97,bottom=.15,top=.90,wspace=.75,hspace=.90)
    axes=[fig.add_subplot(gs[:,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])]
    for ax,letter in zip(axes,'abc'):label(ax,letter)
    return fig,axes

def forest(ax,df,xlabel,labels=None):
    ax.axvline(0,color=GRAY,lw=.65,zorder=0)
    for i,(_,r) in enumerate(df.iterrows()):
        val,lo,hi=r.relative_change_percent,r.relative_ci_low_percent,r.relative_ci_high_percent
        color=BLUE if val>=0 else RED
        macro=('dataset' in r and str(r.dataset).startswith('macro')) or ('display_label' in r and str(r.display_label).startswith('Equal-'))
        ax.plot([lo,hi],[i,i],color=color,lw=1.2 if macro else .9)
        ax.plot(val,i,marker='D' if macro else 'o',ms=4 if macro else 3,color=color,zorder=3)
    if labels is None:labels=list(df.display_label)
    ax.set_yticks(range(len(df)),labels)
    ax.set_ylim(len(df)-.45,-.55)
    ax.tick_params(axis='y',length=0)
    ax.set_xlabel(xlabel,labelpad=7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
    ax.margins(x=.10)
    for tick in ax.get_yticklabels():
        if tick.get_text().startswith('Equal-'):tick.set_fontweight('bold')

def curve(ax,x,y,ylabel,xlabel,highlight=None):
    ax.plot(x,y,'o-',color=BLUE,ms=3,lw=1.1)
    if highlight is not None:ax.axvline(highlight,color=GRAY,ls=':',lw=.7,zorder=0)
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_xlabel(xlabel,labelpad=5);ax.set_ylabel(ylabel,labelpad=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
    ax.margins(x=.08,y=.20)

def save(fig,axes,number):
    fig.canvas.draw()
    fig.savefig(O/f'Extended_Data_Fig_{number}.pdf')
    plt.close(fig)

def fig1():
    fig,axes=grid(2,145)
    forest(axes[0],read('1a'),'Relative PDS increase (%)')
    forest(axes[1],read('1b'),'Relative AUPRC increase (%)')
    axes[0].set_title('Gene–prior correspondence: PDS',pad=12)
    axes[1].set_title('Gene–prior correspondence: AUPRC',pad=12)
    for ax,key,ylabel in zip(axes[2:],['1c','1d'],['Mean cosine PDS','Mean DEG AUPRC']):
        z=read(key).sort_values('size');x=z['size'].to_numpy()
        ax.plot(x,z['mean'],'o-',color=BLUE,ms=3)
        ax.fill_between(x,z.q25,z.q75,color=BLUE,alpha=.15,lw=0)
        ax.set_xticks([1,2,4,6,8,10,12]);ax.set_xlabel('Number of priors');ax.set_ylabel(ylabel)
        ax.set_title('Prior-number dependence',pad=12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4));ax.margins(y=.12)
    save(fig,axes,1)

def fig2():
    fig,axes=grid(1,110);fig.subplots_adjust(bottom=.30,top=.87)
    z=read('2a');a=z.pivot(index='view',columns='dataset',values='relative_loss_percent').reindex(index=V,columns=DS)
    limit=float(np.ceil(np.abs(a.to_numpy()).max()/2)*2)
    im=axes[0].imshow(a,aspect='auto',interpolation='nearest',cmap='RdBu',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
    axes[0].set_yticks(range(12),[v.replace('GO_','GO ') for v in V]);axes[0].set_xticks(range(6),['Tian i','Tian a','Adamson','Norman','Frangieh','Mean'],rotation=40,ha='right',rotation_mode='anchor')
    axes[0].tick_params(length=0);axes[0].spines[['left','bottom']].set_visible(False)
    axes[0].set_title('Leave one prior out',pad=12)
    pos=axes[0].get_position();cax=fig.add_axes([pos.x0,.095,pos.width,.022])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.ax.tick_params(labelsize=6,length=2)
    cb.set_label('Relative PDS loss on removal (%)',size=7,labelpad=5)
    forest(axes[1],read('2b'),'Relative AUPRC increase (%)')
    axes[1].set_title('Full model versus three priors',pad=12)
    save(fig,axes,2)

def fig3():
    fig,axes=grid(1,84)
    forest(axes[0],read('3a'),'Relative AUPRC increase (%)')
    axes[0].set_title('Full model versus high-coverage group',pad=12)
    forest(axes[1],read('3b'),'Relative AUPRC increase (%)',['Low','Intermediate','High'])
    axes[1].set_title('Prior similarity to training genes',pad=12)
    save(fig,axes,3)

def fig4():
    fig,axes=three(112)
    forest(axes[0],read('4a'),'Relative AUPRC increase (%)')
    axes[0].set_title('Alignment versus uniform weights',pad=12)
    curve(axes[1],read('4b').beta,read('4b').pds_query_cosine,'Mean cosine PDS','Alignment fraction β',.5)
    curve(axes[2],read('4c').beta,read('4c').auprc_dg,'Mean DEG AUPRC','Alignment fraction β',.5)
    save(fig,axes,4)

def fig5():
    fig,ax=plt.subplots(figsize=(89/25.4,68/25.4));fig.subplots_adjust(left=.23,right=.96,bottom=.25,top=.85)
    forest(ax,read('5'),'Relative nMSE increase (%)',['Seen 0','Seen 1','Seen 2'])
    ax.set_title('Removal of CORUM',pad=12)
    save(fig,[ax],5)

def fig6():
    fig,axes=three(112)
    forest(axes[0],read('6a'),'Relative L1 PDS increase (%)')
    axes[0].set_title('Full OT versus identity mapping',pad=12)
    curve(axes[1],read('6b').eta,read('6b').pds_l1_exclude_target,'Mean L1 PDS','Transport strength η')
    curve(axes[2],read('6c').eta,read('6c').mse_delta,'Mean expression-change MSE','Transport strength η')
    save(fig,axes,6)

if __name__=='__main__':
    numbers=[int(x) for x in sys.argv[2:]] or list(range(1,7))
    for n in numbers:globals()['fig'+str(n)]();print('Exported figure',n,flush=True)
