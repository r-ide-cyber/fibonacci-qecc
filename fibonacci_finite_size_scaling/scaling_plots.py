"""Finite-size plots; only summary tables are used, never filtered detail rows."""
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE = '#2368aa'
ORANGE = '#d76b26'
GREEN = '#248571'

def _finish(fig, name, directory, save):
    if save:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / (name + '.png'), dpi=170, bbox_inches='tight')
        fig.savefig(directory / (name + '.svg'), bbox_inches='tight')
    return fig


def plot_scaling(summary, kind='rates', output_dir='results/figures', save=False):
    """One figure for each (pair,W), with a panel per m. CI: pointwise Wilson 95%."""
    if summary.empty:
        return []
    if kind not in {'rates', 'gap', 'failure', 'inverse_N'}:
        raise ValueError(kind)
    figures=[]
    for (pair, W), group in summary.groupby(['pair_id','window_size'], sort=True):
        sizes=sorted(group.erasure_size.unique())
        cols=min(3,len(sizes)); rows=math.ceil(len(sizes)/cols)
        fig,axes=plt.subplots(rows,cols,figsize=(4.1*cols,3.0*rows),squeeze=False,layout='constrained')
        for ax,m in zip(axes.flat,sizes):
            g=group[group.erasure_size.eq(m)].sort_values('N')
            x=1/g.N.to_numpy(float) if kind=='inverse_N' else g.N.to_numpy(float)
            a=g.R_KL_arbitrary.to_numpy(float); c=g.R_KL_contiguous.to_numpy(float)
            lo=g.R_KL_arbitrary_ci95_low.to_numpy(float); hi=g.R_KL_arbitrary_ci95_high.to_numpy(float)
            sampled=g.calculation_mode.eq('sampled').to_numpy()
            if kind in {'rates','inverse_N'}:
                y=a; low=lo; high=hi; color=ORANGE
                ax.plot(x,a,color=ORANGE,lw=1.6,label='Arbitrary')
                if kind=='rates':
                    ax.plot(x,c,color=BLUE,lw=1.5,marker='s',ms=3,label='Contiguous (exact)')
                ylabel='KL correction rate'
                ax.set_ylim(-.04,1.06)
            elif kind=='gap':
                y=c-a; low=c-hi; high=c-lo; color=GREEN
                ax.plot(x,y,color=color,lw=1.6)
                ax.axhline(0,color='#67717a',lw=.7)
                ylabel='Contiguous − arbitrary'
                ax.set_ylim(-1.06,1.06)
            else:
                y=1-a; low=1-hi; high=1-lo; color=ORANGE
                ax.plot(x,y,color=color,lw=1.6)
                ylabel='Arbitrary failure rate'
                ax.set_ylim(-.04,1.06)
            for flag,marker in [(False,'o'),(True,'D')]:
                sel=sampled==flag
                ax.scatter(x[sel],y[sel],s=23,marker=marker,color=color,zorder=4)
            if sampled.any():
                ax.errorbar(x[sampled],y[sampled],yerr=np.maximum(0,np.array([y-low,high-y]))[:,sampled],
                            fmt='none',ecolor=color,capsize=3,alpha=.85,lw=1)
            ax.set(title=f'm = {m}',xlabel='1 / N' if kind=='inverse_N' else 'System size N',ylabel=ylabel)
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(sizes):]:
            ax.set_visible(False)
        handles=[Line2D([0],[0],marker='o',color=ORANGE,lw=0,label='Arbitrary: exact'),
                 Line2D([0],[0],marker='D',color=ORANGE,lw=0,label='Arbitrary: sampled; 95% Wilson CI')]
        if kind=='rates':
            handles.insert(0,Line2D([0],[0],marker='s',color=BLUE,label='Contiguous: exact'))
        fig.legend(handles=handles,loc='outside lower center',ncol=min(3,len(handles)),fontsize=9)
        fig.suptitle(f'{pair} | fixed W = {W} | {kind.replace("_", " ")}',fontsize=13)
        figures.append(_finish(fig,f'{pair}_W{W}_{kind}',output_dir,save))
    return figures


def plot_heatmaps(summary, output_dir='results/figures', save=False):
    if summary.empty:
        return []
    figures=[]
    for (pair,W),g in summary.groupby(['pair_id','window_size'],sort=True):
        fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
        for ax,column,label in zip(axes,['R_KL_contiguous','R_KL_arbitrary'],['Contiguous (exact)','Arbitrary (* = sampled)']):
            matrix=g.pivot(index='N',columns='erasure_size',values=column).sort_index()
            mode=g.pivot(index='N',columns='erasure_size',values='calculation_mode').reindex(index=matrix.index,columns=matrix.columns)
            im=ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)),vmin=0,vmax=1,cmap='cividis',aspect='auto',origin='lower')
            ax.set_xticks(range(len(matrix.columns)),[str(x) for x in matrix.columns])
            ax.set_yticks(range(len(matrix.index)),[str(x) for x in matrix.index])
            ax.set(xlabel='Erased qubits m',ylabel='System size N',title=label)
            if matrix.size<=160:
                for i in range(len(matrix)):
                    for j in range(len(matrix.columns)):
                        value=matrix.iloc[i,j]
                        if not np.isfinite(value):
                            continue
                        suffix='*' if column=='R_KL_arbitrary' and mode.iloc[i,j]=='sampled' else ''
                        ax.text(j,i,f'{value:.2f}{suffix}',ha='center',va='center',fontsize=8,
                                color='white' if value<.5 else '#142535')
        fig.colorbar(im,ax=axes,label='KL correction rate',shrink=.8)
        fig.suptitle(f'{pair} | fixed W = {W}')
        figures.append(_finish(fig,f'{pair}_W{W}_heatmaps',output_dir,save))
    return figures
