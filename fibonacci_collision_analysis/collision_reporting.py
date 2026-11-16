"""Scientific figures; grouped comparisons avoid mixing N or erasure ensembles."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS=['#146C94','#E97835','#697C47','#9F5D9C','#5B718C','#BD536A','#997932']

def _figure_dict(out, save_directory=None):
    plt.rcParams.update({'figure.dpi':120,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'font.size':10})
    figs={}; s=out['spectra']; j=out['joint_spectrum']; c=out['comparison']; e=out['expected_collisions']; summaries=out['logical_summary']
    # Show representative configuration clearly; all configurations remain in tables.
    first=summaries.iloc[0]; pid=first.pair_id
    n=int(summaries[summaries.pair_id==pid].inflation_n.max())
    focus=s[(s.pair_id==pid)&(s.inflation_n==n)]
    for name,metric,label in [('01_cross_hamming','cross_hamming','Hamming weight w'),('02_cross_span','cross_span','Cyclic span ell')]:
        data=focus[focus.metric_type==metric]
        fig,ax=plt.subplots(figsize=(7,4));ax.bar(data.value,data['count'],color=COLORS[0]);ax.set(xlabel=label,ylabel='Cross pair count',title=f'{pid}, n={n}');fig.tight_layout();figs[name]=fig
    jf=j[(j.pair_id==pid)&(j.inflation_n==n)&(j.pair_type=='cross')]
    fig,ax=plt.subplots(figsize=(7,4))
    scatter=ax.scatter(jf.hamming_weight,jf.cyclic_span,s=20+15*np.log1p(jf['count']),c=np.log1p(jf['count']),cmap='viridis',alpha=.8)
    fig.colorbar(scatter,ax=ax,label='log(1 + pair count)');ax.set(xlabel='Hamming weight w',ylabel='Cyclic span ell',title=f'{pid}, n={n}');fig.tight_layout();figs['03_joint_spectrum']=fig
    ec=e[(e.pair_id==pid)&(e.inflation_n==n)]
    W=int(ec.window_size.max()); ec=ec[ec.window_size==W].sort_values('erasure_size')
    fig,ax=plt.subplots(figsize=(7,4))
    for col,label,color in [('expected_cross_collision_arbitrary','Arbitrary',COLORS[0]),('expected_cross_collision_contiguous','Contiguous',COLORS[1])]: ax.plot(ec.erasure_size,ec[col],'o-',label=label,color=color)
    ax.set(xlabel='Erasure size m',ylabel='Expected colliding cross pairs',title=f'{pid}, n={n}, W={W}');ax.legend();fig.tight_layout();figs['04_expected_collision']=fig
    for name,x,y,xlabel,ylabel in [('05_distance_vs_arbitrary','d_cross','R_KL_arbitrary','Minimum cross Hamming distance','Arbitrary KL correction rate'),('06_span_vs_contiguous','ell_cross_min','R_KL_contiguous','Minimum cross cyclic span','Contiguous KL correction rate'),('08_geometry_vs_gap','low_weight_large_span_unique_fraction','correction_rate_gap','Low-weight / large-span unique-support fraction','R_contiguous - R_arbitrary')]:
        # Largest equal-N stratum with the most distinct logical pairs. Pick m
        # showing response variation when possible; report exact stratum in title.
        groups=list(c.groupby(['inflation_n','N','window_size','erasure_size']))
        key,g=max(groups,key=lambda item:(item[1].pair_id.nunique(),item[1][y].nunique(),item[0][0],item[0][-1]))
        fig,ax=plt.subplots(figsize=(7,4))
        for i,(p,gg) in enumerate(g.groupby('pair_id')):
            if y=='R_KL_arbitrary':
                errors=np.vstack((gg[y]-gg.R_KL_low_arbitrary,gg.R_KL_high_arbitrary-gg[y]))
                ax.errorbar(gg[x],gg[y],yerr=errors,fmt='o',capsize=3,label=p,color=COLORS[i%len(COLORS)])
            else: ax.scatter(gg[x],gg[y],label=p,color=COLORS[i%len(COLORS)],s=50)
        ax.set(xlabel=xlabel,ylabel=ylabel,title=f'n={key[0]}, N={key[1]}, W={key[2]}, m={key[3]}')
        ax.legend(fontsize=7,loc='best');fig.tight_layout();figs[name]=fig
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,kind in zip(axs,['arbitrary','contiguous']):
        ax.scatter(c['expected_cross_collision_'+kind],c['KL_failure_rate_'+kind],s=20,alpha=.6,color=COLORS[0 if kind=='arbitrary' else 1]);ax.set(xlabel='Expected cross pair count',ylabel='KL failure rate',title=f'{kind}: all configurations (descriptive)')
    fig.tight_layout();figs['07_expected_vs_KL_failure']=fig
    cc=c[(c.pair_id==pid)&(c.inflation_n==n)&(c.window_size==W)].sort_values('erasure_size')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for metric,label,col in [('cross_hamming','Hamming criterion',COLORS[0]),('cross_span','Span criterion',COLORS[1])]:
        data=focus[focus.metric_type==metric]
        frac=[data.loc[data.value<=m,'count'].sum()/data['count'].sum() for m in cc.erasure_size]
        axs[0].plot(cc.erasure_size,frac,'o-',color=col,label=label)
    for kind,col in [('arbitrary',COLORS[0]),('contiguous',COLORS[1])]:
        axs[1].plot(cc.erasure_size,cc['R_KL_'+kind],'o-',color=col,label=kind)
        axs[1].fill_between(cc.erasure_size,cc['R_KL_low_'+kind],cc['R_KL_high_'+kind],color=col,alpha=.18)
    axs[0].set(xlabel='Erasure size m',ylabel='Cumulative cross pair fraction');axs[1].set(xlabel='Erasure size m',ylabel='KL correction rate (sampled: 95% CI)',ylim=(-.05,1.05))
    for ax in axs: ax.legend()
    fig.suptitle(f'{pid}, n={n}, W={W}');fig.tight_layout();figs['09_cumulative_and_KL']=fig
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for i,(p,g) in enumerate(summaries.groupby('pair_id')):
        g=g.sort_values('N');axs[0].plot(g.N,g.d_cross,'o-',label=p,color=COLORS[i%5]);axs[1].plot(g.N,g.ell_cross_min,'o-',label=p,color=COLORS[i%5])
    axs[0].set(xlabel='N',ylabel='d_cross');axs[1].set(xlabel='N',ylabel='ell_cross_min');axs[0].legend(fontsize=7);fig.tight_layout();figs['10_finite_size']=fig
    detail=out['erasure_collisions']; measured=detail.dropna(subset=['I_RE','D_X','D_Z'])
    if len(measured):
        fig,axs=plt.subplots(1,3,figsize=(12,3.6))
        for ax,x,y in zip(axs,['C_cross','C_cross','Delta_C_self'],['D_X','I_RE','D_Z']):
            ax.scatter(measured[x],measured[y],alpha=.35,s=10);ax.set(xlabel=x,ylabel=y)
        fig.suptitle('Measured erasure trials; pooled descriptive view');fig.tight_layout();figs['11_collision_leakage']=fig
    fig,ax=plt.subplots(figsize=(7,4))
    ax.stackplot(cc.erasure_size,cc.P_cross_collision_arbitrary,cc.diagonal_only_failure_rate_arbitrary,labels=['Cross collision failure','Diagonal-only failure'],colors=COLORS[:2],alpha=.8)
    ax.set(xlabel='Erasure size m',ylabel='Arbitrary KL failure rate',ylim=(0,1.02),title=f'{pid}, n={n}, W={W}');ax.legend();fig.tight_layout();figs['12_failure_decomposition']=fig
    if save_directory:
        directory=Path(save_directory);directory.mkdir(parents=True,exist_ok=True)
        for name,fig in figs.items():
            fig.savefig(directory/(name+'.png'),bbox_inches='tight');fig.savefig(directory/(name+'.svg'),bbox_inches='tight')
    return figs


def make_figures(out, config=None):
    config=config or {}
    directory=Path(config.get('OUTPUT_DIR','results'))/'figures' if config.get('SAVE_FIGURES',False) else None
    figures=_figure_dict(out,directory)
    for fig in figures.values(): plt.close(fig)
    return list(figures.values())

from collision_analysis import conclusions
