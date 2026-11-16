"""Plots and conservative Japanese conclusions for logical leakage experiments."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {'contiguous': '#176b87', 'arbitrary': '#cf5c36'}

def _finish(fig, filename, output_dir, save):
    fig.tight_layout()
    if save:
        folder = Path(output_dir) / 'figures'
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / (filename + '.png'), dpi=160, bbox_inches='tight')
        fig.savefig(folder / (filename + '.svg'), bbox_inches='tight')
    return fig

def _axes(count, width=4.4, height=3.3):
    columns = min(3, max(1, count))
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns*width, rows*height), squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, list(axes.flat)

def plot_vs_m(summary, metric='mean_I_RE', output_dir='results', save=False):
    groups = list(summary.groupby(['pair_id','inflation_n','N','window_size'], sort=True))
    if not groups:
        return None
    fig, axes = _axes(len(groups))
    labels = {'mean_I_RE': 'Mean I(R:E) [bit]', 'max_I_RE': 'Maximum observed I(R:E) [bit]',
              'mean_D_Z': 'Mean D_Z', 'mean_D_X': 'Mean D_X'}
    names = {'mean_I_RE':'leakage_vs_m','max_I_RE':'worst_leakage_vs_m',
             'mean_D_Z':'DZ_vs_m','mean_D_X':'DX_vs_m'}
    for ax, (key, data) in zip(axes, groups):
        for kind, sub in data.groupby('erasure_type', sort=True):
            sub = sub.sort_values('erasure_size')
            error = sub['std_I_RE'].fillna(0) if metric == 'mean_I_RE' else None
            ax.errorbar(sub.erasure_size, sub[metric], yerr=error, marker='o', ms=4,
                        capsize=2, color=COLORS[kind], label=kind)
        pair, n, N, W = key
        ax.set_title(f'{pair}\nn={n}, N={N}, W={W}', fontsize=10)
        ax.set(xlabel='Erased sites m', ylabel=labels[metric], ylim=(-0.025, 2.05 if 'I_RE' in metric else 1.025))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    if metric == 'mean_I_RE':
        fig.suptitle('Error bars: pattern standard deviation (not confidence intervals)', y=1.005, fontsize=11)
    return _finish(fig, names[metric], output_dir, save)

def plot_scaling(summary, output_dir='results', save=False):
    data = summary[summary.erasure_domain_mode == 'window']
    groups = list(data.groupby(['pair_id','window_size','erasure_size'], sort=True))
    if not groups:
        return None
    fig, axes = _axes(len(groups))
    for ax, (key, sub) in zip(axes, groups):
        for kind, points in sub.groupby('erasure_type', sort=True):
            points = points.sort_values('N')
            ax.plot(points.N, points.mean_I_RE, 'o-', ms=4, color=COLORS[kind], label=kind)
        pair, W, m = key
        ax.set_title(f'{pair}\nW={W}, m={m}', fontsize=10)
        ax.set(xlabel='Physical length N (from inflation)', ylabel='Mean I(R:E) [bit]', ylim=(-.025,2.05))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed windows; shared relative patterns across sizes', y=1.005, fontsize=11)
    return _finish(fig, 'leakage_vs_N', output_dir, save)

def plot_scatter(detail, kind='DZ_DX', output_dir='results', save=False):
    fig, ax = plt.subplots(figsize=(7,4.5))
    for success, rows in detail.groupby('KL_correctable'):
        label = 'KL correctable' if success else 'KL failure'
        color = '#176b87' if success else '#cf5c36'
        if kind == 'DZ_DX':
            ax.scatter(rows.D_Z, rows.D_X, s=18, alpha=.35, color=color, label=label)
        else:
            ax.scatter(rows.I_RE, rows.KL_correctable.astype(int), s=18, alpha=.25, color=color, label=label)
    if kind == 'DZ_DX':
        ax.set(xlabel='D_Z = 0.5 ||rho00-rho11||_1', ylabel='D_X = ||rho01||_1',
               xlim=(-.025,1.025), ylim=(-.025,1.025))
    else:
        ax.set(xlabel='I(R:E) [bit]', ylabel='KL correctable', xlim=(-.025,2.025), yticks=[0,1])
    ax.grid(alpha=.2)
    ax.legend()
    return _finish(fig, 'DZ_DX_scatter' if kind == 'DZ_DX' else 'I_RE_KL_scatter', output_dir, save)

def plot_distribution(detail, output_dir='results', save=False):
    # A single explicitly labelled setting; combining different sizes would hide heterogeneity.
    key, data = next(iter(detail.groupby(['pair_id','inflation_n','N','window_size'], sort=True)))
    fig, ax = plt.subplots(figsize=(9,4.5))
    groups = list(data.groupby(['erasure_size','erasure_type'], sort=True))
    boxes = ax.boxplot([x.I_RE.to_numpy() for _,x in groups], patch_artist=True, showfliers=True,
                      tick_labels=[f'm={m}\n{t[:3]}' for (m,t),_ in groups])
    for patch, ((m,t),_) in zip(boxes['boxes'],groups):
        patch.set_facecolor(COLORS[t]); patch.set_alpha(.6)
    ax.set(title=f'Leakage distribution: {key[0]}, n={key[1]}, N={key[2]}, W={key[3]}',
           ylabel='I(R:E) [bit]', ylim=(-.025,2.05))
    ax.grid(axis='y',alpha=.2)
    return _finish(fig,'leakage_distribution',output_dir,save)

def make_all_figures(results, output_dir='results', save=False):
    if results['detail'].empty:
        return {}
    summary, detail = results['summary'], results['detail']
    figs = {metric: plot_vs_m(summary,metric,output_dir,save)
            for metric in ('mean_I_RE','max_I_RE','mean_D_Z','mean_D_X')}
    figs['scaling'] = plot_scaling(summary,output_dir,save)
    figs['DZ_DX'] = plot_scatter(detail,'DZ_DX',output_dir,save)
    figs['I_RE_KL'] = plot_scatter(detail,'I_RE_KL',output_dir,save)
    figs['distribution'] = plot_distribution(detail,output_dir,save)
    return {k:v for k,v in figs.items() if v is not None}

def scaling_assessment(summary, tol=1e-10):
    records=[]
    for keys, group in summary[summary.erasure_domain_mode=='window'].groupby(
            ['pair_id','window_size','erasure_size','erasure_type']):
        group=group.sort_values('N')
        vals=group.mean_I_RE.to_numpy()
        records.append(dict(zip(['pair_id','window_size','erasure_size','erasure_type'],keys),
            number_of_sizes=len(vals), first_N=int(group.N.iloc[0]), last_N=int(group.N.iloc[-1]),
            first_mean_I_RE=float(vals[0]),last_mean_I_RE=float(vals[-1]),
            observed_nonincreasing=bool(np.all(np.diff(vals)<=tol)) if len(vals)>1 else None,
            endpoint_decreased=bool(vals[-1]<vals[0]-tol) if len(vals)>1 else None,
            largest_size_all_observed_decoupled=bool(group.decoupled_rate.iloc[-1]==1),
            interpretation='finite observations; no extrapolation or infinite-size proof'))
    return pd.DataFrame(records)

def final_report(results, seed_checks=None, tol=1e-10):
    d,s=results['detail'],results['summary']
    if d.empty:
        return '実行対象の消失パターンがありません。skipped表を確認してください。'
    num=len(d); kl=int(d.KL_correctable.sum())
    bad=int((d.KL_correctable & (d.I_RE>=tol)).sum())
    rec_leak=d.recoverable_both & (d.I_RE>=tol)
    lines=['# 計算結果（実際に実行した設定のみ）','',
        f'- seed pair: '+', '.join(f'{a} / {b}' for a,b in d[['seed_0','seed_1']].drop_duplicates().itertuples(index=False,name=None)),
        f'- n / N: '+', '.join(f'{n} / {N}' for n,N in d[['inflation_n','N']].drop_duplicates().sort_values('N').itertuples(index=False,name=None)),
        f'- 全{num:,}試行（exactとsampledを含む）。KL成功{kl:,}試行、KL成功かつ I_RE ≥ tolerance は{bad}試行。',
        f'- recoverability成功かつ漏洩あり: {int(rec_leak.sum()):,}/{num:,}試行 ({rec_leak.mean():.3%})。',
        f'- 観測した I_RE の範囲: {d.I_RE.min():.8g}〜{d.I_RE.max():.8g} bit。',
        '- 全設定を混合した割合は、設定ごとの率とは別の参考値です。各mのKL率・mean/max I_RE・mean D_Z/D_Xはsummary表を参照してください。',
        '- sampledの最大値は観測最大値であり、未抽出パターンを含む最悪値の証明ではありません。','']
    worst=d.loc[d.I_RE.idxmax()]
    lines += ['## 観測最大の一例','',
        f"pair={worst.pair_id}, n={worst.inflation_n}, N={worst.N}, W={worst.window_size}, m={worst.erasure_size}, type={worst.erasure_type}",
        f"E={worst.erasure_positions_absolute}; I_RE={worst.I_RE:.8g}, D_Z={worst.D_Z:.8g}, D_X={worst.D_X:.8g}",'']
    assess=scaling_assessment(s,tol)
    if not assess.empty:
        varied=assess[assess.number_of_sizes>1]
        lines += ['## 有限サイズ比較','',
            f'固定W,m,typeで2サイズ以上ある{len(varied)}系列中、端点で平均漏洩が減少した系列は{int(varied.endpoint_decreased.sum())}です。',
            '有限個のN・有限サンプルから無限系の収束を断定しません。固定Wとfull_ringは区別します。','']
    if d.pair_id.nunique()<2:
        lines+=['論理pairが1組のため、Q4のpair間の相関は評価できません。LOGICAL_PAIRSに複数pairを指定できます。','']
    if seed_checks is not None:
        lines+=['seedの隣接交換条件および原論文の追加条件はseed_checks表に保存しています。','']
    return '\n'.join(lines)
