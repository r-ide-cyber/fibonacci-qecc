"""Descriptive conclusions for the observed finite sizes; no infinite-N fit."""
import pandas as pd


def convergence_assessment(summary):
    columns=['pair_id','window_size','erasure_size','N_first','N_last','R_first','R_last',
             'observed_change','gap_last','delta_last','observed_nondecreasing','all_sizes_exact','interpretation']
    if summary.empty:
        return pd.DataFrame(columns=columns)
    rows=[]
    for keys,g in summary.groupby(['pair_id','window_size','erasure_size'],sort=True):
        g=g.sort_values('N'); r=g.R_KL_arbitrary
        first,last=g.iloc[0],g.iloc[-1]
        diff=float(last.R_KL_arbitrary-first.R_KL_arbitrary)
        if len(g)<2:
            comment='有効なNが1点のみ。有限サイズ依存性は判定できない。'
        elif diff>0:
            comment='観測範囲では任意消失の訂正率が改善。有限サイズ効果と整合的。'
        elif r.eq(1).all():
            comment='観測した全サイズで成功率1。改善前のサイズはこの系列に含まれない。'
        elif diff==0:
            comment='両端の訂正率は同じ。今回の範囲だけでは有限サイズ効果を確定できない。'
        else:
            comment='両端比較では改善なし。内部の変動とsampling区間を確認する。'
        if g.calculation_mode.eq('sampled').any():
            comment+=' sampling値を含むため、率1のみから全パターン成功は断定しない。'
        rows.append(dict(zip(columns, [*keys,int(first.N),int(last.N),float(first.R_KL_arbitrary),
            float(last.R_KL_arbitrary),diff,float(last.gap_KL),float(r.diff().abs().iloc[-1]) if len(g)>1 else float('nan'),
            bool(r.diff().dropna().ge(0).all()) if len(g)>1 else None,
            bool(g.calculation_mode.eq('exact').all()),comment])))
    result = pd.DataFrame(rows,columns=columns)
    delta_trends = []
    for _, g in summary.groupby(['pair_id','window_size','erasure_size'],sort=True):
        deltas = g.sort_values('N').R_KL_arbitrary.diff().abs().dropna()
        delta_trends.append(bool(deltas.diff().dropna().le(1e-12).all()) if len(deltas) >= 2 else None)
    result['observed_Delta_R_nonincreasing'] = delta_trends
    return result


def conclusion_markdown(prepared, config, result, theory_comparison):
    lines=['### 自動まとめ', '']
    for pair in prepared['pairs']:
        lines.append(f"- `{pair['pair_id']}`: seed 0 = `{pair['seed_0']}` / seed 1 = `{pair['seed_1']}`。指定の隣接交換条件: 合格。")
    lines += [f"- INFLATION_LIST = `{config['INFLATION_LIST']}`",
              f"- WINDOW_SIZE_LIST = `{config['WINDOW_SIZE_LIST']}`",
              f"- ERASURE_SIZE_LIST = `{config['ERASURE_SIZE_LIST']}`",'']
    if result is None:
        lines.append('本計算はまだ実行していません。以下のN一覧・benchmark・validationは準備段階の結果です。率や有限サイズ効果の結論は未計算です。')
        return '\n'.join(lines)
    s=result['summary']
    if s.empty:
        lines.append('有効な解析組がありません。N ≥ W、m ≤ Wを満たす設定を確認してください。')
        return '\n'.join(lines)
    lines.append(f'有効な集計行数: {len(s):,}。各Nの訂正率とgap、thresholdは上の表を参照してください。')
    lines.append(f"任意消失の観測率1: {int(s.R_KL_arbitrary.eq(1).sum())} / {len(s)} 組。exactとsampledを区別して解釈します。")
    if theory_comparison is not None and not theory_comparison.empty and 'KL_correctable_window' in theory_comparison:
        lines.append(f"window全体のKL成功: {int(theory_comparison.KL_correctable_window.sum())} / {len(theory_comparison)} 組。成功したwindow内では未抽出の部分集合を含め訂正可能です。")
    if theory_comparison is not None and not theory_comparison.empty:
        lines.append(f"原論文の条件と保証長を満たすwindow: {int(theory_comparison.theory_window_guaranteed.sum())} / {len(theory_comparison)} 組。条件内では固定W全体とその全ての部分集合の訂正が保証されます。")
    lines += ['', '固定Wの結果は、ring全域から消失位置を選ぶ従来の実験とは異なる問いに答えます。単一Nを無限系と呼ばず、無根拠な1/N fitは行いません。',
              'samplingの95%区間は各点のWilson区間です。同じ試行をサイズ間で共有するため各Nの推定値は相関し、区間の重なりだけで変化の有意性を判定できません。']
    return '\n'.join(lines)
