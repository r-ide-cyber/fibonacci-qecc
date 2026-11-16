"""有限Fibonacci巡回符号。整数カウントのみでKLを判定する。"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import comb, sqrt
from numbers import Integral
from pathlib import Path
import hashlib
import json
import random
import legacy_functions as legacy

cyclic_orbit = legacy.cyclic_orbit
canonical_rotation = legacy.canonical_rotation


def require_int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def inflate(seed, n):
    return legacy.inflate(seed, require_int(n, 'n'), {'1': '10', '0': '1'})


def adjacent_swap_neighbors(seed):
    legacy.validate_bitstring(seed, 'seed')
    result = set()
    for i in range(len(seed)):
        j = (i + 1) % len(seed)
        if seed[i] != seed[j]:
            bits = list(seed)
            bits[i], bits[j] = bits[j], bits[i]
            result.add(''.join(bits))
    return sorted(result)


def satisfies_adjacent_swap_condition(F0, F1):
    for seed in (F0, F1):
        legacy.validate_bitstring(seed, 'seed')
    if len(F0) != len(F1) or F0.count('1') != F1.count('1'):
        return False
    return (canonical_rotation(F0) != canonical_rotation(F1)
            and not legacy.obtainable_by_one_adjacent_swap(F0, F1, periodic=True)
            and not legacy.obtainable_by_one_adjacent_swap(F1, F0, periodic=True))


def validate_pair(F0, F1):
    if not satisfies_adjacent_swap_condition(F0, F1):
        raise ValueError('seed pairが同じk0/k1・巡回非同値・隣接交換条件を満たしません。')
    return {'seed_length': len(F0), 'k0': F0.count('0'), 'k1': F0.count('1'),
            'adjacent_swap_valid': True}


def generate_valid_logical_pairs(seed_length, k0, k1):
    seed_length = require_int(seed_length, 'seed_length', 1)
    k0, k1 = require_int(k0, 'k0'), require_int(k1, 'k1')
    if k0 + k1 != seed_length:
        raise ValueError('k0+k1 must equal seed_length')
    reps = legacy.fixed_weight_representatives(seed_length, k1)
    return [(a, b) for a, b in combinations(reps, 2)
            if satisfies_adjacent_swap_condition(a, b)]


def normalize_erasure(E, N):
    N = require_int(N, 'N', 1)
    raw = tuple(E)
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in raw):
        raise ValueError('消失位置は整数で指定してください。')
    E = tuple(sorted(int(i) for i in raw))
    if len(E) != len(set(E)) or any(i < 0 or i >= N for i in E):
        raise ValueError('消失位置の重複、または0..N-1の範囲外です。')
    return E


def remaining_pattern(x, E):
    legacy.validate_bitstring(x, 'x')
    erased = set(normalize_erasure(E, len(x)))
    return ''.join(b for i, b in enumerate(x) if i not in erased)


def erased_pattern(x, E):
    legacy.validate_bitstring(x, 'x')
    return ''.join(x[i] for i in normalize_erasure(E, len(x)))


def validate_orbit(orbit):
    orbit = tuple(orbit)
    if not orbit:
        raise ValueError('orbit must be nonempty')
    for x in orbit:
        legacy.validate_bitstring(x, 'orbit word')
    if len(set(orbit)) != len(orbit) or len({len(x) for x in orbit}) != 1:
        raise ValueError('orbit must contain distinct words of equal length')
    return orbit


def group_by_remaining(orbit, E):
    orbit = validate_orbit(orbit)
    E = normalize_erasure(E, len(orbit[0]))
    kept = tuple(i for i in range(len(orbit[0])) if i not in E)
    groups = defaultdict(list)
    for x in orbit:
        groups[''.join(x[i] for i in kept)].append(x)
    return dict(groups)


def recoverability(orbit, E):
    return all(len(g) == 1 for g in group_by_remaining(orbit, E).values())


def group_integer_words(words, erased_mask, full_mask):
    """C移植用kernel: 補領域キー→消失部分リスト。bit位置を圧縮しない。"""
    groups = defaultdict(list)
    remaining_mask = full_mask ^ erased_mask
    for word in words:
        groups[word & remaining_mask].append(word & erased_mask)
    return dict(groups)


def collision_counts(groups0, groups1):
    """selfはx!=yの順序なし組、crossはO0×O1の順序付き組。"""
    c0 = sum(len(g) * (len(g) - 1) // 2 for g in groups0.values())
    c1 = sum(len(g) * (len(g) - 1) // 2 for g in groups1.values())
    cross = sum(len(groups0[k]) * len(groups1[k]) for k in groups0.keys() & groups1.keys())
    return c0, c1, cross


def operator_counts(groups_a, groups_b):
    counts = Counter()
    for key in groups_a.keys() & groups_b.keys():
        for a in groups_a[key]:
            for b in groups_b[key]:
                counts[(a, b)] += 1
    return counts


def equal_normalized_counts(c0, size0, c1, size1):
    return all(c0.get(k, 0) * size1 == c1.get(k, 0) * size0
               for k in c0.keys() | c1.keys())


def reduced_operator_counts(orbit_a, orbit_b, E):
    """可読な(x_E,y_E)キーの非正規化整数カウント。"""
    a, b = validate_orbit(orbit_a), validate_orbit(orbit_b)
    if len(a[0]) != len(b[0]):
        raise ValueError('orbit lengths differ')
    E = normalize_erasure(E, len(a[0]))
    ga = {r: [erased_pattern(x, E) for x in g] for r, g in group_by_remaining(a, E).items()}
    gb = {r: [erased_pattern(x, E) for x in g] for r, g in group_by_remaining(b, E).items()}
    return operator_counts(ga, gb)


def reduced_operator(orbit_a, orbit_b, E):
    """rho[(x_E,y_E)]。対角ブロックはFraction、交差ブロックは表示用float。"""
    a, b = validate_orbit(orbit_a), validate_orbit(orbit_b)
    counts = reduced_operator_counts(a, b, E)
    if len(a) == len(b):
        return {k: Fraction(v, len(a)) for k, v in counts.items()}
    return {k: v / sqrt(len(a) * len(b)) for k, v in counts.items()}


def classify_groups(g0, g1, size0, size1):
    c0, c1, cross = collision_counts(g0, g1)
    diag = equal_normalized_counts(operator_counts(g0, g0), size0,
                                   operator_counts(g1, g1), size1)
    off = cross == 0  # 正の等振幅なので相殺はない。
    recover = c0 == c1 == 0
    kl = diag and off
    return dict(recoverable_0=c0 == 0, recoverable_1=c1 == 0,
                recoverable_both=recover, self_collision_0=c0 > 0,
                self_collision_1=c1 > 0, cross_collision=cross > 0,
                self_collision_count_0=c0, self_collision_count_1=c1,
                cross_collision_count=cross, KL_diagonal=diag,
                KL_offdiagonal=off, KL_correctable=kl,
                comparison_type=('A' if kl else 'B') if recover else ('C' if kl else 'D'),
                KL_failure_reason=('none' if kl else 'diagonal+offdiagonal' if not diag and not off
                                   else 'diagonal' if not diag else 'offdiagonal'))


def evaluate_orbits(orbit0, orbit1, E):
    a, b = validate_orbit(orbit0), validate_orbit(orbit1)
    if len(a[0]) != len(b[0]) or set(a) & set(b):
        raise ValueError('logical supports must have equal word length and be disjoint')
    N = len(a[0]); E = normalize_erasure(E, N)
    mask = legacy.erased_site_mask(E, N); full = (1 << N) - 1
    g0 = group_integer_words(tuple(int(x, 2) for x in a), mask, full)
    g1 = group_integer_words(tuple(int(x, 2) for x in b), mask, full)
    return classify_groups(g0, g1, len(a), len(b))


class LogicalPair:
    def __init__(self, seed_0, seed_1, inflation_n):
        self.metadata = validate_pair(seed_0, seed_1)
        self.seed_0, self.seed_1, self.inflation_n = seed_0, seed_1, inflation_n
        self.F0, self.F1 = inflate(seed_0, inflation_n), inflate(seed_1, inflation_n)
        self.O0, self.O1 = cyclic_orbit(self.F0), cyclic_orbit(self.F1)
        if len(self.F0) != len(self.F1) or set(self.O0) & set(self.O1):
            raise ValueError('inflation後の同長性・直交性が成立しません。')
        self.N = len(self.F0)
        self.words0, self.words1 = tuple(int(x, 2) for x in self.O0), tuple(int(x, 2) for x in self.O1)
        self.metadata.update(seed_0=seed_0, seed_1=seed_1, inflation_n=inflation_n,
                             N=self.N, orbit_size_0=len(self.O0), orbit_size_1=len(self.O1))

    def evaluate(self, E):
        E = normalize_erasure(E, self.N)
        mask = legacy.erased_site_mask(E, self.N); full = (1 << self.N) - 1
        g0 = group_integer_words(self.words0, mask, full)
        g1 = group_integer_words(self.words1, mask, full)
        return dict(self.metadata, erasure_size=len(E), erasure_positions=E,
                    **classify_groups(g0, g1, len(self.O0), len(self.O1)))

    def inspect_erasure_pattern(self, E, max_examples=3):
        E = normalize_erasure(E, self.N)
        record = self.evaluate(E)
        report = dict(record, inflated_F0=self.F0, inflated_F1=self.F1)
        groups = [group_by_remaining(o, E) for o in (self.O0, self.O1)]
        def example(x, y):
            return dict(x=x, y=y, x_E=erased_pattern(x, E), y_E=erased_pattern(y, E),
                        x_remaining=remaining_pattern(x, E), y_remaining=remaining_pattern(y, E))
        def limited(pairs):
            from itertools import islice
            return [example(x, y) for x, y in islice(pairs, max_examples)]
        for i in (0, 1):
            pairs = (pair for key in sorted(groups[i]) for pair in combinations(groups[i][key], 2))
            report[f'self_collision_examples_{i}'] = limited(pairs)
        cross = ((x, y) for key in sorted(groups[0].keys() & groups[1].keys())
                 for x in groups[0][key] for y in groups[1][key])
        report['cross_collision_examples'] = limited(cross)
        c0 = reduced_operator_counts(self.O0, self.O0, E)
        c1 = reduced_operator_counts(self.O1, self.O1, E)
        diffs = []
        for key in sorted(c0.keys() | c1.keys()):
            a, b = Fraction(c0[key], len(self.O0)), Fraction(c1[key], len(self.O1))
            if a != b:
                diffs.append(dict(row_E=key[0], col_E=key[1], rho00=str(a), rho11=str(b),
                    words_0=[dict(x=x, x_E=erased_pattern(x, E), x_remaining=remaining_pattern(x, E))
                             for x in self.O0 if erased_pattern(x, E)==key[0]][:max_examples],
                    words_1=[dict(x=x, x_E=erased_pattern(x, E), x_remaining=remaining_pattern(x, E))
                             for x in self.O1 if erased_pattern(x, E)==key[0]][:max_examples]))
                if len(diffs) >= max_examples: break
        report['diagonal_mismatch_examples'] = diffs
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report


@dataclass(frozen=True)
class ErasurePlan:
    N: int
    m: int
    mode: str
    total_patterns: int
    sample_count: int
    random_seed: int

    def __iter__(self):
        if self.mode == 'exact':
            yield from combinations(range(self.N), self.m)
        else:
            # 各試行は全C(N,m)から独立一様抽出（復元抽出）。重複も試行として保持。
            rng = random.Random(f'{self.random_seed}:{self.N}:{self.m}')
            for _ in range(self.sample_count):
                yield tuple(sorted(rng.sample(range(self.N), self.m)))


def generate_erasure_patterns(N, m, enumeration_mode='auto', max_exact_patterns=100000,
                               num_random_samples=10000, random_seed=42):
    N, m = require_int(N, 'N', 1), require_int(m, 'm')
    max_exact_patterns = require_int(max_exact_patterns, 'max_exact_patterns', 1)
    num_random_samples = require_int(num_random_samples, 'num_random_samples', 1)
    random_seed = require_int(random_seed, 'random_seed')
    if m > N or enumeration_mode not in ('auto', 'exact', 'random'):
        raise ValueError('m<=N and mode in auto/exact/random required')
    total = comb(N, m)
    mode = ('exact' if total <= max_exact_patterns else 'random') if enumeration_mode == 'auto' else enumeration_mode
    return ErasurePlan(N, m, mode, total, total if mode == 'exact' else num_random_samples, random_seed)


def load_jsonl(path):
    with Path(path).open() as h:
        return [json.loads(line) for line in h if line.strip()]


def verify_reference(reference_dir):
    ref = Path(reference_dir)
    manifest = json.loads((ref/'provenance.json').read_text())
    for name, digest in manifest['bundled_hashes'].items():
        if hashlib.sha256((ref/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'既存資料のhash不一致: {name}')
    if hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest() != manifest['legacy_functions_sha256']:
        raise RuntimeError('再利用した既存関数のhash不一致')
    cfg = json.loads((ref/'seed_variation_config.json').read_text())
    if cfg['fibonacci']['substitution'] != {'0':'1', '1':'10'} or not cfg['seed_search']['include_periodic_boundary_swap']:
        raise RuntimeError('inflation ruleまたはring boundaryが既存条件と一致しません。')
    return manifest


def reproduce_baseline(reference_dir, results_dir):
    """保存済みパターン・KL残差・重み付き訂正率を照合。不一致時は保存して停止。"""
    import pandas as pd
    ref, dest = Path(reference_dir), Path(results_dir)
    dest.mkdir(parents=True, exist_ok=True)
    verify_reference(ref)
    catalog = {r['code_id']: r for r in load_jsonl(ref/'code_catalog.jsonl')}
    diagnostics, pattern_rows = [], []
    models = {}
    for key, r in catalog.items():
        model = LogicalPair(r['seed_a'], r['seed_b'], r['depth']); models[key] = model
        if model.N != r['n'] or model.F0 != r['inflated_a'] or model.F1 != r['inflated_b']:
            diagnostics.append(f'{key}: seed/inflation回数/規則の不一致')
        for i, orbit in enumerate((model.O0, model.O1)):
            saved = r['logical_states'][i]
            if set(saved) != set(orbit) or any(abs(complex(*a) - 1/sqrt(len(orbit))) > 1e-12 for a in saved.values()):
                diagnostics.append(f'{key}: 巡回軌道/正規化の不一致')
    for kind, name in [('contiguous','contiguous_kl_results'), ('arbitrary','arbitrary_kl_results')]:
        for r in load_jsonl(ref/(name+'.jsonl')):
            model = models[r['code_id']]
            E = tuple(r['representative_sites'] if kind == 'contiguous' else r['sites'])
            exact = model.evaluate(E)
            states = [legacy.deserialize_integer_state(s) for s in catalog[r['code_id']]['logical_states']]
            old = legacy.sparse_kl_residual(states, E, model.N, 1e-12)
            residual_match = all(abs(old[k] - r[k]) <= 1e-12 for k in ('diagonal_residual','off_diagonal_residual','max_residual'))
            match = (exact['KL_correctable'] == r['satisfies_kl'] == old['satisfies_kl']
                     and exact['KL_diagonal'] == (old['diagonal_residual'] <= 1e-12)
                     and exact['KL_offdiagonal'] == (old['off_diagonal_residual'] <= 1e-12)
                     and residual_match)
            weight = (r['translation_multiplicity'] if r['exact_scan'] else r['sample_hits']) if kind=='arbitrary' else model.N
            # 連続消失は全startも評価し、wrap-aroundを含めて一致確認。
            if kind=='contiguous':
                for start in range(model.N):
                    shifted = tuple(sorted((start+j)%model.N for j in range(r['erasure_length'])))
                    if model.evaluate(shifted)['KL_correctable'] != exact['KL_correctable']:
                        match = False
            row = dict(kind=kind, code_id=r['code_id'], seed_0=model.seed_0, seed_1=model.seed_1,
                       inflation_n=model.inflation_n, N=model.N, erasure_size=len(E),
                       erasure_positions=E, weight=weight, expected_KL=r['satisfies_kl'],
                       recomputed_KL=exact['KL_correctable'], legacy_KL=old['satisfies_kl'],
                       residual_match=residual_match, matched=match)
            pattern_rows.append(row)
            if not match: diagnostics.append(f'{kind} {r["code_id"]} E={E}: KL/残差不一致')
    detail = pd.DataFrame(pattern_rows)
    detail.to_csv(dest/'baseline_pattern_checks.csv', index=False)
    historical = pd.read_csv(ref/'arbitrary_success_by_code.csv')
    summaries=[]
    for (kind,code_id,m), group in detail.groupby(['kind','code_id','erasure_size']):
        trials=int(group.weight.sum())
        expected=int((group.weight*group.expected_KL).sum())
        actual=int((group.weight*group.recomputed_KL).sum())
        saved_rate=expected/trials
        if kind=='arbitrary':
            h=historical[(historical.code_id==code_id)&(historical.erasure_length==m)]
            if len(h)!=1: diagnostics.append(f'保存済み集計が一意でない: {code_id} m={m}')
            else:
                saved_rate=float(h.iloc[0].success_probability)
                if int(h.iloc[0].weighted_trials)!=trials: diagnostics.append(f'探索重み不一致: {code_id} m={m}')
        matched=bool(group.matched.all()) and abs(actual/trials-saved_rate)<=1e-12
        if not matched: diagnostics.append(f'訂正率不一致: {code_id} m={m}')
        summaries.append(dict(kind=kind,code_id=code_id,erasure_size=int(m),
                              checked_representatives=len(group),weighted_trials=trials,
                              expected_success_rate=saved_rate,recomputed_success_rate=actual/trials,
                              difference=actual/trials-saved_rate,matched=matched))
    summary=pd.DataFrame(summaries)
    summary.to_csv(dest/'baseline_reproduction.csv',index=False)
    if diagnostics:
        text='\n'.join(diagnostics)+'\n確認項目: inflation, seed, n, cyclic shift, ring boundary, index, adjacent swap, KL definition.'
        (dest/'baseline_failure.txt').write_text(text)
        raise RuntimeError(text)
    (dest/'baseline_failure.txt').unlink(missing_ok=True)
    return summary


def analyze_model(model, m_values, *, baseline_verified, enumeration_mode='auto',
                  max_exact_patterns=100000, num_random_samples=10000, random_seed=42):
    if not baseline_verified:
        raise RuntimeError('既存研究結果の再現確認が必要です。')
    import pandas as pd
    rows=[]
    for m in m_values:
        plan=generate_erasure_patterns(model.N,m,enumeration_mode,max_exact_patterns,num_random_samples,random_seed)
        for sample_index,E in enumerate(plan):
            rows.append(dict(model.evaluate(E),enumeration_method=plan.mode,
                             total_possible_patterns=plan.total_patterns,random_seed=random_seed,
                             sample_index=sample_index))
        print(f'm={m:2d}: {plan.mode}, {plan.sample_count:,} patterns',flush=True)
    if not rows: raise ValueError('m_values is empty')
    return pd.DataFrame(rows)


def wilson(successes, trials):
    z=1.959963984540054; p=successes/trials; den=1+z*z/trials
    center=(p+z*z/(2*trials))/den
    half=z*sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/den
    return max(0,center-half),min(1,center+half)


def summarize(detail):
    import pandas as pd
    rows=[]
    mapping={'recoverability_success_rate':'recoverable_both','KL_success_rate':'KL_correctable',
             'KL_diagonal_success_rate':'KL_diagonal','KL_offdiagonal_success_rate':'KL_offdiagonal'}
    keys=['seed_0','seed_1','inflation_n','N','erasure_size']
    for values,g in detail.groupby(keys,sort=True):
        row=dict(zip(keys,values)); exact=bool((g.enumeration_method=='exact').all())
        row.update(num_patterns=len(g),enumeration_method='exact' if exact else 'random',
                   total_possible_patterns=int(g.total_possible_patterns.iloc[0]),random_seed=int(g.random_seed.iloc[0]))
        for rate,col in mapping.items():
            successes=int(g[col].sum()); row[rate]=successes/len(g)
            lo,hi=(row[rate],row[rate]) if exact else wilson(successes,len(g))
            row[rate+'_ci95_low'],row[rate+'_ci95_high']=lo,hi
        for kind in 'ABCD':
            count=int((g.comparison_type==kind).sum());row[f'Type_{kind}_count']=count;row[f'Type_{kind}_rate']=count/len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(detail,summary,results_dir,metadata):
    import platform
    dest=Path(results_dir);dest.mkdir(parents=True,exist_ok=True)
    export=detail.copy()
    export.erasure_positions=export.erasure_positions.map(lambda E:json.dumps(list(E)))
    export.to_csv(dest/'recoverability_kl_detail.csv',index=False)
    summary.to_csv(dest/'recoverability_kl_summary.csv',index=False)
    (dest/'run_metadata.json').write_text(json.dumps(dict(metadata,python_version=platform.python_version(),
        sampling='independent uniform subsets with replacement; duplicate trials retained',
        self_collision_convention='unordered distinct pairs; cross = O0 x O1',
        exact_KL='integer counts normalized by distinct orbit sizes',
        site_index='0-based from left; periodic boundary'),ensure_ascii=False,indent=2))


def plot_summary(summary,results_dir):
    import matplotlib.pyplot as plt
    dest=Path(results_dir);dest.mkdir(parents=True,exist_ok=True)
    specs=[('recoverability_vs_kl',[('recoverability_success_rate','Classical recoverability'),('KL_success_rate','KL correctability')]),
           ('kl_conditions',[('KL_diagonal_success_rate','KL diagonal'),('KL_offdiagonal_success_rate','KL off-diagonal')]),
           ('type_b_rate',[('Type_B_rate','Type B')])]
    figures=[]
    for filename,series in specs:
        fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
        for col,label in series:
            ax.plot(summary.erasure_size,summary[col],marker='o',markersize=3,label=label)
            if col+'_ci95_low' in summary:
                ax.fill_between(summary.erasure_size,summary[col+'_ci95_low'],summary[col+'_ci95_high'],alpha=.12)
        sampled=summary.enumeration_method.eq('random')
        ax.scatter(summary.loc[sampled,'erasure_size'],[-.035]*int(sampled.sum()),marker='|',color='grey',label='Sampled m')
        ax.set(xlabel='Erased qubits m',ylabel='Fraction of erasure patterns',ylim=(-.06,1.05),
               title=f'Fibonacci cyclic code | N={int(summary.N.iloc[0])}, n={int(summary.inflation_n.iloc[0])}')
        ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.savefig(dest/(filename+'.png'),dpi=180);fig.savefig(dest/(filename+'.svg'))
        figures.append(fig)
    return figures
