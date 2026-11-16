"""Fixed-window finite-size analysis; all quantum tests reuse recoverability_kl.

Sites are indexed from zero on the left; window_end is inclusive. Random trials
are independent uniform subsets WITH replacement; repeated draws remain trials.
No statevector or dense reduced matrix is constructed here.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
from itertools import combinations, islice
import json
from math import ceil, sqrt
from pathlib import Path
import platform
from time import perf_counter
import warnings

import pandas as pd
import recoverability_kl as kl

DEFAULT_CONFIG = dict(
    SEED_0='000111', SEED_1='010101', PAIR_ID='L06_W03_P0001',
    EXTRA_SEED_PAIRS=[], INFLATION_LIST=[6, 7, 8, 9, 10],
    WINDOW_SIZE_LIST=[20, 30, 50], ERASURE_SIZE_LIST=list(range(1, 11)),
    MAX_EXACT_PATTERNS=100000, NUM_RANDOM_SAMPLES=10000, RANDOM_SEED=42,
    SAVE_RESULTS=True, SAVE_FIGURES=True, DETAIL_MODE='failures',
    DETAIL_MEMORY_LIMIT=2000, DETAIL_REPRESENTATIVE_LIMIT=100,
    BENCHMARK_PATTERNS=100, BENCHMARK_SAFETY_FACTOR=1.5,
    MAX_ESTIMATED_SECONDS=600, RUN_FULL_ANALYSIS=False, OUTPUT_DIR='results',
)
_SEMANTIC_KEYS = ('SEED_0', 'SEED_1', 'PAIR_ID', 'EXTRA_SEED_PAIRS',
                  'INFLATION_LIST', 'WINDOW_SIZE_LIST', 'ERASURE_SIZE_LIST',
                  'MAX_EXACT_PATTERNS', 'NUM_RANDOM_SAMPLES', 'RANDOM_SEED')
_RATE_FIELDS = dict(rec='recoverable_both', KL='KL_correctable',
                    KL_diagonal='KL_diagonal', KL_offdiagonal='KL_offdiagonal')
_RESULT_KEYS = ('recoverable_0', 'recoverable_1', 'recoverable_both',
                'self_collision_count_0', 'self_collision_count_1',
                'cross_collision_count', 'KL_diagonal', 'KL_offdiagonal',
                'KL_correctable', 'comparison_type')
DETAIL_COLUMNS = [
    'pair_id', 'seed_0', 'seed_1', 'inflation_n', 'N', 'window_size',
    'window_start', 'window_end', 'erasure_type', 'erasure_size',
    'erasure_positions_relative', 'erasure_positions_absolute',
    'calculation_mode', 'pattern_id', 'pattern_index', 'random_seed',
    'recoverable_both', 'KL_diagonal', 'KL_offdiagonal', 'KL_correctable',
    'recoverable_0', 'recoverable_1', 'self_collision_count_0',
    'self_collision_count_1', 'cross_collision_count', 'comparison_type',
    'KL_failure_reason',
]


def _config(config):
    c = dict(DEFAULT_CONFIG)
    c.update(config)
    for name in ('MAX_EXACT_PATTERNS', 'NUM_RANDOM_SAMPLES', 'BENCHMARK_PATTERNS'):
        c[name] = kl.require_int(c[name], name, 1)
    for name in ('RANDOM_SEED', 'DETAIL_MEMORY_LIMIT', 'DETAIL_REPRESENTATIVE_LIMIT'):
        c[name] = kl.require_int(c[name], name)
    for name, minimum in [('INFLATION_LIST', 0), ('WINDOW_SIZE_LIST', 1), ('ERASURE_SIZE_LIST', 1)]:
        c[name] = sorted(set(kl.require_int(v, name, minimum) for v in c[name]))
        if not c[name]:
            raise ValueError(f'{name} must be nonempty')
    if c['DETAIL_MODE'] not in ('all', 'failures', 'representative', 'none'):
        raise ValueError('DETAIL_MODE must be all/failures/representative/none')
    for name in ('SAVE_RESULTS', 'SAVE_FIGURES', 'RUN_FULL_ANALYSIS'):
        if not isinstance(c[name], bool):
            raise ValueError(f'{name} must be bool')
    for name in ('MAX_ESTIMATED_SECONDS', 'BENCHMARK_SAFETY_FACTOR'):
        if isinstance(c[name], bool) or not isinstance(c[name], (int, float)) or not 0 < c[name] < float('inf'):
            raise ValueError(f'{name} must be finite and positive')
    return c


def _fingerprint(config):
    return hashlib.sha256(json.dumps({k: config[k] for k in _SEMANTIC_KEYS},
                                    sort_keys=True).encode()).hexdigest()


def _checked_config(prepared, config):
    c = _config(config)
    if _fingerprint(c) != prepared['config_fingerprint']:
        raise RuntimeError('解析条件がprepare時から変化しました。prepareから再実行してください。')
    return c


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(config):
    """Validate seeds once, cache one LogicalPair per pair,n, create repeatable plans."""
    c = _config(config)
    pairs = [dict(pair_id=c['PAIR_ID'], seed_0=c['SEED_0'], seed_1=c['SEED_1'])]
    pairs.extend(dict(p) for p in c['EXTRA_SEED_PAIRS'])
    ids = set()
    for p in pairs:
        if not all(k in p for k in ('pair_id', 'seed_0', 'seed_1')):
            raise ValueError('Each extra pair needs pair_id, seed_0, seed_1')
        if not isinstance(p['pair_id'], str) or not p['pair_id'] or p['pair_id'] in ids:
            raise ValueError('pair_id must be a nonempty unique string')
        ids.add(p['pair_id'])
        p.update(kl.validate_pair(p['seed_0'], p['seed_1']))
    models, sizes, windows, plans = {}, [], [], {}
    for p in pairs:
        for n in c['INFLATION_LIST']:
            model = kl.LogicalPair(p['seed_0'], p['seed_1'], n)
            models[p['pair_id'], n] = model
            sizes.append(dict(pair_id=p['pair_id'], **model.metadata))
            for W in c['WINDOW_SIZE_LIST']:
                if model.N < W:
                    warnings.warn(f"Skip {p['pair_id']} n={n}, N={model.N}, W={W}: N<W", RuntimeWarning)
                    continue
                start = (model.N - W) // 2
                windows.append(dict(pair_id=p['pair_id'], seed_0=p['seed_0'], seed_1=p['seed_1'],
                                    inflation_n=n, N=model.N, window_size=W,
                                    window_start=start, window_end=start + W - 1,
                                    window_positions_absolute=tuple(range(start, start + W))))
    for W in c['WINDOW_SIZE_LIST']:
        for m in c['ERASURE_SIZE_LIST']:
            if m > W:
                warnings.warn(f'Skip W={W}, m={m}: m>W', RuntimeWarning)
                continue
            plans[W, m] = kl.generate_erasure_patterns(
                W, m, 'auto', c['MAX_EXACT_PATTERNS'], c['NUM_RANDOM_SAMPLES'], c['RANDOM_SEED'])
    columns = ['pair_id', 'seed_0', 'seed_1', 'inflation_n', 'N', 'window_size',
               'window_start', 'window_end', 'window_positions_absolute']
    prepared = dict(config=c, config_fingerprint=_fingerprint(c), pairs=pairs, models=models,
                    size_table=pd.DataFrame(sizes), windows=pd.DataFrame(windows, columns=columns),
                    plans=plans, translation_verified=False, small_validation_verified=False)
    return prepared


def translation_test(prepared):
    """Check several shapes at central, shifted and ring-wrapping positions."""
    prepared['translation_verified'] = False
    rows = []
    for window in prepared['windows'].to_dict('records'):
        model = prepared['models'][window['pair_id'], window['inflation_n']]
        W = window['window_size']
        shapes = {tuple(range(W)), (0,), tuple(sorted({0, W // 2, W - 1}))}
        starts = sorted({0, window['window_start'], model.N // 2, model.N - 1})
        for relative in sorted(shapes):
            base = model.evaluate(relative)
            for start in starts:
                E = tuple(sorted((start + i) % model.N for i in relative))
                shifted = model.evaluate(E)
                match = all(base[k] == shifted[k] for k in _RESULT_KEYS)
                rows.append(dict(pair_id=window['pair_id'], inflation_n=model.inflation_n,
                                 N=model.N, window_size=W, translated_start=start,
                                 erasure_positions_relative=relative,
                                 erasure_positions_absolute=E,
                                 KL_correctable=shifted['KL_correctable'], matched=match))
                if not match:
                    prepared['translation_verified'] = False
                    raise AssertionError('Translation mismatch: inspect cyclic orbit, index, periodic boundary, relative→absolute mapping')
    prepared['translation_verified'] = True
    return pd.DataFrame(rows)


def validate_small(prepared):
    """Exhaustively cross-check <=10-site seed codes against both prior APIs.

    If a requested seed exceeds ten sites, also use the documented six-site
    reference pair. This validates the implementation but is not a proof for an
    untested larger seed; the larger seed's structural checks still run in prepare.
    """
    prepared['small_validation_verified'] = False
    candidates = [(p['pair_id'], p['seed_0'], p['seed_1'])
                  for p in prepared['pairs'] if len(p['seed_0']) <= 10]
    if not candidates:
        warnings.warn('指定seedのn=0はN>10のため、小規模validationには6-site参照pairを使用します。', RuntimeWarning)
        candidates = [('validation_reference', '000111', '010101')]
    rows = []
    for pair_id, a, b in candidates:
        model = kl.LogicalPair(a, b, 0)
        states = [[(word, 1 / sqrt(len(words))) for word in words]
                  for words in (model.words0, model.words1)]
        for m in range(model.N + 1):
            for E in combinations(range(model.N), m):
                current = model.evaluate(E)
                prior = kl.evaluate_orbits(model.O0, model.O1, E)
                old = kl.legacy.sparse_kl_residual(states, E, model.N, 1e-12)
                matched = (all(current[k] == prior[k] for k in _RESULT_KEYS)
                           and current['recoverable_0'] == kl.recoverability(model.O0, E)
                           and current['recoverable_1'] == kl.recoverability(model.O1, E)
                           and current['KL_correctable'] == old['satisfies_kl']
                           and current['KL_diagonal'] == (old['diagonal_residual'] <= 1e-12)
                           and current['KL_offdiagonal'] == (old['off_diagonal_residual'] <= 1e-12))
                rows.append(dict(pair_id=pair_id, seed_0=a, seed_1=b, inflation_n=0,
                                 N=model.N, erasure_size=m, erasure_positions=E,
                                 KL_correctable=current['KL_correctable'],
                                 legacy_KL_correctable=old['satisfies_kl'], matched=matched))
                if not matched:
                    prepared['small_validation_verified'] = False
                    raise AssertionError(f'Small-code validation failed: {pair_id}, E={E}')
    prepared['small_validation_verified'] = True
    return pd.DataFrame(rows)


def benchmark(prepared, config):
    """At most 1000 arbitrary evaluations per model; predict all planned work.

    A total budget of 100..1000 is distributed over W,m. If there are more cells
    than the budget, untimed cells use the slowest measured cell for that model.
    Timings include the reused evaluator, not dataframe/CSV costs; the explicit
    safety factor therefore multiplies the estimate. Re-run after config changes.
    """
    c = _checked_config(prepared, config)
    budget = max(100, min(1000, c['BENCHMARK_PATTERNS']))
    rows = []
    for (pair_id, n), model in prepared['models'].items():
        ws = prepared['windows']
        model_windows = ws[(ws.pair_id == pair_id) & (ws.inflation_n == n)]
        cells = [(w, m, prepared['plans'][w['window_size'], m])
                 for w in model_windows.to_dict('records')
                 for m in c['ERASURE_SIZE_LIST'] if (w['window_size'], m) in prepared['plans']]
        if not cells:
            continue
        window_cost = {}
        for w in model_windows.to_dict('records'):
            t0 = perf_counter()
            model.evaluate(w['window_positions_absolute'])
            window_cost[w['window_size']] = perf_counter() - t0
        allocations = [budget // len(cells) + (i < budget % len(cells)) for i in range(len(cells))]
        model_rows = []
        for (w, m, plan), count in zip(cells, allocations):
            t0 = perf_counter()
            if count:
                iterator = iter(plan)
                for _ in range(count):
                    try:
                        relative = next(iterator)
                    except StopIteration:
                        iterator = iter(plan)
                        relative = next(iterator)
                    model.evaluate(tuple(w['window_start'] + r for r in relative))
            elapsed = perf_counter() - t0 if count else 0.0
            model_rows.append(dict(
                pair_id=pair_id, inflation_n=n, N=model.N, window_size=w['window_size'],
                erasure_size=m, orbit_size_0=len(model.O0), orbit_size_1=len(model.O1),
                num_benchmark_patterns=count, elapsed_seconds=elapsed,
                seconds_per_pattern=elapsed / count if count else None,
                timing_source='measured' if count else 'slowest_measured_cell',
                planned_arbitrary_patterns=plan.sample_count,
                planned_contiguous_patterns=w['window_size'] - m + 1,
                calculation_mode='exact' if plan.mode == 'exact' else 'sampled',
                config_fingerprint=prepared['config_fingerprint'],
                benchmark_safety_factor=c['BENCHMARK_SAFETY_FACTOR']))
        slowest = max(r['seconds_per_pattern'] for r in model_rows if r['seconds_per_pattern'] is not None)
        for row in model_rows:
            if row['seconds_per_pattern'] is None:
                row['seconds_per_pattern'] = slowest
            row['estimated_seconds_per_10000'] = 10000 * row['seconds_per_pattern']
            # Full-window costs may differ from the small-m pattern costs.
            same_window_cells = sum(r['window_size'] == row['window_size'] for r in model_rows)
            row['full_window_seconds'] = window_cost[row['window_size']]
            row['estimated_full_seconds'] = c['BENCHMARK_SAFETY_FACTOR'] * (
                row['seconds_per_pattern'] * (row['planned_arbitrary_patterns'] + row['planned_contiguous_patterns'])
                + row['full_window_seconds'] / same_window_cells)
        rows.extend(model_rows)
    frame = pd.DataFrame(rows)
    estimated = float(frame.estimated_full_seconds.sum()) if len(frame) else 0.0
    if estimated > c['MAX_ESTIMATED_SECONDS']:
        warnings.warn(f'推定 {estimated:.1f} 秒 > 上限 {c["MAX_ESTIMATED_SECONDS"]:.1f} 秒。'
                      '設定を縮小するかbenchmarkを確認して上限を変更してください。本計算は未実行です。', RuntimeWarning)
    return frame


class _Details:
    def __init__(self, config, path):
        self.config, self.rows, self.selected, self.total = config, [], 0, 0
        self.handle = path.open('w', newline='', encoding='utf-8') if path else None
        self.writer = csv.DictWriter(self.handle, fieldnames=DETAIL_COLUMNS) if self.handle else None
        if self.writer:
            self.writer.writeheader()

    def add(self, row):
        self.total += 1
        mode = self.config['DETAIL_MODE']
        selected = (mode == 'all' or mode == 'failures' and not row['KL_correctable']
                    or mode == 'representative' and self.selected < self.config['DETAIL_REPRESENTATIVE_LIMIT'])
        if not selected:
            return
        self.selected += 1
        record = {key: row[key] for key in DETAIL_COLUMNS}
        if len(self.rows) < self.config['DETAIL_MEMORY_LIMIT']:
            self.rows.append(record)
        if self.writer:
            export = dict(record)
            for key in ('erasure_positions_relative', 'erasure_positions_absolute'):
                export[key] = json.dumps(export[key])
            self.writer.writerow(export)

    def close(self):
        if self.handle:
            self.handle.close()


def _pattern_id(kind, W, m, index):
    # index identifies independent draws, including repeated relative subsets.
    return f'{kind}:W{W}:m{m}:i{index}'


def _evaluate_patterns(model, window, m, kind, plan, details, previous=None):
    W, start = window['window_size'], window['window_start']
    count = W - m + 1 if kind == 'contiguous' else plan.sample_count
    patterns = (tuple(range(j, j + m)) for j in range(count)) if kind == 'contiguous' else iter(plan)
    mode = 'exact' if kind == 'contiguous' or plan.mode == 'exact' else 'sampled'
    successes, types = Counter(), Counter()
    bits = bytearray(ceil(count / 8)) if kind == 'arbitrary' else None
    changes = Counter()
    digest = hashlib.sha256()
    for i, relative in enumerate(patterns):
        absolute = tuple(start + j for j in relative)
        result = model.evaluate(absolute)
        if window['KL_correctable_window'] and not result['KL_correctable']:
            raise AssertionError(f'Monotonicity violated: full window correctable but subset {relative} fails')
        for rate, col in _RATE_FIELDS.items():
            successes[rate] += int(result[col])
        types[result['comparison_type']] += 1
        if bits is not None:
            now = int(result['KL_correctable'])
            if now:
                bits[i // 8] |= 1 << (i % 8)
            if previous is not None:
                old = (previous[i // 8] >> (i % 8)) & 1
                changes['gained' if now > old else 'lost' if now < old else 'unchanged'] += 1
        digest.update(json.dumps(relative, separators=(',', ':')).encode() + b'\n')
        record = dict(result, pair_id=window['pair_id'], window_size=W,
                      window_start=start, window_end=window['window_end'], erasure_type=kind,
                      erasure_positions_relative=relative, erasure_positions_absolute=absolute,
                      calculation_mode=mode, pattern_id=_pattern_id(kind, W, m, i),
                      pattern_index=i, random_seed=plan.random_seed)
        details.add(record)
    result = dict(count=count, mode=mode, types=types, successes=successes,
                  pattern_digest=digest.hexdigest(), bits=bits, paired_changes=changes)
    for rate in _RATE_FIELDS:
        estimate = successes[rate] / count
        lo, hi = (estimate, estimate) if mode == 'exact' else kl.wilson(successes[rate], count)
        result[rate], result[rate + '_ci95_low'], result[rate + '_ci95_high'] = estimate, lo, hi
    return result


def _thresholds(summary):
    rows = []
    for (pair_id, W, m), group in summary.groupby(['pair_id', 'window_size', 'erasure_size'], sort=True):
        group = group.sort_values(['N', 'inflation_n'])
        tests = [
            ('exact_enumeration', (group.calculation_mode == 'exact') & group.R_KL_arbitrary.eq(1)),
            ('full_window_proof', group.KL_correctable_window),
            ('sampled_observed', (group.calculation_mode == 'sampled') & group.R_KL_arbitrary.eq(1)),
        ]
        for scope, mask in tests:
            qualifying = group[mask]
            first = qualifying.iloc[0] if len(qualifying) else None
            rows.append(dict(pair_id=pair_id, seed_0=group.seed_0.iloc[0], seed_1=group.seed_1.iloc[0],
                             window_size=W, erasure_size=m, threshold_kind=scope,
                             found=first is not None,
                             n_star=int(first.inflation_n) if first is not None else None,
                             N_star=int(first.N) if first is not None else None,
                             proven_all_patterns=first is not None and scope != 'sampled_observed',
                             all_later_tested_sizes_pass=(bool(group.loc[group.N >= first.N, 'R_KL_arbitrary'].eq(1).all()) if first is not None else None),
                             later_tested_size_count=(int((group.N > first.N).sum()) if first is not None else 0),
                             interpretation='minimum within tested sizes; not an asymptotic threshold',
                             ci95_low=float(first.R_KL_arbitrary_ci95_low) if first is not None else None,
                             ci95_high=float(first.R_KL_arbitrary_ci95_high) if first is not None else None))
    return pd.DataFrame(rows)


def run_analysis(prepared, config, benchmark_df, validation_passed=False):
    """Run only after explicit opt-in, validation and a complete current benchmark."""
    c = _checked_config(prepared, config)
    if not c['RUN_FULL_ANALYSIS']:
        raise RuntimeError('RUN_FULL_ANALYSIS=False: benchmark確認後にTrueへ変更してください。')
    if not validation_passed or not prepared['translation_verified'] or not prepared['small_validation_verified']:
        raise RuntimeError('translation_testとvalidate_smallの成功後、validation_passed=Trueが必要です。')
    required = {'pair_id', 'inflation_n', 'window_size', 'erasure_size', 'config_fingerprint', 'estimated_full_seconds'}
    if benchmark_df.empty or not required.issubset(benchmark_df.columns):
        raise RuntimeError('有効なbenchmark結果が必要です。')
    if not benchmark_df.config_fingerprint.eq(prepared['config_fingerprint']).all():
        raise RuntimeError('benchmark条件が現在の解析条件と異なります。')
    if 'benchmark_safety_factor' in benchmark_df and not benchmark_df.benchmark_safety_factor.eq(c['BENCHMARK_SAFETY_FACTOR']).all():
        raise RuntimeError('BENCHMARK_SAFETY_FACTORが変わりました。benchmarkを再実行してください。')
    expected = {(w['pair_id'], w['inflation_n'], w['window_size'], m)
                for w in prepared['windows'].to_dict('records')
                for m in c['ERASURE_SIZE_LIST'] if (w['window_size'], m) in prepared['plans']}
    observed = set(benchmark_df[['pair_id', 'inflation_n', 'window_size', 'erasure_size']].itertuples(index=False, name=None))
    if expected != observed or len(observed) != len(benchmark_df):
        raise RuntimeError('benchmark coverage does not match planned cells')
    if benchmark_df.estimated_full_seconds.isna().any() or not benchmark_df.estimated_full_seconds.ge(0).all():
        raise RuntimeError('benchmark estimates must be finite nonnegative values')
    estimate = float(benchmark_df.estimated_full_seconds.sum())
    if estimate > c['MAX_ESTIMATED_SECONDS']:
        warnings.warn(f'推定 {estimate:.1f} 秒が上限を超えるため本計算を開始しません。', RuntimeWarning)
        raise RuntimeError('Benchmark time limit exceeded; reduce settings or review and raise MAX_ESTIMATED_SECONDS.')
    dest = Path(c['OUTPUT_DIR']) if c['SAVE_RESULTS'] else None
    if dest:
        dest.mkdir(parents=True, exist_ok=True)
    detail_path = dest / 'finite_size_detail.csv' if dest and c['DETAIL_MODE'] != 'none' else None
    details = _Details(c, detail_path)
    rows, windows = [], []
    t0 = perf_counter()
    try:
        for window in prepared['windows'].to_dict('records'):
            model = prepared['models'][window['pair_id'], window['inflation_n']]
            result = model.evaluate(window['window_positions_absolute'])
            windows.append(dict(window, KL_correctable_window=result['KL_correctable'],
                                recoverable_window=result['recoverable_both'],
                                KL_diagonal_window=result['KL_diagonal'],
                                KL_offdiagonal_window=result['KL_offdiagonal']))
        window_df = pd.DataFrame(windows)
        for (pair_id, W), group in window_df.groupby(['pair_id', 'window_size'], sort=True):
            group = group.sort_values(['N', 'inflation_n'])
            for m in c['ERASURE_SIZE_LIST']:
                if (W, m) not in prepared['plans']:
                    continue
                plan = prepared['plans'][W, m]
                previous, previous_N, previous_n, previous_digest = None, None, None, None
                for window in group.to_dict('records'):
                    model = prepared['models'][pair_id, window['inflation_n']]
                    contiguous = _evaluate_patterns(model, window, m, 'contiguous', plan, details)
                    arbitrary = _evaluate_patterns(model, window, m, 'arbitrary', plan, details, previous)
                    if previous_digest is not None and arbitrary['pattern_digest'] != previous_digest:
                        raise AssertionError('Relative patterns differ across N')
                    row = {k: window[k] for k in ('pair_id', 'seed_0', 'seed_1', 'inflation_n', 'N',
                                                  'window_size', 'window_start', 'window_end', 'KL_correctable_window')}
                    row.update(erasure_size=m, calculation_mode=arbitrary['mode'],
                               calculation_mode_contiguous='exact', num_patterns_arbitrary=arbitrary['count'],
                               num_patterns_contiguous=contiguous['count'],
                               number_of_patterns=arbitrary['count'] + contiguous['count'],
                               total_possible_arbitrary_patterns=plan.total_patterns,
                               random_seed=plan.random_seed, arbitrary_pattern_digest=arbitrary['pattern_digest'],
                               contiguous_pattern_digest=contiguous['pattern_digest'],
                               previous_N=previous_N, previous_inflation_n=previous_n,
                               proven_by_full_window=window['KL_correctable_window'])
                    for kind, metrics in [('contiguous', contiguous), ('arbitrary', arbitrary)]:
                        for rate in _RATE_FIELDS:
                            col = f'R_{rate}_{kind}'
                            row[col] = metrics[rate]
                            row[col + '_success_count'] = metrics['successes'][rate]
                            row[col + '_ci95_low'] = metrics[rate + '_ci95_low']
                            row[col + '_ci95_high'] = metrics[rate + '_ci95_high']
                        for typ in 'ABCD':
                            row[f'Type_{typ}_count_{kind}'] = metrics['types'][typ]
                            row[f'Type_{typ}_rate_{kind}'] = metrics['types'][typ] / metrics['count']
                    row['gap_KL'] = row['R_KL_contiguous'] - row['R_KL_arbitrary']
                    row['gap_KL_ci95_low'] = row['R_KL_contiguous'] - row['R_KL_arbitrary_ci95_high']
                    row['gap_KL_ci95_high'] = row['R_KL_contiguous'] - row['R_KL_arbitrary_ci95_low']
                    row['arbitrary_failure_rate'] = 1 - row['R_KL_arbitrary']
                    row['arbitrary_failure_rate_ci95_low'] = 1 - row['R_KL_arbitrary_ci95_high']
                    row['arbitrary_failure_rate_ci95_high'] = 1 - row['R_KL_arbitrary_ci95_low']
                    for change in ('gained', 'lost', 'unchanged'):
                        row['paired_' + change + '_count'] = arbitrary['paired_changes'][change] if previous is not None else None
                    if previous is not None:
                        d = (arbitrary['paired_changes']['gained'] - arbitrary['paired_changes']['lost']) / arbitrary['count']
                        changes = arbitrary['paired_changes']['gained'] + arbitrary['paired_changes']['lost']
                        # Sample variance of paired {-1,0,1} changes divided by n.
                        variance_of_mean = max(0, (changes / arbitrary['count'] - d * d) / (arbitrary['count'] - 1)) if arbitrary['count'] > 1 else float('nan')
                        row['paired_delta_R_KL_arbitrary'] = d
                        row['paired_delta_standard_error'] = sqrt(variance_of_mean) if arbitrary['mode'] == 'sampled' else 0.0
                    else:
                        row['paired_delta_R_KL_arbitrary'], row['paired_delta_standard_error'] = None, None
                    rows.append(row)
                    previous, previous_N, previous_n, previous_digest = arbitrary['bits'], model.N, model.inflation_n, arbitrary['pattern_digest']
    finally:
        details.close()
    summary = pd.DataFrame(rows).sort_values(['pair_id', 'window_size', 'erasure_size', 'N']).reset_index(drop=True)
    scaling = summary.copy()
    grouped = scaling.groupby(['pair_id', 'window_size', 'erasure_size'], sort=False)
    for name in ('R_KL_contiguous', 'R_KL_arbitrary', 'gap_KL'):
        scaling['Delta_' + name] = grouped[name].diff().abs()
        scaling['signed_Delta_' + name] = grouped[name].diff()
    scaling['inverse_N'] = 1 / scaling.N
    threshold = _thresholds(summary)
    stats = summary.groupby(['inflation_n', 'N', 'window_size', 'erasure_size'])[
        ['R_KL_contiguous', 'R_KL_arbitrary', 'gap_KL']].agg(['mean', 'median', 'min', 'max', 'count'])
    stats.columns = ['_'.join(col) for col in stats.columns]
    stats = stats.reset_index()
    metadata = dict(config=c, config_fingerprint=prepared['config_fingerprint'],
                    python_version=platform.python_version(), pandas_version=pd.__version__,
                    elapsed_seconds=perf_counter() - t0, estimated_seconds=estimate,
                    evaluated_patterns=details.total, selected_detail_rows=details.selected,
                    retained_detail_rows=len(details.rows), detail_memory_truncated=details.selected > len(details.rows),
                    detail_selection=c['DETAIL_MODE'], summary_uses_all_patterns=True,
                    representative_selection='first rows in deterministic pair/W/m/N/type/pattern traversal',
                    detail_disk_complete_for_selection=bool(detail_path),
                    sampling='independent uniform subsets with replacement; duplicates retained; seed excludes N and pair',
                    paired_standard_error='sample standard error of paired trial differences; no asymptotic extrapolation',
                    window_end='inclusive; site zero at left; periodic ring',
                    threshold='minimum within tested sizes; sampled observed success does not prove all subsets',
                    source_sha256={Path(__file__).name: _sha(__file__),
                                   Path(kl.__file__).name: _sha(kl.__file__),
                                   Path(kl.legacy.__file__).name: _sha(kl.legacy.__file__)})
    paths = {}
    frames = dict(detail=pd.DataFrame(details.rows, columns=DETAIL_COLUMNS), summary=summary,
                  scaling=scaling, windows=window_df, thresholds=threshold, pair_statistics=stats)
    if dest:
        for name, frame in frames.items():
            if name == 'detail':
                continue
            path = dest / f'finite_size_{name}.csv'
            frame.to_csv(path, index=False)
            paths[name] = str(path.resolve())
        if detail_path:
            paths['detail'] = str(detail_path.resolve())
        path = dest / 'run_metadata.json'
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        paths['metadata'] = str(path.resolve())
    return dict(frames, paths=paths, metadata=metadata)


def inspect_pattern(prepared, pair_id, inflation_n, window_size, relative_positions, max_examples=3):
    """Reuse the prior collision/operator witness report for one window subset."""
    model = prepared['models'][pair_id, inflation_n]
    W = kl.require_int(window_size, 'window_size', 1)
    if W > model.N:
        raise ValueError('window_size > N')
    relative = kl.normalize_erasure(relative_positions, W)
    start = (model.N - W) // 2
    return model.inspect_erasure_pattern(tuple(start + i for i in relative), max_examples=max_examples)
