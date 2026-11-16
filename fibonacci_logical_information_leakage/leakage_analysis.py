"""Reproducible experiment pipeline for finite Fibonacci logical leakage.

The unchanged recoverability_kl module owns the legacy definitions. Random
arbitrary patterns are independent uniform subsets, sampled WITH replacement;
duplicate draws remain observations. The same W,m and random seed reproduce
the preceding finite_size_scaling notebook's exact relative pattern sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import islice
import json
from math import comb
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

import recoverability_kl as kl
from theory import paper_seed_conditions
from fibonacci_leakage_core import OrbitPair, analyze_erasure


DEFAULT_CONFIG = dict(
    SEED_0='000111', SEED_1='010101', PAIR_ID='L06_W03_P0001',
    LOGICAL_PAIRS=None, INFLATION_LIST=[0, 1, 2, 4, 6],
    ERASURE_DOMAIN_MODE='window', WINDOW_SIZE_LIST=[6, 20],
    ERASURE_SIZE_LIST=[1, 2, 3, 4, 5, 6],
    ERASURE_TYPE_LIST=['contiguous', 'arbitrary'],
    MAX_EXACT_PATTERNS=100, NUM_RANDOM_SAMPLES=100, RANDOM_SEED=42,
    MAX_DENSE_ERASURE_QUBITS=12, MAX_MATRIX_DIMENSION=1024,
    EIGENVALUE_TOL=1e-12, LEAKAGE_TOL=1e-10,
    BENCHMARK_PATTERNS=100, BENCHMARK_SAFETY_FACTOR=1.5,
    MAX_DETAIL_ROWS=1000000, SAVE_RESULTS=True, SAVE_FIGURES=True,
    OUTPUT_DIR='results',
)
GROUP_KEYS = ['pair_id', 'seed_0', 'seed_1', 'inflation_n', 'N',
              'window_size', 'erasure_domain_mode', 'erasure_type', 'erasure_size']
_EXACT_KEYS = ['recoverable_both', 'KL_diagonal', 'KL_offdiagonal',
               'KL_correctable', 'self_collision_count_0',
               'self_collision_count_1', 'cross_collision_count']
_SEMANTIC_KEYS = [k for k in DEFAULT_CONFIG if k not in
                  ['SAVE_RESULTS', 'SAVE_FIGURES', 'OUTPUT_DIR',
                   'BENCHMARK_PATTERNS', 'BENCHMARK_SAFETY_FACTOR', 'MAX_DETAIL_ROWS']]
DETAIL_COLUMNS = GROUP_KEYS + [
    'window_start', 'window_end', 'erasure_positions_relative',
    'erasure_positions_absolute', 'erasure_positions', 'pattern_id',
    'pattern_index', 'calculation_mode', 'operator_mode', 'random_seed',
    'recoverable_both', 'recoverable_0', 'recoverable_1', 'KL_diagonal',
    'KL_offdiagonal', 'KL_correctable', 'comparison_type', 'S_R', 'S_E',
    'S_RE', 'I_RE', 'D_Z', 'D_X', 'self_collision_0', 'self_collision_1',
    'cross_collision', 'self_collision_count_0', 'self_collision_count_1',
    'cross_collision_count', 'leakage_failure_type', 'decoupled',
    'important_case', 'contiguous_runs_ring', 'max_cyclic_gap',
    'legacy_core_agree', 'KL_implies_decoupling_verified',
]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fingerprint(c):
    return hashlib.sha256(json.dumps(
        {k: _jsonable(c[k]) for k in _SEMANTIC_KEYS},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_config(config):
    """Validate user controls; N is always computed from inflation."""
    if not isinstance(config, dict):
        raise ValueError('config must be a dictionary of uppercase parameter names')
    if 'N' in config:
        raise ValueError('N must not be supplied; use INFLATION_LIST to generate N')
    c = {**DEFAULT_CONFIG, **config}
    for name in ['MAX_EXACT_PATTERNS', 'NUM_RANDOM_SAMPLES',
                 'MAX_MATRIX_DIMENSION', 'BENCHMARK_PATTERNS', 'MAX_DETAIL_ROWS']:
        c[name] = kl.require_int(c[name], name, 1)
    for name in ['RANDOM_SEED', 'MAX_DENSE_ERASURE_QUBITS']:
        c[name] = kl.require_int(c[name], name)
    if c['ERASURE_DOMAIN_MODE'] not in ['window', 'full_ring']:
        raise ValueError('ERASURE_DOMAIN_MODE must be window or full_ring')
    for name, minimum in [('INFLATION_LIST', 0), ('WINDOW_SIZE_LIST', 1),
                          ('ERASURE_SIZE_LIST', 0)]:
        if not isinstance(c[name], (list, tuple)):
            raise ValueError(f'{name} must be a list of integers')
        c[name] = sorted({kl.require_int(x, name, minimum) for x in c[name]})
        if not c[name] and not (name == 'WINDOW_SIZE_LIST' and c['ERASURE_DOMAIN_MODE'] == 'full_ring'):
            raise ValueError(f'{name} must be nonempty')
    types = c['ERASURE_TYPE_LIST']
    if not isinstance(types, (list, tuple)) or not types or any(
            x not in ['contiguous', 'arbitrary'] for x in types):
        raise ValueError('ERASURE_TYPE_LIST must contain contiguous and/or arbitrary')
    c['ERASURE_TYPE_LIST'] = list(dict.fromkeys(types))
    for name in ['EIGENVALUE_TOL', 'LEAKAGE_TOL', 'BENCHMARK_SAFETY_FACTOR']:
        value = c[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ['SAVE_RESULTS', 'SAVE_FIGURES']:
        if not isinstance(c[name], bool):
            raise ValueError(f'{name} must be bool')
    if not isinstance(c['OUTPUT_DIR'], (str, Path)) or not str(c['OUTPUT_DIR']):
        raise ValueError('OUTPUT_DIR must be a nonempty path')
    supplied = c['LOGICAL_PAIRS']
    if supplied is None:
        supplied = [dict(pair_id=c['PAIR_ID'], seed_0=c['SEED_0'], seed_1=c['SEED_1'])]
    if not isinstance(supplied, (list, tuple)) or not supplied:
        raise ValueError('LOGICAL_PAIRS must be None or a nonempty list of pair dictionaries')
    pairs, seen = [], set()
    for p in supplied:
        if not isinstance(p, dict) or not all(k in p for k in ['pair_id', 'seed_0', 'seed_1']):
            raise ValueError('Each logical pair requires pair_id, seed_0, seed_1')
        if not isinstance(p['pair_id'], str) or not p['pair_id'] or p['pair_id'] in seen:
            raise ValueError('pair_id must be a unique nonempty string')
        seen.add(p['pair_id'])
        legacy_checks = kl.validate_pair(p['seed_0'], p['seed_1'])
        paper_checks = paper_seed_conditions(p['seed_0'], p['seed_1'])
        if not paper_checks['paper_seed_condition']:
            raise ValueError(f"{p['pair_id']}: original-paper seed condition fails: "
                             'an adjacent 01/10 swap returns to the same cyclic class '
                             f"(self_swap_0={paper_checks['self_swap_0']}, "
                             f"self_swap_1={paper_checks['self_swap_1']}). "
                             'Choose seeds satisfying the additional self-swap exclusion.')
        pairs.append({**p, **legacy_checks, **paper_checks})
    c['LOGICAL_PAIRS'] = pairs
    return c


@dataclass(frozen=True)
class PatternPlan:
    """A repeatable, lazy plan; does not allocate all C(W,m) tuples."""
    W: int
    m: int
    erasure_type: str
    domain_mode: str
    max_exact_patterns: int
    num_random_samples: int
    random_seed: int

    @property
    def mode(self):
        return 'exact' if self.erasure_type == 'contiguous' or self.total_patterns <= self.max_exact_patterns else 'sampled'

    @property
    def total_patterns(self):
        if self.erasure_type == 'arbitrary':
            return comb(self.W, self.m)
        if self.m in [0, self.W]:
            return 1
        return self.W if self.domain_mode == 'full_ring' else self.W - self.m + 1

    @property
    def count(self):
        return self.total_patterns if self.mode == 'exact' else self.num_random_samples

    def __iter__(self):
        if self.erasure_type == 'arbitrary':
            yield from kl.generate_erasure_patterns(
                self.W, self.m, 'auto', self.max_exact_patterns,
                self.num_random_samples, self.random_seed)
        elif self.m in [0, self.W]:
            yield tuple(range(self.m))
        else:
            for start in range(self.total_patterns):
                yield tuple(sorted((start + j) % self.W for j in range(self.m)))


def prepare(config):
    """Validate seeds and generate N, windows, models and a transparent work plan."""
    c = validate_config(config)
    models, orbit_pairs, plans = {}, {}, {}
    sizes, windows, jobs, skipped = [], [], [], []
    for p in c['LOGICAL_PAIRS']:
        for n in c['INFLATION_LIST']:
            model = kl.LogicalPair(p['seed_0'], p['seed_1'], n)
            key = p['pair_id'], n
            models[key] = model
            orbit_pairs[key] = OrbitPair(model.O0, model.O1, N=model.N)
            sizes.append(dict(pair_id=p['pair_id'], **model.metadata,
                              M0=len(model.O0), M1=len(model.O1)))
            widths = [model.N] if c['ERASURE_DOMAIN_MODE'] == 'full_ring' else c['WINDOW_SIZE_LIST']
            for W in widths:
                base = dict(pair_id=p['pair_id'], seed_0=p['seed_0'], seed_1=p['seed_1'],
                            inflation_n=n, N=model.N, window_size=W,
                            erasure_domain_mode=c['ERASURE_DOMAIN_MODE'])
                if W > model.N:
                    skipped.append(dict(**base, erasure_size=None, reason='W>N'))
                    continue
                start = (model.N - W) // 2
                window = dict(**base, window_start=start, window_end=start + W - 1)
                windows.append(window)
                for m in c['ERASURE_SIZE_LIST']:
                    if m > W:
                        skipped.append(dict(**base, erasure_size=m, reason='m>domain_size'))
                        continue
                    for kind in c['ERASURE_TYPE_LIST']:
                        plan_key = W, m, kind, c['ERASURE_DOMAIN_MODE']
                        if plan_key not in plans:
                            plans[plan_key] = PatternPlan(W, m, kind, c['ERASURE_DOMAIN_MODE'],
                                c['MAX_EXACT_PATTERNS'], c['NUM_RANDOM_SAMPLES'], c['RANDOM_SEED'])
                        plan = plans[plan_key]
                        jobs.append(dict(**window, erasure_type=kind, erasure_size=m,
                            calculation_mode=plan.mode, number_of_patterns=plan.count,
                            total_possible_patterns=plan.total_patterns,
                            hilbert_dimension_E=2**m, plan_key=plan_key))
    job_columns = GROUP_KEYS + ['window_start', 'window_end', 'calculation_mode',
                    'number_of_patterns', 'total_possible_patterns', 'hilbert_dimension_E', 'plan_key']
    return dict(config=c, config_fingerprint=_fingerprint(c),
                models=models, orbit_pairs=orbit_pairs, plans=plans,
                seed_checks=pd.DataFrame(c['LOGICAL_PAIRS']), size_table=pd.DataFrame(sizes),
                windows=pd.DataFrame(windows), jobs=pd.DataFrame(jobs, columns=job_columns),
                skipped=pd.DataFrame(skipped, columns=GROUP_KEYS[:7] + ['erasure_size', 'reason']))


def _checked_config(prepared, config):
    c = prepared['config'] if config is None else validate_config(config)
    if _fingerprint(c) != prepared['config_fingerprint']:
        raise ValueError('Analysis parameters changed since prepare(); rerun prepare and benchmark')
    return c


def _evaluate(prepared, job, relative, c):
    absolute = tuple(job['window_start'] + i for i in relative)
    key = job['pair_id'], job['inflation_n']
    old = prepared['models'][key].evaluate(absolute)
    new = analyze_erasure(prepared['orbit_pairs'][key], absolute,
        max_dense_erasure_qubits=c['MAX_DENSE_ERASURE_QUBITS'],
        max_matrix_dimension=c['MAX_MATRIX_DIMENSION'],
        eigenvalue_tol=c['EIGENVALUE_TOL'], leakage_tol=c['LEAKAGE_TOL'])
    for field in _EXACT_KEYS:
        if field in new and new[field] != old[field]:
            raise AssertionError(f'Legacy/core mismatch {field}: {key}, E={absolute}')
    if old['KL_correctable'] and (new['I_RE'] >= c['LEAKAGE_TOL'] or
            new['D_Z'] >= c['LEAKAGE_TOL'] or new['D_X'] >= c['LEAKAGE_TOL']):
        raise AssertionError(f'KL succeeds but leakage exceeds tolerance: {key}, E={absolute}')
    return absolute, old, new


def benchmark(prepared, config=None):
    """Time 100..1000 total evaluations, distributed over the planned jobs.

    Untimed jobs use the slowest measured job and are explicitly marked as
    extrapolations. Estimates include legacy KL, leakage and validation costs;
    dataframe/CSV/plotting overhead is approximated by a visible safety factor.
    The benchmark does not write files.
    """
    c = _checked_config(prepared, config)
    jobs = prepared['jobs'].to_dict('records')
    if not jobs:
        return pd.DataFrame()
    budget = max(100, min(1000, c['BENCHMARK_PATTERNS']))
    # Spread measurements across sorted N,m, including both smallest and largest.
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i]['N'], jobs[i]['erasure_size'],
                                                   jobs[i]['window_size'], jobs[i]['erasure_type']))
    allocation = [budget // len(jobs)] * len(jobs)
    remainder = budget % len(jobs)
    chosen = np.linspace(0, len(jobs) - 1, remainder, dtype=int) if remainder else []
    for position in chosen:
        allocation[order[int(position)]] += 1
    rows = []
    for job, count in zip(jobs, allocation):
        plan = prepared['plans'][job['plan_key']]
        iterator = iter(plan)
        t0 = perf_counter()
        for _ in range(count):
            try:
                relative = next(iterator)
            except StopIteration:
                iterator = iter(plan)
                relative = next(iterator)
            _evaluate(prepared, job, relative, c)
        elapsed = perf_counter() - t0 if count else 0.
        rows.append({k: v for k, v in job.items() if k != 'plan_key'} | dict(
            pattern_count=count, runtime_seconds=elapsed,
            seconds_per_pattern=elapsed / count if count else np.nan,
            timing_source='measured' if count else 'slowest_measured_job_extrapolation',
            benchmark_safety_factor=c['BENCHMARK_SAFETY_FACTOR'],
            config_fingerprint=prepared['config_fingerprint']))
    slowest = max(r['seconds_per_pattern'] for r in rows if r['pattern_count'])
    for row in rows:
        if not row['pattern_count']:
            row['seconds_per_pattern'] = slowest
        row['estimated_total_seconds'] = (row['seconds_per_pattern'] * row['number_of_patterns']
                                          * c['BENCHMARK_SAFETY_FACTOR'])
    return pd.DataFrame(rows)


def _geometry(E, N):
    if not E:
        return 0, N
    selected = set(E)
    runs = sum((i - 1) % N not in selected for i in selected)
    if len(E) == N:
        runs = 1
    # Gap = number of un-erased sites between successive erased sites on ring.
    gaps = [(E[(i + 1) % len(E)] - E[i] - 1) % N for i in range(len(E))]
    return runs, max(gaps)


def _summarize(detail):
    if detail.empty:
        return pd.DataFrame(columns=GROUP_KEYS + ['mean_I_RE', 'max_I_RE'])
    records = []
    for key, group in detail.groupby(GROUP_KEYS, sort=False, dropna=False):
        values = group.I_RE.to_numpy()
        record = dict(zip(GROUP_KEYS, key))
        record.update(mean_I_RE=float(np.mean(values)), median_I_RE=float(np.median(values)),
            std_I_RE=float(np.std(values, ddof=0)), min_I_RE=float(np.min(values)),
            max_I_RE=float(np.max(values)), mean_D_Z=float(group.D_Z.mean()),
            max_D_Z=float(group.D_Z.max()), mean_D_X=float(group.D_X.mean()),
            max_D_X=float(group.D_X.max()), decoupled_rate=float(group.decoupled.mean()),
            KL_success_rate=float(group.KL_correctable.mean()),
            recoverability_success_rate=float(group.recoverable_both.mean()),
            recoverable_with_leakage_rate=float((group.recoverable_both & ~group.decoupled).mean()),
            number_of_patterns=len(group), number_of_unique_patterns=group.pattern_id.nunique(),
            calculation_mode=group.calculation_mode.iloc[0],
            operator_modes=';'.join(sorted(group.operator_mode.unique())),
            maximum_interpretation='exact_maximum' if group.calculation_mode.iloc[0] == 'exact' else 'observed_sample_maximum',
            KL_decoupling_rate_difference=float(group.KL_correctable.mean() - group.decoupled.mean()),
            KL_decoupling_numerical_mismatch_count=int((group.KL_correctable != group.decoupled).sum()))
        for q in [0.05, 0.25, 0.75, 0.95]:
            record[f'q{int(q*100):02d}_I_RE'] = float(np.quantile(values, q))
        records.append(record)
    return pd.DataFrame(records)


def _worst_patterns(detail, summary, tol):
    if detail.empty:
        return detail.copy()
    maxima = summary[GROUP_KEYS + ['max_I_RE', 'maximum_interpretation']]
    merged = detail.merge(maxima, on=GROUP_KEYS, validate='many_to_one')
    # Keep physically tied patterns despite sub-tolerance eigensolver noise.
    worst = merged[(merged.I_RE - merged.max_I_RE).abs() <= tol].copy()
    worst['is_exact_numeric_maximum'] = worst.I_RE == worst.max_I_RE
    worst['distance_below_numeric_maximum'] = worst.max_I_RE - worst.I_RE
    worst['worst_erasure_positions'] = worst.erasure_positions_absolute
    worst['tied_maximum_draws'] = worst.groupby(GROUP_KEYS, dropna=False).I_RE.transform('size')
    worst['repeated_draw_count'] = worst.groupby(GROUP_KEYS + ['pattern_id'], dropna=False).I_RE.transform('size')
    worst = worst.drop_duplicates(GROUP_KEYS + ['pattern_id']).reset_index(drop=True)
    worst['important_case_D'] = worst.I_RE >= tol
    return worst


def _finite_size(detail, summary):
    """Per-pattern trajectories and paired changes, valid only at fixed W,m."""
    if detail.empty:
        return detail.copy()
    result = detail.copy()
    fixed = result.erasure_domain_mode == 'window'
    result['fixed_relative_pattern_comparison'] = fixed
    result['finite_size_interpretation'] = np.where(fixed,
        'fixed_W_m_relative_pattern', 'domain_changes_with_N_no_fixed_pattern_claim')
    trajectory = ['pair_id', 'window_size', 'erasure_type', 'erasure_size',
                  'pattern_id', 'pattern_index']
    result = result.sort_values(trajectory + ['N']).reset_index(drop=True)
    for name in ['N', 'I_RE', 'D_Z', 'D_X']:
        previous = result.groupby(trajectory, dropna=False)[name].shift(1)
        result[f'previous_{name}'] = previous.where(result.fixed_relative_pattern_comparison)
    result['delta_I_RE'] = result.I_RE - result.previous_I_RE
    result['delta_D_Z'] = result.D_Z - result.previous_D_Z
    result['delta_D_X'] = result.D_X - result.previous_D_X
    result['same_relative_pattern_verified'] = result.previous_N.notna()
    return result


def _comparisons(summary):
    keys = [k for k in GROUP_KEYS if k != 'erasure_type']
    if summary.empty:
        return pd.DataFrame(columns=keys)
    columns = keys + ['mean_I_RE', 'max_I_RE', 'mean_D_Z', 'mean_D_X',
                      'KL_success_rate', 'recoverability_success_rate',
                      'calculation_mode', 'number_of_patterns']
    contiguous = summary[summary.erasure_type == 'contiguous'][columns]
    arbitrary = summary[summary.erasure_type == 'arbitrary'][columns]
    result = contiguous.merge(arbitrary, on=keys, suffixes=('_contiguous', '_arbitrary'),
                              how='outer', validate='one_to_one')
    result['arbitrary_minus_contiguous_mean_I_RE'] = result.mean_I_RE_arbitrary - result.mean_I_RE_contiguous
    return result


def _pair_correlations(summary):
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    strength = summary[summary.erasure_type == 'arbitrary'].copy()
    strength['KL_failure_rate'] = 1 - strength.KL_success_rate
    strength['recoverability_failure_rate'] = 1 - strength.recoverability_success_rate
    keys = ['inflation_n', 'N', 'window_size', 'erasure_domain_mode', 'erasure_size']
    rows = []
    for key, group in strength.groupby(keys, sort=False, dropna=False):
        # Several IDs for the same physical seed pair do not add evidence.
        unique = group.assign(seed_pair_key=group.apply(lambda r: '|'.join(sorted([
            kl.canonical_rotation(r.seed_0), kl.canonical_rotation(r.seed_1)])), axis=1))
        unique = unique.drop_duplicates('seed_pair_key')
        for failure in ['KL_failure_rate', 'recoverability_failure_rate']:
            enough = len(unique) >= 3
            varied = enough and unique[failure].nunique() > 1 and unique.mean_I_RE.nunique() > 1
            rows.append(dict(zip(keys, key)) | dict(
                failure_metric=failure, number_of_distinct_pairs=len(unique),
                pearson_r=float(unique[failure].corr(unique.mean_I_RE)) if varied else np.nan,
                status='descriptive_correlation' if varied else
                       'requires_at_least_3_distinct_pairs' if not enough else 'undefined_constant_data',
                interpretation='descriptive_only_no_causal_or_asymptotic_claim'))
    return strength, pd.DataFrame(rows)


def _collision_summary(detail):
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for field in ['cross_collision', 'self_collision_0', 'self_collision_1']:
        for present, group in detail.groupby(field, sort=True):
            rows.append(dict(collision_field=field, collision_present=bool(present),
                number_of_patterns=len(group), mean_I_RE=group.I_RE.mean(),
                mean_D_Z=group.D_Z.mean(), mean_D_X=group.D_X.mean(),
                note='pooled_descriptive_summary; inspect fixed pair,N,W,m for controlled comparisons'))
    return pd.DataFrame(rows)


def _save_results(results, c):
    target = Path(c['OUTPUT_DIR'])
    target.mkdir(parents=True, exist_ok=True)
    filenames = dict(detail='logical_leakage_detail.csv', summary='logical_leakage_summary.csv',
        worst='logical_leakage_worst_patterns.csv', finite_size='finite_size_leakage.csv',
        cases='logical_leakage_important_cases.csv', comparisons='contiguous_arbitrary_comparison.csv',
        pair_strength='logical_pair_strength.csv', pair_correlations='logical_pair_correlations.csv',
        collisions='collision_leakage_summary.csv', skipped='skipped_configurations.csv',
        seed_checks='seed_conditions.csv', size_table='inflation_sizes.csv', jobs='analysis_plan.csv')
    for name, filename in filenames.items():
        frame = results[name].copy()
        for column in frame.columns:
            if frame[column].map(lambda x: isinstance(x, (tuple, list, dict))).any():
                frame[column] = frame[column].map(lambda x: json.dumps(_jsonable(x), ensure_ascii=False)
                    if isinstance(x, (tuple, list, dict)) else x)
        frame.to_csv(target / filename, index=False)
    (target / 'metadata.json').write_text(json.dumps(_jsonable(results['metadata']),
        indent=2, ensure_ascii=False), encoding='utf-8')


def run_analysis(config, prepared=None):
    """Run every requested pattern; return DataFrames and optionally save CSVs.

    Call prepare()/benchmark() first to inspect the cost. All detail rows are
    retained; MAX_DETAIL_ROWS prevents accidentally materializing an unbounded
    result. Raise that explicit guard only when memory is sufficient.
    """
    prepared = prepare(config) if prepared is None else prepared
    c = _checked_config(prepared, config)
    planned = int(prepared['jobs'].number_of_patterns.sum())
    if planned > c['MAX_DETAIL_ROWS']:
        raise ValueError(f'{planned:,} detail rows exceed MAX_DETAIL_ROWS={c["MAX_DETAIL_ROWS"]:,}; '
                         'reduce sampling/jobs or explicitly increase the memory guard')
    t0 = perf_counter()
    records, digests = [], []
    for job in prepared['jobs'].to_dict('records'):
        digest = hashlib.sha256()
        plan = prepared['plans'][job['plan_key']]
        for index, relative in enumerate(plan):
            encoded = json.dumps(relative, separators=(',', ':'))
            digest.update((encoded + '\n').encode())
            absolute, old, new = _evaluate(prepared, job, relative, c)
            row = {k: job[k] for k in GROUP_KEYS + ['window_start', 'window_end']}
            row.update(old)
            operator_mode = new.get('calculation_mode', new.get('operator_mode', 'unspecified'))
            row.update(new)
            decoupled = row['I_RE'] < c['LEAKAGE_TOL']
            important = ('A' if old['recoverable_both'] and not old['KL_correctable'] and not decoupled
                         else 'B' if old['recoverable_both'] and old['KL_correctable'] and decoupled
                         else 'C' if not old['recoverable_both'] and old['KL_correctable'] else '')
            runs, gap = _geometry(absolute, job['N'])
            pattern_key = f"{job['window_size']}:{job['erasure_size']}:{job['erasure_type']}:{encoded}"
            row.update(erasure_positions_relative=relative, erasure_positions_absolute=absolute,
                erasure_positions=absolute, pattern_id=hashlib.sha256(pattern_key.encode()).hexdigest()[:24],
                pattern_index=index, calculation_mode=plan.mode, operator_mode=operator_mode,
                random_seed=c['RANDOM_SEED'], decoupled=decoupled, important_case=important,
                contiguous_runs_ring=runs, max_cyclic_gap=gap,
                legacy_core_agree=True, KL_implies_decoupling_verified=True)
            records.append(row)
        digests.append({k: job[k] for k in GROUP_KEYS} | dict(
            pattern_sequence_sha256=digest.hexdigest(), number_of_patterns=plan.count))
    detail = pd.DataFrame(records) if records else pd.DataFrame(columns=DETAIL_COLUMNS)
    summary = _summarize(detail)
    worst = _worst_patterns(detail, summary, c['LEAKAGE_TOL'])
    cases = detail[detail.important_case != ''].copy()
    if not worst.empty:
        case_d = worst[worst.important_case_D].copy()
        case_d['important_case'] = 'D'
        case_d['important_case_definition'] = 'positive group maximum; ties within LEAKAGE_TOL retained'
        cases = pd.concat([cases, case_d], ignore_index=True)
    strength, correlations = _pair_correlations(summary)
    hashes = {}
    for name in ['legacy_functions.py', 'recoverability_kl.py', 'theory.py',
                 'fibonacci_leakage_core.py', 'leakage_analysis.py']:
        path = Path(__file__).with_name(name)
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(config=_jsonable(c), config_fingerprint=prepared['config_fingerprint'],
        elapsed_seconds=perf_counter() - t0, python_version=platform.python_version(),
        numpy_version=np.__version__, pandas_version=pd.__version__, source_sha256=hashes,
        pattern_sequences=digests, sampling='uniform independent subsets WITH replacement; duplicates retained',
        std_definition='population standard deviation (ddof=0)',
        units='bits, logarithm base 2', D_X_normalization='raw trace norm ||rho_E^(01)||_1; no additional factor',
        relative_pattern_rule='same W,m,random seed uses preceding legacy generator; independent of N and pair_id',
        maximum_rule='exact maximum for enumeration; observed sample maximum for sampled jobs',
        case_D_rule='positive group maximum; all distinct ties within LEAKAGE_TOL retained',
        comparison_type_legacy='A=rec+KL, B=rec+notKL, C=notRec+KL, D=notRec+notKL',
        important_case='A=rec+notKL+leakage, B=rec+KL+decoupled, C=notRec+KL, D=positive group maximum',
        full_ring_caveat='The erasure domain changes with N; no fixed-relative-pattern asymptotic comparison',
        theory_note='legacy cross-swap validation preserved, with original-paper self-swap exclusion checked separately')
    results = dict(detail=detail, summary=summary, worst=worst,
        finite_size=_finite_size(detail, summary), cases=cases, comparisons=_comparisons(summary),
        pair_strength=strength, pair_correlations=correlations, collisions=_collision_summary(detail),
        skipped=prepared['skipped'], seed_checks=prepared['seed_checks'],
        size_table=prepared['size_table'], jobs=prepared['jobs'], metadata=metadata)
    if c['SAVE_RESULTS']:
        _save_results(results, c)
    return results
