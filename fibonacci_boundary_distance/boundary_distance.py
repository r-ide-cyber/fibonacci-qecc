"""Open-chain boundary-distance workflow; ring and open supports stay separate."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import json
import math
import numbers
import random
import time
import warnings
import platform

import pandas as pd

from boundary_models import build_ring_model, build_open_model, inflate, validate_pair


@dataclass
class Config:
    seed_0: str = '000111'
    seed_1: str = '010101'
    pair_id: str = 'L06_W03_P0001'
    inflation_list: tuple = (7, 8, 9, 10)
    window_size_list: tuple = (20, 30, 50)
    boundary_distance_list: tuple = (0, 5, 10, 20, 40, 80)
    erasure_size_list: tuple = (1, 2, 3, 4, 5)
    max_exact_patterns: int = 100000
    num_random_samples: int = 10000
    random_seed: int = 42
    epsilon_boundary: float = 0.01
    boundary_sides: tuple = ('left', 'right')
    include_center: bool = True
    boundary_fraction_list: tuple = ()
    save_results: bool = True
    save_figures: bool = True
    output_dir: str = 'results'
    figures_dir: str = 'figures'
    detail_memory_limit: int = 2000
    open_model_id: str = 'anchored_singleton'
    open_model_description: str = 'Each logical class contains only its fixed-origin inflated open word.'

    def validate(self):
        validate_pair(self.seed_0, self.seed_1)
        for name, minimum in [('inflation_list', 0), ('window_size_list', 1),
                              ('boundary_distance_list', 0), ('erasure_size_list', 1)]:
            values = tuple(getattr(self, name))
            if name != 'boundary_distance_list' and not values:
                raise ValueError(f'{name} must not be empty')
            if len(values) != len(set(values)):
                raise ValueError(f'{name} contains duplicates')
            for v in values:
                _integer(v, name, minimum)
        for name, minimum in [('max_exact_patterns', 1), ('num_random_samples', 1),
                              ('random_seed', 0), ('detail_memory_limit', 0)]:
            _integer(getattr(self, name), name, minimum)
        if not math.isfinite(self.epsilon_boundary) or not 0 < self.epsilon_boundary <= 1:
            raise ValueError('epsilon_boundary must be in (0,1]')
        if set(self.boundary_sides) - {'left', 'right'} or len(set(self.boundary_sides)) != len(self.boundary_sides):
            raise ValueError('boundary_sides must contain unique left/right entries')
        if len(set(self.boundary_fraction_list)) != len(self.boundary_fraction_list):
            raise ValueError('duplicate boundary fractions')
        for f in self.boundary_fraction_list:
            if isinstance(f, bool) or not isinstance(f, numbers.Real) or not math.isfinite(f) or not 0 <= f < 1:
                raise ValueError('boundary fractions must be in [0,1)')
        if not self.pair_id or not self.open_model_id or not self.open_model_description:
            raise ValueError('pair_id and open model description/id are required')
        return self


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be integer >= {minimum}')
    return int(value)


@dataclass(frozen=True)
class PatternSet:
    patterns: tuple
    mode: str
    digest: str
    total_possible: int
    sampling_scheme: str


def build_pattern_bank(cfg):
    """Cache once per (W,m,type); sampling draws are iid uniform with replacement."""
    cfg.validate()
    bank = {}
    for W in cfg.window_size_list:
        for m in cfg.erasure_size_list:
            if m > W:
                warnings.warn(f'm={m} > W={W}; skipping erasure size', stacklevel=2)
                continue
            for kind in ('contiguous', 'arbitrary'):
                total = W - m + 1 if kind == 'contiguous' else math.comb(W, m)
                if kind == 'contiguous':
                    pats = tuple(tuple(range(start, start + m)) for start in range(W - m + 1))
                    mode, scheme = 'exact', 'enumeration'
                elif total <= cfg.max_exact_patterns:
                    pats = tuple(combinations(range(W), m))
                    mode, scheme = 'exact', 'enumeration'
                else:
                    digest = hashlib.sha256(f'{cfg.random_seed}:{W}:{m}:arbitrary'.encode()).digest()
                    rng = random.Random(int.from_bytes(digest, 'big'))
                    pats = tuple(tuple(sorted(rng.sample(range(W), m))) for _ in range(cfg.num_random_samples))
                    mode, scheme = 'sampled', 'iid_uniform_with_replacement'
                h = hashlib.sha256()
                for pat in pats:
                    h.update((','.join(map(str, pat)) + ';').encode())
                bank[(W, m, kind)] = PatternSet(pats, mode, h.hexdigest(), total, scheme)
    return bank


WINDOW_COLUMNS = ['inflation_n', 'N', 'window_size', 'boundary_side', 'distance_mode',
                  'boundary_distance', 'fraction_requested', 'boundary_fraction_actual',
                  'boundary_distance_left', 'boundary_distance_right',
                  'nearest_boundary_distance', 'window_start', 'window_end']


def windows_for(N, W, cfg):
    """Distances are number of sites outside window; no wrap in open chain."""
    _integer(N, 'N', 1); _integer(W, 'W', 1)
    rows = []
    if W > N:
        warnings.warn(f'W={W} > N={N}; skipping windows', stacklevel=2)
        return rows
    specs = [('sites', B, None) for B in cfg.boundary_distance_list]
    specs += [('fraction', math.floor(f * N), f) for f in cfg.boundary_fraction_list]
    for mode, B, fraction in specs:
        if B + W > N:
            warnings.warn(f'B={B}, W={W}, N={N}: B+W>N; skipping window', stacklevel=2)
            continue
        for side in cfg.boundary_sides:
            start = B if side == 'left' else N - W - B
            left, right = start, N - start - W
            rows.append(dict(N=N, window_size=W, boundary_side=side, distance_mode=mode,
                boundary_distance=B, fraction_requested=fraction, boundary_fraction_actual=B/N,
                boundary_distance_left=left, boundary_distance_right=right,
                nearest_boundary_distance=min(left, right), window_start=start, window_end=start+W-1))
    if cfg.include_center:
        start = (N-W)//2
        rows.append(dict(N=N, window_size=W, boundary_side='center', distance_mode='center',
            boundary_distance=start, fraction_requested=None, boundary_fraction_actual=start/N,
            boundary_distance_left=start, boundary_distance_right=N-start-W,
            nearest_boundary_distance=start, window_start=start, window_end=start+W-1))
    return rows


@dataclass
class Prepared:
    config: Config
    models: dict
    patterns: dict
    windows: pd.DataFrame
    size_table: pd.DataFrame


def prepare(cfg, candidate_factory=None, quantum_state_definition=None, enable_open_kl=False):
    cfg.validate()
    if candidate_factory is not None and cfg.open_model_id == 'anchored_singleton':
        raise ValueError('A custom candidate_factory needs an explicit open_model_id and description.')
    models, sizes, windows = {}, [], []
    for n in cfg.inflation_list:
        F0, F1 = inflate(cfg.seed_0, n), inflate(cfg.seed_1, n)
        if len(F0) != len(F1):
            raise ValueError('Inflated words have different lengths; analysis stopped.')
        N = len(F0)
        sizes.append(dict(inflation_n=n, N=N, k0=cfg.seed_0.count('0'), k1=cfg.seed_0.count('1'),
                          inflated_count_0=F0.count('0'), inflated_count_1=F0.count('1')))
        models[(n, 'ring')] = build_ring_model(cfg.seed_0, cfg.seed_1, n)
        models[(n, 'open')] = build_open_model(cfg.seed_0, cfg.seed_1, n,
            candidate_factory=candidate_factory, model_id=cfg.open_model_id,
            model_description=cfg.open_model_description,
            quantum_state_definition=quantum_state_definition, enable_kl=enable_open_kl)
        for W in cfg.window_size_list:
            windows.extend(dict(inflation_n=n, **r) for r in windows_for(N, W, cfg))
    return Prepared(cfg, models, build_pattern_bank(cfg), pd.DataFrame(windows, columns=WINDOW_COLUMNS),
                    pd.DataFrame(sizes))


CHECK_KEYS = ('recoverable', 'KL_diagonal', 'KL_offdiagonal', 'KL_correctable')


def ring_sanity_check(prepared, max_patterns=5):
    """Actually re-evaluate translations, including wrap across N-1/0; fail closed."""
    _integer(max_patterns, 'max_patterns', 1)
    records = []
    for (n, geometry), model in prepared.models.items():
        if geometry != 'ring':
            continue
        for (W, m, kind), ps in prepared.patterns.items():
            if W > model.N:
                continue
            window_rows = prepared.windows[(prepared.windows.inflation_n == n) &
                                          (prepared.windows.window_size == W)]
            shifts = sorted({0, model.N-1, model.N//2, *map(int, window_rows.window_start)})
            for index, relative in enumerate(ps.patterns[:max_patterns]):
                reference = model.evaluate(relative)
                expected = tuple(reference[key] for key in CHECK_KEYS)
                for shift in shifts:
                    E = tuple(sorted((shift + p) % model.N for p in relative))
                    result = model.evaluate(E)
                    got = tuple(result[key] for key in CHECK_KEYS)
                    passed = got == expected
                    records.append(dict(geometry='ring', inflation_n=n, N=model.N,
                        window_size=W, erasure_size=m, erasure_type=kind, pattern_id=index,
                        shift=shift, wraps=any(shift+p >= model.N for p in relative), passed=passed))
                    if not passed:
                        raise AssertionError(f'Ring translation failed: n={n}, E={relative}, shift={shift}; '
                            'check cyclic orbit, indexing, periodic mask. Do not interpret as boundary effect.')
    return pd.DataFrame(records)


def pattern_plan(prepared):
    rows = []
    for w in prepared.windows.to_dict('records'):
        for (W, m, kind), ps in prepared.patterns.items():
            if W != w['window_size']:
                continue
            for geometry in ('open', 'ring'):
                rows.append(dict(**w, geometry=geometry, erasure_size=m, erasure_type=kind,
                    number_of_patterns=len(ps.patterns), total_possible_patterns=ps.total_possible,
                    calculation_mode=ps.mode, pattern_digest=ps.digest))
    return pd.DataFrame(rows)


def benchmark(prepared, count=100):
    """Measure each n, geometry, W, m, type at one placement.

    Extrapolation excludes CSV/plot overhead; it is an estimate, not a bound.
    """
    _integer(count, 'count', 100)
    if count > 1000:
        raise ValueError('benchmark count must be between 100 and 1000')
    plan = pattern_plan(prepared)
    if plan.empty:
        return pd.DataFrame()
    records = []
    for (n, geometry, W, m, kind), rows in plan.groupby(
            ['inflation_n', 'geometry', 'window_size', 'erasure_size', 'erasure_type'], sort=False):
        w = rows.iloc[0]
        ps = prepared.patterns[(W, m, kind)]
        model = prepared.models[(n, geometry)]
        start = int(w.window_start)
        begin = time.perf_counter()
        for index in range(count):
            relative = ps.patterns[index % len(ps.patterns)]
            model.evaluate(tuple(start+p for p in relative))
        runtime = time.perf_counter()-begin
        tasks = int(rows.number_of_patterns.sum())
        records.append(dict(inflation_n=n, geometry=geometry, N=model.N, W=W, B=int(w.boundary_distance),
            m=m, erasure_type=kind, pattern_count=count, unique_patterns_measured=min(count,len(ps.patterns)),
            runtime=runtime, time_per_pattern=runtime/count, planned_evaluations=tasks,
            estimated_runtime=tasks*runtime/count, calculation_mode=ps.mode,
            estimate_scope='evaluation_only_one_placement; excludes IO and plotting'))
    return pd.DataFrame(records)


def _wilson(success, count, mode):
    if count == 0:
        return None, None
    rate = success/count
    if mode == 'exact':
        return rate, rate
    z = 1.959963984540054
    center = (rate+z*z/(2*count))/(1+z*z/count)
    half = z*math.sqrt(rate*(1-rate)/count+z*z/(4*count*count))/(1+z*z/count)
    return max(0., center-half), min(1., center+half)


@dataclass
class RunResult:
    detail: pd.DataFrame
    summary: pd.DataFrame
    wide_summary: pd.DataFrame
    delta: pd.DataFrame
    threshold: pd.DataFrame
    mirror: pd.DataFrame
    paired: pd.DataFrame
    total_detail_rows: int
    output_dir: str | None


def _model_fields(model, geometry):
    md = getattr(model, 'metadata', {})
    return dict(model_id=md.get('model_id', 'ring_cyclic_orbit' if geometry == 'ring' else 'open'),
                open_support_is_singleton=bool(md.get('open_support_is_singleton', False)),
                model_description=md.get('model_description', ''),
                support_size_0=md.get('support_size_0'), support_size_1=md.get('support_size_1'))


def _paired_stats(a, b, mode):
    """b-a from shared pattern ids. a/b are binary tuples; no independence assumption."""
    if a is None or b is None:
        return dict(delta=None, standard_error=None, ci95_low=None, ci95_high=None)
    if len(a) != len(b) or not a:
        raise ValueError('Paired arrays must have same positive length')
    d = [int(y)-int(x) for x,y in zip(a,b)]
    avg = sum(d)/len(d)
    se = 0. if mode == 'exact' else (math.sqrt(sum((x-avg)**2 for x in d)/(len(d)-1)/len(d)) if len(d)>1 else None)
    return dict(delta=avg, standard_error=se,
        ci95_low=max(-1.,avg-1.96*se) if se is not None else None,
        ci95_high=min(1.,avg+1.96*se) if se is not None else None)


def run_analysis(prepared, progress=None):
    """Stream full detail to CSV, retain a bounded preview. All evaluations are direct."""
    from boundary_reporting import derive_comparisons, make_wide_summary
    cfg = prepared.config
    if cfg.save_results and (prepared.windows.empty or not prepared.patterns or not any(
            W in set(prepared.windows.window_size) for W, _, _ in prepared.patterns)):
        raise ValueError('No valid window/pattern combinations; no output files were changed.')
    # Translation diagnostics also precede every actual run, even if cells were skipped.
    sanity = ring_sanity_check(prepared, max_patterns=1)
    root = Path(cfg.output_dir) if cfg.save_results else None
    writer, stream = None, None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        (root/'geometry_open').mkdir(exist_ok=True)
        (root/'geometry_ring').mkdir(exist_ok=True)
        (root/'run_status.json').write_text(json.dumps({'status':'running'}, indent=2))
        stream = (root/'boundary_distance_detail.csv').open('w', newline='', encoding='utf-8')
    previews, summaries, paired_rows = [], [], []
    total = 0
    # Process one n/W/m/type at a time to bound paired-outcome memory.
    try:
        for n in cfg.inflation_list:
            for (W, m, kind), ps in prepared.patterns.items():
                windows = prepared.windows[(prepared.windows.inflation_n == n) & (prepared.windows.window_size == W)]
                if windows.empty:
                    continue
                unique_count = len(set(ps.patterns))
                for geometry in ('open', 'ring'):
                    model = prepared.models[(n,geometry)]
                    outcomes = []
                    for w in windows.to_dict('records'):
                        base = dict(geometry=geometry, pair_id=cfg.pair_id, seed_0=cfg.seed_0, seed_1=cfg.seed_1,
                            **w, erasure_type=kind, erasure_size=m, calculation_mode=ps.mode,
                            pattern_digest=ps.digest, sampling_scheme=ps.sampling_scheme,
                            **_model_fields(model, geometry))
                        rec_bits, kl_bits, kl_missing = bytearray(), bytearray(), False
                        diag_sum = off_sum = 0
                        for pattern_id, relative in enumerate(ps.patterns):
                            absolute = tuple(int(w['window_start'])+x for x in relative)
                            result = model.evaluate(absolute)
                            rec_bits.append(bool(result['recoverable']))
                            kl = result['KL_correctable']
                            if kl is None:
                                kl_missing = True
                            else:
                                kl_bits.append(bool(kl))
                                diag_sum += bool(result['KL_diagonal'])
                                off_sum += bool(result['KL_offdiagonal'])
                            record = dict(**base, pattern_id=pattern_id,
                                erasure_positions_relative=json.dumps(relative),
                                erasure_positions_absolute=json.dumps(absolute),
                                recoverable=bool(result['recoverable']),
                                recoverable_0=result.get('recoverable_0'), recoverable_1=result.get('recoverable_1'),
                                KL_diagonal=result['KL_diagonal'], KL_offdiagonal=result['KL_offdiagonal'],
                                KL_correctable=kl)
                            total += 1
                            if len(previews) < cfg.detail_memory_limit:
                                previews.append(record)
                            if stream is not None:
                                if writer is None:
                                    writer = csv.DictWriter(stream, fieldnames=list(record)); writer.writeheader()
                                writer.writerow(record)
                        count = len(rec_bits)
                        lo, hi = _wilson(sum(rec_bits),count,ps.mode)
                        klo, khi = (None,None) if kl_missing else _wilson(sum(kl_bits),count,ps.mode)
                        summaries.append(dict(**base, number_of_patterns=count,
                            total_possible_patterns=ps.total_possible, number_of_unique_patterns=unique_count,
                            R_rec=sum(rec_bits)/count, R_KL=None if kl_missing else sum(kl_bits)/count,
                            R_KL_diagonal=None if kl_missing else diag_sum/count,
                            R_KL_offdiagonal=None if kl_missing else off_sum/count,
                            R_rec_ci95_low=lo, R_rec_ci95_high=hi,
                            R_KL_ci95_low=klo, R_KL_ci95_high=khi))
                        outcomes.append((w,rec_bits,None if kl_missing else kl_bits))
                        if progress:
                            progress(f'{geometry}: n={n}, W={W}, m={m}, {kind}, {w["boundary_side"]} B={w["boundary_distance"]}')
                    centers = [x for x in outcomes if x[0]['boundary_side']=='center']
                    if centers:
                        center = centers[0]
                        for w,rb,kb in outcomes:
                            if w['boundary_side']=='center':
                                continue
                            for metric,a,b in [('recoverability',rb,center[1]), ('KL',kb,center[2])]:
                                paired_rows.append(dict(geometry=geometry,pair_id=cfg.pair_id,
                                    **w,erasure_type=kind,erasure_size=m,metric=metric,
                                    calculation_mode=ps.mode,number_of_patterns=len(rb),
                                    pattern_digest=ps.digest,contrast='center_minus_placement',
                                    **_paired_stats(a,b,ps.mode)))
    except Exception as exc:
        if root is not None:
            (root/'run_status.json').write_text(json.dumps({'status':'failed','error':str(exc)},indent=2))
        raise
    finally:
        if stream is not None:
            stream.close()
    summary = pd.DataFrame(summaries)
    # Direct per-pattern translations should imply identical ring aggregates.
    if not summary.empty:
        ring = summary[summary.geometry=='ring']
        for key, rows in ring.groupby(['inflation_n','window_size','erasure_size','erasure_type']):
            for metric in ['R_rec','R_KL','R_KL_diagonal','R_KL_offdiagonal']:
                if rows[metric].nunique(dropna=False) != 1:
                    raise AssertionError(f'Ring rate depends on window placement: {key}, {metric}')
    delta, threshold, mirror = derive_comparisons(summary,cfg.epsilon_boundary)
    result = RunResult(pd.DataFrame(previews),summary,make_wide_summary(summary),delta,threshold,mirror,
                       pd.DataFrame(paired_rows),total,str(root) if root else None)
    if root is not None:
        outputs = {'boundary_distance_summary.csv':result.wide_summary,
                   'boundary_distance_summary_long.csv':summary,
                   'boundary_distance_threshold.csv':threshold,
                   'boundary_distance_delta.csv':delta,'boundary_distance_mirror.csv':mirror,
                   'boundary_distance_paired.csv':result.paired,'ring_translation_sanity.csv':sanity,
                   'inflation_sizes.csv':prepared.size_table,'window_plan.csv':prepared.windows}
        for filename,frame in outputs.items():
            frame.to_csv(root/filename,index=False)
        for geometry in ('open','ring'):
            for filename,frame in outputs.items():
                if 'geometry' in frame.columns:
                    frame[frame.geometry==geometry].to_csv(root/f'geometry_{geometry}'/filename,index=False)
        # Split full detail in chunks; never load all rows into memory.
        if total:
            for chunk_id,chunk in enumerate(pd.read_csv(root/'boundary_distance_detail.csv',chunksize=50000,
                    dtype={'seed_0':str, 'seed_1':str, 'pair_id':str, 'model_id':str})):
                for geometry in ('open','ring'):
                    chunk[chunk.geometry==geometry].to_csv(root/f'geometry_{geometry}'/'boundary_distance_detail.csv',
                        index=False,mode='w' if chunk_id==0 else 'a',header=chunk_id==0)
        manifest = dict(config=asdict(cfg),python=platform.python_version(),
            total_detail_rows=total,detail_preview_rows=len(previews),
            open_model_metadata={str(n):prepared.models[(n,'open')].metadata for n in cfg.inflation_list},
            scientific_scope='Auxiliary open-chain model; singleton classical supports do not identify boundary effects.',
            sampling='iid uniform subsets with replacement; shared across n, B, side, geometry',
            ci='Pointwise Wilson intervals; paired difference normal intervals; no simultaneous threshold inference')
        (root/'open_candidate_sets.json').write_text(json.dumps({str(n): dict(C0=prepared.models[(n,'open')].C0, C1=prepared.models[(n,'open')].C1) for n in cfg.inflation_list}, ensure_ascii=False, indent=2))
        (root/'run_metadata.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
        (root/'run_status.json').write_text(json.dumps({'status':'complete','total_detail_rows':total},indent=2))
    return result
