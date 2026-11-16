"""Reproducible collision experiments; exact KL and explicit sampling denominators."""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
from math import comb, sqrt
import hashlib, json, random, time, platform
import numpy as np
import pandas as pd
from collision_core import CollisionGeometry, benchmark, inflate, validate_pair
from collision_kl import evaluate_erasure


def logical_pairs(config):
    pairs = config.get('LOGICAL_PAIRS') or [dict(pair_id=config.get('PAIR_ID', 'user_pair'), seed_0=config['SEED_0'], seed_1=config['SEED_1'])]
    if len({p['pair_id'] for p in pairs}) != len(pairs):
        raise ValueError('pair_id must be unique')
    for p in pairs:
        validate_pair(p['seed_0'], p['seed_1'])
    return pairs


def validate_config(c):
    logical_pairs(c)
    for name in ['INFLATION_LIST', 'ERASURE_SIZE_LIST']:
        if not c[name] or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in c[name]):
            raise ValueError(name + ' must contain nonnegative integers')
        if len(set(c[name])) != len(c[name]):
            raise ValueError(name + ' contains duplicates')
    if c['ERASURE_DOMAIN_MODE'] not in ('full_ring', 'window'):
        raise ValueError('ERASURE_DOMAIN_MODE must be full_ring or window')
    for key in ['MAX_EXACT_PATTERNS', 'NUM_RANDOM_SAMPLES', 'TOP_K_DANGEROUS_SUPPORTS']:
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if c['ERASURE_DOMAIN_MODE'] == 'window':
        w = c['WINDOW_SIZE_LIST']
        if not w or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in w) or len(set(w)) != len(w):
            raise ValueError('WINDOW_SIZE_LIST must contain distinct positive integers')
    return True


def domains(N, config):
    return [N] if config['ERASURE_DOMAIN_MODE'] == 'full_ring' else config['WINDOW_SIZE_LIST']


def interval_patterns(N, m, W):
    # Fixed window {0,...,W-1}. Whole ring retains all N start positions,
    # including N equivalent trials at m=0 or m=N, per the specified ensemble.
    starts = range(N) if W == N else range(W-m+1)
    for start in starts:
        yield tuple(sorted((start+i) % N for i in range(m))), start


def arbitrary_patterns(N, m, W, config):
    total = comb(W, m)
    if total <= config['MAX_EXACT_PATTERNS']:
        yield from ((E, 'exact') for E in combinations(range(W), m))
    else:
        # Shared random stream across logical pairs with the same domain, W and m.
        token = f"{config['RANDOM_SEED']}|{config['ERASURE_DOMAIN_MODE']}|{W}|{m}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8], 'big'))
        for _ in range(config['NUM_RANDOM_SAMPLES']):
            yield tuple(sorted(rng.sample(range(W), m))), 'sampled'


def build_plan(config):
    validate_config(config)
    rows = []
    for p in logical_pairs(config):
        for n in config['INFLATION_LIST']:
            N = len(inflate(p['seed_0'], n))
            for W in domains(N, config):
                for m in config['ERASURE_SIZE_LIST']:
                    reason = 'window_exceeds_N' if W > N else 'm_exceeds_domain' if m > W else ''
                    count = comb(W, m) if not reason else 0
                    mode = 'exact' if count <= config['MAX_EXACT_PATTERNS'] else 'sampled'
                    for kind in ['arbitrary', 'contiguous']:
                        planned = (count if mode == 'exact' else config['NUM_RANDOM_SAMPLES']) if kind == 'arbitrary' else (N if W == N else W-m+1)
                        rows.append(dict(**p, inflation_n=n, N=N, domain_mode=config['ERASURE_DOMAIN_MODE'], window_size=W,
                            erasure_size=m, erasure_type=kind, calculation_mode=mode if kind == 'arbitrary' else 'exact',
                            population_patterns=count if kind == 'arbitrary' else (N if W == N else max(0,W-m+1)),
                            planned_trials=0 if reason else planned, status='skipped' if reason else 'planned', reason=reason))
    return pd.DataFrame(rows)


def benchmark_plan(config):
    rows = []
    for p in logical_pairs(config):
        for n in config['INFLATION_LIST']:
            row = dict(benchmark(p['seed_0'], p['seed_1'], n))
            row.update(pair_id=p['pair_id'], inflation_n=n)
            F0,F1=inflate(p['seed_0'],n),inflate(p['seed_1'],n)
            w0=tuple(int(F0[i:]+F0[:i],2) for i in range(len(F0)))
            w1=tuple(int(F1[i:]+F1[:i],2) for i in range(len(F1)))
            w0,w1=tuple(dict.fromkeys(w0)),tuple(dict.fromkeys(w1))
            E=tuple(range(min(3,len(F0))))
            start=time.perf_counter()
            for _ in range(3):
                evaluate_erasure(w0,w1,len(F0),E)
            row['KL_seconds_per_pattern']=(time.perf_counter()-start)/3
            rows.append(row)
    result=pd.DataFrame(rows)
    plan=build_plan(config)
    trials=plan.groupby(['pair_id','inflation_n'],as_index=False).planned_trials.sum()
    result=result.merge(trials,on=['pair_id','inflation_n'])
    result['estimated_KL_seconds']=result.KL_seconds_per_pattern*result.planned_trials
    return result


def wilson(k,n):
    z=1.959963984540054
    p=k/n; denom=1+z*z/n
    radius=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    center=(p+z*z/(2*n))/denom
    return max(0,center-radius),min(1,center+radius)


KEYS=['pair_id','seed_0','seed_1','inflation_n','N','domain_mode','window_size','erasure_size']

def summarize_erasure(df):
    rows=[]
    for key,g in df.groupby(KEYS+['erasure_type'],sort=False,dropna=False):
        row=dict(zip(KEYS+['erasure_type'],key)); n=len(g)
        exact=(g.calculation_mode=='exact').all()
        row.update(n_trials=n,n_unique_patterns=g.erasure_positions.nunique(),calculation_mode='exact' if exact else 'sampled',population_patterns=int(g.population_patterns.iloc[0]))
        for source,target in [('KL_correctable','R_KL'),('KL_diagonal','R_KL_diagonal'),('KL_offdiagonal','R_KL_offdiagonal'),('recoverable_both','R_recoverable_both'),('cross_collision','P_cross_collision')]:
            k=int(g[source].sum()); rate=k/n
            row[target]=rate
            row[target+'_low'],row[target+'_high']=(rate,rate) if exact else wilson(k,n)
        row['KL_failure_rate']=1-row['R_KL']
        row['diagonal_only_failure_rate']=float((~g.KL_diagonal & g.KL_offdiagonal).mean())
        row['offdiagonal_failure_rate']=float((~g.KL_offdiagonal).mean())
        for col in ['C_cross','C_self_0','C_self_1','Delta_C_self','I_RE','D_Z','D_X']:
            vals=pd.to_numeric(g[col],errors='coerce').dropna()
            row['mean_'+col]=float(vals.mean()) if len(vals) else np.nan
            row['measured_'+col]=len(vals)
            row['sem_'+col]=float(vals.std(ddof=1)/sqrt(len(vals))) if len(vals)>1 and not exact else 0 if len(vals) else np.nan
        row['offdiagonal_equivalence_verified']=bool((g.cross_collision == ~g.KL_offdiagonal).all())
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_row(g,x,y,min_count=3):
    data=g[[x,y]].dropna()
    if len(data)<min_count: reason='fewer_than_3_pairs'
    elif data[x].nunique()<2 or data[y].nunique()<2: reason='constant_variable'
    else: reason=''
    return dict(x=x,y=y,n=len(data),pearson=np.nan if reason else float(data[x].corr(data[y])),
        spearman=np.nan if reason else float(data[x].rank(method='average').corr(data[y].rank(method='average'))),
        status=reason or 'descriptive_only',note='No causal or out-of-sample prediction claim; shared seeds may create dependence.')


def correlate_pairs(comparison):
    rows=[]
    for key,g in comparison.groupby(['inflation_n','N','domain_mode','window_size','erasure_size'],dropna=False):
        for x,y in [('d_cross','R_KL_arbitrary'),('ell_cross_min','R_KL_contiguous'),('expected_cross_collision_arbitrary','R_KL_arbitrary'),('expected_cross_collision_contiguous','R_KL_contiguous'),('low_weight_large_span_unique_fraction','correction_rate_gap')]:
            rows.append(dict(zip(['inflation_n','N','domain_mode','window_size','erasure_size'],key))|correlation_row(g,x,y))
    return pd.DataFrame(rows)


def correlate_leakage(detail):
    rows=[]
    for key,g in detail.groupby(KEYS+['erasure_type'],dropna=False):
        for x,y in [('C_cross','D_X'),('C_cross','I_RE'),('Delta_C_self','D_Z'),('Delta_C_self','KL_diagonal')]:
            row=dict(zip(KEYS+['erasure_type'],key))|correlation_row(g,x,y)
            row['note']='Descriptive correlation across erasure trials; repeated samples are retained.'
            rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(config, progress=print):
    validate_config(config); plan=build_plan(config)
    if not plan.planned_trials.sum(): raise ValueError('No valid erasure configuration: inspect experiment_plan for skipped settings.')
    if plan.planned_trials.sum()>config.get('MAX_DETAIL_ROWS',1000000):
        raise ValueError('計画がMAX_DETAIL_ROWSを超えています。sampling数・設定数を減らすか上限を明示的に変更してください。')
    tables={k:[] for k in ['logical_summary','spectra','joint_spectrum','unique_supports','dangerous_supports','expected_collisions','erasure_collisions','orbit_pairs','collision_profiles','low_weight_large_span_supports']}
    checks=[]
    for p in logical_pairs(config):
        for n in config['INFLATION_LIST']:
            progress(f"{p['pair_id']}: inflation n={n}, geometry")
            geom=CollisionGeometry(p['seed_0'],p['seed_1'],n,pair_id=p['pair_id'],store_pair_details=config.get('STORE_PAIR_DETAILS',False))
            meta={k:geom.metadata[k] for k in ['pair_id','seed_0','seed_1','inflation_n','N']}
            low=config.get('LOW_WEIGHT_MAX',max(config['ERASURE_SIZE_LIST']))
            large=config.get('LARGE_SPAN_MIN') or int(np.ceil(config.get('LARGE_SPAN_FRACTION',.5)*geom.N))
            tables['logical_summary'].append(geom.summary(low_weight_max=low,large_span_min=large))
            tables['spectra'].extend(geom.spectrum_frame().to_dict('records'))
            tables['joint_spectrum'].extend(geom.joint_frame().to_dict('records'))
            for typ in ['cross','self_0','self_1']:
                tables['unique_supports'].extend(geom.support_frame(typ,include_positions=config.get('INCLUDE_ALL_SUPPORT_POSITIONS',False)).to_dict('records'))
            tables['dangerous_supports'].extend(geom.dangerous_supports(config['TOP_K_DANGEROUS_SUPPORTS']).to_dict('records'))
            selected_masks=[d for d in geom.supports['cross'] if geom.features[d][0]<=low and geom.features[d][1]>=large]
            tables['low_weight_large_span_supports'].extend(geom.support_frame('cross',selected_masks=selected_masks).to_dict('records'))
            if config.get('STORE_PAIR_DETAILS',False): tables['orbit_pairs'].extend(geom.iter_pair_details())
            for W in domains(geom.N,config):
                if W>geom.N: continue
                sizes=[m for m in config['ERASURE_SIZE_LIST'] if m<=W]
                window_positions=tuple((config.get('WINDOW_START',0)+i)%geom.N for i in range(W)) if config['ERASURE_DOMAIN_MODE']=='window' else tuple(range(geom.N))
                ex=geom.expected_frame(sizes,window_positions=window_positions)
                for rec in ex.to_dict('records'):
                    rec.update(meta,domain_mode=config['ERASURE_DOMAIN_MODE'],window_size=W)
                    rec['erasure_size']=rec.get('erasure_size',rec.get('m'))
                    rec['expected_cross_collision_contiguous_full_ring']=rec['expected_cross_collision_contiguous']
                    tables['expected_collisions'].append(rec)
                for m in sizes:
                    progress(f"  N={geom.N}, W={W}, m={m}: arbitrary / contiguous")
                    for kind in ['arbitrary','contiguous']:
                        patterns,calculation_mode=pattern_stream(geom.N,m,window_positions,kind,config)
                        stream=((E,calculation_mode) for E in patterns)
                        population=comb(W,m) if kind=='arbitrary' else (geom.N if W==geom.N else W-m+1)
                        for trial,(E,mode) in enumerate(stream):
                            leak=config.get('COMPUTE_LEAKAGE',False) and trial<config.get('LEAKAGE_MAX_PATTERNS_PER_GROUP',float('inf'))
                            prof=config.get('SAVE_COLLISION_PROFILES',False) and trial<config.get('PROFILE_MAX_PATTERNS_PER_GROUP',float('inf'))
                            r=evaluate_erasure(geom.words0,geom.words1,geom.N,E,compute_leakage=leak,max_leakage_dim=config.get('MAX_LEAKAGE_DIM',256),include_profile=prof)
                            profile=r.pop('collision_profile',None)
                            # Independently compare support-based count with residual-pattern grouping.
                            if trial<config.get('VALIDATE_SUPPORT_PATTERNS',3):
                                c=geom.collision_counts(E)
                                for key in ['C_cross','C_self_0','C_self_1']: assert c[key]==r[key], (key,c,r)
                            r.update(meta,domain_mode=config['ERASURE_DOMAIN_MODE'],window_size=W,erasure_type=kind,erasure_size=m,
                                erasure_positions=json.dumps(list(E)),trial_index=trial,calculation_mode=mode,population_patterns=population)
                            if profile is not None:
                                for item in profile: tables['collision_profiles'].append(dict(meta,window_size=W,erasure_type=kind,erasure_size=m,trial_index=trial,**item))
                            tables['erasure_collisions'].append(r)
            checks.append(dict(meta,geometry_pairs_verified=True))
    out={k:pd.DataFrame(v) for k,v in tables.items()}
    out['analysis_plan']=plan;out['skipped_configurations']=plan[plan.status=='skipped'].copy()
    out['performance']=summarize_erasure(out['erasure_collisions'])
    perf=out['performance']; exp=out['expected_collisions']
    # Matching-domain contiguous ensemble: full ring N starts, fixed window W-m+1 starts.
    cont=perf[perf.erasure_type=='contiguous'][KEYS+['mean_C_cross','mean_C_self_0','mean_C_self_1']]
    exp=exp.drop(columns=['expected_cross_collision_contiguous','expected_self_collision_0_contiguous','expected_self_collision_1_contiguous'],errors='ignore').merge(cont,on=KEYS,validate='one_to_one')
    exp=exp.rename(columns={'mean_C_cross':'expected_cross_collision_contiguous','mean_C_self_0':'expected_self_collision_0_contiguous','mean_C_self_1':'expected_self_collision_1_contiguous'})
    exp['union_bound_arbitrary']=np.minimum(1,exp.expected_cross_collision_arbitrary)
    exp['union_bound_contiguous']=np.minimum(1,exp.expected_cross_collision_contiguous)
    cross_totals=out['logical_summary'][['pair_id','inflation_n','number_cross_pairs']]
    exp=exp.merge(cross_totals,on=['pair_id','inflation_n'],validate='many_to_one')
    exp['collision_probability_lower_bound_arbitrary']=exp.expected_cross_collision_arbitrary/exp.number_cross_pairs
    exp['collision_probability_lower_bound_contiguous']=exp.expected_cross_collision_contiguous/exp.number_cross_pairs
    out['expected_collisions']=exp
    a=perf[perf.erasure_type=='arbitrary'].drop(columns='erasure_type')
    b=perf[perf.erasure_type=='contiguous'].drop(columns='erasure_type')
    comp=a.merge(b,on=KEYS,suffixes=('_arbitrary','_contiguous'),validate='one_to_one')
    comp=comp.merge(out['logical_summary'],on=['pair_id','seed_0','seed_1','inflation_n','N'],validate='many_to_one')
    comp=comp.merge(exp[KEYS+['expected_cross_collision_arbitrary','expected_cross_collision_contiguous','expected_cross_collision_contiguous_full_ring']],on=KEYS,validate='one_to_one')
    comp['correction_rate_gap']=comp.R_KL_contiguous-comp.R_KL_arbitrary
    comp['observed_pair_type']=np.select([(comp.R_KL_contiguous>=config.get('HIGH_RATE_THRESHOLD',.8))&(comp.R_KL_arbitrary<=config.get('LOW_RATE_THRESHOLD',.2)),(comp.R_KL_contiguous>=config.get('HIGH_RATE_THRESHOLD',.8))&(comp.R_KL_arbitrary>=config.get('HIGH_RATE_THRESHOLD',.8)),(comp.R_KL_contiguous<=config.get('LOW_RATE_THRESHOLD',.2))&(comp.R_KL_arbitrary<=config.get('LOW_RATE_THRESHOLD',.2))],['A','B','C'],default='intermediate')
    comp['performance_type']=comp.observed_pair_type
    comp['cross_only_prediction_overestimate']=comp.diagonal_only_failure_rate_arbitrary
    out['comparison']=comp;out['correlations']=correlate_pairs(comp)
    out['correlations']['Pearson']=out['correlations'].pearson
    out['correlations']['Spearman']=out['correlations'].spearman
    out['leakage_correlations']=correlate_leakage(out['erasure_collisions'])
    # This identity isolates the part of KL failure not explained by cross collision.
    assert np.allclose(perf.KL_failure_rate,perf.P_cross_collision+perf.diagonal_only_failure_rate)
    exact=comp.calculation_mode_arbitrary=='exact'
    if exact.any(): assert np.allclose(comp.loc[exact,'mean_C_cross_arbitrary'],comp.loc[exact,'expected_cross_collision_arbitrary'],atol=1e-12)
    assert np.allclose(comp.mean_C_cross_contiguous,comp.expected_cross_collision_contiguous,atol=1e-12)
    out['validation']=pd.DataFrame(checks)
    out['prior_rate_comparison']=compare_prior(out,config)
    for short,long in [('summary','logical_summary'),('supports','unique_supports'),('expected','expected_collisions'),('erasure_detail','erasure_collisions'),('rates','performance')]: out[short]=out[long]
    _,out['skipped']=experiment_plan(config)
    out['finite_size']=comp.sort_values(['pair_id','erasure_size','N']).copy()
    out['rates']['window_start']=config.get('WINDOW_START',0) if config['ERASURE_DOMAIN_MODE']=='window' else 0
    for name in ['R_KL','P_cross_collision']:
        out['rates'][name+'_ci_low']=out['rates'][name+'_low'];out['rates'][name+'_ci_high']=out['rates'][name+'_high']
    if config.get('SAVE_RESULTS',False): save_results(out,config,config.get('OUTPUT_DIR','results'))
    return out


def compare_prior(out,config):
    path=config.get('PRIOR_RATE_CSV')
    if not path: return pd.DataFrame([{'status':'not_requested','note':'Current rates are computed directly with exact legacy-compatible KL.'}])
    old=pd.read_csv(path,dtype={'seed_0':str,'seed_1':str})
    required=KEYS+['erasure_type','R_KL']
    if not set(required)<=set(old):
        raise ValueError('PRIOR_RATE_CSV needs columns: '+','.join(required)+'; use reference adapters, never join mismatched domains.')
    merged=out['performance'][required+['calculation_mode']].merge(old[required],on=KEYS+['erasure_type'],suffixes=('_new','_prior'),how='left',validate='one_to_one')
    merged['difference']=merged.R_KL_new-merged.R_KL_prior
    merged['status']=np.where(merged.R_KL_prior.isna(),'no_matching_prior_row','matched')
    return merged


def save_results(out,config,directory=None):
    directory=directory or config.get('OUTPUT_DIR','results')
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    for name,frame in out.items():
        if isinstance(frame,pd.DataFrame):
            # Never emit a zero-byte/headerless CSV for optional outputs.
            if len(frame.columns)==0: frame=pd.DataFrame(columns=['status'])
            frame.to_csv(directory/(name+('.csv.gz' if name=='orbit_pairs' else '.csv')),index=False)
    (directory/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    meta={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'source_sha256':hashes,
      'site_convention':'site 0 is leftmost / bit N-1','sampling':'uniform iid with replacement; duplicates retained',
      'contiguous_window':'starts 0..W-m; if W=N all N cyclic starts','leakage':'all trials unless LEAKAGE_MAX_PATTERNS_PER_GROUP is explicitly set; missing is not zero'}
    (directory/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (directory/'conclusions.md').write_text(conclusions(out),encoding='utf-8')


def conclusions(out):
    c=out['comparison'];s=out['logical_summary'];a=c[c.observed_pair_type=='A']
    measured=out['erasure_collisions'].I_RE.notna().sum()
    lines=['# Collision解析の観測結果',f"解析したlogical pairは{s.pair_id.nunique()}組、pair×inflationは{len(s)}条件です。",
      f"設定した閾値によるAタイプ（連続に強く任意に弱い）は{len(a)}条件です。",f"情報漏洩I_REの計算済み試行は{measured}件です。未計算値は欠測です。",
      'cross collision と KL off-diagonal failure の一致を全試行で確認しました。KL failure rate = P(cross collision) + P(diagonal failure かつ crossなし) を確認しました。',
      '期待collision数はpair数の期待値です。collision発生確率やKL failure rateとは異なります。',
      '相関は同じn,N,window,mの異なるpair間のみです。3pair未満・定数変数は係数を欠測にします。係数は探索的で、未知pairへの予測精度を示しません。',
      '全巡回orbitの状態は並進不変なので、full_ringの固定m連続消失KL訂正率は0または1です。',
      '設計候補はd_cross増大、低weight supportの削減、期待cross collision数の低減、ell_cross_min維持です。ただしdiagonal条件の評価を必ず併用します。',
      'n=0は検証用です。原論文のinflation後の保証を適用しません。']
    if len(a)==0: lines.append('この実行設定では中心仮説を判定できるAタイプがありません。仮説が確認できたとは結論しません。')
    else: lines.append('Aタイプの存在とlow-weight/large-span supportを比較できますが、機構がKL failure全体を説明するかはdiagonal-only failure率も確認してください。')
    return '\n\n'.join(lines)+'\n'

# Public notebook API.
def contiguous_patterns(N,m,window_positions):
    W=tuple(window_positions)
    if m<0 or m>len(W): raise ValueError('m outside domain')
    if len(W)==N:
        return [tuple(sorted((s+i)%N for i in range(m))) for s in range(N)]
    return [tuple(sorted(W[s:s+m])) for s in range(len(W)-m+1)]


def pattern_stream(N,m,window_positions,erasure_type,config):
    W=tuple(window_positions)
    if erasure_type=='contiguous': return iter(contiguous_patterns(N,m,W)),'exact'
    population=comb(len(W),m)
    if population<=config['MAX_EXACT_PATTERNS']: return (tuple(sorted(E)) for E in combinations(W,m)),'exact'
    token=f"{config['RANDOM_SEED']}|{N}|{W}|{m}".encode()
    rng=random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8],'big'))
    return (tuple(sorted(rng.sample(W,m))) for _ in range(config['NUM_RANDOM_SAMPLES'])),'sampled'


def experiment_plan(config):
    p=build_plan(config)
    skip=p[p.status=='skipped'].copy()
    skip['reason']=skip.reason.replace({'window_exceeds_N':'window_size>N','m_exceeds_domain':'erasure_size>domain_size'})
    plan=p[p.status=='planned'].copy().rename(columns={'planned_trials':'num_patterns'})
    return plan,skip


pair_correlations=correlate_pairs


def compare_reference_rates(rates,reference_directory):
    path=Path(reference_directory)/'recoverability_kl_summary.csv'
    if not path.exists(): return pd.DataFrame([dict(status='reference_not_available')])
    old=pd.read_csv(path,dtype={'seed_0':str,'seed_1':str})
    keys=['seed_0','seed_1','inflation_n','N','erasure_type','erasure_size','domain_mode','window_size']
    current=rates.copy()
    if 'window_start' in current: current=current[current.window_start==0]
    joined=current[keys+['R_KL','calculation_mode','n_trials']].merge(old[keys+['R_KL','calculation_mode','num_patterns','source']],on=keys,how='left',suffixes=('_new','_prior'),validate='one_to_many')
    joined['rate_difference']=joined.R_KL_new-joined.R_KL_prior
    joined['status']=np.where(joined.R_KL_prior.isna(),'no_matching_prior_row','matched')
    joined['comparison_note']='Exact runs should agree; sampled runs can differ because of sampling. Window origin is 0.'
    return joined
