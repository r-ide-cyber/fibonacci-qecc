"""Staged, resumable research workflow. Selection uses training results only."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from itertools import combinations
from math import comb, sqrt
from time import perf_counter
import hashlib
import json
import random
import warnings
import numpy as np
import pandas as pd
from .features import canonical, inflate, orbit, pair_features, collision_spectrum
from .core import (pair_id, validate_pair, representatives, adjacent_swap_valid,
                   pattern_bank, erasure_test, leakage_test, contiguous_masks, wilson)

SCHEMA_VERSION = "fib-design-v1"
MODEL_VERSION = "uniform-distinct-orbit-integer-KL-v1"
FEATURE_DIRECTIONS = {
    "d_cross": 1, "ell_cross_min": 1,
    "low_weight_cross_support_fraction": -1,
    "dangerous_support_multiplicity": -1,
    "expected_cross_collision_arbitrary": -1,
    "local_substring_mismatch": -1, "short_range_C2_mismatch": -1,
    "long_range_C2_mismatch": -1, "C3_mismatch": -1,
}


@dataclass
class Config:
    baseline_seed_0: str = "..."
    baseline_seed_1: str = "..."
    seed_length: int = 6
    k0: int = 3
    k1: int = 3
    candidate_mode: str = "all"
    manual_candidate_pairs: list = field(default_factory=list)
    loaded_candidates_csv: str = ""
    screening_inflation_n: int = 5
    validation_inflation_list: list = field(default_factory=lambda: [6, 7, 8])
    erasure_domain_mode: str = "full_ring"
    window_size_list: list = field(default_factory=lambda: [20, 30, 50])
    erasure_size_list: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    max_exact_patterns: int = 100000
    train_num_samples: int = 5000
    validation_num_samples: int = 10000
    train_random_seed: int = 42
    validation_random_seed: int = 31415
    top_k_candidates: int = 20
    top_k_leakage: int = 5
    leakage_num_samples: int = 200
    max_exact_leakage_patterns: int = 2000
    max_leakage_dimension: int = 2048
    low_weight_threshold: int = 5
    max_contiguous_degradation: float = 0.02
    require_no_contiguous_degradation: bool = False
    max_leakage_degradation: float = 1e-9
    require_statistical_evidence: bool = True
    min_validation_sizes: int = 2
    min_improving_m: int = 2
    normalization: str = "minmax"
    weights: dict = field(default_factory=lambda: {
        "d_cross": 1., "ell_cross_min": .5, "low_weight_cross_support_fraction": 1.,
        "dangerous_support_multiplicity": .5, "expected_cross_collision_arbitrary": 1.,
        "local_substring_mismatch": .5, "short_range_C2_mismatch": .5,
        "long_range_C2_mismatch": 1., "C3_mismatch": 1.})
    substring_lengths: tuple = (2, 3, 4)
    short_lags: tuple = (1, 2, 3)
    long_lags: tuple | None = None
    c3_offsets: tuple | None = None
    max_raw_seeds: int = 1000000
    max_possible_pairs: int = 2000000
    random_pair_count: int = 500
    random_max_attempts: int = 100000
    output_dir: str = "."
    previous_result_dirs: list = field(default_factory=list)
    save_results: bool = True
    save_figures: bool = True
    resume: bool = True
    n_jobs: int = 1

    def validate(self):
        if "..." in (self.baseline_seed_0, self.baseline_seed_1):
            raise ValueError("BASELINE_SEED_0 / BASELINE_SEED_1 に既存研究のseedを入力してください。動作確認のみなら DEMO_MODE=True。")
        validate_pair(self.baseline_seed_0, self.baseline_seed_1, self.seed_length, self.k0, self.k1)
        if self.k0 + self.k1 != self.seed_length:
            raise ValueError("K0 + K1 must equal SEED_LENGTH")
        if self.candidate_mode not in {"all", "manual", "loaded", "random"}:
            raise ValueError("Unknown CANDIDATE_MODE")
        if self.erasure_domain_mode not in {"full_ring", "window"}:
            raise ValueError("Unknown ERASURE_DOMAIN_MODE")
        if not self.validation_inflation_list or min([self.screening_inflation_n] + self.validation_inflation_list) < 0:
            raise ValueError("Inflation counts must be nonnegative")
        if not self.erasure_size_list or min(self.erasure_size_list) < 0:
            raise ValueError("ERASURE_SIZE_LIST must be nonempty and nonnegative")
        if self.erasure_domain_mode == "window" and (not self.window_size_list or min(self.window_size_list) < 1):
            raise ValueError("WINDOW_SIZE_LIST must contain positive sizes")
        if min(self.train_num_samples, self.validation_num_samples, self.top_k_candidates,
               self.top_k_leakage, self.leakage_num_samples, self.n_jobs) < 1:
            raise ValueError("Sample counts, Top-K and n_jobs must be positive")
        if self.normalization not in {"minmax", "zscore"}:
            raise ValueError("normalization must be minmax or zscore")
        if set(self.weights) - FEATURE_DIRECTIONS.keys() or any(v < 0 for v in self.weights.values()):
            raise ValueError("Weights must be nonnegative with known feature names")
        if self.max_exact_patterns < 1 or self.max_exact_leakage_patterns < 1:
            raise ValueError("Exact thresholds must be positive")
        return self


def _plain(value):
    if isinstance(value, dict): return {str(k): _plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def _atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_plain(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


class RecordStore:
    """JSON checkpoints plus matching, versioned CSV records; never loads pickle."""
    def __init__(self, config):
        self.config = config
        self.root = Path(config.output_dir) / "results" / "checkpoints"
        self.memory, self.imported, self.inventory = {}, {}, []
        sources = [Path(p) for p in config.previous_result_dirs]
        if config.resume:
            sources.append(Path(config.output_dir) / "results")
        for directory in sources:
            for path in sorted(directory.glob("*.csv")):
                try:
                    df = pd.read_csv(path, dtype={"seed_0": str, "seed_1": str})
                    usable = 0
                    for row in df.to_dict("records"):
                        if row.get("schema_version") == SCHEMA_VERSION and row.get("model_version") == MODEL_VERSION:
                            key = row.get("record_key")
                            if isinstance(key, str) and len(key) == 64 and isinstance(row.get("payload_json"), str):
                                payload = json.loads(row["payload_json"])
                                if payload.get("record_key") == key:
                                    self.imported[key] = payload
                                    usable += 1
                    self.inventory.append(dict(file=str(path), rows=len(df), compatible_records=usable,
                                               status="reusable" if usable else "reference_only: missing matching provenance"))
                except (ValueError, OSError, pd.errors.EmptyDataError) as exc:
                    self.inventory.append(dict(file=str(path), rows=0, compatible_records=0, status=str(exc)))

    def get(self, kind, signature, compute):
        signature = {"kind": kind, "model": MODEL_VERSION, "schema": SCHEMA_VERSION, **signature}
        key = hashlib.sha256(json.dumps(_plain(signature), sort_keys=True).encode()).hexdigest()
        if key in self.memory:
            return self.memory[key].copy()
        path = self.root / kind / f"{key}.json"
        result = self.imported.get(key) if self.config.resume else None
        if result is None and self.config.resume and path.exists():
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("record_key") != key: result = None
            except (ValueError, OSError):
                result = None
        if result is None:
            result = _plain(compute())
            result.update(record_key=key, schema_version=SCHEMA_VERSION, model_version=MODEL_VERSION)
            if self.config.save_results:
                _atomic_json(path, result)  # One atomic checkpoint per candidate/condition.
        self.memory[key] = result
        return result.copy()


def candidate_table(config):
    config.validate()
    total_raw = comb(config.seed_length, config.k1)
    reps, pairs, rejects = None, [], []
    if config.candidate_mode == "all":
        reps = representatives(config.seed_length, config.k0, config.k1, config.max_raw_seeds)
        if comb(len(reps), 2) > config.max_possible_pairs:
            raise ValueError("Possible pairs exceed MAX_POSSIBLE_PAIRS; choose random/manual/loaded")
        pairs = [(a,b) for a,b in combinations(reps,2) if adjacent_swap_valid(a,b)]
    elif config.candidate_mode == "manual":
        pairs = config.manual_candidate_pairs
    elif config.candidate_mode == "loaded":
        loaded = pd.read_csv(config.loaded_candidates_csv, dtype={"seed_0":str,"seed_1":str})
        if not {"seed_0","seed_1"}.issubset(loaded):
            raise ValueError("Loaded candidates CSV requires seed_0 and seed_1")
        pairs = list(loaded[["seed_0","seed_1"]].itertuples(index=False,name=None))
    else:
        rng, chosen = random.Random(config.train_random_seed), set()
        for _ in range(config.random_max_attempts):
            def draw():
                ones = set(rng.sample(range(config.seed_length), config.k1))
                return canonical("".join("1" if i in ones else "0" for i in range(config.seed_length)))
            pair = tuple(sorted((draw(), draw())))
            if adjacent_swap_valid(*pair): chosen.add(pair)
            if len(chosen) >= config.random_pair_count: break
        pairs = sorted(chosen)
    baseline = validate_pair(config.baseline_seed_0, config.baseline_seed_1, config.seed_length, config.k0, config.k1)
    valid = set()
    for pair in pairs:
        try: valid.add(validate_pair(*pair, config.seed_length, config.k0, config.k1))
        except (ValueError, TypeError) as exc: rejects.append({"pair":str(pair),"reason":str(exc)})
    valid.add(baseline)
    records = [dict(candidate_id=pair_id(a,b), seed_0=a, seed_1=b, seed_length=len(a), k0=config.k0,
                    k1=config.k1, adjacent_swap_valid=True, is_baseline=(a,b)==baseline) for a,b in sorted(valid)]
    stats = dict(raw_bit_strings=total_raw, number_cyclic_classes=len(reps) if reps is not None else None,
                 number_possible_pairs=comb(len(reps),2) if reps is not None else None,
                 number_adjacent_swap_valid_pairs=len(valid), number_valid_seeds=len({x for p in valid for x in p}),
                 exhaustive=config.candidate_mode=="all", rejected_input_pairs=len(rejects),
                 nonbaseline_candidates=len(valid)-1)
    return pd.DataFrame(records), stats, pd.DataFrame(rejects)


def domains(config, n):
    N = len(inflate(config.baseline_seed_0,n))
    widths = [N] if config.erasure_domain_mode == "full_ring" else sorted(set(config.window_size_list))
    return [(N,W,m) for W in widths if W <= N for m in sorted(set(config.erasure_size_list)) if m <= W]


def feature_task(task):
    a,b,n,kwargs = task
    return pair_features(a,b,n,**kwargs)


class Experiment:
    def __init__(self, config):
        self.config = config.validate()
        self.store = RecordStore(config)
        self.candidates, self.counts, self.rejected = candidate_table(config)
        self.baseline = self.candidates.loc[self.candidates.is_baseline].iloc[0].to_dict()
        self.by_id = {r["candidate_id"]:r for r in self.candidates.to_dict("records")}
        self.tables = {"candidates":self.candidates, "rejected":self.rejected,
                       "prior_inventory":pd.DataFrame(self.store.inventory)}
        audit=[]
        for n in sorted(set([config.screening_inflation_n]+config.validation_inflation_list)):
            N=len(inflate(config.baseline_seed_0,n))
            for W in ([N] if config.erasure_domain_mode=="full_ring" else sorted(set(config.window_size_list))):
                for m in sorted(set(config.erasure_size_list)):
                    audit.append(dict(inflation_n=n,N=N,window_size=W,erasure_size=m,
                                      eligible=m<=W<=N,reason="" if m<=W<=N else "W>N or m>W"))
        self.tables["condition_audit"]=pd.DataFrame(audit)
        self.banks = {}

    def _features(self, row, n, W):
        c = self.config
        kwargs = dict(low_weight_threshold=c.low_weight_threshold, erasure_sizes=tuple(c.erasure_size_list),
                      window_size=W,substring_lengths=c.substring_lengths,short_lags=c.short_lags,
                      long_lags=c.long_lags,c3_offsets=c.c3_offsets)
        result = self.store.get("features",dict(a=row["seed_0"],b=row["seed_1"],n=n,kwargs=kwargs),
                                lambda:feature_task((row["seed_0"],row["seed_1"],n,kwargs)))
        result.update({k:row[k] for k in ("candidate_id","seed_0","seed_1","is_baseline")})
        result["inflation_n"] = n
        result["expected_cross_collision_normalized"] = result["expected_cross_collision_arbitrary"]/result["cross_support_count"]
        return result

    def benchmark(self):
        c, b = self.config, self.baseline
        result = dict(self.counts)
        n = c.screening_inflation_n
        available = domains(c,n)
        if not available: raise ValueError("No feasible screening (N,W,m); check windows and erasure sizes")
        N,W,m = available[-1]
        t = perf_counter(); self._features(b,n,W); feature_seconds = perf_counter()-t
        bank = self._bank(N,W,m,"train")
        masks = bank["masks"][:min(30,len(bank["masks"]))]
        t=perf_counter()
        for mask in masks: erasure_test(b["seed_0"],b["seed_1"],n,mask)
        per_pattern=(perf_counter()-t)/len(masks)
        n_detailed = min(c.top_k_candidates, len(self.candidates)-1)+1
        n_patterns=0
        for vn in sorted(set([n]+c.validation_inflation_list)):
            for nN,w,mm in domains(c,vn):
                total=comb(w,mm)
                n_patterns += total if total<=c.max_exact_patterns else c.validation_num_samples
        result.update(orbit_pair_count=len(orbit(b["seed_0"],n))*len(orbit(b["seed_1"],n)),
                      screening_time_per_pair_seconds=feature_seconds,
                      estimated_screening_total_seconds=feature_seconds*len(self.candidates)*len({w for _,w,_ in available}),
                      KL_seconds_per_pattern=per_pattern,
                      KL_time_per_candidate_seconds=per_pattern*sum(min(comb(w,mm),c.train_num_samples) if comb(w,mm)>c.max_exact_patterns else comb(w,mm) for _,w,mm in available),
                      estimated_validation_total_seconds=per_pattern*n_patterns*n_detailed,
                      benchmark_N=N,benchmark_W=W,benchmark_m=m,
                      estimate_note="Baseline pilot only; runtime grows with N, collision structure, and leakage eigensolver. Cached feature timing may be warm.")
        self.tables["benchmark"]=pd.DataFrame([result])
        return self.tables["benchmark"]

    def screen(self):
        c=self.config; n=c.screening_inflation_n
        windows=sorted({W for _,W,_ in domains(c,n)})
        if not windows: raise ValueError("No feasible screening domain")
        # For n_jobs>1 only uncached pure pair feature tasks are dispatched.
        if c.n_jobs>1:
            from concurrent.futures import ProcessPoolExecutor
            tasks=[]
            for row in self.candidates.to_dict("records"):
                for W in windows:
                    kw=dict(low_weight_threshold=c.low_weight_threshold,erasure_sizes=tuple(c.erasure_size_list),window_size=W,
                            substring_lengths=c.substring_lengths,short_lags=c.short_lags,long_lags=c.long_lags,c3_offsets=c.c3_offsets)
                    tasks.append((row["seed_0"],row["seed_1"],n,kw))
            # Workers compute only; parent owns all writes. Resume uses serial cached path.
            if not c.resume:
                with ProcessPoolExecutor(max_workers=c.n_jobs) as pool:
                    for task,result in zip(tasks,pool.map(feature_task,tasks)):
                        a,b,nn,kw=task
                        self.store.get("features",dict(a=a,b=b,n=nn,kwargs=kw),lambda r=result:r)
        rows=[]; detailed=[]
        for row in self.candidates.to_dict("records"):
            parts=[self._features(row,n,W) for W in windows]; detailed.extend(parts)
            result={**row,**parts[0]}
            for metric in ("expected_cross_collision_arbitrary","expected_cross_collision_contiguous","expected_cross_collision_normalized"):
                result[metric]=float(np.mean([p[metric] for p in parts]))
            result.update(screening_inflation_n=n,screening_windows=json.dumps(windows))
            # Aggregated screening rows are not imported as raw feature records.
            for key in ("record_key","schema_version","model_version"): result.pop(key,None)
            rows.append(result)
        df=pd.DataFrame(rows)
        for feature,direction in FEATURE_DIRECTIONS.items():
            values=df[feature].astype(float)
            scale=values.max()-values.min() if c.normalization=="minmax" else values.std(ddof=0)
            centre=values.min() if c.normalization=="minmax" else values.mean()
            df["normalized_"+feature]=((values-centre)/scale if scale>1e-15 else values*0)*direction
        collision=[f for f in FEATURE_DIRECTIONS if f in {"d_cross","ell_cross_min","low_weight_cross_support_fraction","dangerous_support_multiplicity","expected_cross_collision_arbitrary"}]
        sequence=[f for f in FEATURE_DIRECTIONS if f not in collision]
        def score(features): return sum((c.weights.get(f,0)*df["normalized_"+f] for f in features),start=pd.Series(0.,index=df.index))
        df["design_score"]=score(FEATURE_DIRECTIONS)
        df["collision_score"]=score(collision); df["sequence_score"]=score(sequence)
        objectives=np.column_stack([df[f].to_numpy()*FEATURE_DIRECTIONS[f] for f in ("d_cross","expected_cross_collision_arbitrary","ell_cross_min","local_substring_mismatch")])
        df["pareto_optimal"]=[not np.any(np.all(objectives>=point,axis=1)&np.any(objectives>point,axis=1)) for point in objectives]
        self.tables["screening"]=df.sort_values(["design_score","candidate_id"],ascending=[False,True]).reset_index(drop=True)
        self.tables["validation_features"]=pd.DataFrame(detailed)
        methods={"collision_only":"collision_score","sequence_only":"sequence_score","combined":"design_score"}
        for f in FEATURE_DIRECTIONS:
            col="without_"+f; df[col]=score([g for g in FEATURE_DIRECTIONS if g!=f]); methods[col]=col
        ablation=[]
        nonbase=df[~df.is_baseline]
        for method,col in methods.items():
            for rank,row in enumerate(nonbase.sort_values([col,"candidate_id"],ascending=[False,True]).to_dict("records"),1):
                ablation.append(dict(method=method,rank=rank,candidate_id=row["candidate_id"],score=row[col]))
        self.tables["ablation"]=pd.DataFrame(ablation,columns=["method","rank","candidate_id","score"])
        # Quota-free round robin: preserve Pareto and both feature families within K.
        orders=[list(nonbase.sort_values(["pareto_optimal","design_score"],ascending=False).candidate_id)]
        orders += [list(nonbase.sort_values([col,"candidate_id"],ascending=[False,True]).candidate_id) for col in methods.values()]
        selected=[]
        for rank in range(len(nonbase)):
            for order in orders:
                if order[rank] not in selected and len(selected)<c.top_k_candidates: selected.append(order[rank])
        self.selected_ids=[self.baseline["candidate_id"]]+selected
        self.tables["screening"]["selected_for_KL"]=self.tables["screening"].candidate_id.isin(self.selected_ids)
        self.tables["collision_spectrum"]=pd.DataFrame([
            dict(candidate_id=row["candidate_id"],inflation_n=n,N=len(inflate(row["seed_0"],n)),weight=w,multiplicity=v)
            for row in self.candidates.to_dict("records")
            for w,v in collision_spectrum(row["seed_0"],row["seed_1"],n).items()])
        self.save()
        return self.tables["screening"]

    def _bank(self,N,W,m,split):
        key=N,W,m
        c=self.config
        if key not in self.banks:
            self.banks[key]=pattern_bank(N,W,m,c.max_exact_patterns,c.train_num_samples,c.validation_num_samples,c.train_random_seed,c.validation_random_seed)
            if c.save_results:
                for label,entry in self.banks[key].items():
                    data={**entry,"masks":[str(x) for x in entry["masks"]],"N":N,"W":W,"m":m,"split":label}
                    _atomic_json(Path(c.output_dir)/"results"/"samples"/f"N{N}_W{W}_m{m}_{label}_{entry['sample_id']}.json",data)
        return self.banks[key][split]

    def _performance(self,row,n,N,W,m,split):
        bank=self._bank(N,W,m,split)
        def compute():
            flags=[erasure_test(row["seed_0"],row["seed_1"],n,e) for e in bank["masks"]]
            contiguous=erasure_test(row["seed_0"],row["seed_1"],n,(1<<m)-1)
            arr=np.array(flags,dtype=int)
            rate=float(arr[:,0].mean())
            ci= (rate,rate) if bank["calculation_mode"]=="exact" else wilson(int(arr[:,0].sum()),len(arr))
            return dict(R_KL_arbitrary=rate,R_KL_contiguous=float(contiguous[0]),
                        R_rec_arbitrary=float(arr[:,1].mean()),R_rec_contiguous=float(contiguous[1]),
                        R_label_rec_arbitrary=float(arr[:,2].mean()),R_label_rec_contiguous=float(contiguous[2]),
                        KL_ci_low=ci[0],KL_ci_high=ci[1],kl_flags=arr[:,0].tolist())
        result=self.store.get("performance",dict(a=row["seed_0"],b=row["seed_1"],n=n,N=N,W=W,m=m,sample_id=bank["sample_id"]),compute)
        result.update({k:v for k,v in bank.items() if k!="masks"})
        result.update(candidate_id=row["candidate_id"],seed_0=row["seed_0"],seed_1=row["seed_1"],is_baseline=row["is_baseline"],
                      inflation_n=n,N=N,window_size=W,erasure_size=m,sample_split=split,
                      contiguous_calculation_mode="exact_translation_symmetry",
                      contiguous_population_patterns=len(contiguous_masks(N,W,m)))
        return result

    def evaluate(self,ids,inflations,split):
        records=[]
        for n in sorted(set(inflations)):
            for N,W,m in domains(self.config,n):
                baseline=self._performance(self.baseline,n,N,W,m,split)
                for cid in ids:
                    r=baseline.copy() if cid==self.baseline["candidate_id"] else self._performance(self.by_id[cid],n,N,W,m,split)
                    delta=np.array(r["kl_flags"])-np.array(baseline["kl_flags"])
                    difference=float(delta.mean())
                    if r["calculation_mode"]=="exact" or r["is_baseline"]:
                        lower=upper=difference
                    else:
                        # Distribution-free 95% paired Hoeffding bound, conservative
                        # for sampling without replacement. Valid even zero discordance.
                        radius=sqrt(2*np.log(40)/len(delta))
                        lower=max(-1.,difference-radius); upper=min(1.,difference+radius)
                    r.update(Delta_R_arbitrary=difference,Delta_R_contiguous=r["R_KL_contiguous"]-baseline["R_KL_contiguous"],
                             paired_delta_ci_low=lower,paired_delta_ci_high=upper,
                             paired_candidate_only_success=int((delta==1).sum()),paired_baseline_only_success=int((delta==-1).sum()),
                             gap_KL=r["R_KL_contiguous"]-r["R_KL_arbitrary"])
                    records.append(r)
        frame=pd.DataFrame(records)
        old=self.tables.get("performance",pd.DataFrame())
        self.tables["performance"]=pd.concat([old,frame],ignore_index=True).drop_duplicates(
            ["candidate_id","inflation_n","window_size","erasure_size","sample_split"],keep="last") if not frame.empty else old
        self.save()
        return frame

    def train(self):
        result=self.evaluate(self.selected_ids,[self.config.screening_inflation_n],"train")
        grouped=result.groupby("candidate_id").agg(gain=("Delta_R_arbitrary","mean"),worst_contiguous=("Delta_R_contiguous","min"))
        tol=0 if self.config.require_no_contiguous_degradation else self.config.max_contiguous_degradation
        chosen=grouped[(grouped.index!=self.baseline["candidate_id"]) & (grouped.worst_contiguous>=-tol)].sort_values("gain",ascending=False).head(self.config.top_k_leakage)
        # Freeze all validation and leakage decisions before looking at held-out data.
        self.leakage_ids=[self.baseline["candidate_id"]]+chosen.index.tolist()
        self.frozen_selection=dict(KL_ids=self.selected_ids,leakage_ids=self.leakage_ids,
                                   basis="surrogate Top-K then training KL only",validation_used_for_selection=False)
        if self.config.save_results:
            _atomic_json(Path(self.config.output_dir)/"results"/"frozen_selection.json",self.frozen_selection)
        return result

    def leakage(self,ids,inflations,split):
        c=self.config; rows=[]
        for n in sorted(set(inflations)):
            for N,W,m in domains(c,n):
                bank=self._bank(N,W,m,split)
                masks=bank["masks"]
                if not(bank["calculation_mode"]=="exact" and len(masks)<=c.max_exact_leakage_patterns):
                    rng=random.Random(int(hashlib.sha256((bank["sample_id"]+"leakage").encode()).hexdigest()[:16],16))
                    indices=sorted(rng.sample(range(len(masks)),min(c.leakage_num_samples,len(masks))))
                    masks=tuple(masks[i] for i in indices)
                lid=hashlib.sha256(",".join(map(str,masks)).encode()).hexdigest()[:20]
                exact=bank["calculation_mode"]=="exact" and len(masks)==bank["population_patterns"]
                for cid in ids:
                    row=self.by_id[cid]
                    def compute(row=row,masks=masks,n=n):
                        values=pd.DataFrame([leakage_test(row["seed_0"],row["seed_1"],n,e,c.max_leakage_dimension) for e in masks])
                        result={f"{stat}_{metric}":float(getattr(values[metric],stat)()) for metric in ("I_RE","D_Z","D_X","D_Y") for stat in ("mean","median","max")}
                        return result
                    r=self.store.get("leakage",dict(a=row["seed_0"],b=row["seed_1"],n=n,N=N,W=W,m=m,sample_id=lid),compute)
                    r.update(candidate_id=cid,is_baseline=row["is_baseline"],seed_0=row["seed_0"],seed_1=row["seed_1"],inflation_n=n,N=N,window_size=W,
                             erasure_size=m,sample_split=split,n_leakage_patterns=len(masks),sample_id=lid,
                             leakage_mode="exact" if exact else "sampled",worst_case_certified=exact,
                             max_I_RE_interpretation="global_maximum" if exact else "observed_maximum_lower_bound")
                    rows.append(r)
        frame=pd.DataFrame(rows)
        self.tables["leakage"]=pd.concat([self.tables.get("leakage",pd.DataFrame()),frame],ignore_index=True).drop_duplicates(
            ["candidate_id","inflation_n","window_size","erasure_size","sample_split"],keep="last")
        self.save()
        return frame

    def validate(self):
        # Includes an independent split at screening N, and the requested new N.
        ns=sorted(set([self.config.screening_inflation_n]+self.config.validation_inflation_list))
        results=self.evaluate(self.selected_ids,ns,"validation")
        features=[]
        for n in ns:
            for W in sorted({w for _,w,_ in domains(self.config,n)}):
                for cid in self.selected_ids: features.append(self._features(self.by_id[cid],n,W))
        self.tables["validation_features"]=pd.DataFrame(features)
        self.leakage(self.leakage_ids,ns,"validation")
        self.summarize()
        self.save()
        return results

    def summarize(self):
        p=self.tables["performance"]; c=self.config
        # One AUC per N,W first, then equal-weight averaging over conditions.
        agg=[]
        for (cid,split),group in p.groupby(["candidate_id","sample_split"]):
            auc=[]
            for _,curve in group.groupby(["inflation_n","window_size"]):
                curve=curve.sort_values("erasure_size"); x=curve.erasure_size.to_numpy(); y=curve.R_KL_arbitrary.to_numpy()
                auc.append(float(np.trapezoid(y,x)/(x[-1]-x[0])) if len(x)>1 else float(y[0]))
            agg.append(dict(candidate_id=cid,sample_split=split,mean_R_KL_arbitrary=group.R_KL_arbitrary.mean(),
                            mean_R_KL_contiguous=group.R_KL_contiguous.mean(),AUC_arbitrary=np.mean(auc),
                            worst_R_KL_arbitrary=group.R_KL_arbitrary.min()))
        self.tables["aggregate"]=pd.DataFrame(agg)
        pv=p[(p.sample_split=="validation") & p.inflation_n.isin(c.validation_inflation_list)]
        lv=self.tables.get("leakage",pd.DataFrame())
        lv=lv[(lv.sample_split=="validation") & lv.inflation_n.isin(c.validation_inflation_list)] if not lv.empty else lv
        keys=["inflation_n","N","window_size","erasure_size"]
        baseline_leak=lv[lv.is_baseline] if not lv.empty else lv
        tol=0 if c.require_no_contiguous_degradation else c.max_contiguous_degradation
        rows=[]
        features=self.tables["screening"].set_index("candidate_id")
        for cid,g in pv.groupby("candidate_id"):
            if cid==self.baseline["candidate_id"]: continue
            r=self.by_id[cid]; own=lv[lv.candidate_id==cid] if not lv.empty else lv
            paired=own.merge(baseline_leak,on=keys,suffixes=("","_baseline")) if not own.empty else pd.DataFrame()
            covered=not paired.empty and len(paired)==len(g)
            leakage_delta=float((paired.max_I_RE-paired.max_I_RE_baseline).max()) if covered else np.nan
            leakage_ok=covered and leakage_delta<=c.max_leakage_degradation
            improving=g[g.Delta_R_arbitrary>1e-12]
            all_sizes=all((g[g.inflation_n==n].Delta_R_arbitrary>1e-12).any() for n in set(c.validation_inflation_list))
            improved_m=improving.erasure_size.nunique()
            shape_ok=g.inflation_n.nunique()>=c.min_validation_sizes and all_sizes and improved_m>=c.min_improving_m
            empirical_ok=g.Delta_R_arbitrary.min()>=-1e-12 and g.Delta_R_arbitrary.mean()>1e-12
            contiguous_ok=g.Delta_R_contiguous.min()>=-tol-1e-12
            # Family-wise union bound across tested condition/candidate differences.
            comparisons=max(1,len(pv[~pv.is_baseline]))
            lower=g.Delta_R_arbitrary-np.sqrt(2*np.log(2*comparisons/.05)/g.n_patterns)
            lower=g.Delta_R_arbitrary.where(g.calculation_mode=="exact",lower)
            supported=(lower>0)
            evidence_ok=g.calculation_mode.eq("exact").all() or (supported.groupby(g.inflation_n).any().all() and supported.groupby(g.erasure_size).any().sum()>=c.min_improving_m)
            adjacent=adjacent_swap_valid(r["seed_0"],r["seed_1"])
            passed=bool(shape_ok and empirical_ok and contiguous_ok and leakage_ok and adjacent and (evidence_ok or not c.require_statistical_evidence))
            category=("A" if g.Delta_R_contiguous.min()>=-1e-12 else "B") if shape_ok and empirical_ok and contiguous_ok else ("D" if len(improving) else "C")
            reasons=[]
            for ok,label in [(shape_ok,"insufficient improvements across N/m"),(empirical_ok,"arbitrary non-improvement or regression"),(contiguous_ok,"contiguous degradation"),(leakage_ok,"leakage missing or worse"),(evidence_ok,"sample evidence insufficient")]:
                if not ok: reasons.append(label)
            feat=features.loc[cid].to_dict()
            rows.append({**feat,"candidate_id":cid,"classification":category,"adjacent_swap_valid":adjacent,
                "average_arbitrary_improvement":g.Delta_R_arbitrary.mean(),"worst_arbitrary_improvement":g.Delta_R_arbitrary.min(),
                "contiguous_change":g.Delta_R_contiguous.min(),"improving_size_count":improving.inflation_n.nunique(),
                "improving_m_count":improved_m,"max_I_RE":own.max_I_RE.max() if not own.empty else np.nan,
                "worst_leakage_change":leakage_delta,"leakage_coverage_complete":covered,"leakage_not_worse_observed":leakage_ok,
                "leakage_worst_case_certified":bool(covered and paired.worst_case_certified.all() and paired.worst_case_certified_baseline.all()),
                "statistical_evidence":bool(evidence_ok),"validation_pass":passed,"validation_scope":"tested finite N,W,m only",
                "baseline_comparison":"; ".join(reasons) if reasons else "passes configured finite validation"})
        final=pd.DataFrame(rows) if rows else pd.DataFrame(columns=["rank","candidate_id","seed_0","seed_1","validation_pass","baseline_comparison"])
        if not final.empty:
            final=final.sort_values(["validation_pass","average_arbitrary_improvement","worst_arbitrary_improvement"],ascending=False).reset_index(drop=True)
            final.insert(0,"rank",range(1,len(final)+1))
        self.tables["final"]=final
        comparison=[]
        ids=[self.baseline["candidate_id"]]+(final.candidate_id.head(3).tolist() if not final.empty else [])
        for cid in ids:
            g=pv[pv.candidate_id==cid]; f=features.loc[cid]
            own=lv[lv.candidate_id==cid] if not lv.empty else lv
            comparison.append(dict(label="Baseline" if cid==self.baseline["candidate_id"] else f"Candidate {ids.index(cid)}",
                candidate_id=cid,seed_pair=f"{f.seed_0} / {f.seed_1}",d_cross=f.d_cross,ell_cross_min=f.ell_cross_min,
                R_KL_arbitrary=g.R_KL_arbitrary.mean(),R_KL_contiguous=g.R_KL_contiguous.mean(),gap_KL=g.gap_KL.mean(),
                max_I_RE=own.max_I_RE.max() if not own.empty else np.nan,expected_collision_count=f.expected_cross_collision_arbitrary,
                long_range_C2_mismatch=f.long_range_C2_mismatch,C3_mismatch=f.C3_mismatch,
                performance_scope="mean over requested validation n,W,m",feature_inflation_n=c.screening_inflation_n))
        self.tables["comparison"]=pd.DataFrame(comparison)
        train=self.tables["aggregate"].query("sample_split == 'train'")
        joined=self.tables["screening"].merge(train,on="candidate_id")
        cor=[]
        for f in ["design_score","collision_score","sequence_score",*FEATURE_DIRECTIONS]:
            if len(joined)>=3 and joined[f].nunique()>1 and joined.mean_R_KL_arbitrary.nunique()>1:
                value=joined[f].rank().corr(joined.mean_R_KL_arbitrary.rank())
            else: value=np.nan
            cor.append(dict(feature=f,spearman=value,n_evaluated=len(joined),scope="selected candidates only; selection-biased exploratory correlation"))
        self.tables["correlations"]=pd.DataFrame(cor)
        if not self.tables["ablation"].empty:
            ab=self.tables["ablation"].drop(columns=["mean_R_KL_arbitrary","mean_R_KL_contiguous","AUC_arbitrary","worst_R_KL_arbitrary","sample_split"],errors="ignore")
            self.tables["ablation"]=ab.merge(train,on="candidate_id",how="left")
        validn=sorted(set(pv.inflation_n)) if not pv.empty else []
        self.conclusion=dict(baseline=[self.baseline["seed_0"],self.baseline["seed_1"]],counts=self.counts,
                             passed_candidates=int(final.validation_pass.sum()) if not final.empty else 0,
                             tested_validation_inflations=validn,
                             statement=("No alternative valid pair in this exhaustive seed space." if self.counts["exhaustive"] and self.counts["nonbaseline_candidates"]==0 else
                                        "Report only tested conditions; no untested candidate or infinite-size guarantee."))
        if self.config.save_results:
            _atomic_json(Path(c.output_dir)/"results"/"conclusion.json",self.conclusion)
        return final

    def save(self):
        if not self.config.save_results: return
        out=Path(self.config.output_dir)/"results"; out.mkdir(parents=True,exist_ok=True)
        names={"candidates":"code_design_candidates","screening":"code_design_screening",
               "performance":"code_design_KL_validation","leakage":"code_design_leakage",
               "final":"code_design_final_ranking","comparison":"code_design_baseline_comparison"}
        for key,df in self.tables.items():
            export=df.copy()
            if "record_key" in export:
                export["payload_json"]=[json.dumps(_plain(self.store.memory.get(k,self.store.imported.get(k,{}))),ensure_ascii=False) for k in export.record_key]
            export=export.drop(columns=["kl_flags"],errors="ignore")
            path=out/(names.get(key,"code_design_"+key)+".csv")
            temp=path.with_suffix(".csv.tmp"); export.to_csv(temp,index=False); temp.replace(path)
        _atomic_json(out/"manifest.json",dict(schema_version=SCHEMA_VERSION,model_version=MODEL_VERSION,config=asdict(self.config),
            logical_state="equal positive superposition over distinct cyclic orbit",recoverability="injective classical whole-word reconstruction",
            D_X="0.5 trace_norm(rho01+rho10)",source="https://arxiv.org/html/2311.13040v2#A5",
            validation_note="Sampled maxima are observed lower bounds. CSV seeds must be read with dtype=str."))

    def run(self):
        self.benchmark(); self.screen(); self.train()
        self.leakage(self.leakage_ids,[self.config.screening_inflation_n],"train")
        self.validate()
        return self.tables
