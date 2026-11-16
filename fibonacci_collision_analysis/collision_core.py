"""Collision geometry for the legacy 0->1, 1->10 Fibonacci orbit code.

Site i maps to integer bit N-1-i. Absolute supports are never identified under
rotation; only the rotation-invariant weight/span feature calculation is reused.
"""
from collections import Counter
from itertools import combinations, islice
from math import comb
from numbers import Integral
from time import perf_counter
from pathlib import Path
import csv, gzip, json
import numpy as np
import pandas as pd


def integer(x,name,minimum=0):
    if isinstance(x,bool) or not isinstance(x,Integral) or x<minimum: raise ValueError(name)
    return int(x)


def bitstring(x):
    if not isinstance(x,str) or not x or set(x)-{'0','1'}: raise ValueError('nonempty binary string required')
    return x


def inflate(seed,n):
    word=bitstring(seed)
    for _ in range(integer(n,'inflation_n')): word=''.join('1' if x=='0' else '10' for x in word)
    return word


def orbit_with_shifts(word):
    bitstring(word); seen=set(); result=[]
    for shift in range(len(word)):
        x=word[shift:]+word[:shift]
        if x not in seen: result.append((shift,x)); seen.add(x)
    return result


def cyclic_orbit(word): return tuple(x for s,x in orbit_with_shifts(word))

def adjacent_swaps(seed):
    out=set()
    for i in range(len(seed)):
        j=(i+1)%len(seed)
        if seed[i]!=seed[j]:
            x=list(seed); x[i],x[j]=x[j],x[i]; out.add(''.join(x))
    return out


def validate_pair(seed_0,seed_1):
    a,b=bitstring(seed_0),bitstring(seed_1)
    o0,o1=set(cyclic_orbit(a)),set(cyclic_orbit(b))
    s0,s1=adjacent_swaps(a),adjacent_swaps(b)
    valid=len(a)==len(b) and a.count('1')==b.count('1') and not o0&o1 and not s0&o1 and not s1&o0
    if not valid: raise ValueError('seed pair must have equal k0/k1, different cyclic classes, and no single periodic adjacent swap between classes')
    return dict(seed_length=len(a),k0=a.count('0'),k1=a.count('1'),adjacent_swap_valid=True,
                paper_no_self_swap_0=not bool(s0&o0),paper_no_self_swap_1=not bool(s1&o1))


def sites_to_mask(E,N):
    N=integer(N,'N',1); sites=tuple(integer(i,'site') for i in E)
    if len(set(sites))!=len(sites) or any(i>=N for i in sites): raise ValueError('sites must be distinct and in [0,N)')
    return sum(1<<(N-1-i) for i in sites)


def mask_to_sites(mask,N):
    N=integer(N,'N',1); mask=integer(mask,'mask')
    if mask.bit_length()>N: raise ValueError('mask exceeds N')
    out=[]
    while mask:
        p=mask.bit_length()-1; out.append(N-1-p); mask^=1<<p
    return tuple(out)


def difference_support(x,y):
    bitstring(x); bitstring(y)
    if len(x)!=len(y): raise ValueError('word lengths differ')
    return set(mask_to_sites(int(x,2)^int(y,2),len(x)))


def cyclic_span(D,N):
    sites=mask_to_sites(sites_to_mask(D,N),N)
    if not sites: return 0
    gaps=[sites[i+1]-sites[i] for i in range(len(sites)-1)]+[N+sites[0]-sites[-1]]
    return N-max(gaps)+1


def _features(mask,N):
    return mask.bit_count(),cyclic_span(mask_to_sites(mask,N),N)


def interval_mask(start,m,N):
    return sites_to_mask(((start+i)%N for i in range(m)),N)


def probability_by_weight(w,L,m):
    if m<0 or m>L: raise ValueError('m outside domain')
    return comb(L-w,m-w)/comb(L,m) if 0<=w<=m else 0.


def support_probability(D,N,m,window_positions=None):
    mask=sites_to_mask(D,N); m=integer(m,'m')
    W=tuple(range(N)) if window_positions is None else tuple(window_positions)
    wm=sites_to_mask(W,N)
    if m>len(W): raise ValueError('m outside window')
    if mask&~wm: return 0.
    return probability_by_weight(mask.bit_count(),len(W),m)


def contiguous_probability(D,N,m):
    N=integer(N,'N',1); m=integer(m,'m')
    if m>N: raise ValueError('m>N')
    mask=sites_to_mask(D,N)
    return sum(mask&~interval_mask(s,m,N)==0 for s in range(N))/N


def _weighted_median(hist):
    if not hist: return np.nan
    total=sum(hist.values()); targets=((total-1)//2,total//2); vals=[]; c=0
    for v,num in sorted(hist.items()):
        for target in targets:
            if c<=target<c+num: vals.append(v)
        c+=num
    return sum(vals)/2


class CollisionGeometry:
    def __init__(self,seed_0,seed_1,inflation_n,pair_id=None,store_pair_details=False):
        self.metadata=validate_pair(seed_0,seed_1)
        self.seed_0,self.seed_1,self.inflation_n=seed_0,seed_1,inflation_n
        self.F0,self.F1=inflate(seed_0,inflation_n),inflate(seed_1,inflation_n)
        self.N=len(self.F0)
        a,b=orbit_with_shifts(self.F0),orbit_with_shifts(self.F1)
        self.shifts0,self.O0=tuple(s for s,x in a),tuple(x for s,x in a)
        self.shifts1,self.O1=tuple(s for s,x in b),tuple(x for s,x in b)
        if len(self.F1)!=self.N or set(self.O0)&set(self.O1): raise ValueError('inflated logical supports must be disjoint, same length')
        self.words0,self.words1=tuple(map(lambda x:int(x,2),self.O0)),tuple(map(lambda x:int(x,2),self.O1))
        self.metadata.update(pair_id=pair_id or f'{seed_0}__{seed_1}',seed_0=seed_0,seed_1=seed_1,
                             inflation_n=inflation_n,N=self.N,orbit_size_0=len(a),orbit_size_1=len(b))
        self.supports={}; self.features={}; self.examples={}; self._pairs=[] if store_pair_details else None
        start=perf_counter()
        for kind,words_a,words_b,shifts_a,shifts_b in [
            ('cross',self.words0,self.words1,self.shifts0,self.shifts1),
            ('self_0',self.words0,self.words0,self.shifts0,self.shifts0),
            ('self_1',self.words1,self.words1,self.shifts1,self.shifts1)]:
            counter=Counter(); features_by_relative_shift={}; examples={}
            for i,x in enumerate(words_a):
                for j in range(0 if kind=='cross' else i+1,len(words_b)):
                    mask=x^words_b[j]; counter[mask]+=1
                    relative=(shifts_b[j]-shifts_a[i])%self.N
                    if relative not in features_by_relative_shift: features_by_relative_shift[relative]=_features(mask,self.N)
                    self.features[mask]=features_by_relative_shift[relative]
                    if mask not in examples: examples[mask]=(shifts_a[i],shifts_b[j])
                    if self._pairs is not None: self._pairs.append((kind,shifts_a[i],shifts_b[j],mask))
            self.supports[kind]=counter; self.examples[kind]=examples
        self.elapsed_seconds=perf_counter()-start

    def summary(self,low_weight_max=None,large_span_min=None):
        low_weight_max=4 if low_weight_max is None else low_weight_max
        large_span_min=max(1,(self.N+1)//2) if large_span_min is None else large_span_min
        row=dict(self.metadata)
        for kind,counter in self.supports.items():
            weights=Counter(); spans=Counter()
            for mask,c in counter.items():
                w,s=self.features[mask]; weights[w]+=c; spans[s]+=c
            row['d_'+kind]=min(weights) if weights else np.nan
            row['ell_'+kind+'_min']=min(spans) if spans else np.nan
            if kind=='cross':
                total=sum(counter.values()); hit=[d for d in counter if self.features[d][0]<=low_weight_max and self.features[d][1]>=large_span_min]
                row.update(mean_cross_hamming=sum(w*c for w,c in weights.items())/total,
                           median_cross_hamming=_weighted_median(weights),mean_cross_span=sum(s*c for s,c in spans.items())/total,
                           median_cross_span=_weighted_median(spans),number_cross_pairs=total,number_unique_cross_supports=len(counter),
                           low_weight_max=low_weight_max,large_span_min=large_span_min,
                           low_weight_large_span_pair_fraction=sum(counter[d] for d in hit)/total,
                           low_weight_large_span_unique_fraction=len(hit)/len(counter),
                           low_weight_pair_fraction=sum(c for w,c in weights.items() if w<=low_weight_max)/total)
        row['geometry_seconds']=self.elapsed_seconds
        return row

    def spectrum_frame(self):
        rows=[]
        for kind,counter in self.supports.items():
            for feature,index in [('hamming',0),('span',1)]:
                for unique in (False,True):
                    counts=Counter()
                    for d,num in counter.items(): counts[self.features[d][index]]+=1 if unique else num
                    total=sum(counts.values()); cumulative=0
                    for value,count in sorted(counts.items()):
                        cumulative+=count
                        rows.append(dict(self.metadata,metric_type=kind+'_'+feature+('_unique' if unique else ''),value=value,
                                         count=count,fraction=count/total,cumulative_count=cumulative,cumulative_fraction=cumulative/total))
        return pd.DataFrame(rows)

    def joint_frame(self):
        rows=[]
        for kind,counter in self.supports.items():
            hist=Counter(); unique=Counter()
            for d,num in counter.items(): hist[self.features[d]]+=num; unique[self.features[d]]+=1
            total=sum(hist.values())
            for (w,s),c in sorted(hist.items()): rows.append(dict(self.metadata,pair_type=kind,hamming_weight=w,cyclic_span=s,count=c,fraction=c/total,unique_count=unique[w,s]))
        return pd.DataFrame(rows)

    def support_frame(self,pair_type='cross',include_positions=True,selected_masks=None):
        rows=[]
        for d in (self.supports[pair_type] if selected_masks is None else selected_masks):
            c=self.supports[pair_type][d]
            w,s=self.features[d]; a,b=self.examples[pair_type][d]
            rows.append(dict(self.metadata,pair_type=pair_type,support_mask=hex(d),difference_positions=json.dumps(mask_to_sites(d,self.N)) if include_positions else None,
                             hamming_weight=w,cyclic_span=s,multiplicity=c,support_multiplicity=c,shift_index_a=a,shift_index_b=b))
        return pd.DataFrame(rows)

    def dangerous_supports(self,top_k=20):
        selected=sorted(self.supports['cross'],key=lambda d:(self.features[d][0],-self.supports['cross'][d],-self.features[d][1]))[:top_k]
        return self.support_frame('cross',selected_masks=selected).reset_index(drop=True)

    def collision_counts(self,E):
        mask=sites_to_mask(E,self.N)
        return {'C_'+kind:sum(c for d,c in counter.items() if d&~mask==0) for kind,counter in self.supports.items()}

    def expected_frame(self,m_values,window_positions=None):
        W=tuple(range(self.N)) if window_positions is None else tuple(window_positions)
        wm=sites_to_mask(W,self.N); histograms={}
        for kind,counter in self.supports.items():
            hist=Counter()
            for d,c in counter.items():
                if not d&~wm: hist[self.features[d][0]]+=c
            histograms[kind]=hist
        rows=[]
        for m in m_values:
            m=integer(m,'m')
            if m>len(W): raise ValueError('m exceeds domain')
            # A complete cyclic orbit makes the sum over ALL pair events invariant
            # under the interval start. Thus the N-start mean equals start 0.
            full_counts=self.collision_counts(tuple(range(m)))
            row=dict(self.metadata,erasure_size=m)
            for kind,hist in histograms.items():
                expected=sum(c*probability_by_weight(w,len(W),m) for w,c in hist.items() if w<=m)
                if kind=='cross': row['expected_cross_collision_arbitrary']=expected; row['expected_cross_collision_contiguous']=full_counts['C_cross']
                else:
                    i=kind[-1]; row[f'expected_self_collision_{i}']=expected; row[f'expected_self_collision_{i}_arbitrary']=expected
                    row[f'expected_self_collision_{i}_contiguous']=full_counts['C_'+kind]
            rows.append(row)
        return pd.DataFrame(rows)

    def iter_pair_details(self):
        if self._pairs is None: raise ValueError('construct with store_pair_details=True')
        for index,(kind,a,b,d) in enumerate(self._pairs):
            w,s=self.features[d]
            yield dict(self.metadata,pair_type=kind,cross_pair_id=index if kind=='cross' else None,
                       shift_index_a=a,shift_index_b=b,hamming_weight=w,cyclic_span=s,
                       difference_positions=json.dumps(mask_to_sites(d,self.N)),support_mask=hex(d),support_multiplicity=self.supports[kind][d])

    def export_pair_details(self,path):
        path=Path(path); rows=self.iter_pair_details(); first=next(rows,None)
        if first is None: return
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'wt',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(first)); writer.writeheader(); writer.writerow(first); writer.writerows(rows)


def benchmark(seed_0,seed_1,inflation_n,sample_pairs=1000):
    validate_pair(seed_0,seed_1)
    a,b=inflate(seed_0,inflation_n),inflate(seed_1,inflation_n); N=len(a)
    o0,o1=cyclic_orbit(a),cyclic_orbit(b); words0,words1=tuple(int(x,2) for x in o0),tuple(int(x,2) for x in o1)
    pairs=list(islice(((x,y) for x in words0 for y in words1),sample_pairs))
    t=perf_counter()
    for x,y in pairs: _features(x^y,N)
    elapsed=perf_counter()-t
    cross=len(o0)*len(o1); self_count=comb(len(o0),2)+comb(len(o1),2)
    return dict(inflation_n=inflation_n,N=N,orbit_size_0=len(o0),orbit_size_1=len(o1),number_cross_pairs=cross,
                number_self_pairs=self_count,sampled_pairs=len(pairs),difference_support_seconds=elapsed,
                estimated_total_seconds_naive=(cross+self_count)*elapsed/max(1,len(pairs)),
                estimate_scope='naive feature computation; excludes KL/leakage/serialization; actual core reuses relative shifts')
