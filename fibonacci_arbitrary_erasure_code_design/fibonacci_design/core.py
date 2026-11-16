"""Exact sparse erasure KL tests; no 2**N state vector is constructed."""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from math import comb, sqrt
from bisect import bisect_right
import hashlib
import json
import random
import numpy as np
from .features import canonical, inflate, orbit


def pair_id(a, b):
    a, b = sorted((canonical(a), canonical(b)))
    return "pair_" + hashlib.sha256(f"{a}|{b}".encode()).hexdigest()[:12]


@lru_cache(maxsize=8192)
def swap_neighbors(seed):
    """All cyclic classes from one unequal adjacent swap, including boundary."""
    out = set()
    for i in range(len(seed)):
        j = (i + 1) % len(seed)
        if seed[i] != seed[j]:
            word = list(seed)
            word[i], word[j] = word[j], word[i]
            out.add(canonical("".join(word)))
    return frozenset(out)


def adjacent_swap_valid(a, b):
    """Li–Boyle Appendix E.1: no one-swap image belongs to {a,b}."""
    a, b = canonical(a), canonical(b)
    chosen = {a, b}
    return a != b and not (swap_neighbors(a) & chosen or swap_neighbors(b) & chosen)


def validate_pair(a, b, length=None, k0=None, k1=None):
    a, b = sorted((canonical(a), canonical(b)))
    if len(a) != len(b) or a.count("0") != b.count("0"):
        raise ValueError("Seeds must have the same length and composition")
    if length is not None and len(a) != length:
        raise ValueError("Seed length differs from SEED_LENGTH")
    if k0 is not None and a.count("0") != k0:
        raise ValueError("Seed zero count differs from K0")
    if k1 is not None and a.count("1") != k1:
        raise ValueError("Seed one count differs from K1")
    if not adjacent_swap_valid(a, b):
        raise ValueError(f"Pair {a}/{b} fails cyclic inequivalence or adjacent-swap condition")
    return a, b


def representatives(length, k0, k1, max_raw=1_000_000):
    if length < 1 or min(k0, k1) < 0 or k0 + k1 != length:
        raise ValueError("Require SEED_LENGTH = K0 + K1 > 0")
    if comb(length, k1) > max_raw:
        raise ValueError("Seed space exceeds MAX_RAW_SEEDS; use random/manual/loaded mode")
    words = set()
    for positions in combinations(range(length), k1):
        word = ["0"] * length
        for i in positions:
            word[i] = "1"
        words.add(canonical("".join(word)))
    return sorted(words)


@lru_cache(maxsize=1024)
def logical_orbits(a, b, n):
    nsites = len(inflate(a, n))
    if len(inflate(b, n)) != nsites:
        raise ValueError("Different physical lengths")
    o0, o1 = orbit(a, n), orbit(b, n)
    if set(o0) & set(o1):
        raise ValueError("Logical states are not orthogonal: intersecting orbits")
    return nsites, o0, o1


def _groups(values, erased, full):
    groups = defaultdict(list)
    retained = full ^ erased
    for x in values:
        groups[x & retained].append(x & erased)
    return groups


def _counts(groups):
    counts = Counter()
    for states in groups.values():
        for x in states:
            for y in states:
                counts[x, y] += 1
    return counts


def erasure_test(a, b, n, erased):
    """(KL, full-string recoverability, label recoverability), exact integers.

    R_rec in this package denotes injectivity of O0 union O1 -> retained bits.
    Label recoverability only requires disjoint retained logical supports.
    Neither classical indicator is a replacement for quantum correctability.
    """
    size, o0, o1 = logical_orbits(a, b, n)
    if erased < 0 or erased >> size:
        raise ValueError("Erasure mask outside physical ring")
    g0, g1 = _groups(o0, erased, (1 << size) - 1), _groups(o1, erased, (1 << size) - 1)
    label_rec = g0.keys().isdisjoint(g1)
    rec = label_rec and all(len(v) == 1 for v in g0.values()) and all(len(v) == 1 for v in g1.values())
    if not label_rec:
        return False, False, False  # Positive amplitudes forbid cancellation in rho01.
    c0, c1 = _counts(g0), _counts(g1)
    diagonal = all(len(o1) * c0[key] == len(o0) * c1[key] for key in c0.keys() | c1.keys())
    return diagonal, rec, label_rec


def reduced_blocks(a, b, n, erased, max_dimension=2048):
    size, o0, o1 = logical_orbits(a, b, n)
    if erased < 0 or erased >> size:
        raise ValueError("Erasure mask outside physical ring")
    full = (1 << size) - 1
    g0, g1 = _groups(o0, erased, full), _groups(o1, erased, full)
    support = sorted({x & erased for x in o0 + o1})
    if 2 * len(support) > max_dimension:
        raise ValueError(f"rho_RE dimension {2*len(support)} exceeds MAX_LEAKAGE_DIMENSION")
    index = {x: i for i, x in enumerate(support)}
    def block(ga, gb, norm):
        result = np.zeros((len(support), len(support)))
        for retained in ga.keys() & gb.keys():
            ia, ib = [index[x] for x in ga[retained]], [index[x] for x in gb[retained]]
            result[np.ix_(ia, ib)] += 1 / norm
        return result
    return block(g0, g0, len(o0)), block(g1, g1, len(o1)), block(g0, g1, sqrt(len(o0)*len(o1)))


def entropy(matrix):
    eig = np.linalg.eigvalsh(matrix)
    if eig.min(initial=0) < -1e-9 or not np.isclose(eig.sum(), 1, atol=1e-9):
        raise ArithmeticError("Invalid reduced density matrix")
    eig = eig[eig > 1e-14]
    return float(-np.sum(eig * np.log2(eig)))


def leakage_test(a, b, n, erased, max_dimension=2048):
    r0, r1, r01 = reduced_blocks(a, b, n, erased, max_dimension)
    joint = np.block([[r0, r01], [r01.T, r1]]) / 2
    value = 1 + entropy((r0 + r1) / 2) - entropy(joint)
    if value < -1e-9 or value > 2 + 1e-9:
        raise ArithmeticError("Mutual information outside [0,2]")
    return {
        "I_RE": float(np.clip(value, 0, 2)),
        "D_Z": float(np.abs(np.linalg.eigvalsh(r0-r1)).sum()/2),
        "D_X": float(np.abs(np.linalg.eigvalsh(r01+r01.T)).sum()/2),
        "D_Y": float(np.abs(np.linalg.eigvalsh(1j*(r01.T-r01))).sum()/2),
    }


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], "big")


def unrank_combination(rank, width, m):
    """Lexicographic combinadic inverse, supports populations > int64."""
    positions = []
    start = 0
    for remaining in range(m, 0, -1):
        for x in range(start, width):
            count = comb(width-x-1, remaining-1)
            if rank < count:
                positions.append(x)
                start = x + 1
                break
            rank -= count
    return sum(1 << i for i in positions)


def _unique_ranks(population, count, rng):
    # Floyd sampling, O(count) space, no rejection blow-up near full enumeration.
    selected = set()
    for j in range(population-count, population):
        t = rng.randrange(j+1)
        selected.add(j if t in selected else t)
    return tuple(sorted(selected))


def _allowed_rank(rank, excluded):
    # Map [0, population-len(excluded)) monotonically into unexcluded ranks.
    lo, hi = rank, rank + len(excluded)
    while lo < hi:
        mid = (lo + hi) // 2
        if mid - bisect_right(excluded, mid) < rank:
            lo = mid + 1
        else:
            hi = mid
    return lo


def pattern_bank(N, W, m, max_exact, train_count, validation_count, train_seed, validation_seed):
    """Shared deterministic, uniform samples, without replacement or split overlap.

    Exact sets are intentionally identical across splits and explicitly marked.
    For sampled sets validation excludes training ranks. Thus its population is
    the held-out complement, recorded separately from the full population.
    """
    if not (0 <= m <= W <= N):
        raise ValueError("Require 0 <= m <= W <= N")
    if min(train_count, validation_count) < 1:
        raise ValueError("Sample counts must be positive")
    total = comb(W, m)
    if total <= max_exact:
        masks = tuple(sum(1 << i for i in p) for p in combinations(range(W), m))
        return {s: _bank_entry(masks, "exact", total, total, False) for s in ("train", "validation")}
    if total < 2:
        raise ValueError("No held-out patterns; raise MAX_EXACT_PATTERNS")
    nt = min(train_count, total-1)
    tr = _unique_ranks(total, nt, random.Random(stable_seed(train_seed, N, W, m, "train")))
    nv = min(validation_count, total-nt)
    vr = _unique_ranks(total-nt, nv, random.Random(stable_seed(validation_seed, N, W, m, "validation")))
    vr = tuple(_allowed_rank(r, tr) for r in vr)
    return {
        "train": _bank_entry(tuple(unrank_combination(r,W,m) for r in tr), "sampled", total, total, True),
        "validation": _bank_entry(tuple(unrank_combination(r,W,m) for r in vr), "sampled", total, total-nt, True),
    }


def _bank_entry(masks, mode, population, effective_population, disjoint):
    sample_id = hashlib.sha256(",".join(map(str, masks)).encode()).hexdigest()[:20]
    return dict(masks=masks, calculation_mode=mode, population_patterns=population,
                effective_population=effective_population, n_patterns=len(masks), sample_id=sample_id,
                splits_disjoint=disjoint)


def contiguous_masks(N, W, m):
    if not 0 <= m <= W <= N:
        raise ValueError("Invalid contiguous domain")
    starts = range(N if W == N else W-m+1)
    return tuple(sorted({sum(1 << ((i+j) % N) for j in range(m)) for i in starts}))


def wilson(successes, count):
    z = 1.959963984540054
    p = successes/count
    denominator = 1+z*z/count
    centre = (p+z*z/(2*count))/denominator
    half = z*sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return max(0.,centre-half), min(1.,centre+half)
