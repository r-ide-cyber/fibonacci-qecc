"""Exact, cached collision and cyclic sequence statistics (standard library only).

Bit ``i`` of an integer represents site ``i`` of the written binary string.
Orbits contain each distinct rotation once, including for periodic strings.
These screening statistics are surrogates; they do not test KL conditions.
"""

from collections import Counter
from functools import lru_cache
from math import comb
from numbers import Integral


# A transparent guard against unexpectedly materializing millions of supports.
# Change this module constant deliberately for a larger-memory experiment.
MAX_CROSS_ORBIT_PAIRS = 4_000_000


def _check_seed(seed):
    if not isinstance(seed, str) or not seed or set(seed) - {"0", "1"}:
        raise ValueError("seed must be a nonempty binary string")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@lru_cache(maxsize=1024)
def inflate(seed: str, n: int) -> str:
    """Apply ``1 -> 10, 0 -> 1`` exactly n times (n=0 is the seed)."""
    _check_seed(seed)
    n = _integer(n, "n")
    word = seed
    for _ in range(n):
        word = "".join("10" if bit == "1" else "1" for bit in word)
    return word


@lru_cache(maxsize=8192)
def canonical(seed: str) -> str:
    """Lexicographically least cyclic rotation, with no reflection quotient."""
    _check_seed(seed)
    return min(seed[i:] + seed[:i] for i in range(len(seed)))


def _rotate(value, shift, size):
    shift %= size
    if not shift:
        return value
    return (value >> shift) | ((value & ((1 << shift) - 1)) << (size - shift))


@lru_cache(maxsize=512)
def orbit(seed: str, n: int) -> tuple[int, ...]:
    """Sorted distinct rotations; string position i is integer bit position i."""
    word = inflate(seed, n)
    size = len(word)
    start = int(word[::-1], 2)
    values, value = {start}, _rotate(start, 1, size)
    while value != start:
        values.add(value)
        value = _rotate(value, 1, size)
    return tuple(sorted(values))


def _pair_orbits(seed0, seed1, n):
    size = len(inflate(seed0, n))
    if len(inflate(seed1, n)) != size:
        raise ValueError("inflated strings must have equal length")
    return size, orbit(seed0, n), orbit(seed1, n)


@lru_cache(maxsize=4)
def cross_supports(seed0: str, seed1: str, n: int) -> dict[int, int]:
    """Return support -> multiplicity over O0 x O1; DO NOT mutate cached result.

    A support is the XOR mask x^y. Multiplicity counts pairs of *distinct*
    orbit elements, so periodic words do not receive extra rotation weights.
    """
    _, orbit0, orbit1 = _pair_orbits(seed0, seed1, n)
    count = len(orbit0) * len(orbit1)
    if count > MAX_CROSS_ORBIT_PAIRS:
        raise ValueError(
            f"{count:,} cross-orbit pairs exceed MAX_CROSS_ORBIT_PAIRS="
            f"{MAX_CROSS_ORBIT_PAIRS:,}; lower screening inflation or increase "
            "the guard deliberately after estimating memory"
        )
    return dict(Counter(x ^ y for x in orbit0 for y in orbit1))


@lru_cache(maxsize=1024)
def collision_spectrum(seed0: str, seed1: str, n: int) -> dict[int, int]:
    """Weight -> pair multiplicity; DO NOT mutate cached result.

    Rotation invariance reduces the work to one fixed x in O0 and all y in
    O1, multiplied by |O0|. This also holds when either orbit is periodic.
    """
    _, orbit0, orbit1 = _pair_orbits(seed0, seed1, n)
    counts = Counter((orbit0[0] ^ y).bit_count() for y in orbit1)
    return {weight: counts[weight] * len(orbit0) for weight in sorted(counts)}


def _positions(mask):
    positions = []
    while mask:
        bit = mask & -mask
        positions.append(bit.bit_length() - 1)
        mask ^= bit
    return positions


def _cyclic_gaps(mask, size):
    positions = _positions(mask)
    if not positions:
        return ()
    return tuple(b - a for a, b in zip(positions, positions[1:])) + (
        size + positions[0] - positions[-1],
    )


def cyclic_span(mask: int, size: int) -> int:
    """Minimum covering cyclic interval cardinality, N-max(gap)+1; empty=0."""
    size = _integer(size, "size", 1)
    mask = _integer(mask, "mask")
    if mask >> size:
        raise ValueError("mask contains positions outside size")
    return size - max(_cyclic_gaps(mask, size)) + 1 if mask else 0


def _contiguous_count(mask, size, window, m):
    """Number of distinct allowed m-interval masks that contain support mask."""
    if m in (0, window):
        return int(mask & ~((1 << m) - 1) == 0)
    if not mask:
        return size if window == size else window - m + 1
    if window == size:
        # The complement interval of length N-m must lie in a gap.
        return sum(max(0, gap - (size - m)) for gap in _cyclic_gaps(mask, size))
    if mask >> window:
        return 0
    first = (mask & -mask).bit_length() - 1
    last = mask.bit_length() - 1
    return max(0, min(first, window - m) - max(0, last - m + 1) + 1)


def _contiguous_pattern_count(size, window, m):
    if m in (0, window):
        return 1
    return size if size == window else window - m + 1


def _expected_collisions(seed0, seed1, n, sizes, window, supports):
    size, orbit0, orbit1 = _pair_orbits(seed0, seed1, n)
    if window == size:
        # Both expectations are rotation invariant, so use one reference x.
        entries = ((orbit0[0] ^ y, len(orbit0)) for y in orbit1)
    else:
        entries = supports.items()
    max_m = max(sizes)
    weight_counts = Counter()
    contiguous_totals = {m: 0 for m in sizes}
    for support, multiplicity in entries:
        weight = support.bit_count()
        if weight > max_m or support >> window:
            continue
        weight_counts[weight] += multiplicity
        for m in sizes:
            if weight <= m:
                contiguous_totals[m] += multiplicity * _contiguous_count(
                    support, size, window, m
                )
    arbitrary = {
        m: sum(
            count * (comb(window - weight, m - weight) / comb(window, m))
            for weight, count in weight_counts.items()
            if weight <= m
        )
        for m in sizes
    }
    contiguous = {
        m: contiguous_totals[m] / _contiguous_pattern_count(size, window, m)
        for m in sizes
    }
    return sum(arbitrary.values()) / len(sizes), sum(contiguous.values()) / len(sizes)


@lru_cache(maxsize=4096)
def _substring_distribution(seed, n, length):
    word = inflate(seed, n)
    extended = word + word[: length - 1]
    counts = Counter(extended[i : i + length] for i in range(len(word)))
    return tuple((substring, count / len(word)) for substring, count in counts.items())


@lru_cache(maxsize=8192)
def _c2(seed, n, lag):
    """C2(r)=N^-1 sum_i s_i s_(i+r), with s_i=2*b_i-1."""
    word = inflate(seed, n)
    value, size = int(word[::-1], 2), len(word)
    return 1 - 2 * (value ^ _rotate(value, lag, size)).bit_count() / size


@lru_cache(maxsize=8192)
def _c3(seed, n, r, t):
    """C3(r,t)=N^-1 sum_i s_i s_(i+r) s_(i+t), s_i=2*b_i-1."""
    word = inflate(seed, n)
    value, size = int(word[::-1], 2), len(word)
    parity = value ^ _rotate(value, r, size) ^ _rotate(value, t, size)
    return 2 * parity.bit_count() / size - 1


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _sequence_mismatches(seed0, seed1, n, lengths, short_lags, long_lags, c3_offsets):
    substring_tv = []
    for length in lengths:
        p = dict(_substring_distribution(seed0, n, length))
        q = dict(_substring_distribution(seed1, n, length))
        substring_tv.append(sum(abs(p.get(s, 0) - q.get(s, 0)) for s in p.keys() | q.keys()) / 2)
    local = _mean(substring_tv)
    short = _mean([abs(_c2(seed0, n, r) - _c2(seed1, n, r)) for r in short_lags])
    long = _mean([abs(_c2(seed0, n, r) - _c2(seed1, n, r)) for r in long_lags])
    three = _mean([abs(_c3(seed0, n, r, t) - _c3(seed1, n, r, t)) for r, t in c3_offsets])
    return {
        "local_substring_mismatch": local,
        "short_range_C2_mismatch": short,
        "long_range_C2_mismatch": long,
        "C3_mismatch": three,
        "local_sequence_mismatch": (local + short) / 2,
    }


@lru_cache(maxsize=4096)
def _pair_features_cached(seed0, seed1, n, threshold, sizes, window, lengths, short, long, triples):
    size, orbit0, orbit1 = _pair_orbits(seed0, seed1, n)
    supports = cross_supports(seed0, seed1, n)
    low_counts = [count for support, count in supports.items() if support.bit_count() <= threshold]
    arbitrary, contiguous = _expected_collisions(seed0, seed1, n, sizes, window, supports)
    relative = [orbit0[0] ^ y for y in orbit1]
    total = len(orbit0) * len(orbit1)
    result = {
        "N": size,
        "window_size": window,
        "orbit_size_0": len(orbit0),
        "orbit_size_1": len(orbit1),
        "d_cross": min(support.bit_count() for support in relative),
        "ell_cross_min": min(cyclic_span(support, size) for support in relative),
        "low_weight_cross_support_fraction": len(low_counts) / len(supports),
        "number_unique_cross_supports": len(supports),
        "support_multiplicity": total / len(supports),
        "max_support_multiplicity": max(supports.values()),
        "dangerous_support_multiplicity": max(low_counts, default=0),
        "cross_support_count": total,
        "expected_cross_collision_arbitrary": arbitrary,
        "expected_cross_collision_contiguous": contiguous,
        "expected_cross_collision_count": arbitrary,
    }
    result.update(_sequence_mismatches(seed0, seed1, n, lengths, short, long, triples))
    return tuple(result.items())


def pair_features(
    seed0: str,
    seed1: str,
    n: int,
    *,
    low_weight_threshold=5,
    erasure_sizes=(1, 2, 3),
    window_size=None,
    substring_lengths=(2, 3, 4),
    short_lags=(1, 2, 3),
    long_lags=None,
    c3_offsets=None,
) -> dict:
    """Exact structural surrogates, with no KL calculation or sampling.

    The low-weight fraction is over distinct supports. ``support_multiplicity``
    is the mean pair count per distinct support; ``cross_support_count`` is
    |O0|*|O1|. Expected collisions count pairs, rather than collision probability.
    They average equally over distinct requested feasible m values (0<=m<=W).
    For arbitrary erasure the inclusion probability of a support S inside the
    window is C(W-|S|,m-|S|)/C(W,m), and is zero outside the window.

    ``window_size=None`` (or N) uses all distinct cyclic intervals on the ring.
    W<N uses sites 0..W-1 with nonwrapping intervals. m=0 and m=W each have one
    distinct interval. The full support multiset is used even in window mode.

    Substring mismatch is mean total variation over valid lengths 1..N. C2/C3
    mismatch is mean absolute spin-correlation difference over distinct lags.
    Defaults for long lags are N//4,N//3,N//2; C3 offsets are (1,2), (1,N//2),
    (N//3,2*N//3). Offsets are reduced modulo N and repeated sites omitted.
    An empty sequence-feature family contributes 0. The local sequence score
    is (substring mismatch + short-range C2 mismatch)/2. Return is a fresh dict.
    """
    size, original_orbit0, original_orbit1 = _pair_orbits(seed0, seed1, n)
    n = _integer(n, "n")
    threshold = _integer(low_weight_threshold, "low_weight_threshold")
    window = size if window_size is None else _integer(window_size, "window_size", 1)
    if window > size:
        raise ValueError("window_size cannot exceed inflated size N")
    requested_sizes = {_integer(m, "erasure size") for m in erasure_sizes}
    sizes = tuple(sorted(m for m in requested_sizes if m <= window))
    if not sizes:
        raise ValueError("erasure_sizes contains no feasible values in 0..window_size")
    requested_lengths = {_integer(k, "substring length", 1) for k in substring_lengths}
    lengths = tuple(sorted(k for k in requested_lengths if k <= size))

    def lag_tuple(values):
        return tuple(sorted({int(_integer(r, "lag") % size) for r in values} - {0}))

    short = lag_tuple(short_lags)
    long = lag_tuple((size // 4, size // 3, size // 2) if long_lags is None else long_lags)
    if c3_offsets is None:
        c3_offsets = ((1, 2), (1, size // 2), (size // 3, 2 * size // 3))
    triples = set()
    for offsets in c3_offsets:
        if len(offsets) != 2:
            raise ValueError("each c3_offsets entry must be a pair (r,t)")
        r, t = (_integer(offset, "C3 offset") % size for offset in offsets)
        if r and t and r != t:
            triples.add(tuple(sorted((r, t))))
    # Rotation and label exchange leave all returned metrics invariant.
    seed0, seed1 = sorted((canonical(seed0), canonical(seed1)))
    result = dict(_pair_features_cached(seed0, seed1, n, threshold, sizes, window, lengths, short, long, tuple(sorted(triples))))
    result["orbit_size_0"] = len(original_orbit0)
    result["orbit_size_1"] = len(original_orbit1)
    return result
