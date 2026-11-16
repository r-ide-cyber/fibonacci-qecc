"""Exact collision/KL diagnostics for positive equal-amplitude orbit codes.

Site 0 is the leftmost binary digit. Self collisions are unordered pairs of
distinct words; cross collisions are pairs in O0 x O1. No physical 2**N state
vector or per-erasure word XOR table is constructed. Collision counts and
group construction take O(|O0|+|O1|) integer operations. Exact diagonal KL
comparison may additionally visit the within-group operator entries; it exits
as soon as a mismatch is found and does not materialize a dense matrix.

Leakage conventions match the preceding leakage notebook:
    D_Z = 0.5 * ||rho00-rho11||_1
    D_X = ||rho01||_1
    I_RE = 1 + S(rho_E) - S(rho_RE), in bits.
Only leakage uses floating point; both KL decisions use integer arithmetic.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from numbers import Integral

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _inputs(words0, words1, N, E):
    N = _integer(N, "N", 1)
    sites = tuple(_integer(site, "erasure site") for site in E)
    if len(set(sites)) != len(sites) or any(site >= N for site in sites):
        raise ValueError("E must contain distinct sites in [0,N)")
    E = tuple(sorted(sites))

    def convert(words):
        result = []
        for word in words:
            if isinstance(word, str):
                if len(word) != N or set(word) - {"0", "1"}:
                    raise ValueError("String words must have exactly N binary digits")
                word = int(word, 2)
            else:
                word = _integer(word, "orbit word")
                if word.bit_length() > N:
                    raise ValueError("Orbit word does not fit in N bits")
            result.append(word)
        if not result or len(set(result)) != len(result):
            raise ValueError("Each orbit must contain nonempty, distinct words")
        return tuple(result)

    a, b = convert(words0), convert(words1)
    if set(a) & set(b):
        raise ValueError("Logical supports must be disjoint")
    return a, b, N, E


def _groups(words, erased_mask, remaining_mask):
    result = defaultdict(list)
    for word in words:
        result[word & remaining_mask].append(word & erased_mask)
    return dict(result)


def _prepare(words0, words1, N, E):
    a, b, N, E = _inputs(words0, words1, N, E)
    erased_mask = sum(1 << (N - 1 - site) for site in E)
    remaining_mask = ((1 << N) - 1) ^ erased_mask
    return (a, b, N, E, _groups(a, erased_mask, remaining_mask),
            _groups(b, erased_mask, remaining_mask))


def operator_counts(groups_a, groups_b):
    """Non-normalized Tr_remaining |a><b| counts, keyed by erased bit masks.

    Each positive entry is an integer; no cancellation is possible. This public
    helper is useful for independent exact checks on small examples. Evaluation
    avoids expanding the cross operator unless dense leakage is requested.
    """
    counts = Counter()
    for remaining in groups_a.keys() & groups_b.keys():
        for a in groups_a[remaining]:
            for b in groups_b[remaining]:
                counts[a, b] += 1
    return counts


def reduced_operator_counts(words0, words1, N, E):
    """Return integer rho00/rho11/rho01 counts and their normalization sizes.

    Keys retain original bit positions. Divide diagonal counts by M0/M1 and
    cross counts by sqrt(M0*M1) to obtain reduced operators.
    """
    a, b, N, E, g0, g1 = _prepare(words0, words1, N, E)
    return {"rho00": operator_counts(g0, g0), "rho11": operator_counts(g1, g1),
            "rho01": operator_counts(g0, g1), "M0": len(a), "M1": len(b),
            "N": N, "E": E}


def _diagonal_equal(g0, g1, M0, M1):
    # Here diagonal KL means equality of the COMPLETE logical diagonal blocks,
    # including their off-diagonal computational-basis matrix entries.
    rows0, rows1 = defaultdict(list), defaultdict(list)
    for groups, rows in ((g0, rows0), (g1, rows1)):
        for remaining, erased_words in groups.items():
            for erased in erased_words:
                rows[erased].append(remaining)
    if rows0.keys() != rows1.keys():
        return False
    if any(len(rows0[e]) * M1 != len(rows1[e]) * M0 for e in rows0):
        return False
    if all(len(group) == 1 for group in g0.values()) and all(
            len(group) == 1 for group in g1.values()):
        return True
    for erased in rows0:
        c0, c1 = Counter(), Counter()
        for remaining in rows0[erased]:
            c0.update(g0[remaining])
        for remaining in rows1[erased]:
            c1.update(g1[remaining])
        if any(c0.get(e, 0) * M1 != c1.get(e, 0) * M0
               for e in c0.keys() | c1.keys()):
            return False
    return True


def _cross_operator_nonzero(g0, g1):
    # Inspect operator support independently of the scalar collision count.
    # A shared remaining key contributes a strictly positive outer product.
    # Its first (erased_0, erased_1) entry is therefore a nonzero witness.
    for remaining, entries0 in g0.items():
        entries1 = g1.get(remaining)
        if entries1:
            return True, (entries0[0], entries1[0])
    return False, None


def _basis(g0, g1):
    return tuple(sorted({e for entries in g0.values() for e in entries}
                        | {e for entries in g1.values() for e in entries}))


def _compress(mask, N, E):
    value = 0
    for site in E:
        value = (value << 1) | ((mask >> (N - 1 - site)) & 1)
    return value


def _dense(g0, g1, M0, M1, N, E, maximum, full_basis=False):
    basis = _basis(g0, g1)
    dimension = (1 << len(E)) if full_basis else len(basis)
    if 2 * dimension > maximum:
        raise MemoryError(f"rho_RE dimension {2 * dimension} exceeds cap {maximum}")
    if full_basis:
        row = lambda e: _compress(e, N, E)
        exported_basis = tuple(range(dimension))
    else:
        row = {e: i for i, e in enumerate(basis)}.__getitem__
        exported_basis = tuple(_compress(e, N, E) for e in basis)
    rho00 = np.zeros((dimension, dimension), dtype=float)
    rho11 = np.zeros_like(rho00)
    rho01 = np.zeros_like(rho00)
    for groups, target in ((g0, rho00), (g1, rho11)):
        for entries in groups.values():
            rows = [row(e) for e in entries]
            # Distinct original words imply distinct erased words in a group.
            target[np.ix_(rows, rows)] += 1
    for remaining in g0.keys() & g1.keys():
        rows0 = [row(e) for e in g0[remaining]]
        rows1 = [row(e) for e in g1[remaining]]
        rho01[np.ix_(rows0, rows1)] += 1
    rho00 /= M0
    rho11 /= M1
    rho01 /= sqrt(M0 * M1)
    rho_E = (rho00 + rho11) / 2
    rho_RE = np.block([[rho00, rho01], [rho01.T, rho11]]) / 2
    return {"rho00": rho00, "rho11": rho11, "rho01": rho01,
            "rho_E": rho_E, "rho_RE": rho_RE, "basis": exported_basis,
            "basis_is_full": bool(full_basis)}


def build_reduced_operators(words0, words1, N, E, *, max_matrix_dimension=256,
                            full_basis=False):
    """Materialize reduced operators only when the declared cap permits it.

    The cap applies to the largest matrix, rho_RE, of dimension 2d. The default
    uses the exact observed erased support, omitting only identically zero rows
    and columns. No spectral truncation is used. basis contains compact erased
    bit labels in increasing site order; full_basis includes all 2**|E| labels.
    """
    maximum = _integer(max_matrix_dimension, "max_matrix_dimension", 1)
    a, b, N, E, g0, g1 = _prepare(words0, words1, N, E)
    return _dense(g0, g1, len(a), len(b), N, E, maximum, full_basis)


def _entropy(matrix):
    values = np.linalg.eigvalsh(matrix)
    if not np.all(np.isfinite(values)) or values.min(initial=0) < -1e-11:
        raise FloatingPointError("Density matrix has invalid eigenvalues")
    trace = float(values.sum())
    if abs(trace - 1) > 1e-10:
        raise FloatingPointError(f"Density matrix trace is {trace}, not one")
    # Clip only roundoff negatives after validating them; retain tiny positives.
    values = np.maximum(values, 0)
    values /= values.sum()
    positive = values[values > 0]
    return float(-np.dot(positive, np.log2(positive)))


def _leakage(operators, correctable):
    S_E, S_RE = _entropy(operators["rho_E"]), _entropy(operators["rho_RE"])
    raw = 1 + S_E - S_RE
    dz = float(np.abs(np.linalg.eigvalsh(operators["rho00"] - operators["rho11"])).sum() / 2)
    dx = float(np.linalg.svd(operators["rho01"], compute_uv=False).sum())
    if not -1e-10 <= raw <= 2 + 1e-10 or not -1e-10 <= dz <= 1 + 1e-10 or not -1e-10 <= dx <= 1 + 1e-10:
        raise FloatingPointError("Leakage measure lies outside its physical range")
    information, dz, dx = min(2.0, max(0.0, raw)), min(1.0, dz), min(1.0, dx)
    if correctable and max(information, dz, dx) > 1e-9:
        raise FloatingPointError("Exact KL and numerical leakage disagree")
    return {"I_RE": information, "I_RE_raw": raw, "D_Z": dz, "D_X": dx,
            "S_R": 1.0, "S_E": S_E, "S_RE": S_RE,
            "leakage_status": "computed", "leakage_method": "exact_compressed_dense"}


def evaluate_erasure(words0, words1, N, E, *, compute_leakage=False,
                     max_leakage_dim=256, include_profile=False):
    """Return exact collision counts/KL flags and optional leakage diagnostics.

    max_leakage_dim bounds the dimension of rho_RE (2 times observed erased
    support size), NOT the erased qubit count. When the cap is exceeded, exact
    KL/collision results are still returned; leakage values are None with
    leakage_status='dimension_limit'. This is a skipped computation, never a
    zero-leakage result. Not requesting leakage gives status='not_requested'.
    """
    maximum = _integer(max_leakage_dim, "max_leakage_dim", 1)
    a, b, N, E, g0, g1 = _prepare(words0, words1, N, E)
    c0 = sum(len(entries) * (len(entries) - 1) // 2 for entries in g0.values())
    c1 = sum(len(entries) * (len(entries) - 1) // 2 for entries in g1.values())
    cross = sum(len(g0[r]) * len(g1[r]) for r in g0.keys() & g1.keys())
    diagonal = _diagonal_equal(g0, g1, len(a), len(b))
    nonzero, witness = _cross_operator_nonzero(g0, g1)
    if nonzero != (cross > 0):
        raise AssertionError("Positive cross operator and collision count disagree")
    offdiagonal = not nonzero
    correctable = diagonal and offdiagonal
    support_dimension = len(_basis(g0, g1))
    result = {
        "erasure_size": len(E), "erasure_positions": E,
        "C_cross": cross, "C_self_0": c0, "C_self_1": c1,
        "Delta_C_self": abs(c0 - c1),
        "cross_collision": cross > 0,
        "self_collision_0": c0 > 0, "self_collision_1": c1 > 0,
        "recoverable_0": c0 == 0, "recoverable_1": c1 == 0,
        "recoverable_both": c0 == c1 == 0,
        "KL_diagonal": diagonal, "KL_offdiagonal": offdiagonal,
        "KL_correctable": correctable,
        "KL_failure_reason": ("none" if correctable else "diagonal+offdiagonal"
                              if not diagonal and not offdiagonal else "diagonal"
                              if not diagonal else "offdiagonal"),
        "rho01_nonzero": nonzero, "cross_collision_operator_consistent": True,
        "erasure_support_dimension": support_dimension,
        "leakage_matrix_dimension": 2 * support_dimension,
        "I_RE": None, "I_RE_raw": None, "D_Z": None, "D_X": None,
        "S_R": None, "S_E": None, "S_RE": None,
        "leakage_status": "not_requested", "leakage_method": None,
    }
    if include_profile:
        kept = tuple(site for site in range(N) if site not in set(E))
        profile = []
        for r in sorted(g0.keys() | g1.keys()):
            n0, n1 = len(g0.get(r, ())), len(g1.get(r, ()))
            profile.append({"remaining_pattern": "".join(str((r >> (N - 1 - site)) & 1) for site in kept),
                            "remaining_mask_hex": hex(r), "n_0": n0, "n_1": n1,
                            "C_cross_r": n0 * n1,
                            "C_self_0_r": n0 * (n0 - 1) // 2,
                            "C_self_1_r": n1 * (n1 - 1) // 2})
        result["collision_profile"] = profile
        result["rho01_nonzero_witness"] = (
            None if witness is None else tuple(_compress(e, N, E) for e in witness))
    if compute_leakage:
        if 2 * support_dimension > maximum:
            result["leakage_status"] = "dimension_limit"
        else:
            result.update(_leakage(_dense(g0, g1, len(a), len(b), N, E, maximum), correctable))
    return result
