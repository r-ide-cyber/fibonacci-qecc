"""Exact orbit-based information leakage for one logical qubit.

No Fibonacci convention is defined here: pass the *unchanged* orbits from the
existing experiment. Site 0 is the leftmost bit. Production calculations never
construct a physical 2**N statevector. NumPy is the only dependency.

The large-erasure path uses sparse grouping factors and their connected
components. It diagonalizes bounded Gram matrices without truncating spectra.
If an exact component exceeds the configured bound, it raises MemoryError.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable
import warnings

import numpy as np


class NumericalValidationError(FloatingPointError):
    """A density matrix or a decoupling identity failed numerical validation."""


def _positive_int(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} is outside its allowed range")
    return value


class OrbitPair:
    """Two normalized equal-positive-amplitude, orthogonal orbit states.

    O0 and O1 contain unique binary strings, or integers with explicit N.
    Distinct orbits must have disjoint supports. Duplicates are rejected rather
    than silently changing the definition of orbit cardinality.
    """

    def __init__(self, O0: Iterable, O1: Iterable, N=None):
        raw0, raw1 = tuple(O0), tuple(O1)
        if not raw0 or not raw1:
            raise ValueError("Both logical orbits must be nonempty")
        if N is None:
            if not isinstance(raw0[0], str):
                raise ValueError("N is required for integer orbit states")
            N = len(raw0[0])
        self.N = _positive_int(N, "N")

        def convert(states):
            result = []
            for state in states:
                if isinstance(state, str):
                    if len(state) != self.N or set(state) - {"0", "1"}:
                        raise ValueError("Every orbit string must contain N binary digits")
                    result.append(int(state, 2))
                elif isinstance(state, Integral) and not isinstance(state, bool):
                    state = int(state)
                    if state < 0 or state.bit_length() > self.N:
                        raise ValueError("Integer orbit state does not fit in N bits")
                    result.append(state)
                else:
                    raise ValueError("Orbit states must be binary strings or integers")
            if len(set(result)) != len(result):
                raise ValueError("An orbit contains duplicate basis states")
            return tuple(result)

        self.O0, self.O1 = convert(raw0), convert(raw1)
        self.M0, self.M1 = len(self.O0), len(self.O1)
        if set(self.O0).intersection(self.O1):
            raise ValueError("Logical orbit supports overlap: the basis is not orthogonal")
        self.full_mask = (1 << self.N) - 1


def _validate_erasure(pair, erasure):
    E = tuple(_positive_int(x, "erasure site", allow_zero=True) for x in erasure)
    if len(set(E)) != len(E):
        raise ValueError("Erasure positions contain duplicates")
    if any(x >= pair.N for x in E):
        raise ValueError("Erasure positions must lie in [0, N)")
    return tuple(sorted(E))


def _group_orbits(pair, E):
    shifts = tuple(pair.N - 1 - site for site in E)
    erasure_mask = sum(1 << shift for shift in shifts)
    remaining_mask = pair.full_mask ^ erasure_mask

    def group(states):
        groups = defaultdict(list)
        for state in states:
            erased = 0
            for shift in shifts:
                erased = (erased << 1) | ((state >> shift) & 1)
            # Keeping zeroed-out gaps preserves equality and avoids projecting
            # all N-m surviving bits for every orbit element.
            groups[state & remaining_mask].append(erased)
        return dict(groups)

    return group(pair.O0), group(pair.O1)


def _exact_diagonal_equal(groups0, groups1, M0, M1):
    """Compare every integer Gram entry, without floating-point KL decisions."""
    row0, row1 = defaultdict(list), defaultdict(list)
    for remaining, entries in groups0.items():
        for erased in entries:
            row0[erased].append(remaining)
    for remaining, entries in groups1.items():
        for erased in entries:
            row1[erased].append(remaining)
    if row0.keys() != row1.keys():
        return False
    # Cheap normalized diagonal comparison precedes off-diagonal checks.
    if any(len(row0[e]) * M1 != len(row1[e]) * M0 for e in row0):
        return False
    if (all(len(entries) == 1 for entries in groups0.values())
            and all(len(entries) == 1 for entries in groups1.values())):
        return True  # Both reduced operators are already diagonal.
    for erased in row0:
        counts0, counts1 = Counter(), Counter()
        for remaining in row0[erased]:
            counts0.update(groups0[remaining])
        for remaining in row1[erased]:
            counts1.update(groups1[remaining])
        if counts0.keys() != counts1.keys():
            return False
        if any(counts0[e] * M1 != counts1[e] * M0 for e in counts0):
            return False
    return True


def _psd_eigenvalues(eigenvalues, tol, name):
    values = np.asarray(eigenvalues, dtype=float)
    if not np.all(np.isfinite(values)):
        raise NumericalValidationError(f"{name}: non-finite eigenvalues")
    minimum = float(values.min(initial=0.0))
    if minimum < -tol:
        message = f"{name}: negative eigenvalue {minimum:.3g} exceeds tolerance {tol:.3g}"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise NumericalValidationError(message)
    return np.maximum(values, 0.0), minimum


def _entropy_from_values(values, tol, name):
    values, minimum = _psd_eigenvalues(values, tol, name)
    trace = float(values.sum())
    if not np.isfinite(trace) or abs(trace - 1.0) > max(1e-10, 100 * tol):
        raise NumericalValidationError(f"{name}: trace {trace:.17g} is not one")
    probabilities = values / trace  # Only roundoff-sized renormalization.
    positive = probabilities[probabilities > 0]
    entropy = float(-np.dot(positive, np.log2(positive)))
    return entropy, {"trace": trace, "min_eigenvalue": minimum}


def von_neumann_entropy(rho, eigenvalue_tol=1e-12, *, return_diagnostics=False):
    """Entropy in bits using eigvalsh; reject significant negative eigenvalues.

    Tiny negative eigenvalues in [-tol, 0) are clipped only after inspection.
    A trace different from one beyond roundoff is an error, not normalized away.
    """
    if not np.isfinite(eigenvalue_tol) or eigenvalue_tol <= 0:
        raise ValueError("eigenvalue_tol must be positive and finite")
    matrix = np.asarray(rho)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError("rho must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise NumericalValidationError("rho contains non-finite entries")
    hermitian_error = float(np.max(np.abs(matrix - matrix.conj().T)))
    if hermitian_error > max(1e-12, 10 * eigenvalue_tol):
        raise NumericalValidationError(f"rho is not Hermitian: error {hermitian_error}")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2)
    entropy, diagnostic = _entropy_from_values(eigenvalues, eigenvalue_tol, "rho")
    diagnostic["hermitian_error"] = hermitian_error
    return (entropy, diagnostic) if return_diagnostics else entropy


@dataclass
class _Factor:
    """Coordinate sparse matrix with unique (row, column) entries."""
    nrows: int
    ncols: int
    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray

    def components(self):
        """Yield compact edge arrays; disjoint row and column supports split exactly."""
        parent = list(range(self.nrows + self.ncols))
        sizes = [1] * len(parent)

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for row, col in zip(self.rows, self.cols):
            a, b = root(int(row)), root(self.nrows + int(col))
            if a != b:
                if sizes[a] < sizes[b]:
                    a, b = b, a
                parent[b] = a
                sizes[a] += sizes[b]
        components = defaultdict(list)
        for index, row in enumerate(self.rows):
            components[root(int(row))].append(index)
        for indices in components.values():
            index = np.asarray(indices, dtype=np.intp)
            original_rows, rows = np.unique(self.rows[index], return_inverse=True)
            original_cols, cols = np.unique(self.cols[index], return_inverse=True)
            yield (_Factor(len(original_rows), len(original_cols), rows, cols,
                           self.data[index]), original_rows, original_cols)


def _guard_dimension(dimension, maximum, operation):
    if dimension > maximum:
        # Even an intentionally requested full 2**m export can have m in the
        # thousands; formatting the diagnostic must not overflow first.
        memory_text = (f"~{8 * dimension * dimension / 2**20:.1f} MiB"
                       if dimension.bit_length() < 500 else "astronomical memory")
        dimension_text = str(dimension) if dimension.bit_length() < 1000 else f"about 2**{dimension.bit_length()-1}"
        raise MemoryError(
            f"{operation} needs an exact dense component of dimension {dimension_text} "
            f"({memory_text} per real matrix), above "
            f"max_matrix_dimension={maximum}. Reduce the erasure/orbit workload "
            "or deliberately increase the memory limit. No approximate spectrum was used."
        )


def _gram(factor, side, maximum, column_signs=None):
    """Bounded Gram matrix formed directly from sparse factor groups."""
    if side == "rows":
        dimension, output, group = factor.nrows, factor.rows, factor.cols
    elif side == "cols":
        dimension, output, group = factor.ncols, factor.cols, factor.rows
        if column_signs is not None:
            raise ValueError("Signed Gram construction uses the row side")
    else:
        raise ValueError("Unknown Gram side")
    _guard_dimension(dimension, maximum, "Gram diagonalization")
    result = np.zeros((dimension, dimension), dtype=float)
    order = np.argsort(group, kind="stable")
    sorted_groups = group[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1, len(order)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        indices = order[start:stop]
        positions, values = output[indices], factor.data[indices]
        sign = 1.0 if column_signs is None else column_signs[int(group[indices[0]])]
        result[np.ix_(positions, positions)] += sign * np.outer(values, values)
    return result


def _factor_entropy(components, maximum, tol, name):
    spectrum = []
    for factor, _, _ in components:
        if min(factor.nrows, factor.ncols) == 1:
            spectrum.append(np.array([np.dot(factor.data, factor.data)]))
        else:
            side = "rows" if factor.nrows <= factor.ncols else "cols"
            spectrum.append(np.linalg.eigvalsh(_gram(factor, side, maximum)))
    values = np.concatenate(spectrum) if spectrum else np.array([0.0])
    return _entropy_from_values(values, tol, name)


def _signed_factor_metrics(components, split_column, maximum, tol):
    """Trace/Frobenius norms of G0 G0† - G1 G1† from unsigned factor /sqrt(2)."""
    trace_norm, norm_squared = 0.0, 0.0
    for factor, _, original_cols in components:
        signs = np.where(original_cols < split_column, 1.0, -1.0)
        if factor.nrows <= factor.ncols:
            signed = 2 * _gram(factor, "rows", maximum, signs)
        else:
            gram = 2 * _gram(factor, "cols", maximum)
            values, vectors = np.linalg.eigh(gram)
            values, _ = _psd_eigenvalues(values, tol, "signed-factor Gram")
            root = (vectors * np.sqrt(values)) @ vectors.T
            signed = (root * signs) @ root
        eigenvalues = np.linalg.eigvalsh((signed + signed.T) / 2)
        trace_norm += float(np.abs(eigenvalues).sum())
        norm_squared += float(np.dot(eigenvalues, eigenvalues))
    return 0.5 * trace_norm, 0.5 * np.sqrt(norm_squared)


def _factor_sqrt_gram(factor, maximum, tol):
    gram = _gram(factor, "cols", maximum)
    values, vectors = np.linalg.eigh(gram)
    values, _ = _psd_eigenvalues(values, tol, "cross-factor Gram")
    return (vectors * np.sqrt(values)) @ vectors.T


def _cross_metrics(components, split_row, maximum, tol):
    """Exact singular spectrum of G0 G1†, split over bipartite components."""
    trace_norm, norm_squared = 0.0, 0.0
    for factor, original_rows, _ in components:
        is_left = original_rows[factor.rows] < split_row
        left_original = np.unique(factor.rows[is_left])
        right_original = np.unique(factor.rows[~is_left])
        if not len(left_original) or not len(right_original):
            continue
        left = _Factor(len(left_original), factor.ncols,
                       np.searchsorted(left_original, factor.rows[is_left]),
                       factor.cols[is_left], factor.data[is_left])
        right = _Factor(len(right_original), factor.ncols,
                        np.searchsorted(right_original, factor.rows[~is_left]),
                        factor.cols[~is_left], factor.data[~is_left])
        if max(left.nrows, right.nrows) <= maximum:
            # Materialize only a bounded occupied-support rectangle.
            product = np.zeros((left.nrows, right.nrows))
            lcols, rcols = defaultdict(list), defaultdict(list)
            for row, col, value in zip(left.rows, left.cols, left.data):
                lcols[int(col)].append((int(row), value))
            for row, col, value in zip(right.rows, right.cols, right.data):
                rcols[int(col)].append((int(row), value))
            for col in lcols.keys() & rcols.keys():
                lr, lv = zip(*lcols[col])
                rr, rv = zip(*rcols[col])
                product[np.ix_(lr, rr)] += np.outer(lv, rv)
            singular = np.linalg.svd(product, compute_uv=False)
        elif factor.ncols <= maximum:
            root_left = _factor_sqrt_gram(left, maximum, tol)
            root_right = _factor_sqrt_gram(right, maximum, tol)
            singular = np.linalg.svd(root_left @ root_right, compute_uv=False)
        else:
            _guard_dimension(min(max(left.nrows, right.nrows), factor.ncols),
                             maximum, "Cross-operator trace norm")
            raise AssertionError("Unreachable memory guard")
        trace_norm += float(singular.sum())
        norm_squared += float(np.dot(singular, singular))
    return trace_norm, np.sqrt(norm_squared)


@dataclass
class ReducedOperators:
    """Exact sparse grouping representation, with guarded dense export.

    basis lists occupied E patterns as binary integers. groups contain the
    unnormalized integer grouping vectors. No 2**m matrix is stored here.
    """
    N: int
    E: tuple
    M0: int
    M1: int
    groups_0: dict
    groups_1: dict
    basis: tuple
    max_matrix_dimension: int = 1024

    @property
    def m(self):
        return len(self.E)

    def to_dense(self, *, full_basis=False, max_matrix_dimension=None):
        """Return rho00/rho11/rho01/rho_RE/rho_E/rho_R in a declared E basis."""
        maximum = self.max_matrix_dimension if max_matrix_dimension is None else max_matrix_dimension
        maximum = _positive_int(maximum, "max_matrix_dimension")
        dimension = (1 << self.m) if full_basis else len(self.basis)
        _guard_dimension(2 * dimension, maximum, "Dense rho_RE export")
        row = (lambda e: e) if full_basis else {e: i for i, e in enumerate(self.basis)}.__getitem__
        rho00 = np.zeros((dimension, dimension))
        rho11 = np.zeros_like(rho00)
        rho01 = np.zeros_like(rho00)
        for groups, target in ((self.groups_0, rho00), (self.groups_1, rho11)):
            singletons = []
            for entries in groups.values():
                if len(entries) == 1:
                    singletons.append(row(entries[0]))
                    continue
                rows = [row(e) for e in entries]
                target[np.ix_(rows, rows)] += 1.0
            if singletons:
                np.add.at(target, (singletons, singletons), 1.0)
        for remaining in self.groups_0.keys() & self.groups_1.keys():
            rows0 = [row(e) for e in self.groups_0[remaining]]
            rows1 = [row(e) for e in self.groups_1[remaining]]
            rho01[np.ix_(rows0, rows1)] += 1.0
        rho00 /= self.M0
        rho11 /= self.M1
        rho01 /= np.sqrt(self.M0 * self.M1)
        rho_E = (rho00 + rho11) / 2
        rho_RE = np.block([[rho00, rho01], [rho01.T, rho11]]) / 2
        rho_R = np.array([[np.trace(rho00), np.trace(rho01)],
                          [np.trace(rho01.T), np.trace(rho11)]]) / 2
        return {"rho00": rho00, "rho11": rho11, "rho01": rho01,
                "rho_E": rho_E, "rho_RE": rho_RE, "rho_R": rho_R,
                "basis": tuple(range(dimension)) if full_basis else self.basis}

    def factors(self):
        remaining = sorted(self.groups_0.keys() | self.groups_1.keys())
        columns = {r: i for i, r in enumerate(remaining)}
        rows = {e: i for i, e in enumerate(self.basis)}
        row0, col0, row1, col1 = [], [], [], []
        for groups, rr, cc in ((self.groups_0, row0, col0), (self.groups_1, row1, col1)):
            for r, entries in groups.items():
                for e in entries:
                    rr.append(rows[e])
                    cc.append(columns[r])
        row0, col0, row1, col1 = [np.asarray(x, dtype=np.intp) for x in (row0, col0, row1, col1)]
        d, q = len(self.basis), len(remaining)
        data0 = np.full(len(row0), 1 / np.sqrt(2 * self.M0))
        data1 = np.full(len(row1), 1 / np.sqrt(2 * self.M1))
        factor_E = _Factor(d, 2 * q, np.r_[row0, row1], np.r_[col0, col1 + q],
                           np.r_[data0, data1])
        factor_RE = _Factor(2 * d, q, np.r_[row0, row1 + d], np.r_[col0, col1],
                            np.r_[data0, data1])
        shared = self.groups_0.keys() & self.groups_1.keys()
        shared_columns = np.array([columns[r] for r in shared], dtype=np.intp)
        mask = np.isin(factor_RE.cols, shared_columns)
        cross = _Factor(2 * d, q, factor_RE.rows[mask], factor_RE.cols[mask],
                        factor_RE.data[mask] * np.sqrt(2))
        return factor_E, factor_RE, cross, q


def analyze_erasure(pair, erasure, *, max_dense_erasure_qubits=12,
                    max_matrix_dimension=1024, eigenvalue_tol=1e-12,
                    leakage_tol=1e-10, return_operators=False):
    """Return one-pattern entropies, trace norms, exact KL tests, and diagnostics.

    D_Z = .5 ||rho00-rho11||_1; D_X = ||rho01||_1 (raw trace norm, no
    additional factor). Both lie in [0,1]. KL diagonal equality uses integer
    counts; KL off-diagonal equality is exactly absence of cross collisions
    because all orbit amplitudes are positive. Large-m results are exact up to
    floating-point arithmetic, with no sparse-eigensolver truncation.
    """
    if not isinstance(pair, OrbitPair):
        raise TypeError("pair must be an OrbitPair")
    max_dense_erasure_qubits = _positive_int(max_dense_erasure_qubits,
                                           "max_dense_erasure_qubits", allow_zero=True)
    maximum = _positive_int(max_matrix_dimension, "max_matrix_dimension")
    for name, value in (("eigenvalue_tol", eigenvalue_tol), ("leakage_tol", leakage_tol)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    E = _validate_erasure(pair, erasure)
    groups0, groups1 = _group_orbits(pair, E)
    basis = tuple(sorted({e for entries in groups0.values() for e in entries}
                         | {e for entries in groups1.values() for e in entries}))
    operators = ReducedOperators(pair.N, E, pair.M0, pair.M1, groups0, groups1, basis, maximum)
    self0 = sum(len(entries) * (len(entries) - 1) // 2 for entries in groups0.values())
    self1 = sum(len(entries) * (len(entries) - 1) // 2 for entries in groups1.values())
    cross = sum(len(groups0[r]) * len(groups1[r]) for r in groups0.keys() & groups1.keys())
    diagonal = _exact_diagonal_equal(groups0, groups1, pair.M0, pair.M1)
    offdiagonal = cross == 0
    dense = len(E) <= max_dense_erasure_qubits and (1 << (len(E) + 1)) <= maximum

    if dense:
        rho = operators.to_dense(full_basis=True)
        S_E, diagnostic_E = von_neumann_entropy(rho["rho_E"], eigenvalue_tol, return_diagnostics=True)
        S_RE, diagnostic_RE = von_neumann_entropy(rho["rho_RE"], eigenvalue_tol, return_diagnostics=True)
        S_R = von_neumann_entropy(rho["rho_R"], eigenvalue_tol)
        difference_values = np.linalg.eigvalsh(rho["rho00"] - rho["rho11"])
        singular = np.linalg.svd(rho["rho01"], compute_uv=False)
        D_Z = float(np.abs(difference_values).sum() / 2)
        D_X = float(singular.sum())
        diagonal_residual = float(np.linalg.norm(difference_values) / 2)
        offdiagonal_residual = float(np.linalg.norm(singular))
        mode = "dense_erasure"
    else:
        factor_E, factor_RE, factor_cross, q = operators.factors()
        components_E = list(factor_E.components())
        components_RE = list(factor_RE.components())
        S_E, diagnostic_E = _factor_entropy(components_E, maximum, eigenvalue_tol, "rho_E")
        S_RE, diagnostic_RE = _factor_entropy(components_RE, maximum, eigenvalue_tol, "rho_RE")
        S_R = 1.0  # Verified orthogonal supports and unit norms in OrbitPair.
        if diagonal:
            D_Z, diagonal_residual = 0.0, 0.0
        else:
            D_Z, diagonal_residual = _signed_factor_metrics(components_E, q, maximum, eigenvalue_tol)
        if offdiagonal:
            D_X, offdiagonal_residual = 0.0, 0.0
        else:
            D_X, offdiagonal_residual = _cross_metrics(factor_cross.components(), len(basis),
                                                      maximum, eigenvalue_tol)
        mode = "grouping_components_gram"

    raw_information = float(S_R + S_E - S_RE)
    roundoff_bound = max(1e-10, 100 * eigenvalue_tol)
    if raw_information < -roundoff_bound or raw_information > 2 + roundoff_bound:
        raise NumericalValidationError(f"I(R:E)={raw_information} lies outside [0,2]")
    information = min(2.0, max(0.0, raw_information))
    for name, value in (("D_Z", D_Z), ("D_X", D_X)):
        if value < -roundoff_bound or value > 1 + roundoff_bound:
            raise NumericalValidationError(f"{name}={value} lies outside [0,1]")
    D_Z, D_X = float(np.clip(D_Z, 0, 1)), float(np.clip(D_X, 0, 1))
    correctable = diagonal and offdiagonal
    if correctable and (information >= leakage_tol or D_Z >= leakage_tol or D_X >= leakage_tol):
        raise NumericalValidationError(
            "Exact KL passes but numerical leakage exceeds LEAKAGE_TOL: "
            f"I={information}, DZ={D_Z}, DX={D_X}. Inspect normalization, "
            "operator construction and eigenspectra before using this result."
        )
    z, x = D_Z >= leakage_tol, D_X >= leakage_tol
    failure_type = "Type ZX" if z and x else "Type Z" if z else "Type X" if x else "Type 0"
    if failure_type == "Type 0" and information >= leakage_tol:
        failure_type = "Type unresolved (tolerances)"
    result = {
        "S_R": float(S_R), "S_E": float(S_E), "S_RE": float(S_RE),
        "I_RE": information, "I_RE_raw": raw_information,
        "I_RE_roundoff_clipped": information != raw_information,
        "D_Z": D_Z, "D_X": D_X,
        "KL_diagonal": diagonal, "KL_offdiagonal": offdiagonal,
        "KL_correctable": correctable,
        "KL_diagonal_residual": float(diagonal_residual),
        "KL_offdiagonal_residual": float(offdiagonal_residual),
        "self_collision_0": self0 > 0, "self_collision_1": self1 > 0,
        "self_collision_count_0": self0, "self_collision_count_1": self1,
        "cross_collision": cross > 0, "cross_collision_count": cross,
        "recoverable_both": self0 == 0 and self1 == 0,
        "decoupled": information < leakage_tol,
        "leakage_failure_type": failure_type, "calculation_mode": mode,
        "erasure_support_dimension": len(basis),
        "rho_E_trace": diagnostic_E["trace"], "rho_RE_trace": diagnostic_RE["trace"],
        "rho_E_min_eigenvalue": diagnostic_E["min_eigenvalue"],
        "rho_RE_min_eigenvalue": diagnostic_RE["min_eigenvalue"],
        "KL_MI_consistent": not correctable or information < leakage_tol,
    }
    if return_operators:
        result["operators"] = operators
    return result


def mutual_information_reference_erasure(pair, erasure, **kwargs):
    """Public entry point returning S_R, S_E, S_RE, I_RE and leakage diagnostics."""
    return analyze_erasure(pair, erasure, **kwargs)
