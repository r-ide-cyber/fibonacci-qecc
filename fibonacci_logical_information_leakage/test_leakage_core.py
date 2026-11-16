"""Independent small-system statevector tests; production core uses no statevector."""
from itertools import combinations
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fibonacci_leakage_core import (OrbitPair, analyze_erasure,
                                    von_neumann_entropy, NumericalValidationError)


def explicit(pair, E):
    """Independent physical vectors + tensor partial trace, N <= 8 only."""
    if pair.N > 8:
        raise ValueError("Statevector validation is restricted to N <= 8")
    E = tuple(sorted(E))
    remaining = tuple(i for i in range(pair.N) if i not in E)
    matrices = []
    for states in (pair.O0, pair.O1):
        vector = np.zeros(2**pair.N)
        vector[list(states)] = 1 / np.sqrt(len(states))
        matrix = vector.reshape((2,) * pair.N).transpose(E + remaining).reshape(2**len(E), -1)
        matrices.append(matrix)
    A, B = matrices
    rho00, rho11, rho01 = A @ A.T, B @ B.T, A @ B.T
    phi = np.concatenate([A, B], axis=0) / np.sqrt(2)
    rho_RE = phi @ phi.T
    rho_E = (rho00 + rho11) / 2
    entropy = lambda r: von_neumann_entropy(r)
    return dict(rho00=rho00, rho11=rho11, rho01=rho01, rho_RE=rho_RE,
                rho_E=rho_E, rho_R=np.eye(2)/2,
                I_RE=1+entropy(rho_E)-entropy(rho_RE),
                D_Z=np.linalg.norm(rho00-rho11, ord="nuc")/2,
                D_X=np.linalg.norm(rho01, ord="nuc"))


class CoreValidation(unittest.TestCase):
    def check_pair(self, pair, E):
        direct = explicit(pair, E)
        result = analyze_erasure(pair, E, return_operators=True)
        matrices = result["operators"].to_dense(full_basis=True)
        for key in ("rho00", "rho11", "rho01", "rho_RE", "rho_E", "rho_R"):
            np.testing.assert_allclose(matrices[key], direct[key], atol=2e-13, rtol=2e-13)
        for key in ("I_RE", "D_Z", "D_X"):
            self.assertAlmostEqual(result[key], direct[key], places=11)
        forced = analyze_erasure(pair, E, max_dense_erasure_qubits=0)
        for key in ("I_RE", "D_Z", "D_X", "S_E", "S_RE"):
            self.assertAlmostEqual(result[key], forced[key], places=10)
        for key in ("KL_diagonal", "KL_offdiagonal", "KL_correctable"):
            self.assertEqual(result[key], forced[key])
        self.assertEqual(result["KL_diagonal"], np.allclose(direct["rho00"], direct["rho11"]))
        self.assertEqual(result["KL_offdiagonal"], np.allclose(direct["rho01"], 0))
        return result

    def test_all_erasure_subsets_random_small_orbits(self):
        rng = np.random.default_rng(24)
        for N in (2, 3, 4, 5):
            for _ in range(5):
                permutation = rng.permutation(2**N)
                size0 = int(rng.integers(1, 2**(N-1)))
                size1 = int(rng.integers(1, 2**N-size0+1))
                pair = OrbitPair(permutation[:size0], permutation[size0:size0+size1], N=N)
                for m in range(N+1):
                    for E in combinations(range(N), m):
                        with self.subTest(N=N, M0=size0, M1=size1, E=E):
                            self.check_pair(pair, E)

    def test_interpretable_failure_types(self):
        cases = [
            ((["00"], ["01"]), (0,), "Type 0", 0., 0., 0.),
            ((["00"], ["11"]), (0,), "Type Z", 1., 1., 0.),
            ((["00", "11"], ["01", "10"]), (0,), "Type X", 1., 0., 1.),
            ((["00"], ["11"]), (0, 1), "Type ZX", 2., 1., 1.),
        ]
        for orbits, E, failure, info, dz, dx in cases:
            result = self.check_pair(OrbitPair(*orbits), E)
            self.assertEqual(result["leakage_failure_type"], failure)
            np.testing.assert_allclose([result["I_RE"], result["D_Z"], result["D_X"]],
                                       [info, dz, dx], atol=1e-12)

    def test_empty_and_full_erasure(self):
        pair = OrbitPair(["000", "111"], ["001", "010", "011"])
        empty = self.check_pair(pair, ())
        full = self.check_pair(pair, (0, 1, 2))
        self.assertEqual(empty["I_RE"], 0.)
        self.assertTrue(empty["KL_correctable"])
        self.assertAlmostEqual(full["I_RE"], 2.)
        self.assertAlmostEqual(full["D_Z"], 1.)
        self.assertAlmostEqual(full["D_X"], 1.)
        self.assertEqual(full["cross_collision_count"], pair.M0*pair.M1)

    def test_large_m_low_rank_no_hilbert_space_allocation(self):
        pair = OrbitPair(range(64), range(64, 128), N=1000)
        result = analyze_erasure(pair, range(1000), max_matrix_dimension=4,
                                 return_operators=True)
        self.assertEqual(result["calculation_mode"], "grouping_components_gram")
        np.testing.assert_allclose([result["I_RE"], result["D_Z"], result["D_X"]],
                                   [2, 1, 1], atol=1e-12)
        with self.assertRaises(MemoryError):
            result["operators"].to_dense()

    def test_memory_guard_requires_exact_component(self):
        pair = OrbitPair(["000", "101", "110"], ["001", "010", "111"])
        with self.assertRaises(MemoryError):
            analyze_erasure(pair, (0, 1), max_dense_erasure_qubits=0,
                            max_matrix_dimension=1)

    def test_low_rank_gram_trace_norm_branch(self):
        rng = np.random.default_rng(103)
        for _ in range(12):
            states = rng.permutation(256)
            pair = OrbitPair(states[:50], states[50:130], N=8)
            dense = analyze_erasure(pair, tuple(range(7)))
            # Two surviving patterns -> <=4 grouping columns despite large E support.
            low_rank = analyze_erasure(pair, tuple(range(7)), max_matrix_dimension=4)
            for key in ("I_RE", "D_Z", "D_X", "KL_diagonal_residual", "KL_offdiagonal_residual"):
                self.assertAlmostEqual(dense[key], low_rank[key], places=10)

    def test_normalization_hermiticity_and_negative_eigenvalue_checks(self):
        self.assertAlmostEqual(von_neumann_entropy(np.eye(2)/2), 1.)
        self.assertAlmostEqual(von_neumann_entropy(np.diag([-1e-14, 1+1e-14])), 0.)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(NumericalValidationError):
                von_neumann_entropy(np.diag([-0.1, 1.1]))
        with self.assertRaises(NumericalValidationError):
            von_neumann_entropy(np.eye(2))
        with self.assertRaises(NumericalValidationError):
            von_neumann_entropy(np.array([[.5, 1], [0, .5]]))

    def test_orbit_and_erasure_validation(self):
        invalid = [(["0"], ["0"]), (["00", "00"], ["11"]),
                   ([], ["00"]), (["02"], ["11"])]
        for a, b in invalid:
            with self.assertRaises(ValueError):
                OrbitPair(a, b)
        pair = OrbitPair(["00"], ["11"])
        for E in ([0, 0], [-1], [2], [0.5]):
            with self.assertRaises(ValueError):
                analyze_erasure(pair, E)


if __name__ == "__main__":
    unittest.main(verbosity=2)
