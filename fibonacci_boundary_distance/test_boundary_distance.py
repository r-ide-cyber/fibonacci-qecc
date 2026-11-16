"""Scientific regression tests. Run: python -m unittest -v test_boundary_distance.

The tiny dense partial-trace oracle below is deliberately independent of the
integer-count implementation used by the research models.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import boundary_models as models
import boundary_distance as workflow
import boundary_reporting as reporting


SEEDS = ("000111", "010101")
KL_DEFINITION = {
    "family": "equal_positive_uniform_support",
    "description": "Explicit tiny support model used only to test partial traces.",
}


def all_erasures(N):
    for m in range(N + 1):
        yield from combinations(range(N), m)


def dense_oracle(C0, C1, E):
    """Form normalized state tensors and trace the complement with matrix products."""
    N = len(C0[0])
    E = tuple(sorted(E))
    kept = tuple(i for i in range(N) if i not in E)
    matrices = []
    for candidates in (C0, C1):
        state = np.zeros(2 ** N)
        for word in candidates:
            state[int(word, 2)] = 1 / np.sqrt(len(candidates))
        matrices.append(state.reshape((2,) * N).transpose(E + kept)
                        .reshape(2 ** len(E), 2 ** len(kept)))
    A, B = matrices
    diagonal = np.allclose(A @ A.T, B @ B.T, atol=1e-12, rtol=0)
    offdiagonal = np.allclose(A @ B.T, 0, atol=1e-12, rtol=0)
    recoverable = all(
        len({"".join(word[i] for i in kept) for word in candidates}) == len(candidates)
        for candidates in (C0, C1)
    )
    return dict(recoverable=recoverable, KL_diagonal=bool(diagonal),
                KL_offdiagonal=bool(offdiagonal),
                KL_correctable=bool(diagonal and offdiagonal))


class MathematicalModelTests(unittest.TestCase):
    def test_prior_inflation_rule_and_derived_length(self):
        self.assertEqual(models.inflate("01", 1), "110")
        self.assertEqual(models.inflate("01", 2), "10101")
        for n, expected_length in ((0, 6), (1, 9), (2, 15)):
            model = models.build_ring_model(*SEEDS, n)
            self.assertEqual(model.N, expected_length)
            self.assertEqual(model.N, len(model.F0))
            self.assertEqual(model.N, len(model.F1))

    def test_invalid_logical_seeds_rejected(self):
        invalid = [
            ("000111", "001110"),  # Same cyclic class.
            ("000111", "001011"),  # One adjacent 01/10 exchange.
            ("000111", "001111"),  # Different k0/k1.
            ("000111", "00111x"),
        ]
        for seed0, seed1 in invalid:
            for build in (models.build_ring_model, models.build_open_model):
                with self.subTest(pair=(seed0, seed1), constructor=build.__name__):
                    with self.assertRaises(ValueError):
                        build(seed0, seed1, 0)

    def test_ring_integer_KL_matches_independent_dense_partial_trace(self):
        model = models.build_ring_model(*SEEDS, 0)
        # The supports have different sizes (6 and 2), testing normalization too.
        self.assertNotEqual(len(model.C0), len(model.C1))
        for E in all_erasures(model.N):
            expected = dense_oracle(model.C0, model.C1, E)
            actual = model.evaluate(E)
            with self.subTest(E=E):
                for field, value in expected.items():
                    self.assertEqual(actual[field], value, field)

    def test_ring_translation_including_wraparound(self):
        fields = ("recoverable", "KL_diagonal", "KL_offdiagonal", "KL_correctable")
        for n in (0, 1):
            model = models.build_ring_model(*SEEDS, n)
            for E in ((), (0,), (0, 1), (0, 2, 4), tuple(range(model.N))):
                reference = model.evaluate(E)
                for shift in range(model.N):
                    moved = tuple(sorted((i + shift) % model.N for i in E))
                    actual = model.evaluate(moved)
                    with self.subTest(n=n, E=E, shift=shift):
                        for field in fields:
                            self.assertEqual(actual[field], reference[field])

    def test_singleton_open_is_trivial_and_has_no_quantum_observables(self):
        model = models.build_open_model(*SEEDS, 0)
        self.assertEqual(model.C0, (model.F0,))
        self.assertEqual(model.C1, (model.F1,))
        self.assertFalse(model.metadata["periodic_adjacency"])
        self.assertTrue(model.metadata["open_support_is_singleton"])
        for E in all_erasures(model.N):
            result = model.evaluate(E)
            self.assertTrue(result["recoverable"])
            for field in ("KL_diagonal", "KL_offdiagonal", "KL_correctable"):
                self.assertIsNone(result[field])
        # Logical distinguishability is intentionally not classical recoverability.
        erased_all = model.evaluate(range(model.N))
        self.assertGreater(erased_all["cross_collision_count"], 0)
        self.assertTrue(erased_all["recoverable"])

    def test_open_within_class_collision_and_fixed_site_coordinates(self):
        factory = lambda F0, F1, n: (("000000", "100000"), ("111111",))
        model = models.build_open_model(*SEEDS, 0, candidate_factory=factory,
                                        model_description="Two fixed words differ at the left site.")
        self.assertFalse(model.evaluate((0,))["recoverable"])
        self.assertTrue(model.evaluate((5,))["recoverable"])
        self.assertEqual(model.evaluate((0,))["self_collision_count_0"], 1)

    def test_open_KL_requires_an_explicit_supported_definition(self):
        for definition in (None, "cyclic orbit", {},
                           {"family": "equal_positive_uniform_support", "description": ""}):
            with self.subTest(definition=definition):
                with self.assertRaises(ValueError):
                    models.build_open_model(*SEEDS, 0, enable_kl=True,
                                            quantum_state_definition=definition)
        with self.assertRaises(ValueError):
            models.build_open_model(*SEEDS, 0, quantum_state_definition=KL_DEFINITION)

    def test_explicit_open_KL_matches_dense_oracle(self):
        supports = (("000000", "100001"), ("010000", "110001", "001100"))
        factory = lambda F0, F1, n: supports
        model = models.build_open_model(
            *SEEDS, 0, candidate_factory=factory,
            model_description="Test model with fixed explicit orthogonal supports.",
            enable_kl=True, quantum_state_definition=KL_DEFINITION)
        for E in all_erasures(model.N):
            expected = dense_oracle(*supports, E)
            with self.subTest(E=E):
                actual = model.evaluate(E)
                for field, value in expected.items():
                    self.assertEqual(actual[field], value, field)

    def test_overlapping_classes_are_classical_only(self):
        factory = lambda F0, F1, n: ((F0,), (F0,))
        kwargs = dict(candidate_factory=factory, model_description="Overlapping classical classes.")
        classical = models.build_open_model(*SEEDS, 0, **kwargs)
        self.assertTrue(classical.evaluate(())["recoverable"])
        with self.assertRaises(ValueError):
            models.build_open_model(*SEEDS, 0, **kwargs, enable_kl=True,
                                    quantum_state_definition=KL_DEFINITION)

    def test_candidate_and_erasure_validation(self):
        invalid_supports = [(("0",), ("111111",)),
                            (("000000", "000000"), ("111111",)),
                            ((), ("111111",))]
        for supports in invalid_supports:
            with self.subTest(supports=supports):
                with self.assertRaises(ValueError):
                    models.build_open_model(
                        *SEEDS, 0, candidate_factory=lambda *args: supports,
                        model_description="Invalid candidate test.")
        for build in (models.build_ring_model, models.build_open_model):
            model = build(*SEEDS, 0)
            for E in ((0, 0), (-1,), (model.N,), (0.5,), (True,)):
                with self.subTest(constructor=build.__name__, E=E):
                    with self.assertRaises(ValueError):
                        model.evaluate(E)


class WorkflowTests(unittest.TestCase):
    def config(self, **updates):
        values = dict(seed_0=SEEDS[0], seed_1=SEEDS[1], inflation_list=(0,),
                      window_size_list=(4,), boundary_distance_list=(0, 1),
                      erasure_size_list=(1, 2), max_exact_patterns=100,
                      num_random_samples=13, random_seed=73,
                      save_results=False, save_figures=False,
                      detail_memory_limit=10000)
        values.update(updates)
        return workflow.Config(**values)

    def test_exact_and_sampled_patterns_are_reproducible_and_order_independent(self):
        cfg = self.config(window_size_list=(4, 6), erasure_size_list=(3,),
                          max_exact_patterns=5)
        first = workflow.build_pattern_bank(cfg)
        second = workflow.build_pattern_bank(self.config(
            window_size_list=(6, 4), erasure_size_list=(3,), max_exact_patterns=5))
        for key in first:
            self.assertEqual(first[key], second[key])
        self.assertEqual(first[(4, 3, "arbitrary")].patterns,
                         tuple(combinations(range(4), 3)))
        self.assertEqual(first[(4, 3, "arbitrary")].mode, "exact")
        sampled = first[(6, 3, "arbitrary")]
        self.assertEqual(sampled.mode, "sampled")
        self.assertEqual(len(sampled.patterns), 13)
        self.assertEqual(sampled.total_possible, 20)
        self.assertEqual(sampled.sampling_scheme, "iid_uniform_with_replacement")
        self.assertEqual(first[(6, 3, "contiguous")].patterns,
                         ((0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)))
        self.assertEqual(first[(6, 3, "contiguous")].mode, "exact")
        for E in sampled.patterns:
            self.assertEqual(len(set(E)), 3)
            self.assertTrue(all(0 <= i < 6 for i in E))

    def test_window_coordinates_and_invalid_combinations_are_skipped(self):
        cfg = self.config(inflation_list=(1,), window_size_list=(4, 10),
                          boundary_distance_list=(0, 1, 6), erasure_size_list=(1, 5))
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            prepared = workflow.prepare(cfg)
        self.assertGreaterEqual(len(notices), 3)
        self.assertEqual(prepared.size_table.N.tolist(), [9])
        self.assertTrue(prepared.windows.window_size.eq(4).all())
        self.assertFalse(prepared.windows.boundary_distance.eq(6).any())
        self.assertNotIn((4, 5, "arbitrary"), prepared.patterns)
        for row in prepared.windows.itertuples():
            self.assertEqual(row.window_end, row.window_start + row.window_size - 1)
            self.assertEqual(row.boundary_distance_left, row.window_start)
            self.assertEqual(row.boundary_distance_right, row.N - row.window_end - 1)
            self.assertTrue(0 <= row.window_start <= row.window_end < row.N)
            if row.boundary_side == "left":
                self.assertEqual(row.window_start, row.boundary_distance)
            elif row.boundary_side == "right":
                self.assertEqual(row.window_start, row.N - row.window_size - row.boundary_distance)
            else:
                self.assertEqual(row.window_start, (row.N - row.window_size) // 2)

    def test_invalid_configuration_is_rejected(self):
        for changes in ({"boundary_distance_list": (-1,)},
                        {"window_size_list": (0,)}, {"erasure_size_list": (True,)},
                        {"inflation_list": (0, 0)}, {"epsilon_boundary": 0},
                        {"boundary_fraction_list": (1.0,)}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    workflow.prepare(self.config(**changes))

    def test_shared_relative_samples_and_absolute_coordinates(self):
        cfg = self.config(inflation_list=(1,), window_size_list=(6,),
                          erasure_size_list=(3,), max_exact_patterns=5)
        result = workflow.run_analysis(workflow.prepare(cfg))
        arbitrary = result.detail[result.detail.erasure_type.eq("arbitrary")]
        for _, group in arbitrary.groupby("pattern_id"):
            self.assertEqual(group.erasure_positions_relative.nunique(), 1)
            self.assertEqual(group.pattern_digest.nunique(), 1)
            for row in group.itertuples():
                relative = json.loads(row.erasure_positions_relative)
                absolute = json.loads(row.erasure_positions_absolute)
                self.assertEqual(absolute, [row.window_start + i for i in relative])
                self.assertTrue(all(row.window_start <= i <= row.window_end for i in absolute))
        self.assertTrue(arbitrary.calculation_mode.eq("sampled").all())

    def test_ring_sanity_check_and_benchmark_cover_both_geometries(self):
        prepared = workflow.prepare(self.config())
        sanity = workflow.ring_sanity_check(prepared)
        self.assertTrue(sanity.passed.all())
        self.assertTrue(sanity.wraps.any())
        measured = workflow.benchmark(prepared, count=100)
        self.assertEqual(set(measured.geometry), {"open", "ring"})
        self.assertTrue(measured.pattern_count.eq(100).all())
        self.assertTrue(measured.runtime.gt(0).all())
        np.testing.assert_allclose(measured.estimated_runtime,
                                   measured.time_per_pattern * measured.planned_evaluations)

    def test_no_invented_open_KL_and_ring_rates_independent_of_position(self):
        result = workflow.run_analysis(workflow.prepare(self.config()))
        opened = result.summary[result.summary.geometry.eq("open")]
        self.assertTrue(opened.R_rec.eq(1).all())
        self.assertTrue(opened.R_KL.isna().all())
        self.assertTrue(result.threshold.B_star_KL.isna().all())
        self.assertTrue(result.threshold.interpretation.eq(
            "trivial_singleton_support_not_boundary_evidence").all())
        self.assertEqual(set(result.threshold.geometry), {"open"})
        ring = result.summary[result.summary.geometry.eq("ring")]
        for _, group in ring.groupby(["inflation_n", "window_size", "erasure_size", "erasure_type"]):
            self.assertEqual(group.R_rec.nunique(), 1)
            self.assertEqual(group.R_KL.nunique(), 1)
        for field in ("R_rec_contiguous", "R_rec_arbitrary", "R_KL_contiguous", "R_KL_arbitrary"):
            self.assertIn(field, result.wide_summary)

    def test_streamed_export_is_complete_even_when_preview_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "results"
            cfg = self.config(output_dir=str(destination), save_results=True, detail_memory_limit=3)
            result = workflow.run_analysis(workflow.prepare(cfg))
            saved = pd.read_csv(destination / "boundary_distance_detail.csv")
            self.assertEqual(len(result.detail), 3)
            self.assertEqual(len(saved), result.total_detail_rows)
            self.assertGreater(len(saved), len(result.detail))
            self.assertEqual(int(result.summary.number_of_patterns.sum()), len(saved))
            expected_fields = {"geometry", "pair_id", "seed_0", "seed_1", "inflation_n", "N",
                               "window_size", "boundary_side", "boundary_distance_left",
                               "boundary_distance_right", "window_start", "window_end", "erasure_type",
                               "erasure_size", "erasure_positions_relative", "erasure_positions_absolute",
                               "recoverable", "KL_diagonal", "KL_offdiagonal", "KL_correctable",
                               "calculation_mode"}
            self.assertTrue(expected_fields <= set(saved.columns))
            split_count = 0
            for geometry in ("open", "ring"):
                split = pd.read_csv(destination / f"geometry_{geometry}" / "boundary_distance_detail.csv",
                                    dtype={"seed_0": str, "seed_1": str})
                self.assertEqual(set(split.seed_0), {cfg.seed_0})
                self.assertEqual(set(split.seed_1), {cfg.seed_1})
                self.assertEqual(set(split.geometry), {geometry})
                split_count += len(split)
            self.assertEqual(split_count, len(saved))
            status = json.loads((destination / "run_status.json").read_text())
            self.assertEqual(status["status"], "complete")

    def test_save_results_false_creates_no_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "must_not_exist"
            cfg = self.config(output_dir=str(destination), detail_memory_limit=0)
            result = workflow.run_analysis(workflow.prepare(cfg))
            self.assertFalse(destination.exists())
            self.assertEqual(len(result.detail), 0)
            self.assertGreater(result.total_detail_rows, 0)
            self.assertIsNone(result.output_dir)

    def test_no_work_export_leaves_previous_run_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(output_dir=str(Path(tmp)/"results"), save_results=True)
            workflow.run_analysis(workflow.prepare(cfg))
            before = {str(p): p.read_bytes() for p in Path(cfg.output_dir).rglob("*") if p.is_file()}
            cfg.window_size_list = (1000,)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prepared = workflow.prepare(cfg)
            with self.assertRaisesRegex(ValueError, "No valid"):
                workflow.run_analysis(prepared)
            after = {str(p): p.read_bytes() for p in Path(cfg.output_dir).rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_seed_counts_and_inflated_counts_are_distinct(self):
        prepared = workflow.prepare(self.config(inflation_list=(1,)))
        row = prepared.size_table.iloc[0]
        self.assertEqual((row.k0, row.k1), (3, 3))
        self.assertEqual((row.inflated_count_0, row.inflated_count_1), (3, 6))

    def test_empty_scan_is_supported_after_all_windows_are_skipped(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = workflow.prepare(self.config(window_size_list=(7,)))
            result = workflow.run_analysis(prepared)
        self.assertTrue(result.summary.empty)
        self.assertTrue(result.threshold.empty)
        self.assertEqual(result.total_detail_rows, 0)


class ComparisonTests(unittest.TestCase):
    def test_threshold_uses_strict_crossing_and_does_not_claim_monotone_convergence(self):
        rows = []
        base = dict(geometry="open", pair_id="synthetic", inflation_n=0, N=20,
                    window_size=4, erasure_size=1, erasure_type="arbitrary",
                    R_KL=np.nan, number_of_patterns=4, calculation_mode="exact")
        for B, rate in enumerate((0.5, 0.75, 0.25, 0.75)):
            rows.append(dict(base, boundary_side="left", distance_mode="sites",
                             boundary_distance=B, boundary_distance_left=B,
                             boundary_distance_right=16 - B, window_start=B,
                             window_end=B + 3, R_rec=rate))
        rows.append(dict(base, boundary_side="center", distance_mode="center",
                         boundary_distance=8, boundary_distance_left=8,
                         boundary_distance_right=8, window_start=8, window_end=11, R_rec=0.75))
        delta, threshold, _ = reporting.derive_comparisons(pd.DataFrame(rows), epsilon=0.25)
        self.assertEqual(threshold.iloc[0].B_star_recoverability, 1)
        self.assertEqual(threshold.iloc[0].B_star_sustained_grid_recoverability, 3)
        self.assertTrue(pd.isna(threshold.iloc[0].B_star_KL))
        self.assertEqual(delta.iloc[0].Delta_boundary_recoverability, 0.25)

    def test_missing_center_does_not_create_a_threshold(self):
        cfg = workflow.Config(inflation_list=(0,), window_size_list=(4,),
                              erasure_size_list=(1,), boundary_distance_list=(0,),
                              include_center=False, save_results=False, save_figures=False)
        result = workflow.run_analysis(workflow.prepare(cfg))
        self.assertTrue(result.threshold.B_star_recoverability.isna().all())
        self.assertTrue(result.threshold.status_recoverability.eq("center_missing").all())


if __name__ == "__main__":
    unittest.main()
