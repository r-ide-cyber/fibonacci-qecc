"""Small, executable regression checks for fixed-window finite-size scaling.

Run with ``python -m unittest -v test_finite_size_scaling`` from this directory.
The test cases use N <= 15 for erasure scans. Larger inflation lengths are
checked without evaluating erasures or constructing dense quantum states.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import contextlib
import hashlib
import io
import math
import tempfile
import unittest
import warnings

import pandas as pd
from pandas.testing import assert_frame_equal

import finite_size_scaling as scaling
import recoverability_kl as baseline
import theory


HERE = Path(__file__).resolve().parent
BASELINE_HASHES = {
    "recoverability_kl.py": "a0d592e6b01a7eee67d70679bfdedfb918dd58a61ee03b76accd41fe5ad9c0b1",
    "legacy_functions.py": "2eebd80bde75a6ef61526006fcfe00ef82dbc800362427d00453d6afa769b17b",
}


def small_config(**changes):
    config = dict(
        PAIR_ID="small_reference",
        SEED_0="000111",
        SEED_1="010101",
        EXTRA_SEED_PAIRS=[],
        INFLATION_LIST=[0, 1, 2],
        WINDOW_SIZE_LIST=[4],
        ERASURE_SIZE_LIST=[1, 2, 4],
        NUM_RANDOM_SAMPLES=17,
        MAX_EXACT_PATTERNS=4,
        RANDOM_SEED=42,
        SAVE_RESULTS=False,
        SAVE_FIGURES=False,
        DETAIL_MODE="all",
        DETAIL_MEMORY_LIMIT=10_000,
        DETAIL_REPRESENTATIVE_LIMIT=3,
        BENCHMARK_PATTERNS=2,
        MAX_ESTIMATED_SECONDS=600.0,
        RUN_FULL_ANALYSIS=True,
        OUTPUT_DIR=HERE / "unused_test_output",
    )
    config.update(changes)
    return config


def ready(config):
    """Exercise the actual preflight, including both validation gates."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prepared = scaling.prepare(config)
        translation = scaling.translation_test(prepared)
        validation = scaling.validate_small(prepared)
        measured = scaling.benchmark(prepared, config)
    return prepared, measured, translation, validation


def run_small(config):
    prepared, measured, _, _ = ready(config)
    with contextlib.redirect_stdout(io.StringIO()):
        result = scaling.run_analysis(prepared, config, measured, validation_passed=True)
    return prepared, result


def comparable_summary(frame):
    """Compare scientific counts/rates without timings or detail-retention metadata."""
    keys = ["pair_id", "inflation_n", "N", "window_size", "erasure_size"]
    values = [c for c in frame.columns if c.startswith("R_") or c in {
        "gap_KL", "arbitrary_failure_rate", "calculation_mode",
        "num_patterns_arbitrary", "num_patterns_contiguous", "number_of_patterns",
    }]
    return frame[keys + values].sort_values(keys).reset_index(drop=True)


class FiniteSizeScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.prepared, cls.measured, cls.translation, cls.validation = ready(cls.config)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = scaling.run_analysis(
                cls.prepared, cls.config, cls.measured, validation_passed=True
            )

    def test_original_implementations_and_reference_are_unchanged(self):
        for name, digest in BASELINE_HASHES.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest)
        baseline.verify_reference(HERE / "reference")

    def test_inflation_lengths_and_same_seed_series(self):
        self.assertEqual(
            [len(baseline.inflate("000111", n)) for n in [6, 7, 8, 9, 10]],
            [102, 165, 267, 432, 699],
        )
        self.assertEqual(
            [len(baseline.inflate("010101", n)) for n in [6, 7, 8, 9, 10]],
            [102, 165, 267, 432, 699],
        )
        for (pair_id, n), model in self.prepared["models"].items():
            self.assertEqual(pair_id, self.config["PAIR_ID"])
            self.assertEqual(model.F0, baseline.inflate(self.config["SEED_0"], n))
            self.assertEqual(model.F1, baseline.inflate(self.config["SEED_1"], n))
            self.assertEqual(model.N, len(model.F0))
            self.assertEqual(model.N, len(model.F1))

    def test_rejects_invalid_pairs_including_periodic_boundary_swap(self):
        # This target needs the ring edge (last, first); an open-chain check misses it.
        self.assertFalse(baseline.legacy.obtainable_by_one_adjacent_swap(
            "000111", "100110", periodic=False))
        self.assertTrue(baseline.legacy.obtainable_by_one_adjacent_swap(
            "000111", "100110", periodic=True))
        for seed1 in ["100110", "001011", "111000", "000011", "01010x"]:
            with self.subTest(seed1=seed1), self.assertRaises(ValueError):
                scaling.prepare(small_config(SEED_1=seed1))

    def test_theory_index_and_additional_seed_condition(self):
        self.assertEqual([theory.L_theory(n) for n in [6, 7, 8, 9, 10]],
                         [14, 22, 35, 56, 90])
        default = theory.paper_seed_conditions("000111", "010101")
        self.assertTrue(default["adjacent_swap_valid"])
        self.assertTrue(default["paper_seed_condition"])
        counterexample = theory.paper_seed_conditions("0001001", "0000011")
        self.assertTrue(baseline.satisfies_adjacent_swap_condition("0001001", "0000011"))
        self.assertTrue(counterexample["adjacent_swap_valid"])
        self.assertTrue(counterexample["self_swap_0"])
        self.assertFalse(counterexample["paper_seed_condition"])
        table = theory.theory_table(self.prepared)
        zero = table.query("inflation_n == 0")
        self.assertFalse(zero.theory_guarantee_applicable.any())
        self.assertTrue(zero.predicted_window_correctable.isna().all())
        # L_theory(0) is a formula value, and must not be presented as a guarantee.
        initial = self.prepared["models"][(self.config["PAIR_ID"], 0)]
        self.assertFalse(initial.evaluate(range(theory.L_theory(0)))["KL_correctable"])

    def test_invalid_window_and_erasure_combinations_are_skipped(self):
        config = small_config(INFLATION_LIST=[0, 1], WINDOW_SIZE_LIST=[4, 8],
                              ERASURE_SIZE_LIST=[1, 5])
        with warnings.catch_warnings(record=True) as captured, contextlib.redirect_stdout(io.StringIO()) as out:
            warnings.simplefilter("always")
            prepared = scaling.prepare(config)
            scaling.translation_test(prepared)
            scaling.validate_small(prepared)
            measured = scaling.benchmark(prepared, config)
            result = scaling.run_analysis(prepared, config, measured, validation_passed=True)
        summary = result["summary"]
        self.assertTrue((summary.window_size <= summary.N).all())
        self.assertTrue((summary.erasure_size <= summary.window_size).all())
        actual = set(zip(summary.N, summary.window_size, summary.erasure_size))
        self.assertEqual(actual, {(6, 4, 1), (9, 4, 1), (9, 8, 1), (9, 8, 5)})
        self.assertTrue(captured or out.getvalue(), "Skipped settings must produce a warning.")

    def test_translation_and_small_validation_run_before_analysis(self):
        self.assertFalse(self.translation.empty)
        self.assertFalse(self.validation.empty)
        for model in self.prepared["models"].values():
            # Include a wrapping translation independent of center-window placement.
            for relative in [(0,), (0, 2), (0, 1, 2, 3)]:
                expected = model.evaluate(relative)["KL_correctable"]
                for shift in [1, model.N // 2, model.N - 1]:
                    translated = tuple(sorted((i + shift) % model.N for i in relative))
                    self.assertEqual(model.evaluate(translated)["KL_correctable"], expected)

    def test_exact_and_sampled_patterns_have_correct_counts_and_geometry(self):
        detail = self.result["detail"]
        summary = self.result["summary"]
        self.assertEqual(set(summary.calculation_mode), {"exact", "sampled"})
        for row in summary.itertuples(index=False):
            m, W = row.erasure_size, row.window_size
            expected_arbitrary = math.comb(W, m) if math.comb(W, m) <= 4 else 17
            self.assertEqual(row.num_patterns_arbitrary, expected_arbitrary)
            self.assertEqual(row.num_patterns_contiguous, W - m + 1)
            # Translation invariance makes the contiguous rate binary at fixed m.
            self.assertIn(row.R_KL_contiguous, (0.0, 1.0))
        for row in detail.itertuples(index=False):
            relative = tuple(row.erasure_positions_relative)
            absolute = tuple(row.erasure_positions_absolute)
            self.assertEqual(len(relative), row.erasure_size)
            self.assertEqual(len(set(relative)), row.erasure_size)
            self.assertTrue(all(0 <= i < row.window_size for i in relative))
            self.assertEqual(absolute, tuple(row.window_start + i for i in relative))
            self.assertEqual(row.window_end - row.window_start + 1, row.window_size)
            self.assertEqual(row.window_start, (row.N - row.window_size) // 2)
            if row.erasure_type == "contiguous":
                self.assertEqual(relative, tuple(range(relative[0], relative[0] + row.erasure_size)))

    def test_identical_sample_sequence_across_system_sizes_with_duplicate_trials(self):
        detail = self.result["detail"]
        sampled = detail[(detail.erasure_type == "arbitrary") & (detail.erasure_size == 2)]
        sequences = [list(group.erasure_positions_relative)
                     for _, group in sampled.groupby("inflation_n", sort=True)]
        self.assertEqual(len(sequences), 3)
        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(sequences[0], sequences[2])
        self.assertEqual(len(sequences[0]), 17)
        self.assertLess(len(set(sequences[0])), len(sequences[0]))
        plan = self.prepared["plans"][(4, 2)]
        self.assertEqual(list(plan), list(plan))
        self.assertEqual(list(plan), sequences[0])

    def test_summary_matches_independent_baseline_enumeration(self):
        for row in self.result["summary"].itertuples(index=False):
            model = self.prepared["models"][(row.pair_id, row.inflation_n)]
            start = (model.N - row.window_size) // 2
            relative_groups = {
                "contiguous": (tuple(range(s, s + row.erasure_size))
                               for s in range(row.window_size - row.erasure_size + 1)),
                "arbitrary": iter(self.prepared["plans"][(row.window_size, row.erasure_size)]),
            }
            for kind, relatives in relative_groups.items():
                counts = Counter()
                for relative in relatives:
                    evaluated = baseline.evaluate_orbits(model.O0, model.O1,
                                                         tuple(start + i for i in relative))
                    counts["trials"] += 1
                    counts["KL"] += int(evaluated["KL_correctable"])
                    counts["rec"] += int(evaluated["recoverable_both"])
                self.assertEqual(getattr(row, f"R_KL_{kind}"), counts["KL"] / counts["trials"])
                self.assertEqual(getattr(row, f"R_rec_{kind}"), counts["rec"] / counts["trials"])
            self.assertAlmostEqual(row.gap_KL, row.R_KL_contiguous - row.R_KL_arbitrary)
            self.assertAlmostEqual(row.arbitrary_failure_rate, 1 - row.R_KL_arbitrary)

    def test_full_window_correctability_implies_all_subsets(self):
        model = self.prepared["models"][(self.config["PAIR_ID"], 2)]
        window = tuple(range((model.N - 4) // 2, (model.N - 4) // 2 + 4))
        self.assertTrue(model.evaluate(window)["KL_correctable"])
        for m in range(5):
            for subset in combinations(window, m):
                self.assertTrue(model.evaluate(subset)["KL_correctable"])
        last = self.result["summary"].query("inflation_n == 2")
        self.assertTrue(last.R_KL_arbitrary.eq(1.0).all())
        self.assertTrue(last.R_KL_contiguous.eq(1.0).all())

    def test_sampled_all_success_ci_does_not_claim_certainty(self):
        row = self.result["summary"].query("inflation_n == 2 and erasure_size == 2").iloc[0]
        self.assertEqual(row.R_KL_arbitrary, 1.0)
        low, high = baseline.wilson(17, 17)
        self.assertLess(low, 1.0)
        self.assertAlmostEqual(high, 1.0)
        self.assertAlmostEqual(row["R_KL_arbitrary_ci95_low"], low)
        self.assertAlmostEqual(row["R_KL_arbitrary_ci95_high"], high)
        failed = self.result["summary"].query("inflation_n == 0 and erasure_size == 2").iloc[0]
        self.assertEqual(failed.R_KL_arbitrary, 0.0)
        low_zero, high_zero = baseline.wilson(0, 17)
        self.assertAlmostEqual(failed["R_KL_arbitrary_ci95_low"], low_zero)
        self.assertAlmostEqual(failed["R_KL_arbitrary_ci95_high"], high_zero)
        self.assertGreater(high_zero, 0.0)
        for _, exact in self.result["summary"].query("calculation_mode == 'exact'").iterrows():
            self.assertEqual(exact["R_KL_arbitrary_ci95_low"], exact.R_KL_arbitrary)
            self.assertEqual(exact["R_KL_arbitrary_ci95_high"], exact.R_KL_arbitrary)

    def test_explicit_run_validation_and_benchmark_gates(self):
        disabled = small_config()
        disabled.pop("RUN_FULL_ANALYSIS")  # The default must leave a full run disabled.
        p_disabled, b_disabled, _, _ = ready(disabled)
        with self.assertRaises(RuntimeError):
            scaling.run_analysis(p_disabled, disabled, b_disabled, validation_passed=True)
        with self.assertRaises(RuntimeError):
            scaling.run_analysis(self.prepared, self.config, self.measured, validation_passed=False)
        fresh = scaling.prepare(self.config)
        with self.assertRaises(RuntimeError):
            scaling.run_analysis(fresh, self.config, self.measured, validation_passed=True)
        with self.assertRaises(RuntimeError):
            scaling.run_analysis(self.prepared, self.config, self.measured.iloc[:0], validation_passed=True)
        with self.assertRaises(RuntimeError):
            scaling.run_analysis(self.prepared, small_config(RANDOM_SEED=43), self.measured,
                                 validation_passed=True)
        incomplete = self.measured.iloc[1:]
        with self.assertRaises(RuntimeError):
            scaling.run_analysis(self.prepared, self.config, incomplete, validation_passed=True)

    def test_scaling_deltas_and_threshold_evidence_are_not_conflated(self):
        scaled = self.result["scaling"]
        for _, group in scaled.groupby(["pair_id", "window_size", "erasure_size"]):
            group = group.sort_values("N")
            self.assertTrue(pd.isna(group.iloc[0].Delta_R_KL_arbitrary))
            for previous, current in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
                expected = current.R_KL_arbitrary - previous.R_KL_arbitrary
                self.assertAlmostEqual(current.Delta_R_KL_arbitrary, abs(expected))
                self.assertAlmostEqual(current.paired_delta_R_KL_arbitrary, expected)
                self.assertEqual(current.paired_gained_count + current.paired_lost_count
                                 + current.paired_unchanged_count, current.num_patterns_arbitrary)
        thresholds = self.result["thresholds"]
        observed = thresholds.query("erasure_size == 2 and threshold_kind == 'sampled_observed'").iloc[0]
        exact = thresholds.query("erasure_size == 2 and threshold_kind == 'exact_enumeration'").iloc[0]
        full = thresholds.query("erasure_size == 2 and threshold_kind == 'full_window_proof'").iloc[0]
        self.assertTrue(observed.found)
        self.assertFalse(observed.proven_all_patterns)
        self.assertFalse(exact.found)
        self.assertTrue(pd.isna(exact.N_star))
        self.assertTrue(full.found)
        self.assertTrue(full.proven_all_patterns)
        self.assertEqual((full.n_star, full.N_star), (2, 15))

    def test_multiple_pairs_remain_separate_and_reuse_relative_patterns(self):
        extra = dict(pair_id="second_pair", seed_0="0001001", seed_1="0000011")
        prepared, result = run_small(small_config(EXTRA_SEED_PAIRS=[extra], INFLATION_LIST=[0, 1]))
        summary = result["summary"]
        self.assertEqual(set(summary.pair_id), {"small_reference", "second_pair"})
        self.assertEqual(len(summary), 12)
        self.assertTrue(summary.loc[summary.pair_id.eq("second_pair"), "seed_0"].eq(extra["seed_0"]).all())
        arbitrary = result["detail"].query("erasure_type == 'arbitrary' and erasure_size == 2")
        sequences = [list(group.erasure_positions_relative)
                     for _, group in arbitrary.groupby(["pair_id", "inflation_n"])]
        self.assertEqual(len(sequences), 4)
        for sequence in sequences[1:]:
            self.assertEqual(sequence, sequences[0])

    def test_single_sample_does_not_report_zero_paired_sampling_error(self):
        _, result = run_small(small_config(INFLATION_LIST=[0, 1], NUM_RANDOM_SAMPLES=1,
                                          ERASURE_SIZE_LIST=[2]))
        later = result["summary"].query("inflation_n == 1").iloc[0]
        self.assertEqual(later.num_patterns_arbitrary, 1)
        self.assertTrue(pd.isna(later.paired_delta_standard_error))

    def test_estimated_time_cap_stops_before_computation(self):
        config = small_config(MAX_ESTIMATED_SECONDS=1e-12)
        prepared, measured, _, _ = ready(config)
        self.assertGreater(measured.estimated_full_seconds.sum(), config["MAX_ESTIMATED_SECONDS"])
        with self.assertWarns(RuntimeWarning), self.assertRaises(RuntimeError):
            scaling.run_analysis(prepared, config, measured, validation_passed=True)

    def test_detail_retention_is_bounded_and_does_not_change_summary(self):
        expected = comparable_summary(self.result["summary"])
        for mode in ["all", "failures", "representative", "none"]:
            with self.subTest(mode=mode):
                _, limited = run_small(small_config(DETAIL_MODE=mode, DETAIL_MEMORY_LIMIT=2,
                                                    DETAIL_REPRESENTATIVE_LIMIT=1))
                self.assertLessEqual(len(limited["detail"]), 2)
                if mode == "none":
                    self.assertTrue(limited["detail"].empty)
                if mode == "failures" and not limited["detail"].empty:
                    self.assertTrue((~limited["detail"].KL_correctable).all())
                assert_frame_equal(comparable_summary(limited["summary"]), expected)

    def test_save_results_false_creates_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "must_not_exist"
            _, result = run_small(small_config(OUTPUT_DIR=output, SAVE_RESULTS=False))
            self.assertFalse(output.exists())
            self.assertEqual(list(root.iterdir()), [])
            self.assertFalse(result["paths"])

    def test_all_details_stream_to_disk_despite_memory_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            _, result = run_small(small_config(OUTPUT_DIR=output, SAVE_RESULTS=True,
                                              DETAIL_MODE="all", DETAIL_MEMORY_LIMIT=2))
            self.assertLessEqual(len(result["detail"]), 2)
            candidates = []
            for path in output.rglob("*.csv"):
                frame = pd.read_csv(path)
                if "erasure_positions_relative" in frame and "erasure_type" in frame:
                    candidates.append(frame)
            self.assertEqual(len(candidates), 1, "One streamed detail CSV is expected.")
            streamed = candidates[0]
            total = int((result["summary"].num_patterns_arbitrary
                         + result["summary"].num_patterns_contiguous).sum())
            self.assertEqual(len(streamed), total)
            assert_frame_equal(comparable_summary(result["summary"]),
                               comparable_summary(self.result["summary"]))


def run_tests():
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(FiniteSizeScalingTests)
    )
    if not result.wasSuccessful():
        raise RuntimeError("finite-size scaling regression tests failed")
    return {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}


if __name__ == "__main__":
    run_tests()
