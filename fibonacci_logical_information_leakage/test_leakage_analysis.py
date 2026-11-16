"""Integration checks for experimental meaning and reproducibility."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import leakage_analysis as a
import recoverability_kl as kl


class PipelineTests(unittest.TestCase):
    def config(self, **updates):
        return dict(INFLATION_LIST=[0, 1], WINDOW_SIZE_LIST=[6],
                    ERASURE_SIZE_LIST=[0, 1, 2, 6], SAVE_RESULTS=False,
                    SAVE_FIGURES=False, **updates)

    def test_legacy_generator_exact_and_sampled_reproducibility(self):
        for limit in [1, 100]:
            plan = a.PatternPlan(6, 2, 'arbitrary', 'window', limit, 200, 42)
            old = kl.generate_erasure_patterns(6, 2, 'auto', limit, 200, 42)
            self.assertEqual(list(plan), list(old))
            self.assertEqual(list(plan), list(plan))
            if limit == 1:
                self.assertEqual(plan.count, 200)
                self.assertLess(len(set(plan)), plan.count)

    def test_full_ring_includes_wrap_and_no_duplicate_empty_or_full(self):
        plan = a.PatternPlan(6, 2, 'contiguous', 'full_ring', 1, 5, 42)
        self.assertEqual(list(plan), [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])
        self.assertEqual(list(a.PatternPlan(6, 2, 'contiguous', 'window', 1, 5, 42)), list(plan)[:-1])
        for m in [0, 6]:
            self.assertEqual(len(list(a.PatternPlan(6, m, 'contiguous', 'full_ring', 1, 5, 42))), 1)

    def test_physical_endpoints_and_legacy_preservation(self):
        c = self.config()
        results = a.run_analysis(c)
        data = results['detail']
        empty = data[data.erasure_size == 0]
        np.testing.assert_allclose(empty.I_RE, 0, atol=1e-12)
        full = data[(data.N == 6) & (data.erasure_size == 6)]
        np.testing.assert_allclose(full.I_RE, 2, atol=1e-12)
        np.testing.assert_allclose(full.D_Z, 1, atol=1e-12)
        np.testing.assert_allclose(full.D_X, 1, atol=1e-12)
        one = data[data.erasure_size == 1]
        self.assertTrue(one.KL_correctable.all())
        self.assertTrue(one.decoupled.all())
        # n=0 adjacent 2-site erasure recovers strings but leaks the Z basis.
        known = data[(data.inflation_n == 0) & (data.erasure_size == 2)
                     & (data.erasure_type == 'contiguous')]
        self.assertTrue(known.recoverable_both.all())
        self.assertFalse(known.KL_correctable.any())
        np.testing.assert_allclose(known.D_Z, 2/3, atol=1e-12)
        self.assertTrue((known.comparison_type == 'B').all())
        self.assertTrue((known.important_case == 'A').all())
        self.assertTrue(data.legacy_core_agree.all())
        self.assertTrue(results['pair_correlations'].pearson_r.isna().all())

    def test_same_draws_across_N_and_paired_changes(self):
        c = dict(self.config(), ERASURE_SIZE_LIST=[2], MAX_EXACT_PATTERNS=1,
                 NUM_RANDOM_SAMPLES=31, ERASURE_TYPE_LIST=['arbitrary'])
        r = a.run_analysis(c)
        groups = list(r['detail'].groupby('N'))
        self.assertEqual(groups[0][1].erasure_positions_relative.tolist(),
                         groups[1][1].erasure_positions_relative.tolist())
        finite = r['finite_size']
        self.assertEqual(int(finite.same_relative_pattern_verified.sum()), 31)
        self.assertTrue(finite[finite.N == 9].previous_N.eq(6).all())
        pd.testing.assert_series_equal(
            finite[finite.N == 9].delta_I_RE.reset_index(drop=True),
            (finite[finite.N == 9].I_RE.reset_index(drop=True) -
             finite[finite.N == 6].I_RE.reset_index(drop=True)).rename('delta_I_RE'))
        self.assertTrue(r['summary'].maximum_interpretation.eq('observed_sample_maximum').all())

    def test_full_ring_has_no_fixed_pattern_scaling_claim(self):
        c = dict(self.config(), ERASURE_DOMAIN_MODE='full_ring', WINDOW_SIZE_LIST=[],
                 ERASURE_SIZE_LIST=[1])
        r = a.run_analysis(c)
        self.assertFalse(r['finite_size'].fixed_relative_pattern_comparison.any())
        self.assertTrue(r['finite_size'].delta_I_RE.isna().all())
        self.assertEqual(sorted(r['detail'].window_size.unique()), [6, 9])

    def test_explicit_skip_table_and_all_skipped(self):
        c = dict(self.config(), WINDOW_SIZE_LIST=[20], INFLATION_LIST=[0])
        p = a.prepare(c)
        self.assertEqual(p['skipped'].reason.tolist(), ['W>N'])
        r = a.run_analysis(c, p)
        self.assertTrue(r['detail'].empty)
        self.assertTrue(r['summary'].empty)

    def test_no_output_when_disabled_and_json_geometry_when_enabled(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'results'
            c = dict(self.config(), OUTPUT_DIR=destination)
            a.run_analysis(c)
            self.assertFalse(destination.exists())
            saved = a.run_analysis(dict(c, SAVE_RESULTS=True))
            table = pd.read_csv(destination / 'logical_leakage_detail.csv')
            self.assertEqual(json.loads(table.erasure_positions_relative.iloc[0]), [])
            metadata = json.loads((destination / 'metadata.json').read_text())
            self.assertTrue(metadata['source_sha256']['recoverability_kl.py'])
            self.assertEqual(len(table), len(saved['detail']))

    def test_input_validation_and_changed_preparation(self):
        c = self.config()
        for bad in [dict(N=6), dict(INFLATION_LIST=[True]), dict(EIGENVALUE_TOL=float('nan')),
                    dict(NUM_RANDOM_SAMPLES=0), dict(ERASURE_DOMAIN_MODE='invalid'),
                    dict(SEED_1='000111')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                a.prepare(dict(c, **bad))
        p = a.prepare(c)
        with self.assertRaisesRegex(ValueError, 'parameters changed'):
            a.run_analysis(dict(c, RANDOM_SEED=41), p)

    def test_benchmark_global_budget_and_memory_guard(self):
        c = self.config()
        p = a.prepare(c)
        timing = a.benchmark(p)
        self.assertEqual(int(timing.pattern_count.sum()), 100)
        self.assertTrue((timing.estimated_total_seconds > 0).all())
        with self.assertRaisesRegex(ValueError, 'MAX_DETAIL_ROWS'):
            a.run_analysis(dict(c, MAX_DETAIL_ROWS=1), p)

    def test_all_tied_pattern_maxima_retained_with_tolerance(self):
        c = dict(self.config(), INFLATION_LIST=[0], ERASURE_SIZE_LIST=[2],
                 ERASURE_TYPE_LIST=['arbitrary'])
        r = a.run_analysis(c)
        # All 15 two-site subsets have equal I; roundoff must not select only some.
        self.assertEqual(len(r['worst']), 15)
        self.assertEqual(int((r['cases'].important_case == 'D').sum()), 15)


if __name__ == '__main__':
    unittest.main(verbosity=2)
