import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import combinations
import pandas as pd
import numpy as np
from collision_analysis import run_analysis, pattern_stream, contiguous_patterns, experiment_plan, pair_correlations


def config(**overrides):
    c=dict(SEED_0='000111',SEED_1='010101',PAIR_ID='baseline',LOGICAL_PAIRS=None,
           INFLATION_LIST=[0],ERASURE_DOMAIN_MODE='full_ring',WINDOW_SIZE_LIST=[4],WINDOW_START=0,
           ERASURE_SIZE_LIST=[0,1,2,3,6],MAX_EXACT_PATTERNS=100,NUM_RANDOM_SAMPLES=50,RANDOM_SEED=42,
           TOP_K_DANGEROUS_SUPPORTS=5,SAVE_RESULTS=False,SAVE_FIGURES=False,OUTPUT_DIR='unused',
           STORE_PAIR_DETAILS=False,SAVE_COLLISION_PROFILES=False,COMPUTE_LEAKAGE=False,MAX_LEAKAGE_DIM=64,
           MAX_DETAIL_ROWS=10000,LOW_WEIGHT_MAX=2,LARGE_SPAN_FRACTION=0.5,VALIDATE_SUPPORT_PATTERNS=3)
    c.update(overrides); return c


class ExperimentTests(unittest.TestCase):
    def test_exact_expectations_and_KL_decomposition(self):
        r=run_analysis(config(),progress=lambda _:None)
        g=r['comparison']
        np.testing.assert_allclose(g.mean_C_cross_arbitrary,g.expected_cross_collision_arbitrary,atol=1e-12)
        np.testing.assert_allclose(g.mean_C_cross_contiguous,g.expected_cross_collision_contiguous_full_ring,atol=1e-12)
        np.testing.assert_allclose(1-g.R_KL_arbitrary,g.P_cross_collision_arbitrary+g.diagonal_only_failure_rate_arbitrary)
        self.assertTrue(set(g.R_KL_contiguous).issubset({0.,1.}))
        # A constant or single-pair curve does not produce a correlation coefficient.
        self.assertTrue(r['correlations'].Pearson.isna().all())
        self.assertTrue(g.loc[g.erasure_size==0,'R_KL_arbitrary'].eq(1).all())
        self.assertTrue(g.loc[g.erasure_size==6,'R_KL_arbitrary'].eq(0).all())

    def test_window_wrap_and_expectation(self):
        r=run_analysis(config(ERASURE_DOMAIN_MODE='window',WINDOW_START=4,ERASURE_SIZE_LIST=[0,2,4,6],SAVE_COLLISION_PROFILES=True),progress=lambda _:None)
        np.testing.assert_allclose(r['comparison'].mean_C_cross_arbitrary,r['comparison'].expected_cross_collision_arbitrary,atol=1e-12)
        self.assertEqual(contiguous_patterns(6,2,(4,5,0,1)),[(4,5),(0,5),(0,1)])
        self.assertFalse(r['collision_profiles'].empty)
        self.assertTrue(r['skipped'].reason.eq('erasure_size>domain_size').all())

    def test_sampling_reproducibility_and_duplicates(self):
        c=config(MAX_EXACT_PATTERNS=1,NUM_RANDOM_SAMPLES=200)
        a,mode=pattern_stream(6,2,tuple(range(6)),'arbitrary',c)
        b,_=pattern_stream(6,2,tuple(range(6)),'arbitrary',c)
        a,b=list(a),list(b)
        self.assertEqual(a,b); self.assertEqual(mode,'sampled')
        self.assertEqual(len(a),200); self.assertLess(len(set(a)),200)
        self.assertTrue(all(len(E)==2 for E in a))

    def test_small_sample_has_nonzero_uncertainty(self):
        r=run_analysis(config(ERASURE_SIZE_LIST=[2],MAX_EXACT_PATTERNS=1,NUM_RANDOM_SAMPLES=8),progress=lambda _:None)
        a=r['rates'].query("erasure_type=='arbitrary'").iloc[0]
        self.assertLess(a.R_KL_ci_low,a.R_KL_ci_high)

    def test_plan_and_row_guard(self):
        c=config(ERASURE_DOMAIN_MODE='window',WINDOW_SIZE_LIST=[4,20])
        p,s=experiment_plan(c)
        self.assertTrue(s.reason.eq('window_size>N').any())
        with self.assertRaises(ValueError): run_analysis(config(MAX_DETAIL_ROWS=1),progress=lambda _:None)

    def test_save_flags(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp)/'nothing'
            run_analysis(config(ERASURE_SIZE_LIST=[1],OUTPUT_DIR=str(out)),progress=lambda _:None)
            self.assertFalse(out.exists())


if __name__=='__main__': unittest.main()
