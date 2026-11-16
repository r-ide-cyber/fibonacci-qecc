"""Independent dense-reference and workflow regression tests; python -m unittest."""
import unittest
from itertools import combinations, product
from math import comb
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pandas as pd
from fibonacci_design.core import *
from fibonacci_design.features import pair_features, cross_supports, cyclic_span
from fibonacci_design.design import Config, Experiment, candidate_table


def dense_blocks(a,b,n,erased):
    N,o0,o1=logical_orbits(a,b,n)
    E=[i for i in range(N) if erased>>i&1]
    B=[i for i in range(N) if not erased>>i&1]
    def coefficients(values):
        state=np.zeros((2**len(E),2**len(B)))
        for x in values:
            ei=sum(((x>>site)&1)<<i for i,site in enumerate(E))
            bi=sum(((x>>site)&1)<<i for i,site in enumerate(B))
            state[ei,bi]=1/np.sqrt(len(values))
        return state
    x,y=coefficients(o0),coefficients(o1)
    return x@x.T,y@y.T,x@y.T


class ExactMathematics(unittest.TestCase):
    def test_dense_kl_all_patterns(self):
        for a,b,n in [('001','011',0),('0011','0101',0),('000111','010101',0),('01','00',1)]:
            # Last pair inflates to different lengths and is tested separately.
            if len(inflate(a,n))!=len(inflate(b,n)): continue
            N,_,_=logical_orbits(a,b,n)
            for mask in range(1<<N):
                r0,r1,r01=dense_blocks(a,b,n,mask)
                expected=np.allclose(r0,r1,atol=1e-12) and np.allclose(r01,0,atol=1e-12)
                self.assertEqual(erasure_test(a,b,n,mask)[0],expected,(a,b,mask))
                q0,q1,q01=reduced_blocks(a,b,n,mask)
                active=np.flatnonzero(np.diag(r0+r1)>0)
                np.testing.assert_allclose(q0,r0[np.ix_(active,active)],atol=1e-12)
                np.testing.assert_allclose(q1,r1[np.ix_(active,active)],atol=1e-12)
                np.testing.assert_allclose(q01,r01[np.ix_(active,active)],atol=1e-12)
                leak=leakage_test(a,b,n,mask)
                self.assertEqual(leak['I_RE']<1e-10,expected)
                self.assertGreaterEqual(leak['D_X'],-1e-12)
                self.assertLessEqual(leak['D_X'],1+1e-12)

    def test_endpoints_and_classical_distinction(self):
        a,b='000111','010101'
        self.assertEqual(erasure_test(a,b,1,0),(True,True,True))
        self.assertAlmostEqual(leakage_test(a,b,1,(1<<9)-1)['I_RE'],2,places=10)
        # Distinct repetition words are classically recoverable but leak Z.
        self.assertEqual(erasure_test('000','111',0,1),(False,True,True))
        with self.assertRaises(ValueError): logical_orbits('001','010',0)
        with self.assertRaises(ValueError): erasure_test(a,b,0,1<<6)

    def test_generation_and_local_guarantee(self):
        for length,count in [(6,1),(8,27),(10,261)]:
            reps=representatives(length,length//2,length//2)
            valid=[(a,b) for a,b in combinations(reps,2) if adjacent_swap_valid(a,b)]
            self.assertEqual(len(valid),count)
        self.assertFalse(adjacent_swap_valid('001','011'))  # self-swap into same class
        c=Config(baseline_seed_0='000111',baseline_seed_1='010101')
        df,stats,_=candidate_table(c)
        self.assertEqual((stats['raw_bit_strings'],stats['number_cyclic_classes'],len(df)),(20,4,1))
        for n in [1,2,3]:
            fn=[1,1,2,3][n]
            N=len(inflate('000111',n))
            for m in range(1,fn+2):
                for mask in contiguous_masks(N,N,m):
                    self.assertTrue(erasure_test('000111','010101',n,mask)[0])

    def test_collision_features_against_enumeration(self):
        a,b='000111','010101'; N=6
        supports=cross_supports(a,b,0)
        for W in [4,6]:
            for m in range(W+1):
                f=pair_features(a,b,0,erasure_sizes=(m,),window_size=W)
                arbitrary=[sum(1<<i for i in E) for E in combinations(range(W),m)]
                contiguous=contiguous_masks(N,W,m)
                def mean_collision(masks):
                    return np.mean([sum(count for d,count in supports.items() if d&~e==0) for e in masks])
                self.assertAlmostEqual(f['expected_cross_collision_arbitrary'],mean_collision(arbitrary))
                self.assertAlmostEqual(f['expected_cross_collision_contiguous'],mean_collision(contiguous))
        for N in range(1,7):
            for support in range(1,1<<N):
                exact=min(m for m in range(1,N+1) if any(support&~e==0 for e in contiguous_masks(N,N,m)))
                self.assertEqual(cyclic_span(support,N),exact)


class Sampling(unittest.TestCase):
    def test_unranking(self):
        for W in range(1,9):
            for m in range(W+1):
                expected=[sum(1<<i for i in E) for E in combinations(range(W),m)]
                self.assertEqual([unrank_combination(r,W,m) for r in range(comb(W,m))],expected)

    def test_disjoint_reproducible_and_exact(self):
        bank=pattern_bank(12,10,4,20,50,60,42,31415)
        self.assertEqual(bank,pattern_bank(12,10,4,20,50,60,42,31415))
        a,b=bank['train'],bank['validation']
        self.assertFalse(set(a['masks'])&set(b['masks']))
        self.assertEqual(len(set(a['masks'])),50)
        self.assertEqual(len(set(b['masks'])),60)
        for E in a['masks']+b['masks']:
            self.assertEqual(E.bit_count(),4); self.assertLess(E,1<<10)
        exact=pattern_bank(8,8,2,100,30,40,1,2)
        self.assertEqual(exact['train'],exact['validation'])
        self.assertEqual(exact['train']['calculation_mode'],'exact')
        near=pattern_bank(6,6,3,1,19,20,1,2)
        self.assertEqual(near['validation']['n_patterns'],1)
        self.assertFalse(set(near['train']['masks'])&set(near['validation']['masks']))
        large=pattern_bank(267,267,40,1,5,7,1,2)
        self.assertGreater(large['train']['population_patterns'],2**63)
        self.assertEqual(large['validation']['n_patterns'],7)


class Workflow(unittest.TestCase):
    def config(self,directory,**kwargs):
        values=dict(baseline_seed_0='00001111',baseline_seed_1='01010101',seed_length=8,k0=4,k1=4,
            screening_inflation_n=0,validation_inflation_list=[1,2],erasure_size_list=[1,2,3],max_exact_patterns=30,
            train_num_samples=10,validation_num_samples=15,leakage_num_samples=5,max_exact_leakage_patterns=20,
            top_k_candidates=3,top_k_leakage=2,output_dir=directory)
        values.update(kwargs)
        return Config(**values)

    def test_resume_csv_and_common_samples(self):
        with TemporaryDirectory() as directory:
            e=Experiment(self.config(directory)); e.run()
            p=e.tables['performance']
            self.assertTrue((p.groupby(['inflation_n','window_size','erasure_size','sample_split']).sample_id.nunique()==1).all())
            base=p[p.is_baseline]
            self.assertTrue((base.Delta_R_arbitrary==0).all())
            self.assertTrue((p.R_KL_contiguous.isin([0.,1.])).all())
            self.assertEqual(len(e.tables['final']),3)
            # Resume must not call expensive evaluation paths.
            with patch('fibonacci_design.design.erasure_test',side_effect=AssertionError('recomputed')), patch('fibonacci_design.design.leakage_test',side_effect=AssertionError('recomputed')):
                r=Experiment(self.config(directory)); r.screen(); r.train()
                r.leakage(r.leakage_ids,[0],'train'); r.validate()
            pd.testing.assert_frame_equal(e.tables['comparison'],r.tables['comparison'])
            with TemporaryDirectory() as second:
                # Compatible CSV payloads alone must also reuse the calculations.
                c=self.config(second,previous_result_dirs=[str(Path(directory)/'results')])
                with patch('fibonacci_design.design.erasure_test',side_effect=AssertionError('CSV recomputed')), patch('fibonacci_design.design.leakage_test',side_effect=AssertionError('CSV recomputed')):
                    q=Experiment(c); q.screen(); q.train(); q.leakage(q.leakage_ids,[0],'train'); q.validate()
                self.assertTrue(q.store.imported)

    def test_window_no_save_and_empty_alternatives(self):
        with TemporaryDirectory() as directory:
            c=self.config(directory,baseline_seed_0='000111',baseline_seed_1='010101',seed_length=6,k0=3,k1=3,
                          erasure_domain_mode='window',window_size_list=[4,7,99],save_results=False)
            e=Experiment(c); e.run()
            self.assertTrue(e.tables['final'].empty)
            self.assertEqual(e.counts['nonbaseline_candidates'],0)
            self.assertFalse(list(Path(directory).iterdir()))
            self.assertTrue((e.tables['performance'].window_size<=e.tables['performance'].N).all())


if __name__=='__main__': unittest.main(verbosity=2)
