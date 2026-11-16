"""2^Nの状態ベクトルを使わない、独立な総当たりペア和との照合。"""
from collections import Counter
from itertools import combinations, product
from fractions import Fraction
from pathlib import Path
import contextlib
import io
import json
import random
import tempfile
import shutil
import hashlib
import unittest
from recoverability_kl import *


def brute_counts(a, b, E):
    # 群分け・integer kernelに依存しない参照実装。
    keep=[i for i in range(len(a[0])) if i not in E]
    counts=Counter()
    for x in a:
        for y in b:
            if all(x[i]==y[i] for i in keep):
                counts[(''.join(x[i] for i in E),''.join(y[i] for i in E))]+=1
    return counts


class CoreTests(unittest.TestCase):
    def test_inflate(self):
        self.assertEqual(inflate('10',0),'10')
        self.assertEqual(inflate('10',1),'101')
        self.assertEqual(inflate('10',2),'10110')
        for n in (-1,True,1.2):
            with self.assertRaises(ValueError): inflate('10',n)

    def test_orbit_and_canonical(self):
        self.assertEqual(cyclic_orbit('0101'),['0101','1010'])
        self.assertEqual(canonical_rotation('1100'),'0011')
        self.assertEqual(cyclic_orbit('000'),['000'])

    def test_ring_swaps(self):
        self.assertEqual(set(adjacent_swap_neighbors('000111')),{'001011','100110'})
        self.assertEqual(adjacent_swap_neighbors('00'),[])
        self.assertTrue(satisfies_adjacent_swap_condition('000111','010101'))
        self.assertFalse(satisfies_adjacent_swap_condition('000111','001011'))
        self.assertFalse(satisfies_adjacent_swap_condition('000111','100110'))
        self.assertFalse(satisfies_adjacent_swap_condition('0101','1010'))
        self.assertFalse(satisfies_adjacent_swap_condition('000111','000011'))
        for n in range(2,7):
            for a in map(''.join,product('01',repeat=n)):
                neighbors={canonical_rotation(b) for b in adjacent_swap_neighbors(a)}
                for b in legacy.fixed_weight_representatives(n,a.count('1')):
                    expected=canonical_rotation(a)!=b and b not in neighbors
                    self.assertEqual(satisfies_adjacent_swap_condition(a,b),expected)

    def test_generate_pairs(self):
        self.assertEqual(generate_valid_logical_pairs(6,3,3),[('000111','010101')])
        with self.assertRaises(ValueError): generate_valid_logical_pairs(6,2,3)

    def test_projection_validation(self):
        self.assertEqual(remaining_pattern('10101',[3,0]),'011')
        self.assertEqual(erased_pattern('10101',[3,0]),'10')
        self.assertEqual(remaining_pattern('10101',range(5)),'')
        for E in ([0,0],[-1],[5],[True],[0.5]):
            with self.assertRaises(ValueError): remaining_pattern('10101',E)

    def test_recoverability_and_counts(self):
        a=['000','100'];b=['011','111']
        self.assertFalse(recoverability(a,[0]))
        self.assertTrue(recoverability(a,[1]))
        g0=group_by_remaining(a,[0]);g1=group_by_remaining(b,[0])
        self.assertEqual(collision_counts(g0,g1),(1,1,0))
        self.assertEqual(collision_counts(g0,g0),(1,1,4))

    def test_reduced_operator(self):
        a=['000','111']
        self.assertEqual(reduced_operator(a,a,[0]),{('0','0'):Fraction(1,2),('1','1'):Fraction(1,2)})
        a=['000','100'];b=['011','111']
        self.assertEqual(reduced_operator(a,b,[0]),{})
        self.assertEqual(reduced_operator(a,a,[0]),dict.fromkeys(product('01',repeat=2),Fraction(1,2)))

    def test_unequal_normalization(self):
        self.assertTrue(equal_normalized_counts(Counter(a=1,b=1),2,Counter(a=2,b=2),4))
        model=LogicalPair('000111','010101',1)
        self.assertEqual((len(model.O0),len(model.O1)),(9,3))
        self.assertTrue(model.evaluate([])['KL_correctable'])
        self.assertFalse(model.evaluate(range(model.N))['KL_correctable'])

    def test_type_c_is_retained(self):
        # 人工的な等振幅状態。Fibonacciの発見としては扱わない。
        r=evaluate_orbits(['000','100'],['011','111'],[0])
        self.assertEqual(r['comparison_type'],'C')
        self.assertFalse(r['recoverable_both']);self.assertTrue(r['KL_correctable'])

    def test_type_b_diagonal_and_off(self):
        self.assertEqual(evaluate_orbits(['00'],['11'],[0])['KL_failure_reason'],'diagonal')
        self.assertEqual(evaluate_orbits(['00','11'],['01','10'],[0])['KL_failure_reason'],'offdiagonal')

    def test_sparse_against_pair_sum(self):
        model=LogicalPair('000111','010101',1)
        for m in range(model.N+1):
            for E in combinations(range(model.N),m):
                c0=brute_counts(model.O0,model.O0,E);c1=brute_counts(model.O1,model.O1,E)
                c01=brute_counts(model.O0,model.O1,E)
                self.assertEqual(reduced_operator_counts(model.O0,model.O1,E),c01)
                row=model.evaluate(E)
                diagonal=all(Fraction(c0[k],len(model.O0))==Fraction(c1[k],len(model.O1)) for k in c0.keys()|c1.keys())
                self.assertEqual(row['KL_diagonal'],diagonal)
                self.assertEqual(row['KL_offdiagonal'],not c01)
                self.assertEqual(row['self_collision_count_0'],(sum(c0.values())-len(model.O0))//2)
                self.assertEqual(row['cross_collision_count'],sum(c01.values()))
                self.assertEqual(row['recoverable_both'],recoverability(model.O0,E) and recoverability(model.O1,E))

    def test_patterns(self):
        self.assertEqual(list(generate_erasure_patterns(4,0)),[()])
        self.assertEqual(list(generate_erasure_patterns(4,4)),[(0,1,2,3)])
        self.assertEqual(len(list(generate_erasure_patterns(5,2))),10)
        p=generate_erasure_patterns(10,5,max_exact_patterns=5,num_random_samples=40)
        self.assertEqual(p.mode,'random');self.assertEqual(list(p),list(p));self.assertEqual(len(list(p)),40)
        self.assertTrue(all(len(set(E))==5 for E in p))
        q=generate_erasure_patterns(10,5,max_exact_patterns=5,num_random_samples=40,random_seed=43)
        self.assertNotEqual(list(p),list(q))
        # m別seedなので呼出順に依存しない。
        self.assertEqual(list(p),list(generate_erasure_patterns(10,5,max_exact_patterns=5,num_random_samples=40)))

    def test_reference_hash_gate(self):
        ref=Path(__file__).parent/'reference'
        with tempfile.TemporaryDirectory() as tmp:
            copy=Path(tmp)/'reference';shutil.copytree(ref,copy)
            with (copy/'seed_pairs.jsonl').open('a') as h: h.write('\n')
            with self.assertRaisesRegex(RuntimeError,'hash'):
                verify_reference(copy)

    def test_baseline_mismatch_stops(self):
        # 改訂した履歴結果との不一致を、hash検査とは別に検出する。
        ref=Path(__file__).parent/'reference'
        with tempfile.TemporaryDirectory() as tmp:
            copy=Path(tmp)/'reference';shutil.copytree(ref,copy)
            target=copy/'contiguous_kl_results.jsonl'
            rows=load_jsonl(target);rows[0]['satisfies_kl']=False
            target.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest=json.loads((copy/'provenance.json').read_text())
            manifest['bundled_hashes'][target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
            (copy/'provenance.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError,'不一致'):
                reproduce_baseline(copy,Path(tmp)/'results')
            self.assertTrue((Path(tmp)/'results'/'baseline_failure.txt').exists())

    def test_analysis_gate(self):
        with self.assertRaises(RuntimeError):
            analyze_model(LogicalPair('000111','010101',1),[0],baseline_verified=False)

    def test_n_above_100(self):
        model=LogicalPair('000111','010101',7)
        self.assertGreater(model.N,100)
        self.assertTrue(model.evaluate([])['KL_correctable'])
        self.assertTrue(model.evaluate([0])['KL_correctable'])

    def test_inspection_and_summary(self):
        model=LogicalPair('000111','010101',1)
        with contextlib.redirect_stdout(io.StringIO()):
            report=model.inspect_erasure_pattern(range(9))
            df=analyze_model(model,[0,1,9],baseline_verified=True)
        self.assertTrue(report['cross_collision_examples'])
        for witness in report['cross_collision_examples']:
            self.assertEqual(witness['x_remaining'],witness['y_remaining'])
        summary=summarize(df)
        for _,r in summary.iterrows():
            self.assertAlmostEqual(sum(r[f'Type_{t}_rate'] for t in 'ABCD'),1)


def run_tests():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise RuntimeError('unit test failed; 新解析を停止します。')
    return {'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}


if __name__=='__main__': run_tests()
