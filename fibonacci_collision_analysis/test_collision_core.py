import unittest
from itertools import combinations,product
from collections import Counter
import numpy as np
from collision_core import *


class GeometryTests(unittest.TestCase):
    def test_definitions(self):
        self.assertEqual(inflate('01',2),'10101')
        self.assertEqual(difference_support('010010','011000'),{2,4})
        self.assertEqual(cyclic_span({5,6,8},20),4)
        self.assertEqual(cyclic_span({2,10},20),9)
        self.assertEqual(cyclic_span([],20),0)
        self.assertEqual(cyclic_span([3],20),1)
        self.assertEqual(cyclic_span(range(20),20),20)
        self.assertEqual(cyclic_span({19,0},20),2)
        self.assertTrue(validate_pair('000111','010101')['adjacent_swap_valid'])
        for a,b in [('0011','0101'),('000111','111000'),('001','011')]:
            with self.assertRaises(ValueError): validate_pair(a,b)

    def test_span_exhaustive_against_intervals(self):
        for N in range(1,7):
            for mask in range(1<<N):
                D=mask_to_sites(mask,N)
                target=next(m for m in range(N+1) if any(set(D).issubset({(s+i)%N for i in range(m)}) for s in range(N)))
                self.assertEqual(cyclic_span(D,N),target)

    def test_complete_pair_tables_periodic_and_aperiodic(self):
        for a,b,n in [('000111','010101',0),('000111','010101',2),('00001111','00011011',1)]:
            g=CollisionGeometry(a,b,n,store_pair_details=True)
            for kind,o0,o1 in [('cross',g.O0,g.O1),('self_0',g.O0,g.O0),('self_1',g.O1,g.O1)]:
                pairs=product(o0,o1) if kind=='cross' else combinations(o0,2)
                brute=Counter()
                for x,y in pairs:
                    d=int(x,2)^int(y,2); brute[d]+=1
                    self.assertEqual(g.features[d],(len(difference_support(x,y)),cyclic_span(difference_support(x,y),g.N)))
                self.assertEqual(g.supports[kind],brute)
            self.assertEqual(len(list(g.iter_pair_details())),len(g.O0)*len(g.O1)+comb(len(g.O0),2)+comb(len(g.O1),2))

    def test_collision_equivalence_and_expected_count(self):
        g=CollisionGeometry('000111','010101',0)
        for m in range(g.N+1):
            vals=[]
            for E in combinations(range(g.N),m):
                kept=[i for i in range(g.N) if i not in E]
                direct=sum(all(x[i]==y[i] for i in kept) for x in g.O0 for y in g.O1)
                self.assertEqual(g.collision_counts(E)['C_cross'],direct); vals.append(direct)
            ex=g.expected_frame([m]).iloc[0]
            self.assertAlmostEqual(sum(vals)/len(vals),ex.expected_cross_collision_arbitrary)
            direct=sum(g.collision_counts(tuple((s+i)%g.N for i in range(m)))['C_cross'] for s in range(g.N))/g.N
            self.assertEqual(direct,ex.expected_cross_collision_contiguous)

    def test_window_and_event_overlap(self):
        g=CollisionGeometry('000111','010101',0); W=(4,5,0,1)
        for m in range(5):
            ex=g.expected_frame([m],W).iloc[0]
            counts=[g.collision_counts(E)['C_cross'] for E in combinations(W,m)]
            self.assertAlmostEqual(np.mean(counts),ex.expected_cross_collision_arbitrary)
        self.assertEqual(support_probability([2],6,2,W),0)
        self.assertAlmostEqual(support_probability([0,1],6,2,W),1/6)
        self.assertGreater(g.expected_frame([6]).iloc[0].expected_cross_collision_arbitrary,1)
        for D in [[],[0],[0,2],[0,5],list(range(6))]:
            for m in range(7):
                brute=sum(set(D).issubset({(s+i)%6 for i in range(m)}) for s in range(6))/6
                self.assertEqual(contiguous_probability(D,6,m),brute)

if __name__=='__main__': unittest.main()
