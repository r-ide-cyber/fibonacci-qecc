import unittest
from itertools import combinations
import numpy as np
from collision_core import cyclic_orbit,inflate
from collision_kl import evaluate_erasure,build_reduced_operators,reduced_operator_counts


def literal_reduced(a,b,N,E):
    kept=[i for i in range(N) if i not in E]
    def tensor(words):
        result=np.zeros((1<<len(E),1<<len(kept)))
        for x in words:
            x=f'{x:0{N}b}'
            i=int(''.join(x[s] for s in E) or '0',2)
            j=int(''.join(x[s] for s in kept) or '0',2)
            result[i,j]+=1/np.sqrt(len(words))
        return result
    aa,bb=tensor(a),tensor(b)
    return aa@aa.T,bb@bb.T,aa@bb.T


class KLTests(unittest.TestCase):
    def test_all_erasure_subsets_dense_partial_trace(self):
        N=6; a=tuple(int(x,2) for x in cyclic_orbit('000111')); b=(int('010101',2),int('101010',2))
        for m in range(7):
            for E in combinations(range(N),m):
                rho0,rho1,cross=literal_reduced(a,b,N,E)
                r=evaluate_erasure(a,b,N,E,compute_leakage=True,max_leakage_dim=128)
                self.assertEqual(r['KL_diagonal'],bool(np.allclose(rho0,rho1,atol=1e-13)))
                self.assertEqual(r['KL_offdiagonal'],bool(np.allclose(cross,0,atol=1e-13)))
                self.assertEqual(r['cross_collision'],bool(np.any(cross)))
                self.assertAlmostEqual(r['D_Z'],sum(abs(np.linalg.eigvalsh(rho0-rho1)))/2)
                self.assertAlmostEqual(r['D_X'],sum(np.linalg.svd(cross,compute_uv=False)))
                rhoE=(rho0+rho1)/2; rhoRE=np.block([[rho0,cross],[cross.T,rho1]])/2
                def entropy(r):
                    w=np.linalg.eigvalsh(r); w=w[w>1e-13]; return -sum(w*np.log2(w))
                self.assertAlmostEqual(r['I_RE'],1+entropy(rhoE)-entropy(rhoRE),places=10)

    def test_diagonal_failure_without_any_collision(self):
        a=tuple(int(x,2) for x in cyclic_orbit(inflate('000111',1)))
        b=tuple(int(x,2) for x in cyclic_orbit(inflate('010101',1)))
        rows=[evaluate_erasure(a,b,9,E) for E in combinations(range(9),2)]
        self.assertEqual(sum(r['cross_collision'] for r in rows),0)
        self.assertEqual(sum(r['KL_correctable'] for r in rows),9)
        self.assertTrue(evaluate_erasure(a,b,9,(0,1))['KL_correctable'])
        self.assertTrue(any(r['recoverable_both'] and not r['KL_correctable'] for r in rows))

    def test_self_collision_does_not_imply_diagonal_failure(self):
        # Plus/minus is excluded by positive-amplitude model; two disjoint Bell orbits
        # instead have equal one-site marginal and no within-orbit collisions.
        # A genuine positive-amplitude example with equal full reduced operators:
        a=(0b000,0b010); b=(0b101,0b111)
        r=evaluate_erasure(a,b,3,(1,))
        self.assertTrue(r['self_collision_0']); self.assertTrue(r['self_collision_1'])
        self.assertTrue(r['KL_diagonal']); self.assertTrue(r['KL_correctable'])

    def test_caps_empty_full_and_profiles(self):
        a=(0,3); b=(1,2)
        r=evaluate_erasure(a,b,2,(),compute_leakage=True,max_leakage_dim=8)
        self.assertTrue(r['KL_correctable']); self.assertAlmostEqual(r['I_RE'],0)
        r=evaluate_erasure(a,b,2,(0,1),compute_leakage=True,max_leakage_dim=8,include_profile=True)
        self.assertFalse(r['KL_correctable']); self.assertAlmostEqual(r['I_RE'],2)
        self.assertEqual(r['C_cross'],4); self.assertEqual(len(r['collision_profile']),1)
        limited=evaluate_erasure(a,b,2,(0,1),compute_leakage=True,max_leakage_dim=2)
        self.assertEqual(limited['leakage_status'],'dimension_limit'); self.assertIsNone(limited['I_RE'])
        self.assertEqual(limited['KL_correctable'],r['KL_correctable'])

if __name__=='__main__': unittest.main()
