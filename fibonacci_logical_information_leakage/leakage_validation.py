"""Independent explicit-statevector checks, restricted to small physical systems."""
from itertools import combinations
import numpy as np
import pandas as pd
from fibonacci_leakage_core import OrbitPair, analyze_erasure
from recoverability_kl import LogicalPair


def _explicit(pair, E):
    if pair.N > 8:
        raise ValueError('Explicit statevector validation is limited to N<=8')
    E = tuple(sorted(E))
    keep = tuple(i for i in range(pair.N) if i not in E)
    factors = []
    for support in (pair.O0, pair.O1):
        psi = np.zeros(2**pair.N)
        psi[list(support)] = 1 / np.sqrt(len(support))
        factors.append(psi.reshape((2,)*pair.N).transpose(E+keep).reshape(2**len(E), -1))
    A, B = factors
    rho00, rho11, rho01 = A@A.T, B@B.T, A@B.T
    # Form the full reference+physical vector independently, then trace the remaining axes.
    phi = np.zeros((2, 2**pair.N))
    phi[0,list(pair.O0)] = 1/np.sqrt(2*pair.M0)
    phi[1,list(pair.O1)] = 1/np.sqrt(2*pair.M1)
    tensor=phi.reshape((2,)*(pair.N+1))
    factor=tensor.transpose((0,)+tuple(i+1 for i in E+keep)).reshape(2**(len(E)+1),-1)
    rhoRE=factor@factor.T
    d=2**len(E)
    blocks=rhoRE.reshape(2,d,2,d)
    rhoR=np.einsum('aibi->ab',blocks)
    rhoE=np.einsum('aiaj->ij',blocks)
    def entropy(rho):
        eigen=np.linalg.eigvalsh(rho)
        assert eigen.min() > -1e-12
        positive=eigen[eigen>0]
        return float(-np.sum(positive*np.log2(positive)))
    SR,SE,SRE=map(entropy,(rhoR,rhoE,rhoRE))
    return dict(rho00=rho00,rho11=rho11,rho01=rho01,rho_RE=rhoRE,rho_E=rhoE,rho_R=rhoR,
                S_R=SR,S_E=SE,S_RE=SRE,I_RE=SR+SE-SRE,
                D_Z=float(np.linalg.svd(rho00-rho11,compute_uv=False).sum()/2),
                D_X=float(np.linalg.svd(rho01,compute_uv=False).sum()))


def run_validation():
    """Return a compact numerical-error table; any failed comparison raises."""
    model=LogicalPair('000111','010101',0)
    orbit=OrbitPair(model.O0,model.O1)
    cases=[('Fibonacci n=0; unequal M0/M1; all 64 subsets', orbit,
             [E for m in range(7) for E in combinations(range(6),m)])]
    for label,O0,O1,E in [
        ('Type 0',['00'],['01'],(0,)),
        ('Type Z',['00'],['11'],(0,)),
        ('Type X',['00','11'],['01','10'],(0,)),
        ('Type ZX / full loss',['00'],['11'],(0,1)),
        ('Unequal orbits / empty and full',['000','111'],['001','010','011'],None),
    ]:
        p=OrbitPair(O0,O1)
        cases.append((label,p,[E] if E is not None else [(),tuple(range(p.N))]))
    rows=[]
    matrices=('rho00','rho11','rho01','rho_RE','rho_E','rho_R')
    scalars=('S_R','S_E','S_RE','I_RE','D_Z','D_X')
    for label,pair,patterns in cases:
        errors={f'max_error_{key}':0. for key in matrices+scalars}
        branch_error=0.
        for E in patterns:
            direct=_explicit(pair,E)
            result=analyze_erasure(pair,E,return_operators=True)
            actual=result['operators'].to_dense(full_basis=True)
            grouped=analyze_erasure(pair,E,max_dense_erasure_qubits=0)
            for key in matrices:
                error=float(np.max(np.abs(actual[key]-direct[key])))
                errors[f'max_error_{key}']=max(errors[f'max_error_{key}'],error)
                np.testing.assert_allclose(actual[key],direct[key],atol=2e-12,rtol=0)
            for key in scalars:
                error=abs(result[key]-direct[key])
                errors[f'max_error_{key}']=max(errors[f'max_error_{key}'],error)
                assert error<1e-10,(label,E,key,error)
                branch_error=max(branch_error,abs(result[key]-grouped[key]))
            assert branch_error<1e-10
            assert result['KL_correctable']==grouped['KL_correctable']
            if label.startswith('Fibonacci'):
                assert result['KL_correctable']==model.evaluate(E)['KL_correctable']
        rows.append(dict(case=label,N=pair.N,M0=pair.M0,M1=pair.M1,
                         patterns_checked=len(patterns),**errors,
                         max_dense_grouping_error=branch_error,passed=True))
    # Hilbert dimension 2^1000 is never allocated; only rank <=2 Gram matrices are needed.
    huge=analyze_erasure(OrbitPair(range(64),range(64,128),N=1000),range(1000),max_matrix_dimension=4)
    np.testing.assert_allclose([huge['I_RE'],huge['D_Z'],huge['D_X']],[2,1,1],atol=1e-11)
    rows.append(dict(case='Large m exact grouping: known full-loss endpoint',N=1000,M0=64,M1=64,
                     patterns_checked=1,operator_mode=huge['calculation_mode'],passed=True))
    return pd.DataFrame(rows)
