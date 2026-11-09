"""Compare actual compiled C++ to frozen project code and independent oracles."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/reference/code'))
sys.path.insert(0, str(ROOT / 'tests/reference/tests'))
from dham import DHAM, count_transitions_one
from util import mfpt_state_reduction, mean_return_time, stationary_gth
import legacy_reference

def flat(a):
    return ' '.join(format(float(x), '.17g') for x in np.asarray(a).ravel())

def probe(mode, text, ok=True):
    p = subprocess.run([str(ROOT/'build/core_probe'), mode], input=text, text=True, capture_output=True)
    if ok:
        assert p.returncode == 0, p.stderr
        return np.fromstring(p.stdout, sep=' ')
    assert p.returncode != 0
    return p.stderr

def matrix_probe(mode, p, s=0, t=1, dt=.002):
    return probe(mode, f'{len(p)} {s} {t} {dt}\n' + flat(p))[0]

@pytest.mark.parametrize('lag', [1,3,11])
def test_directed_counts_match_project(lag):
    rng = np.random.default_rng(77)
    bins = rng.integers(-1, 9, 400)
    got = probe('counts',f'8 {lag} {len(bins)}\n'+ ' '.join(map(str,bins))).reshape(8,8)
    np.testing.assert_array_equal(got,count_transitions_one(bins,8,lag))
    assert not np.array_equal(got,got.T)

@pytest.mark.parametrize('windows',[1,4])
def test_dham_matches_corrected_reference(windows):
    rng=np.random.default_rng(19);n=9;q=np.linspace(0,8,n)
    params=np.stack([np.column_stack([rng.uniform(.1,5,3),rng.uniform(0,8,3),rng.uniform(.7,2,3)]) for _ in range(windows)])
    ref=DHAM(params,n);ref.qspace=q
    cs=rng.integers(0,20,(windows,n,n)).astype(float)
    cs[:,0,:]=0 # terminal-only row remains zero
    cs[:,2,7]=0 # absent directed edge remains absent
    expected=ref.build_MM(cs.sum(0),cs.sum(2),True)
    text=f'{n} {windows} {ref.KbT}\n'
    for c,u in zip(cs,ref.bias_on_grid()):text+=flat(c)+'\n'+flat(u)+'\n'
    got=probe('dham',text).reshape(n,n)
    np.testing.assert_allclose(got,expected,rtol=3e-14,atol=1e-15)
    assert got[2,7]==0 and np.all(got[0]==0)

def test_nonreversible_recovery_and_bias_offset():
    p=np.array([[.7,.3,0],[0,.8,.2],[.4,0,.6]])
    u=np.array([.4,2,.7]);kbt=.5961612777
    pb=p*np.exp((u[:,None]-u[None,:])/(2*kbt));pb/=pb.sum(1)[:,None]
    c=np.array([200,300,100])[:,None]*pb
    for offset in [0,10000]:
        got=probe('dham',f'3 1 {kbt}\n'+flat(c)+'\n'+flat(u+offset)).reshape(3,3)
        np.testing.assert_allclose(got,p,atol=5e-13,rtol=5e-13)

def test_extreme_bias_dham_stays_finite_without_erasing_support():
    c=np.eye(3)*10
    got=probe('dham','3 1 1\n'+flat(c)+'\n0 2000 4000').reshape(3,3)
    np.testing.assert_array_equal(got,np.eye(3))
    # A mathematically positive edge below double range must produce a diagnostic.
    assert 'underflow' in probe('tilt','3 1\n0 2000 4000\n'+flat(np.full((3,3),1/3)),ok=False)

def test_tilt_matches_legacy_orientation_and_retains_diagonal():
    rng=np.random.default_rng(7);p=np.diag(np.full(8,.8))+np.diag(np.full(7,.1),1)+np.diag(np.full(7,.1),-1)
    p/=p.sum(1)[:,None];u=rng.uniform(0,10,8)
    got=probe('tilt','8 .5981\n'+flat(u)+'\n'+flat(p)).reshape(8,8)
    expected=legacy_reference.bias_M_direct(p.T,u).T
    np.testing.assert_allclose(got,expected,rtol=5e-15,atol=1e-16)
    assert np.all(np.diag(got)>0)

@pytest.mark.parametrize('seed',[0,1,2])
def test_mfpt_matches_independent_linear_solve(seed):
    rng=np.random.default_rng(seed);p=rng.random((9,9));p/=p.sum(1)[:,None]
    for target in [0,4,8]:
        keep=np.flatnonzero(np.arange(9)!=target)
        exact=np.linalg.solve(np.eye(8)-p[np.ix_(keep,keep)],np.ones(8))*.002
        for k,start in enumerate(keep):
            got=matrix_probe('mfpt',p,start,target)
            assert got==pytest.approx(exact[k],rel=2e-14)
            assert got==pytest.approx(mfpt_state_reduction(p.T,start,target,.002),rel=2e-14)

def birth_death(f,kbt=.5981):
    n=len(f);p=np.zeros((n,n))
    for i in range(n-1):
        p[i,i+1]=.05*min(1,np.exp(-(f[i+1]-f[i])/kbt))
        p[i+1,i]=.05*min(1,np.exp(-(f[i]-f[i+1])/kbt))
    np.fill_diagonal(p,1-p.sum(1));return p

def test_40_kcal_barrier_against_60_digit_arithmetic():
    import mpmath as mp
    mp.mp.dps=60
    p=birth_death(np.r_[np.zeros(10),np.linspace(0,40,20)])
    n=len(p);a=mp.matrix(n-1);rhs=mp.matrix([1]*(n-1))
    for i in range(n-1):
        for j in range(n-1):
            a[i,j]=-mp.mpf(p[i,j]) if i!=j else mp.fsum(mp.mpf(p[i,k]) for k in range(n) if k!=i)
    exact=float(mp.lu_solve(a,rhs)[2])*.002
    assert matrix_probe('mfpt',p,2,29)==pytest.approx(exact,rel=2e-14)

@pytest.mark.parametrize('p,start,target',[
    ([[0,.5,.5],[0,1,0],[0,0,1]],0,1), # escape trap
    ([[1,0],[0,1]],0,1), # unreachable target
    ([[0,1,0],[0,.8,.2],[0,.3,.7]],1,0),
])
def test_unreachable_or_trapped_means_infinite(p,start,target):
    assert np.isinf(matrix_probe('mfpt',np.array(p),start,target))

def test_transient_predecessor_return_and_irrelevant_trap():
    p=np.array([[0,1,0],[0,.8,.2],[0,.3,.7]])
    assert matrix_probe('return',p,1,1,1)==pytest.approx(1/.6)
    assert np.isinf(matrix_probe('return',p,0,0,1))
    assert matrix_probe('mfpt',p,1,2,1)==pytest.approx(5)
    assert matrix_probe('mfpt',p,1,1,1)==0

def test_stationary_matches_analytic_distribution():
    f=np.random.default_rng(42).normal(size=12);p=birth_death(f)
    got=probe('stationary','12 0 0 1\n'+flat(p));expected=np.exp(-f/.5981);expected/=expected.sum()
    np.testing.assert_allclose(got,expected,rtol=2e-14,atol=0)

def test_ten_hill_energy_and_derivative():
    params=np.r_[np.linspace(.1,5,10),np.linspace(3,5,10),np.linspace(.7,2,10)]
    x=np.linspace(0,8,101)
    got=probe('bias','10 101\n'+flat(params)+'\n'+flat(x)).reshape(-1,2)
    a,b,s=params.reshape(3,-1)
    v=a*np.exp(-.5*((x[:,None]-b)/s)**2)
    np.testing.assert_allclose(got[:,0],v.sum(1),rtol=2e-15)
    np.testing.assert_allclose(got[:,1],(-v*(x[:,None]-b)/s**2).sum(1),rtol=2e-14,atol=2e-15)

def test_optimizer_improves_finite_objective_within_bounds():
    p=birth_death(np.r_[0,1,2,4,5,6,8,9])
    text='8 0 7 91234\n'+flat(p)
    got=probe('optimize',text);repeat=probe('optimize',text)
    np.testing.assert_array_equal(got,repeat)
    random,selected,status=got[:3];params=got[3:]
    assert 0<selected<random
    assert np.all((params[:2]>=.1)&(params[:2]<=5))
    assert np.all((params[2:4]>=0)&(params[2:4]<=7))
    assert np.all((params[4:]>=.7)&(params[4:]<=2))
    u=np.sum(params[:2]*np.exp(-.5*((np.arange(8)[:,None]-params[2:4])/params[4:])**2),axis=1)
    tilted=p*np.exp((u[:,None]-u[None,:])/(2*.5981));tilted/=tilted.sum(1)[:,None]
    assert selected==pytest.approx(mfpt_state_reduction(tilted.T,0,7,.002),rel=1e-13)

def test_optimizer_refuses_all_infinite_candidates():
    assert 'no finite MFPT' in probe('optimize','3 0 1 1\n'+flat([[0,.5,.5],[0,1,0],[0,0,1]]),ok=False)
