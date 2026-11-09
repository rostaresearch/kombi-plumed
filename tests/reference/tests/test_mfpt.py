"""The optimiser's model MFPT: stable state reduction equals the as-run fundamental-matrix formula
where that formula is well conditioned, and stays exact (against 50-digit arithmetic) where it fails,
as after the excursion of pilot run A3_L20ps_run001 (12 Oct 2026)."""
import numpy as np
import pytest

import config
from MSM import MSM
from util import mfpt_state_reduction, try_and_optim_M


def birth_death(F, kT=0.5981, p0=0.05):
    """column-stochastic nearest-neighbour chain with stationary distribution ~ exp(-F/kT)"""
    n = len(F)
    M = np.zeros((n, n))
    for i in range(n - 1):
        up = p0 * min(1.0, np.exp(-(F[i + 1] - F[i]) / kT))      # Metropolis i -> i+1
        down = p0 * min(1.0, np.exp(-(F[i] - F[i + 1]) / kT))    # i+1 -> i
        M[i + 1, i] = up
        M[i, i + 1] = down
    M[np.arange(n), np.arange(n)] = 1.0 - M.sum(axis=0)
    return M


def legacy_mfpt(M, s, e, dt):
    msm = MSM(); msm.build_MSM_from_M(M, dim=1, time_step=dt)
    msm._compute_peq_fes_M(); msm._build_mfpt_matrix_M()
    return float(np.real(msm.mfpts[s, e]))


def mpmath_mfpt(M, s, e, dt):
    """hitting time by 50-digit linear algebra; the diagonal of I - Q is formed from the
    off-diagonal sums, as in the audit's check_stable_mfpt.py"""
    import mpmath as mp
    mp.mp.dps = 50
    P = M.T
    n = P.shape[0]
    keep = [i for i in range(n) if i != e]
    Q = mp.matrix(n - 1, n - 1)
    for a, i in enumerate(keep):
        for b, j in enumerate(keep):
            if a != b:
                Q[a, b] = -mp.mpf(P[i, j])
        Q[a, a] = mp.fsum(mp.mpf(P[i, j]) for j in range(n) if j != i)
    m = mp.lu_solve(Q, mp.matrix([1] * (n - 1)))
    return float(m[keep.index(s)]) * dt


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_state_reduction_equals_legacy_when_well_conditioned(seed):
    rng = np.random.default_rng(seed)
    F = np.cumsum(rng.normal(0, 0.4, 40))          # rough landscape spanning a few kcal/mol
    M = birth_death(F)
    for s, e in [(3, 30), (10, 39), (35, 2)]:
        a, b = legacy_mfpt(M, s, e, 2e-6), mfpt_state_reduction(M, s, e, 2e-6)
        assert b == pytest.approx(a, rel=1e-8)


def test_state_reduction_exact_on_ill_conditioned_chain():
    """a 40 kcal/mol climb to the target (its stationary probability ~1e-29 of the start's):
    state reduction matches 50-digit arithmetic; the as-run formula is already off (0.2% here; on
    the DHAM model after the A3 excursion it gave -1e18 ns, see local_checks/a3_run001_excursion)"""
    F = np.concatenate([np.zeros(10), np.linspace(0, 40, 20)])
    M = birth_death(F)
    s, e, dt = 2, 29, 2e-6
    ref = mpmath_mfpt(M, s, e, dt)
    sr = mfpt_state_reduction(M, s, e, dt)
    assert sr == pytest.approx(ref, rel=1e-10)
    leg = legacy_mfpt(M, s, e, dt)
    assert abs(leg - ref) > 1e3 * abs(sr - ref)


def test_stationary_gth_equals_eig_when_well_conditioned():
    from util import stationary_gth
    rng = np.random.default_rng(7)
    F = np.cumsum(rng.normal(0, 0.4, 30))
    M = birth_death(F)
    msm = MSM(); msm.build_MSM_from_M(M, dim=1); msm._compute_peq_fes_M()
    exact = np.exp(-F/0.5981); exact /= exact.sum()
    # The analytic Metropolis stationary distribution is a stronger reference
    # than a LAPACK eig vector (SciPy versions differ at ~4e-13 absolute here).
    np.testing.assert_allclose(stationary_gth(M), exact, rtol=1e-12, atol=0)
    np.testing.assert_allclose(stationary_gth(M), np.real(msm.peq), rtol=1e-9, atol=1e-12)


def test_start_equal_to_target_keeps_the_as_run_objective():
    """when the run sits on the frontier state it is aimed at, the as-run formula returns the mean
    recurrence time dt / peq[start]; the stable path must return the same objective, not 0 or a failure"""
    rng = np.random.default_rng(8)
    M = birth_death(np.cumsum(rng.normal(0, 0.3, 25))).astype(np.longdouble)
    wi = np.arange(30, 55)
    out = {}
    for method in ("legacy_diag", "state_reduction"):
        config.mfpt_method = method
        try:
            p, info = try_and_optim_M(M, wi, num_gaussian=10, start_index=54, end_index=54, qspace=config.qspace,
                                      rng=np.random.default_rng(1))
        finally:
            config.mfpt_method = "state_reduction"
        out[method] = info
    assert out["state_reduction"]["n_nonfinite_mfpt"] == 0
    assert out["state_reduction"]["best_random_mfpt_ns"] == pytest.approx(out["legacy_diag"]["best_random_mfpt_ns"], rel=1e-6)


def test_optimiser_never_ranks_a_failed_evaluation_best():
    """on the ill-conditioned chain the optimiser returns a finite positive model MFPT and a bias
    that is not just a random start"""
    F = np.concatenate([np.zeros(10), np.linspace(0, 25, 20)])
    M = birth_death(F).astype(np.longdouble)
    wi = np.arange(30, 60)
    p, info = try_and_optim_M(M, wi, num_gaussian=10, start_index=32, end_index=59, qspace=config.qspace,
                              rng=np.random.default_rng(4))
    assert info["mfpt_method"] == "state_reduction"
    assert np.isfinite(info["optimised_mfpt_ns"]) and info["optimised_mfpt_ns"] > 0
    assert info["n_nonfinite_mfpt"] == 0
    assert not np.allclose(p[:10], config.random_start_amp)
