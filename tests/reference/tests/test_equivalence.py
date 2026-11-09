"""The vectorised DHAM counting/matrix and MSM tilt/MFPT give the same numbers as the legacy loops."""
import numpy as np
import pytest

import config
import legacy_reference as ref
from dham import DHAM, count_transitions, count_transitions_one
from MSM import MSM

QS = config.qspace
NB = config.DHAM_num_bins
KBT = 0.001987204259 * 300


def random_walk_bins(rng, n_rows, n_steps, start=33):
    """bin trajectories that wander like the per-step record (mostly stay, +-1 moves, some exits to 100)"""
    b = np.empty((n_rows, n_steps), dtype=np.int64)
    for k in range(n_rows):
        x = start + rng.integers(-3, 4)
        for i in range(n_steps):
            x = int(np.clip(x + rng.choice([-1, 0, 0, 0, 1]), 25, 100))
            b[k, i] = x
    return b


def random_params(rng, n_props):
    n = config.num_gaussian
    out = np.empty((n_props, n, 3))
    for k in range(n_props):
        out[k, :, 0] = rng.uniform(0.1, 1.0, n)
        out[k, :, 1] = rng.uniform(1.0, 8.0, n)
        out[k, :, 2] = rng.uniform(0.7, 2.0, n)
    return out


@pytest.mark.parametrize("lag", [1, 3])
def test_count_transitions(lag):
    rng = np.random.default_rng(1)
    b = random_walk_bins(rng, 4, 3000)
    s_ref, t_ref = ref.count_transitions(b, NB, lag)
    s_new, t_new = count_transitions(b, NB, lag)
    assert np.array_equal(s_ref, s_new) and np.array_equal(t_ref, t_new)
    Ntr = np.stack([count_transitions_one(b[k], NB, lag) for k in range(b.shape[0])])
    s_c, t_c = count_transitions(None, NB, lag, Ntr=Ntr)
    assert np.array_equal(s_ref, s_c) and np.array_equal(t_ref, t_c)


def test_build_MM_matches_loops():
    rng = np.random.default_rng(2)
    b = random_walk_bins(rng, 5, 4000)
    sumtr, trvec = ref.count_transitions(b, NB, 1)
    gp = random_params(rng, b.shape[0])
    d = DHAM(gp, NB); d.qspace = QS; d.KbT = KBT
    MM_ref = ref.build_MM(sumtr, trvec, gp, QS, KBT)
    MM_new = d.build_MM(sumtr, trvec, biased=True)
    assert np.isfinite(MM_ref).all()
    np.testing.assert_allclose(np.asarray(MM_new, float), np.asarray(MM_ref, float), rtol=1e-12, atol=0)
    assert d.diagnostics["zero_sump_with_counts"] == 0


def test_build_MM_guard_for_states_without_outgoing_counts():
    """A terminal arrival must not manufacture outgoing or reversed evidence."""
    b = np.array([[33, 33, 34, 34, 33, 33, 35]])   # 35 entered at the last step only
    sumtr, trvec = ref.count_transitions(b, NB, 1)
    gp = random_params(np.random.default_rng(3), 1)
    d = DHAM(gp, NB); d.qspace = QS; d.KbT = KBT
    MM = d.build_MM(sumtr, trvec, biased=True)
    assert np.isfinite(np.asarray(MM, float)).all()
    assert d.diagnostics["zero_sump_with_counts"] == 0
    assert np.all(MM[35] == 0)
    assert MM[33, 35] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        MM_ref = ref.build_MM(sumtr, trvec, gp, QS, KBT)
    assert np.isfinite(np.asarray(MM_ref, float)).all()


def random_column_stochastic(rng, n):
    """tridiagonal-plus-diagonal column-stochastic matrix like a per-step DHAM model"""
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = rng.uniform(0.8, 0.98)
        if i > 0: M[i - 1, i] = rng.uniform(0.0, 0.1)
        if i < n - 1: M[i + 1, i] = rng.uniform(0.0, 0.1)
    return M / M.sum(axis=0)


@pytest.mark.parametrize("dtype", [np.float64, np.longdouble])
def test_bias_M_direct_matches_loops(dtype):
    rng = np.random.default_rng(4)
    M = random_column_stochastic(rng, 40).astype(dtype)
    M[:, 7] = 0   # an empty column, as the working matrix can have after truncation elsewhere
    bias = rng.uniform(0, 10, 40)
    msm = MSM(); msm.build_MSM_from_M(M, dim=1)
    msm._bias_M(bias, method="direct_bias")
    np.testing.assert_allclose(np.asarray(msm.M, float), np.asarray(ref.bias_M_direct(M, bias), float), rtol=1e-13, atol=1e-300)


def test_mfpt_matrix_matches_loops():
    rng = np.random.default_rng(5)
    M = random_column_stochastic(rng, 30)
    msm = MSM(); msm.build_MSM_from_M(M, dim=1, time_step=2e-6)
    msm._compute_peq_fes_M()
    msm._build_mfpt_matrix_M()
    m_ref = ref.mfpt_matrix_diag(M, msm.peq, 2e-6)
    assert np.array_equal(np.isinf(m_ref), np.isinf(msm.mfpts))
    fin = np.isfinite(m_ref)
    np.testing.assert_allclose(msm.mfpts[fin], m_ref[fin], rtol=1e-12)


def test_dham_run_end_to_end():
    """DHAM.run on raw data equals the legacy count + build_MM + eigenvector pipeline"""
    rng = np.random.default_rng(6)
    b = random_walk_bins(rng, 3, 5000)
    s = QS[b - 1] + 0.5 * (QS[1] - QS[0])   # digitize(s) == b: bin b is [QS[b-1], QS[b]); bin 100 is s >= 8
    assert np.array_equal(np.digitize(s, QS), b)
    gp = random_params(rng, 3)
    d = DHAM(gp, NB); d.setup(s, 300, 2, "t"); d.lagtime = 1
    _, F_new, M_new = d.run(biased=True)
    bb = np.digitize(s, QS)
    sumtr, trvec = ref.count_transitions(bb, NB, 1)
    MM_ref = ref.build_MM(sumtr, trvec, gp, QS, KBT)
    msm = MSM(); msm.M = MM_ref.T; msm._compute_peq_fes_M()
    np.testing.assert_allclose(np.asarray(M_new, float), np.asarray(MM_ref.T, float), rtol=1e-12, atol=0)
    fin = np.isfinite(np.real(msm.free_energy))
    np.testing.assert_allclose(np.real(F_new)[fin], np.real(msm.free_energy)[fin], rtol=1e-10, atol=1e-10)
