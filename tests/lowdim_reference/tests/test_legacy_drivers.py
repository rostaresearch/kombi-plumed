"""Regression tests for the legacy KOMBI drivers used for the paper's NaCl, 1D and 2D examples
(legacy_examples/1D_NaCl_TW, 1D_langevin_sim, 2D_langevin_sim).

These scripts are not part of the kombi package. Their util.py and dham.py (and, where needed,
the driver) are loaded by path, with a stub ``config`` module, for the duration of each test.

Faults covered (all were in the code that produced the published numbers):
- the optimiser and DHAM used a Gaussian sqrt(2) wider than the bias applied in OpenMM;
- DHAM read the stored [a, b, c] parameter blocks as scrambled (a, b, c) rows;
- the NaCl and 1D optimisers received DHAM's matrix transposed (column-stochastic) and treated
  it as row-stochastic;
- np.exp(-u_ij / 2*kT) evaluates as exp(-u_ij * kT / 2);
- run times counted the last propagation in full and used the last visit to the target;
- the 2D DHAM dropped the amplitude and had the exponent sign reversed (it doubled the bias);
- the 2D local target was ravelled in the other index order; the 2D walls and barrier.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")  # util.py of the 1D/2D drivers imports openmm

LEGACY = Path(__file__).resolve().parents[1] / "legacy_examples"
KT = 0.5981  # kcal/mol, as hard-coded in the optimisers
KT_DHAM = 0.001987204259 * 300


def _load(monkeypatch, sysdir: str, modules=("util", "dham"), **cfg):
    shim = types.ModuleType("config")
    shim.__dict__.update(dict(num_bins=100, num_gaussian=10, propagation_step=5000, dcdfreq=500,
                              dcdfreq_mfpt=1, T=300, amp=6, pbc=False, platform=None,
                              nonbonded_method="NoCutoff", box_nm=None, target_distance=8.0,
                              target_radius=0.1, fpt_stride=10,
                              stepsize=0.002 * openmm.unit.picoseconds,
                              seed=None, deterministic_forces=False, fine_every=0, ring_ps=0.0))
    shim.__dict__.update(cfg)
    monkeypatch.setitem(sys.modules, "config", shim)
    out = {}
    for name in ("mfpt_safety",) + modules:
        spec = importlib.util.spec_from_file_location(name, LEGACY / sysdir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        out[name] = mod
    return types.SimpleNamespace(**out)


@pytest.fixture
def nacl(monkeypatch):
    return _load(monkeypatch, "1D_NaCl_TW")


@pytest.fixture
def one_d(monkeypatch):
    return _load(monkeypatch, "1D_langevin_sim", num_gaussian=20)


@pytest.fixture
def two_d(monkeypatch):
    return _load(monkeypatch, "2D_langevin_sim", num_bins=20, num_gaussian=20)


def _nn_chain(F: np.ndarray, kT: float, scale: float = 0.1) -> np.ndarray:
    """Row-stochastic nearest-neighbour chain, P[i, j] = P(i -> j), stationary exp(-F/kT)."""
    n = F.size
    P = np.zeros((n, n))
    for i in range(n - 1):
        P[i, i + 1] = scale * np.exp(-(F[i + 1] - F[i]) / (2 * kT))
        P[i + 1, i] = scale * np.exp(-(F[i] - F[i + 1]) / (2 * kT))
    P[np.diag_indices(n)] = 1 - P.sum(1)
    return P


def _stationary(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(P.T)
    p = np.real(v[:, np.argmax(np.real(w))])
    return p / p.sum()


def _openmm_energy(expression: str, globals_: dict, x_nm: float, y_nm: float = 0.0) -> float:
    """Energy (kJ/mol) of one CustomExternalForce on one particle, Reference platform."""
    system = openmm.System()
    system.addParticle(12.0)
    force = openmm.CustomExternalForce(expression)
    for k, v in globals_.items():
        force.addGlobalParameter(k, v)
    force.addParticle(0)
    system.addForce(force)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([openmm.Vec3(x_nm, y_nm, 0.0)])
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)


# ------------------------------------------------------------------------------------------
# sqrt(2): the Gaussian of the optimiser and DHAM is the one OpenMM applies
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["nacl", "one_d"])
def test_gaussian_matches_openmm_bias(fixture, request) -> None:
    util = request.getfixturevalue(fixture).util
    a, b, c = 1.3, 3.0, 0.4  # kcal/mol, length units of the CV
    for x in (2.6, 3.0, 3.4, 3.9):
        e_openmm = _openmm_energy("A*exp(-(x-b)^2/(2*c^2))", {"A": a * 4.184, "b": b, "c": c}, x)
        assert util.gaussian(x, a, b, c) * 4.184 == pytest.approx(e_openmm, rel=1e-6)
    # one standard deviation away the Gaussian has fallen to exp(-1/2), not exp(-1/4)
    assert util.gaussian(b + c, a, b, c) == pytest.approx(a * np.exp(-0.5))


def test_1d_apply_bias_matches_util_gaussian(one_d) -> None:
    """The bias the 1D driver puts into OpenMM equals the sum the optimiser and DHAM use."""
    n = 20
    rng = np.random.default_rng(3)
    params = np.concatenate((rng.uniform(0.1, 1.2, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0.6, 2, n)))
    system = openmm.System()
    system.addParticle(12.0)
    one_d.util.apply_bias(system, 0, params, num_gaussians=n)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    rows = one_d.util.params_to_gaussians(params, n)
    for x in (0.7, 2.0, 3.3, 5.2):
        ctx.setPositions([openmm.Vec3(x, 0, 0)])
        e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        assert sum(one_d.util.gaussian(x, *r) for r in rows) * 4.184 == pytest.approx(e, rel=1e-6)


def test_nacl_add_bias_matches_util_gaussian(monkeypatch) -> None:
    """The bias mfpt_NaCl.add_bias puts into OpenMM (once, one bond) equals the sum the
    optimiser and DHAM use, for parameters stored as [a (kcal/mol), b (A), c (A)]."""
    pytest.importorskip("mdtraj")
    pytest.importorskip("PIL")
    m = _load(monkeypatch, "1D_NaCl_TW", modules=("util", "dham", "mfpt_NaCl"))
    rng = np.random.default_rng(7)
    params = np.concatenate((rng.uniform(0.1, 2, 10), rng.uniform(2.5, 8, 10), rng.uniform(0.3, 1.5, 10)))
    system = openmm.System()
    system.addParticle(23.0)
    system.addParticle(35.45)
    m.mfpt_NaCl.add_bias(system, params)
    bias = system.getForce(system.getNumForces() - 1)
    assert bias.getNumBonds() == 1
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    rows = m.util.params_to_gaussians(params, 10)
    for r in (2.6, 3.5, 5.0, 7.9):  # A
        ctx.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(r / 10, 0, 0)])
        e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        assert sum(m.util.gaussian(r, *row) for row in rows) * 4.184 == pytest.approx(e, rel=1e-6)


def test_2d_dham_gaussian_keeps_amplitude_and_matches_openmm(two_d) -> None:
    g = (2.5, 1.0, 1.5, 0.5, 0.8)  # a, bx, by, cx, cy
    e = _openmm_energy("A*exp(-(x-bx)^2/(2*cx^2) - (y-by)^2/(2*cy^2))",
                       dict(A=g[0] * 4.184, bx=g[1], by=g[2], cx=g[3], cy=g[4]), 1.3, 1.1)
    assert two_d.dham.gaussian_2d(1.3, 1.1, *g) * 4.184 == pytest.approx(e, rel=1e-6)
    assert two_d.dham.gaussian_2d(1.3, 1.1, *g) == pytest.approx(2.5 / 1.0 * two_d.dham.gaussian_2d(1.3, 1.1, 1.0, *g[1:]))


# ------------------------------------------------------------------------------------------
# parameter layout
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["nacl", "one_d"])
def test_params_to_gaussians_rows(fixture, request) -> None:
    util = request.getfixturevalue(fixture).util
    a, b, c = np.arange(10) + 0.5, np.arange(10) + 20.0, np.arange(10) + 40.0
    rows = util.params_to_gaussians(np.concatenate((a, b, c)), 10)
    np.testing.assert_array_equal(rows, np.stack((a, b, c), axis=1))


def test_params_to_gaussians_2d_rows(two_d) -> None:
    blocks = [np.arange(20) + 100 * k for k in range(5)]
    rows = two_d.util.params_to_gaussians_2d(np.concatenate(blocks), 20)
    np.testing.assert_array_equal(rows, np.stack(blocks, axis=1))


@pytest.mark.parametrize("path, bad", [
    ("1D_NaCl_TW/mfpt_NaCl.py", r'.txt").reshape(-1, 3)'),
    ("1D_langevin_sim/langevin_sim_mfpt_opt.py", "= gaussian_params.reshape(1, config.num_gaussian, 3)"),
    ("1D_langevin_sim/langevin_sim_mfpt_opt.py", "(global_gaussian_params, gaussian_params.reshape(1"),
    ("2D_langevin_sim/langevin_sim_mfpt_opt.py", '.txt").reshape(-1,5)'),
])
def test_drivers_no_longer_reshape_raw_parameter_vector(path, bad) -> None:
    """REGRESSION GUARD: the drivers reshaped [a(n), b(n), c(n)] straight into (n, 3)."""
    assert bad not in (LEGACY / path).read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------
# matrix orientation: DHAM returns M[i, j] = P(i -> j), which is what the optimiser assumes
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["nacl", "one_d"])
def test_dham_returns_row_stochastic_matrix(fixture, request) -> None:
    mods = request.getfixturevalue(fixture)
    n = 100
    F = np.sin(np.linspace(0, 3 * np.pi, n)) + 0.3 * np.linspace(0, 1, n) ** 2
    P = _nn_chain(F, KT_DHAM, scale=0.2)
    pi = _stationary(P)
    C = pi[:, None] * P * 1e7  # expected counts of an unbiased run (symmetric: detailed balance)
    trvec = C.sum(1)[None, :]
    no_bias = np.zeros((1, 10, 3))
    no_bias[..., 2] = 1.0  # a = 0, width 1
    d = mods.dham.DHAM(no_bias, num_bins=n)
    d.qspace = np.linspace(2.0, 9, n + 1) if fixture == "nacl" else np.linspace(0, 2 * np.pi, n + 1)
    d.KbT = KT_DHAM
    MM = np.asarray(d.build_MM(C, trvec, biased=True), dtype=float)
    np.testing.assert_allclose(MM, P, atol=1e-12)
    assert not np.allclose(MM, P.T, atol=1e-6)
    src = inspect.getsource(mods.dham.DHAM.run)
    assert "return mU2, MM.T" not in src


@pytest.mark.parametrize("fixture", ["nacl", "one_d"])
def test_optimiser_sees_the_true_landscape(fixture, request) -> None:
    """With zero bias, the free energy the optimiser computes from DHAM's matrix is F, not -F."""
    util = request.getfixturevalue(fixture).util
    n = 60
    F = 1.5 * np.cos(np.linspace(0, 2 * np.pi, n)) + np.linspace(0, 1, n)
    P = _nn_chain(F, KT)
    Mb = util.bias_M_1D(P, np.zeros(n), kT=KT)
    peq = util.compute_free_energy(Mb.T, kT=KT)[0]
    np.testing.assert_allclose(-KT * np.log(peq) + KT * np.log(peq[0]), F - F[0], atol=1e-8)


# ------------------------------------------------------------------------------------------
# operator precedence in the bias tilt
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["nacl", "one_d"])
def test_bias_M_1D_gives_boltzmann_reweighted_stationary_distribution(fixture, request) -> None:
    """With the tilt exp(-(u_j - u_i)/(2kT)) (diagonal kept, rows renormalised) the biased
    stationary distribution is exactly pi_i exp(-u_i/kT) r_i, r_i the row sums before
    renormalising. The old exp(-u_ij / 2*kT) = exp(-u_ij*kT/2) fails this."""
    util = request.getfixturevalue(fixture).util
    n = 40
    F = np.cos(np.linspace(0, 2 * np.pi, n))
    u = 2.0 * np.exp(-((np.arange(n) - 15) / 5.0) ** 2)
    P = _nn_chain(F, KT)
    Mb = util.bias_M_1D(P, u, kT=KT)
    r = (P * np.exp(-(u[None, :] - u[:, None]) / (2 * KT)))
    np.fill_diagonal(r, np.diag(P))
    expected = _stationary(P) * np.exp(-u / KT) * r.sum(1)
    np.testing.assert_allclose(_stationary(Mb), expected / expected.sum(), rtol=1e-9)


def test_2d_bias_M_matches_1d_tilt(two_d, one_d) -> None:
    rng = np.random.default_rng(5)
    M = rng.random((30, 30))
    M /= M.sum(1, keepdims=True)
    u = rng.uniform(0, 3, 30)
    np.testing.assert_allclose(two_d.util.bias_M(M, u, kT=KT), one_d.util.bias_M_1D(M, u, kT=KT), atol=1e-14)


@pytest.mark.parametrize("fixture", ["nacl", "one_d", "two_d"])
def test_bias_and_mfpt_accept_dham_longdouble(fixture, request) -> None:
    """DHAM.build_MM returns np.longdouble, which np.linalg.inv rejects on some platforms."""
    util = request.getfixturevalue(fixture).util
    P = _nn_chain(np.cos(np.linspace(0, 3, 15)), KT).astype(np.longdouble)
    bias = util.bias_M_1D if fixture != "two_d" else util.bias_M
    Mb = bias(P, np.linspace(0, 1, 15), kT=KT)
    assert Mb.dtype == np.float64
    peq = util.compute_free_energy(Mb.T, kT=KT)[0]
    assert np.isfinite(util.Markov_mfpt_calc(peq, P)).all()


@pytest.mark.parametrize("path", ["1D_NaCl_TW/util.py", "1D_langevin_sim/util.py", "2D_langevin_sim/util.py"])
def test_no_unparenthesised_kT_divisions(path) -> None:
    """REGRESSION GUARD: np.exp(-u_ij / 2*kT) and np.exp(-u_ij / 2*0.5981)."""
    code = [line.split("#")[0] for line in (LEGACY / path).read_text(encoding="utf-8").splitlines()]
    assert not any("/ 2*kT" in c or "/ 2*0.5981" in c for c in code)


# ------------------------------------------------------------------------------------------
# vectorised MFPT and DHAM against the original loops
# ------------------------------------------------------------------------------------------
def _mfpt_loops(peq, M):
    """Markov_mfpt_calc as it was (double loop)."""
    N = M.shape[0]
    Idn = np.eye(N)
    Qinv = np.linalg.inv(Idn + np.ones((N, 1)) @ peq.reshape(1, -1) - M)
    mfpt = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            term1 = Qinv[j, j] - Qinv[i, j] + Idn[i, j]
            mfpt[i, j] = 1e12 if peq[j] * term1 == 0 else term1 / peq[j]
    return mfpt


@pytest.mark.parametrize("fixture", ["nacl", "one_d", "two_d"])
def test_markov_mfpt_calc_matches_loops_and_absorbing_solve(fixture, request) -> None:
    util = request.getfixturevalue(fixture).util
    n = 25
    P = _nn_chain(np.sin(np.linspace(0, 4, n)), KT, scale=0.3)
    pi = _stationary(P)
    m = util.Markov_mfpt_calc(pi, P)
    np.testing.assert_allclose(m, _mfpt_loops(pi, P), rtol=1e-10)
    j = n - 3
    keep = [k for k in range(n) if k != j]
    t = np.linalg.solve(np.eye(n - 1) - P[np.ix_(keep, keep)], np.ones(n - 1))
    np.testing.assert_allclose(m[keep, j], t, rtol=1e-8)


def test_nacl_build_MM_matches_loop_reference(nacl) -> None:
    """Vectorised DHAM against the published loop (with the per-Gaussian rows it should get)."""
    rng = np.random.default_rng(11)
    n, K = 100, 3
    sumtr = rng.integers(0, 5, (n, n)).astype(float)
    sumtr = 0.5 * (sumtr + sumtr.T) * (rng.random((n, n)) < 0.05)
    trvec = rng.integers(0, 50, (K, n)).astype(float)
    gp = np.stack([np.stack((rng.uniform(0.1, 2, 10), rng.uniform(2, 9, 10), rng.uniform(0.3, 1.5, 10)), 1)
                   for _ in range(K)])
    d = nacl.dham.DHAM(gp, num_bins=n)
    d.qspace = np.linspace(2.0, 9, n + 1)
    d.KbT = KT_DHAM
    MM = np.asarray(d.build_MM(sumtr, trvec, biased=True), dtype=float)
    xs = nacl.util.state_centres(d.qspace)
    ref = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if sumtr[i, j] > 0:
                sump1 = 0.0
                for k in range(K):
                    u = sum(nacl.util.gaussian(xs, *gp[k, g]) for g in range(10))
                    if trvec[k, i] > 0:
                        sump1 += trvec[k, i] * np.exp(-(u[j] - u[i]) / (2 * KT_DHAM))
                if sump1 > 0:
                    ref[i, j] = sumtr[i, j] / sump1
    rs = ref.sum(1, keepdims=True)
    ref = np.divide(ref, rs, out=np.zeros_like(ref), where=rs > 0)
    np.testing.assert_allclose(MM, ref, rtol=1e-12, atol=1e-15)


# ------------------------------------------------------------------------------------------
# 2D DHAM: the bias is removed, not doubled
# ------------------------------------------------------------------------------------------
def test_2d_dham_removes_the_bias(two_d) -> None:
    N = 20
    iy = 10
    ix = np.arange(1, N)
    states = np.ravel_multi_index((ix, np.full_like(ix, iy)), (N, N), order="F")
    xc = two_d.dham.state_centres(N)
    F = 1.2 * np.cos(1.3 * xc[ix]) + 0.15 * xc[ix]
    g = np.array([1.5, np.pi, xc[iy], 0.8, 5.0])  # one Gaussian, a = 1.5 kcal/mol
    u = two_d.dham.gaussian_2d(xc[ix], xc[iy], *g)
    Pb = _nn_chain(F + u, KT_DHAM, scale=0.02)  # biased dynamics along one row of the grid
    C_row = _stationary(Pb)[:, None] * Pb * 1e8
    C = np.zeros((N * N, N * N))
    C[np.ix_(states, states)] = C_row
    trvec = C.sum(1)[None, :]
    d = two_d.dham.DHAM(g.reshape(1, 1, 5))
    d.KbT = KT_DHAM
    MM = np.asarray(d.build_MM(C, trvec, biased=True), dtype=float)
    Frec = -KT_DHAM * np.log(_stationary(MM[np.ix_(states, states)]))
    dF = (Frec - Frec.min()) - (F - F.min())
    slope = np.polyfit(u, dF, 1)[0]
    assert abs(slope) < 0.05, slope  # the old sign gave a slope of 2 (F + 2u)
    assert np.sqrt(np.mean((dF - dF.mean()) ** 2)) < 0.02


def test_2d_counts_are_from_to(two_d) -> None:
    b = np.array([[[0], [1], [1], [2]]])
    sumtr, trvec = two_d.dham.count_transitions(b, 3, 1)
    np.testing.assert_array_equal(trvec[0], [1, 2, 0])  # transitions out of states 0, 1, 2
    np.testing.assert_array_equal(sumtr, [[0, 1, 0], [0, 1, 1], [0, 0, 0]])


@pytest.mark.parametrize("fixture", ["nacl", "one_d", "two_d"])
def test_directed_counts_do_not_invent_reverse_edges(fixture, request):
    mods = request.getfixturevalue(fixture)
    b = np.array([[[0], [1], [1], [2]]])
    counts, occupancy = mods.dham.count_transitions(b, 3, 1)
    np.testing.assert_array_equal(counts, [[0, 1, 0], [0, 1, 1], [0, 0, 0]])
    np.testing.assert_array_equal(occupancy, [[1, 2, 0]])
    assert "use_symmetry" not in inspect.signature(mods.dham.count_transitions).parameters


@pytest.mark.parametrize("fixture", ["nacl", "one_d", "two_d"])
def test_directed_dham_recovers_nonreversible_tilted_model(fixture, request):
    """Exact one-window check; deliberately violates detailed balance.

    This tests the estimator's stated half-bias model, not its physical accuracy
    for arbitrary finite-lag MD transitions.
    """
    mods = request.getfixturevalue(fixture)
    if fixture == "two_d":
        d = mods.dham.DHAM(np.array([[[0.8, 1.4, 2.1, 0.9, 1.2]]]))
        n = d.numbins ** 2
        U = mods.dham.gaussian_2d(d.x, d.y, *d.gaussian_params[0, 0]).ravel()
    else:
        n = 7
        d = mods.dham.DHAM(np.array([[[0.8, 1.4, 0.9]]]), num_bins=n)
        d.qspace = np.linspace(0, 4, n+1)
        U = mods.util.gaussian(mods.util.state_centres(d.qspace)[:n], *d.gaussian_params[0, 0])
    d.KbT = KT_DHAM
    P = np.eye(n)*0.7
    P[np.arange(n), (np.arange(n)+1)%n] = 0.3
    Pb = P*np.exp(-(U[None, :]-U[:, None])/(2*KT_DHAM))
    Pb /= Pb.sum(1, keepdims=True)
    exposure = np.linspace(10, 100, n)
    C = exposure[:, None]*Pb
    recovered = np.asarray(d.build_MM(C, exposure[None, :], biased=True), dtype=float)
    np.testing.assert_allclose(recovered, P, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------------------------------
# 2D target index, walls and barrier
# ------------------------------------------------------------------------------------------
def test_2d_target_index_is_the_bin_of_the_target(monkeypatch) -> None:
    pytest.importorskip("mdtraj")
    m = _load(monkeypatch, "2D_langevin_sim", modules=("util", "dham", "langevin_sim_mfpt_opt"),
              num_bins=20, num_gaussian=20)
    N = 20
    target = np.array([[1.0, 1.5]])
    s = int(m.util.digitize_xy(target, N)[0])
    ix, iy = np.unravel_index(s, (N, N), order="F")
    grid = np.linspace(0, 2 * np.pi, N)
    assert grid[ix - 1] <= 1.0 < grid[ix] and grid[iy - 1] <= 1.5 < grid[iy]
    working = np.array([s, s + 1, s + N, s + 2 * N + 3])
    assert m.langevin_sim_mfpt_opt.find_closest_index(working, s, N) == s
    # the transposed bin is not chosen when the target bin itself is unvisited
    t_swapped = int(np.ravel_multi_index((iy, ix), (N, N), order="F"))
    near_target = int(np.ravel_multi_index((ix, iy + 1), (N, N), order="F"))
    assert m.langevin_sim_mfpt_opt.find_closest_index(np.array([t_swapped, near_target]), s, N) == near_target


def test_2d_walls_and_barrier(two_d) -> None:
    system = openmm.System()
    system.addParticle(12.0)
    system, _ = two_d.util.apply_fes(system, 0, mode="multiwell", amp=6, plot=False)
    forces = [system.getForce(k) for k in range(system.getNumForces())]
    left, bottom = forces[0], forces[2]
    for f, (x, y) in ((left, (0.0, 3.0)), (bottom, (3.0, 0.0))):
        e = _openmm_energy(f.getEnergyFunction(), {}, x, y)
        assert e == pytest.approx(100 / (1 + np.exp(5 * 0.7)), rel=1e-6)  # wall centred at -0.7 nm
    barrier = [f for f in forces if isinstance(f, openmm.CustomExternalForce)
               and any(f.getGlobalParameterName(k) == "A9" for k in range(f.getNumGlobalParameters()))][0]
    assert barrier.getGlobalParameterDefaultValue(0) == pytest.approx(0.3 * 6 * 4.184)


# ------------------------------------------------------------------------------------------
# first-passage accounting
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["nacl", "one_d", "two_d"])
def test_first_passage_steps_counts_to_the_first_frame(fixture, request) -> None:
    util = request.getfixturevalue(fixture).util
    L = 5
    record = np.array([2.6, 3, 4, 3, 2.9, 5, 6, 7, 7.5, 7.9, 6, 8.1, 7, 8.4, 9.0])  # per-step CV
    props = record.reshape(3, L)
    if fixture == "nacl":
        hits = [util.first_hit(p, 8.0) for p in props]
    else:
        hits = [util.first_hit_within(p.reshape(-1, 1), [8.25], 0.2) for p in props]
    p_hit = next(k for k, h in enumerate(hits) if h is not None)  # the driver stops here
    steps = util.first_passage_steps(p_hit, hits[p_hit], 1, L)
    first = int(np.flatnonzero(record >= 8.0)[0]) if fixture == "nacl" else \
        int(np.flatnonzero(np.abs(record - 8.25) < 0.2)[0])
    assert steps == first + 1  # steps from the start to the first frame in the target
    assert steps < (p_hit + 1) * L + hits[p_hit]  # the old formula: i_prop*L + reach


@pytest.mark.parametrize("path, bad", [
    ("1D_NaCl_TW/mfpt_NaCl.py", "total_steps = i*config.propagation_step + reach"),
    ("1D_langevin_sim/langevin_sim_mfpt_opt.py", "total_steps = i_prop * config.propagation_step + reach"),
    ("2D_langevin_sim/langevin_sim_mfpt_opt.py", "total_steps = i_prop * config.propagation_step + reach"),
])
def test_drivers_no_longer_count_the_last_propagation_in_full(path, bad) -> None:
    """REGRESSION GUARD for the csv time accounting."""
    assert bad not in (LEGACY / path).read_text(encoding="utf-8")


def test_record_first_passage_csv(nacl, tmp_path) -> None:
    f = tmp_path / "fp.csv"
    nacl.util.record_first_passage(str(f), "t1", 6, 25123, True, 0.002)
    nacl.util.record_first_passage(str(f), "t2", 150, 755000, False, 0.002)
    lines = f.read_text().splitlines()
    assert lines[0] == "time_tag,n_propagations,steps,time_ps,reached"
    assert lines[1].split(",")[2:] == ["25123", "50.246", "1"]
    assert lines[2].split(",")[-1] == "0"


# ------------------------------------------------------------------------------------------
# NaCl system: explicit nonbonded treatment
# ------------------------------------------------------------------------------------------
class _FakePsf:
    def __init__(self):
        self.box = None
        self.kwargs = None

    def setBox(self, a, b, c):  # noqa: N802 (OpenMM name)
        self.box = (a, b, c)

    def createSystem(self, params, **kwargs):  # noqa: N802
        self.kwargs = kwargs
        return kwargs


def test_nacl_distance_minimum_image(nacl, monkeypatch) -> None:
    pos = np.array([[0.1, 0.1, 0.1], [2.4, 0.1, 0.1]])  # nm
    assert nacl.util.nacl_distance(pos) == pytest.approx(23.0)  # no box: plain distance
    monkeypatch.setattr(sys.modules["config"], "nonbonded_method", "PME")
    monkeypatch.setattr(sys.modules["config"], "box_nm", 2.5)
    assert nacl.util.nacl_distance(pos) == pytest.approx(2.0)  # 2.3 nm apart = 0.2 nm through the boundary
    assert nacl.util.nacl_distance(openmm.unit.Quantity(pos, openmm.unit.nanometers)) == pytest.approx(2.0)


def test_nacl_add_bias_is_periodic_with_pme(monkeypatch) -> None:
    pytest.importorskip("mdtraj")
    pytest.importorskip("PIL")
    m = _load(monkeypatch, "1D_NaCl_TW", modules=("util", "dham", "mfpt_NaCl"),
              nonbonded_method="PME", box_nm=2.5)
    system = openmm.System()
    system.addParticle(23.0)
    system.addParticle(35.45)
    m.mfpt_NaCl.add_bias(system, np.concatenate((np.ones(10), np.linspace(3, 7, 10), np.full(10, 0.5))))
    assert system.getForce(system.getNumForces() - 1).usesPeriodicBoundaryConditions()


def test_nacl_create_system_sets_nonbonded_method(nacl, monkeypatch) -> None:
    import openmm.app as app

    psf = _FakePsf()
    nacl.util.create_nacl_system(psf, None)
    assert psf.kwargs["nonbondedMethod"] is app.NoCutoff and psf.box is None
    monkeypatch.setattr(sys.modules["config"], "nonbonded_method", "PME")
    monkeypatch.setattr(sys.modules["config"], "box_nm", 2.57)
    psf = _FakePsf()
    nacl.util.create_nacl_system(psf, None)
    assert psf.kwargs["nonbondedMethod"] is app.PME and psf.box is not None


# ------------------------------------------------------------------------------------------
# numba unbiased reference: same potential as the OpenMM system the drivers build
# ------------------------------------------------------------------------------------------
def _numba_module(sysdir):
    """Load unbiased_numba.py once per system and register it in sys.modules (numba's
    cache=True looks the module up by name)."""
    pytest.importorskip("numba")
    name = f"unbiased_numba_{sysdir}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, LEGACY / sysdir / "unbiased_numba.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _openmm_fes_context(util, extra=()):
    system = openmm.System()
    system.addParticle(12.0)
    system, _ = util.apply_fes(system, 0, mode="multiwell", amp=6, plot=False)
    for expr in extra:
        f = openmm.CustomExternalForce(expr)
        f.addParticle(0)
        system.addForce(f)
    return openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))


def test_numba_1d_matches_openmm_potential(one_d) -> None:
    nb = _numba_module("1D_langevin_sim")
    ctx = _openmm_fes_context(one_d.util, ("1e3 * y^2", "1e3 * z^2"))
    for x in np.linspace(-0.2, 2 * np.pi + 0.6, 37):
        ctx.setPositions([openmm.Vec3(x, 0, 0)])
        st = ctx.getState(getEnergy=True, getForces=True)
        e = st.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        fx = st.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)[0, 0]
        assert nb.energy(x) == pytest.approx(e, rel=1e-9, abs=1e-6)
        assert nb.force(x) == pytest.approx(fx, rel=1e-6, abs=1e-6)


def test_numba_2d_matches_openmm_potential(two_d) -> None:
    nb = _numba_module("2D_langevin_sim")
    ctx = _openmm_fes_context(two_d.util, ("1e3 * z^2",))
    rng = np.random.default_rng(3)
    for x, y in rng.uniform(-0.6, 2 * np.pi + 0.6, (40, 2)):
        ctx.setPositions([openmm.Vec3(x, y, 0)])
        st = ctx.getState(getEnergy=True, getForces=True)
        e = st.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        f = st.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)[0]
        assert nb.energy(x, y) == pytest.approx(e, rel=1e-9, abs=1e-6)
        assert nb.force(x, y) == pytest.approx((f[0], f[1]), rel=1e-6, abs=1e-6)


def test_numba_first_passage_counts_steps() -> None:
    nb = _numba_module("1D_langevin_sim")
    # a start inside the target is reported at the first check
    assert nb.first_passage(5.2, 0.0, 1, 100, 5.2, 0.1, 1) == 1
    assert nb.first_passage(2.0, 0.0, 1, 50, 5.2, 0.1, 1) == -1  # not reached in 50 steps


# ------------------------------------------------------------------------------------------
# reproducible, fine-sampled NaCl runs (18 Oct 2026)
# ------------------------------------------------------------------------------------------
def test_nacl_run_seeds_are_derived_deterministically(nacl) -> None:
    a, b = nacl.util.run_seeds(7), nacl.util.run_seeds(7)
    assert a == b and a["integrator_seed"] != a["velocity_seed"]
    assert nacl.util.run_seeds(8)["integrator_seed"] != a["integrator_seed"]
    assert 0 < a["integrator_seed"] < 2**31 and 0 < a["velocity_seed"] < 2**31
    assert nacl.util.run_seeds(None) is None


def test_nacl_optimiser_is_reproducible_with_a_seeded_generator(nacl) -> None:
    n = 12
    F = np.cos(np.linspace(0, 3, n))
    P = _nn_chain(F, KT, scale=0.3)
    work = np.arange(20, 20 + n)
    runs = [nacl.util.try_and_optim_M(P, work, num_gaussian=10, start_state=work[1], end_state=work[-1],
                                      rng=np.random.default_rng([5, 3])) for _ in range(2)]
    np.testing.assert_array_equal(runs[0], runs[1])


def test_nacl_fine_frames_ring_and_stream(nacl, tmp_path) -> None:
    mdtraj = pytest.importorskip("mdtraj")
    from mdtraj.formats import XTCTrajectoryFile
    rng = np.random.default_rng(0)
    frames = [(50 * (k + 1), rng.random((4, 3)).astype(np.float32)) for k in range(5)]
    for keep, expect in ((3, frames[2:]), (0, frames)):
        path = str(tmp_path / f"f{keep}.xtc")
        w = nacl.util.FineFrames(path, keep=keep)
        for step, xyz in frames:
            w.add(step, xyz, 2.5)
        w.add(frames[-1][0], frames[-1][1], 2.5)  # a repeated step is not written twice
        w.close()
        with XTCTrajectoryFile(path) as f:
            xyz, time, step, box = f.read()
        assert step.tolist() == [s for s, _ in expect]
        np.testing.assert_allclose(time, [s * 0.002 for s, _ in expect], rtol=1e-6)
        np.testing.assert_allclose(xyz, np.stack([x for _, x in expect]), atol=1e-3)
        np.testing.assert_allclose(box[:, 0, 0], 2.5, rtol=1e-6)


def test_nacl_write_run_info(nacl, tmp_path, monkeypatch) -> None:
    """write_run_info must run end to end (it failed on a missing import in the 18 Oct pilots)."""
    import json
    monkeypatch.setattr(sys.modules["config"], "seed", 11)
    system = openmm.System()
    system.addParticle(1.0)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    path = tmp_path / "info.json"
    nacl.util.write_run_info(str(path), ctx, {"method": "test"})
    info = json.loads(path.read_text())
    assert info["seeds"]["run_seed"] == 11 and info["platform"] == "Reference" and info["method"] == "test"
