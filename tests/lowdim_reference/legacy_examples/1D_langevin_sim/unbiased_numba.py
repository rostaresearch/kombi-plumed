#unbiased first-passage reference for the 1D model potential, with numba.
#the OpenMM driver (langevin_sim_unbiased.py) is too slow for first passages of ~1e10-1e11 steps,
#so this integrates the same dynamics directly:
#  potential: util.apply_fes(mode="multiwell", amp=config.amp), i.e. the flat term, 9 Gaussian
#             hills and the two sigmoid walls (kJ/mol, nm). y and z only feel their harmonic
#             restraints, which are independent of x, so only x is integrated.
#  integrator: OpenMM's LangevinIntegrator (leapfrog) update,
#             v <- a v + (1 - a) F/(m gamma) + sqrt(kT (1 - a^2)/m) xi,  x <- x + v dt,  a = exp(-gamma dt),
#             m = 12 amu, gamma = 1/ps, dt = 2 fs, T = 300 K, as in the drivers.
#  start:     the energy minimum reached from config.start_state (the drivers minimise first),
#             velocity drawn at T.
#  target:    |x - end_state| < config.target_radius, checked every step (as KOMBI).
#usage: python unbiased_numba.py <n_runs> <first_seed> <processes> [max_steps]
#writes first_passage_unbiased.csv (same columns as the drivers; n_propagations = seed).
import csv
import os
import sys
import time

import numba
import numpy as np

KJ = 4.184
AMP = 6.0                                   #config.amp
A = np.array([0.9, 0.3, 0.7, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * AMP * KJ #util.apply_fes, kJ/mol
X0 = np.array([1.12, 1, 3, 4.15, 4, 5.27, 4.75, 6, 1])
SX = np.array([0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5])
WALL, K, OFF = 100.0, 5.0, 0.4              #walls: 100/(1+exp(5(x-(1-0.4)))) and 100/(1+exp(-5(x-(2pi+0.4))))
FLAT = AMP * KJ
MASS, GAMMA, DT, T = 12.0, 1.0, 0.002, 300.0 #amu, 1/ps, ps, K
KT = 0.00831446261815324 * T                #kJ/mol
START, TARGET, RADIUS = 2.0, 5.2, 0.1       #config.start_state, end_state, target_radius (nm)


@numba.njit
def energy(x):
    e = FLAT
    for i in range(A.shape[0]):
        e += A[i] * np.exp(-(x - X0[i]) ** 2 / (2 * SX[i] ** 2))
    e += WALL / (1 + np.exp(K * (x - (1 - OFF))))
    e += WALL / (1 + np.exp(-K * (x - (2 * np.pi + OFF))))
    return e


@numba.njit
def force(x):
    f = 0.0
    for i in range(A.shape[0]):
        f += A[i] * np.exp(-(x - X0[i]) ** 2 / (2 * SX[i] ** 2)) * (x - X0[i]) / SX[i] ** 2
    el = np.exp(K * (x - (1 - OFF)))
    f += WALL * K * el / (1 + el) ** 2
    er = np.exp(-K * (x - (2 * np.pi + OFF)))
    f -= WALL * K * er / (1 + er) ** 2
    return f


@numba.njit
def first_passage(x, v, seed, max_steps, target, radius, check_every):
    np.random.seed(seed)
    a = np.exp(-GAMMA * DT)
    fs = (1 - a) / GAMMA
    ns = np.sqrt(KT * (1 - a * a) / MASS)
    for step in range(1, max_steps + 1):
        v = a * v + fs * force(x) / MASS + ns * np.random.standard_normal()
        x = x + v * DT
        if step % check_every == 0 and abs(x - target) < radius:
            return step
    return -1


def minimum_from(x):
    """steepest descent with a backtracking step, to the local minimum (the drivers minimise)."""
    h = 1e-3
    for _ in range(200000):
        f = force(x)
        if abs(f) < 1e-6:
            break
        xn = x + h * f
        if energy(xn) < energy(x):
            x, h = xn, h * 1.2
        else:
            h *= 0.5
    return x


def one(seed, max_steps=int(2e12), target=TARGET, radius=RADIUS, check_every=1):
    rng = np.random.default_rng(seed)
    x0 = minimum_from(START)
    v0 = rng.normal(0, np.sqrt(KT / MASS))
    t = time.time()
    steps = first_passage(x0, v0, seed, max_steps, target, radius, check_every)
    return seed, steps, time.time() - t


if __name__ == "__main__":
    from functools import partial
    from multiprocessing import Pool
    n, seed0, procs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    max_steps = int(float(sys.argv[4])) if len(sys.argv) > 4 else int(2e12)
    first_passage(1.0, 0.0, 0, 10, TARGET, RADIUS, 1)  #compile once before forking
    new = not os.path.exists("first_passage_unbiased.csv")
    with Pool(procs) as pool, open("first_passage_unbiased.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_tag", "n_propagations", "steps", "time_ps", "reached"])
        for seed, steps, wall in pool.imap_unordered(partial(one, max_steps=max_steps), [seed0 + k for k in range(n)]):
            reached = steps > 0
            s = steps if reached else max_steps
            w.writerow([f"numba_seed{seed}", seed, s, s * DT, int(reached)])
            f.flush()
            print(f"seed {seed}: steps {s} ({s * DT / 1e6:.3f} us) reached {reached} wall {wall:.0f} s", flush=True)
