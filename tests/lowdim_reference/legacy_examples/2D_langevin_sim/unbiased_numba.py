#unbiased first-passage reference for the 2D model potential, with numba.
#the OpenMM driver (langevin_sim_unbiased.py) is too slow for long first passages, so this
#integrates the same dynamics directly:
#  potential: util.apply_fes(mode="multiwell", amp=config.amp) as fixed: the flat term, 9 Gaussian
#             wells, the barrier (in kJ/mol) and the four sigmoid walls centred 0.7 nm outside
#             [0, 2pi] (kJ/mol, nm). z only feels its harmonic restraint, independent of x and y,
#             so only x and y are integrated.
#  integrator: OpenMM's LangevinIntegrator (leapfrog) update, per coordinate,
#             v <- a v + (1 - a) F/(m gamma) + sqrt(kT (1 - a^2)/m) xi,  r <- r + v dt,  a = exp(-gamma dt),
#             m = 12 amu, gamma = 1/ps, dt = 2 fs, T = 300 K, as in the drivers.
#  start:     the energy minimum reached from config.start_state (the drivers minimise first),
#             velocities drawn at T.
#  target:    |(x, y) - end_state| < config.target_radius, checked every step (as KOMBI).
#usage: python unbiased_numba.py <n_runs> <first_seed> <processes> [max_steps]
#writes first_passage_unbiased.csv (same columns as the drivers; n_propagations = seed).
import csv
import os
import sys
import time

import numba
import numpy as np

KJ = 4.184
AMP = 6.0                                    #config.amp
#wells (negative) and the barrier (positive), util.apply_fes, kJ/mol
C = np.concatenate((-np.array([0.9, 0.3, 0.5, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * AMP * KJ, [0.3 * AMP * KJ]))
X0 = np.array([1.12, 1, 3, 4.15, 4, 5.27, 5.5, 6, 1, np.pi])
Y0 = np.array([1.34, 2.25, 2.31, 3.62, 5, 4.14, 4.5, 1.52, 5, np.pi])
SX = np.array([0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5, 3])
SY = np.array([0.5, 0.3, 1, 0.8, 0.2, 0.3, 1, 0.6, 0.7, 0.3])
WALL, K, OFF = 100.0, 5.0, 0.7
FLAT = AMP * KJ
MASS, GAMMA, DT, T = 12.0, 1.0, 0.002, 300.0
KT = 0.00831446261815324 * T
START = np.array([5.0, 4.0])
TARGET = np.array([1.0, 1.5])
RADIUS = 0.1


@numba.njit
def energy(x, y):
    e = FLAT
    for i in range(C.shape[0]):
        e += C[i] * np.exp(-(x - X0[i]) ** 2 / (2 * SX[i] ** 2) - (y - Y0[i]) ** 2 / (2 * SY[i] ** 2))
    for r in (x, y):
        e += WALL / (1 + np.exp(K * (r + OFF)))
        e += WALL / (1 + np.exp(-K * (r - (2 * np.pi + OFF))))
    return e


@numba.njit
def wall_force(r):
    el = np.exp(K * (r + OFF))
    er = np.exp(-K * (r - (2 * np.pi + OFF)))
    return WALL * K * el / (1 + el) ** 2 - WALL * K * er / (1 + er) ** 2


@numba.njit
def force(x, y):
    fx = wall_force(x)
    fy = wall_force(y)
    for i in range(C.shape[0]):
        g = C[i] * np.exp(-(x - X0[i]) ** 2 / (2 * SX[i] ** 2) - (y - Y0[i]) ** 2 / (2 * SY[i] ** 2))
        fx += g * (x - X0[i]) / SX[i] ** 2
        fy += g * (y - Y0[i]) / SY[i] ** 2
    return fx, fy


@numba.njit
def first_passage(x, y, vx, vy, seed, max_steps, tx, ty, radius, check_every):
    np.random.seed(seed)
    a = np.exp(-GAMMA * DT)
    fs = (1 - a) / GAMMA
    ns = np.sqrt(KT * (1 - a * a) / MASS)
    r2 = radius * radius
    for step in range(1, max_steps + 1):
        fx, fy = force(x, y)
        vx = a * vx + fs * fx / MASS + ns * np.random.standard_normal()
        vy = a * vy + fs * fy / MASS + ns * np.random.standard_normal()
        x = x + vx * DT
        y = y + vy * DT
        if step % check_every == 0 and (x - tx) ** 2 + (y - ty) ** 2 < r2:
            return step
    return -1


def minimum_from(p):
    """steepest descent with a backtracking step, to the local minimum (the drivers minimise)."""
    x, y = float(p[0]), float(p[1])
    h = 1e-3
    for _ in range(400000):
        fx, fy = force(x, y)
        if np.hypot(fx, fy) < 1e-6:
            break
        xn, yn = x + h * fx, y + h * fy
        if energy(xn, yn) < energy(x, y):
            x, y, h = xn, yn, h * 1.2
        else:
            h *= 0.5
    return x, y


def one(seed, max_steps=int(2e12), target=TARGET, radius=RADIUS, check_every=1):
    rng = np.random.default_rng(seed)
    x0, y0 = minimum_from(START)
    vx, vy = rng.normal(0, np.sqrt(KT / MASS), 2)
    t = time.time()
    steps = first_passage(x0, y0, vx, vy, seed, max_steps, target[0], target[1], radius, check_every)
    return seed, steps, time.time() - t


if __name__ == "__main__":
    from functools import partial
    from multiprocessing import Pool
    n, seed0, procs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    max_steps = int(float(sys.argv[4])) if len(sys.argv) > 4 else int(2e12)
    first_passage(1.0, 1.0, 0.0, 0.0, 0, 10, TARGET[0], TARGET[1], RADIUS, 1)  #compile once before forking
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
