"""Directed finite-chain MFPT safety, 22 October 2026.

No reversibility, count averaging, artificial edges, or finite penalty for an
unreachable target. Input to row_mfpt is row-stochastic. Equal endpoints use
first RETURN (at least one step), preserving the historical KOMBI objective.
"""
import numpy as np
from scipy.optimize import minimize as _scipy_minimize

def _reachable(P, start, stop=None):
    """Reachability in a ROW graph; retain every positive edge, however small."""
    seen = np.zeros(len(P), dtype=bool)
    seen[start] = True
    todo = [int(start)]
    while todo:
        i = todo.pop()
        if i == stop:
            continue
        for j in np.flatnonzero(P[i] > 0):
            if not seen[j]:
                seen[j] = True
                todo.append(int(j))
    return seen


def mean_return_time(M, state, time_step):
    """First return (>=1 step), not zero-time hitting, on a directed chain.

    Restrict GTH to the closed communicating class reachable from this state.
    Transient states elsewhere have exactly zero stationary mass and cannot
    affect the return. If escape without return has positive probability, the
    unconditional return time is infinite. No edge or pseudocount is added.
    """
    P = np.asarray(M, dtype=float).T
    if not np.isfinite(P).all() or np.any(P < 0):
        return np.inf
    forward = _reachable(P, state)
    backward = _reachable(P.T, state)
    if np.any(forward & ~backward):
        return np.inf
    idx = np.flatnonzero(forward)
    block = np.asarray(M)[np.ix_(idx, idx)]
    if not np.allclose(block.sum(axis=0), 1., rtol=0., atol=1e-10):
        return np.inf
    pi = stationary_gth(block)
    probability = pi[np.flatnonzero(idx == state)[0]]
    return float(time_step / probability) if probability > 0 else np.inf

def stationary_gth(M):
    """stationary distribution of the column-stochastic M by Grassmann-Taksar-Heyman state reduction,
    which keeps full relative accuracy when the probabilities span many orders of magnitude.
    Copied from the October 2026 audit (kombi_audit_20261011/kombi_core.py: stationary_gth)."""
    A = np.array(M.T, dtype=np.float64)           # row-stochastic
    n = A.shape[0]
    for k in range(n - 1, 0, -1):
        s = A[k, :k].sum()                        # > 0 for an irreducible chain; else NaN, counted as a failure
        A[:k, k] /= s
        A[:k, :k] += np.outer(A[:k, k], A[k, :k])
    r = np.zeros(n); r[0] = 1.0
    for k in range(1, n):
        r[k] = r[:k] @ A[:k, k]
    return r / r.sum()

def mfpt_state_reduction(M, start, end, time_step):
    """MFPT start -> end of the column-stochastic chain M by stable state reduction (censoring,
    GTH type): every other state is eliminated in turn, and only sums of positive numbers are formed
    (a state's escape probability is the sum of its off-diagonal entries, never 1 - P_kk), so the
    result keeps full relative accuracy however long the MFPT. Copied from the October 2026 audit
    (kombi_audit_20261011/kombi_core.py: mfpt_absorbing), where it matched 50-digit arithmetic to
    ~1e-16 on models for which the fundamental-matrix formula of MSM._build_mfpt_matrix_M failed.
    That formula divides by the target's stationary probability, which eig cannot resolve below
    ~1e-16 of the largest: after an excursion into barely sampled states it returned MFPTs of
    -1e18 ns and the optimiser kept a random start (pilot run A3_L20ps_run001, 12 Oct 2026)."""
    A = np.array(M.T, dtype=np.float64)            # row-stochastic
    n = A.shape[0]
    if start == end:
        return 0.0
    # Ignore only states that cannot be visited before hitting the target.
    # If a reachable branch can avoid it forever, the unconditional MFPT is
    # genuinely infinite; skipping that closed class would give a false finite result.
    forward = _reachable(A, start, stop=end)
    backward = _reachable(A.T, end)
    if np.any(forward & ~backward):
        return np.inf
    if not np.all(forward):
        idx = np.flatnonzero(forward)
        start = int(np.flatnonzero(idx == start)[0])
        end = int(np.flatnonzero(idx == end)[0])
        A = A[np.ix_(idx, idx)]
        n = len(idx)
    tau = np.ones(n)
    alive = np.ones(n, bool)
    np.fill_diagonal(A, 0.0)                       # only off-diagonal entries are used
    for k in range(n):
        if k in (start, end):
            continue
        alive[k] = False
        idx = np.flatnonzero(alive)
        s_k = A[k, idx].sum()
        if s_k <= 0:                               # isolated state: never re-entered
            continue
        col = A[idx, k].copy()
        A[np.ix_(idx, idx)] += np.outer(col, A[k, idx] / s_k)
        tau[idx] += col * tau[k] / s_k
        A[idx, k] = 0.0
    A[start, start] = 0.0
    if A[start, end] <= 0:
        return np.inf
    return float(tau[start] / A[start, end]) * time_step


def row_mfpt(P, start, target, time_step=1.0, recurrence=True):
    P = np.asarray(P, dtype=float)
    if (P.ndim != 2 or P.shape[0] != P.shape[1] or not np.isfinite(P).all()
            or np.any(P < 0) or not np.allclose(P.sum(1), 1., rtol=0, atol=1e-10)):
        raise ValueError("Expected a finite, nonnegative row-stochastic matrix")
    if not (0 <= start < len(P) and 0 <= target < len(P)):
        raise IndexError("MFPT endpoint outside working model")
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("Lag time must be finite and positive")
    value = (mean_return_time(P.T, start, time_step)
             if recurrence and start == target
             else mfpt_state_reduction(P.T, start, target, time_step))
    if value < 0 or np.isnan(value):
        return np.inf
    return value


def checked_minimize(fun, x0, args=(), **kwargs):
    """Keep a finite in-bounds initial candidate if local optimization fails.

    No random replacement, clipping, new force bound, or connectivity repair.
    A genuinely infinite initial objective stops this decision before MD.
    """
    if x0 is None:
        raise RuntimeError("No finite positive MFPT candidate on the directed graph")
    x0 = np.asarray(x0, dtype=float)
    bounds = kwargs.get("bounds")
    def valid(x):
        if np.shape(x) != x0.shape or not np.isfinite(x).all():
            return False
        if bounds is not None:
            if len(bounds) != len(x):
                return False
            for v, (lo, hi) in zip(x, bounds):
                if (lo is not None and v < lo) or (hi is not None and v > hi):
                    return False
        return True
    if not valid(x0):
        raise RuntimeError("Initial candidate is outside the declared bias bounds")
    baseline = float(fun(x0, *args))
    if not (np.isfinite(baseline) and baseline > 0):
        raise RuntimeError("No finite positive MFPT candidate; refusing to apply bias")
    result = _scipy_minimize(fun, x0, args=args, **kwargs)
    value = float(fun(result.x, *args)) if valid(result.x) else np.inf
    if not (np.isfinite(value) and value > 0):
        result.x = x0.copy()
        result.fun = baseline
        result.success = False
        result.message = str(result.message) + "; retained verified finite initial candidate"
        result.selected_candidate_source = "finite_initial_fallback"
    else:
        result.fun = value
        result.selected_candidate_source = "local_optimizer"
    return result
