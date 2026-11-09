#util file for applying mfpt method on 1-d NaCl system.
#by TW 26th July 2023
#Ras rerun (Oct 2026): the optimiser is as run, except that it takes a seeded generator and its
#bounds from config, returns what it found, and evaluates the model MFPT by stable state reduction
#(config.mfpt_method; "legacy_diag" restores the fundamental-matrix formula as run). apply_fes / apply_bias / update_bias (Langevin and
#NaCl leftovers, never called by the Ras driver) and the dead plotting block are removed.

import numpy as np
from MSM import *
from scipy.optimize import minimize
import openmm
import config

def gaussian(x, a, b, c): #self-defined gaussian function, same form as the OpenMM bias
        return a * np.exp(-(x - b)**2 / (2 * c**2))

##############################################################
# all MSM related functions now using MSM class in MSM.py
##############################################################

def try_and_optim_M(M, working_indices, num_gaussian=10, start_index=0, end_index=0, qspace = config.qspace, plot = False, rng = None):
    #print("inside try and optim_M")
    """
    here we try different gaussian params 1000 times
    and use the best one (lowest mfpt) to local optimise the gaussian_params

    returns the best gaussian params, as [a(num_gaussian), b(num_gaussian), c(num_gaussian)]
    in kcal/mol and A, and a dict describing the search

    input:
    M: the working transition matrix, square matrix.
    working_indices: the indices of the working states.
    num_gaussian: number of gaussian functions to use.
    start_state: the starting state. note this has to be converted into the index space.
    end_state: the ending state. note this has to be converted into the index space.
    rng: numpy Generator for the random starts (the legacy code drew from a fresh unseeded
         default_rng() at every try; same distributions, now reproducible).
    """
    if rng is None:
        rng = np.random.default_rng()
    best_mfpt = np.inf
    best_params = None
    time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds)

    start_state_working_index = np.argmin(np.abs(working_indices - start_index))
    end_state_working_index = np.argmin(np.abs(working_indices - end_index))
    print("optimizing to get g_param from start state:", start_state_working_index, "to end state:", end_state_working_index, "in working indices.")
    print("converted to xspace that's from:", qspace[working_indices[start_state_working_index]], "to", qspace[working_indices[end_state_working_index]])

    upper = qspace[working_indices[-1]]
    lower = qspace[working_indices[0]]
    print("upper bound:", upper, "lower bound:", lower)

    #initialize msm object
    msm = MSM()
    msm.qspace = qspace
    qspace_low = qspace[0]
    qspace_high = qspace[-1]
    n_nonfinite = [0]

    def biased_mfpt(params, step):
        """model MFPT start -> end of the working chain tilted by the bias (config.mfpt_method)"""
        msm.build_MSM_from_M(M, dim=1, time_step = step)
        total_bias = optimiser_total_bias(qspace, params, num_gaussian)
        #project the total_bias into the pca space.
        msm._bias_M(total_bias[working_indices], method = "direct_bias")
        if config.mfpt_method == "legacy_diag":
            msm._compute_peq_fes_M()
            msm._build_mfpt_matrix_M()
            m = float(np.real(msm.mfpts[start_state_working_index, end_state_working_index]))
        elif start_state_working_index == end_state_working_index:
            # Preserve the existing first-return objective at an already reached
            # frontier, but do not apply irreducible GTH to a reducible graph.
            m = mean_return_time(msm.M, start_state_working_index, step)
        else:
            m = mfpt_state_reduction(msm.M, start_state_working_index, end_state_working_index, step)
        if not (np.isfinite(m) and m > 0):
            n_nonfinite[0] += 1
            m = np.inf      #never let a numerical failure look like the best bias
        return m

    for try_num in range(config.n_random_starts):
        a = np.ones(num_gaussian) * config.random_start_amp
        b = rng.uniform(qspace_low, qspace_high, num_gaussian)
        c = rng.uniform(config.random_start_width[0], config.random_start_width[1], num_gaussian)

        mfpt_biased = biased_mfpt(np.concatenate((a, b, c)), time_step)
        if try_num % 100 == 0:
            print("random try:", try_num, "mfpt:", mfpt_biased)
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = np.concatenate((a, b, c)) #we concatenate the params into a single array. in shape (3*num_gaussian,)

    print("best mfpt:", best_mfpt)
    if best_params is None:
        raise RuntimeError("No finite MFPT candidate on the directed working graph; "
                           "do not apply an unoptimized random bias")

    def mfpt_helper(params, M, start_state = start_index, end_state = end_index, kT=0.5981, working_indices=working_indices):
        #time_step left at the MSM default (0.1), as run: only scales the objective
        return biased_mfpt(params, 0.1)

    res = minimize(mfpt_helper, #minimize comes from scipy.
                   best_params, #gaussian params
                   args=(M,
                         start_state_working_index,
                         end_state_working_index,
                         working_indices),
                   #method='Nelder-Mead',
                   method="L-BFGS-B",
                   bounds= [tuple(config.amp_bounds)]*config.num_gaussian + [tuple(config.center_bounds)]*config.num_gaussian + [tuple(config.width_bounds)]*config.num_gaussian, #add bounds to the parameters
                   tol=1e-6)

    #model MFPT of the chosen bias in ns (per-step chain, one step = time_step x lag)
    bounds = np.asarray([tuple(config.amp_bounds)]*num_gaussian
                        + [tuple(config.center_bounds)]*num_gaussian
                        + [tuple(config.width_bounds)]*num_gaussian)
    candidate = np.asarray(res.x, dtype=float)
    valid_parameters = (candidate.shape == best_params.shape
                        and np.isfinite(candidate).all()
                        and np.all(candidate >= bounds[:, 0])
                        and np.all(candidate <= bounds[:, 1]))
    optimised = (biased_mfpt(candidate, time_step * config.dham_lag_steps)
                 if valid_parameters else np.inf)
    candidate_source = "local_optimizer"
    if not np.isfinite(optimised):
        # A failed line search must not overwrite a valid random-search result.
        candidate = best_params.copy()
        if not (np.all(candidate >= bounds[:, 0]) and np.all(candidate <= bounds[:, 1])):
            raise RuntimeError("Finite random candidate lies outside the unchanged bias bounds")
        optimised = biased_mfpt(candidate, time_step * config.dham_lag_steps)
        candidate_source = "finite_random_fallback"
    if not (np.isfinite(optimised) and optimised > 0):
        raise RuntimeError("Selected bias has no finite positive objective; refusing to apply it")
    info = {"best_random_mfpt_ns": float(np.real(best_mfpt)) * config.dham_lag_steps,
            "best_random_params": best_params,
            "optimised_mfpt_ns": optimised,
            "mfpt_method": config.mfpt_method,
            "n_nonfinite_mfpt": int(n_nonfinite[0]),
            "optimiser_success": bool(res.success),
            "optimiser_message": str(res.message),
            "optimiser_nit": int(res.nit),
            "selected_candidate_source": candidate_source,
            "start_working_index": int(start_state_working_index),
            "end_working_index": int(end_state_working_index)}
    return candidate, info


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

def optimiser_total_bias(x, params, num_gaussian=10):
    """the bias as the optimiser evaluates it: [a, b, c] layout, util.gaussian summed in order
    (same arithmetic as the legacy loops). The start-up guard compares this function with OpenMM."""
    a = params[:num_gaussian]
    b = params[num_gaussian:2*num_gaussian]
    c = params[2*num_gaussian:3*num_gaussian]
    total_bias = np.zeros_like(np.asarray(x, dtype=float))
    for j in range(num_gaussian):
        total_bias += gaussian(x, a[j], b[j], c[j])
    return total_bias

def get_total_bias(x, gaussian_param, num_gaussians = 20):
    """
    this function returns the total bias given the gaussian_param.
    note this is used for plotting the total bias.
    """
    assert len(gaussian_param) == 3 * num_gaussians, "gaussian_param should be in A, x0, sigma_x, format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, cx))
    num_gaussians = len(gaussian_param)//3
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    sigma_x = gaussian_param[2*num_gaussians:3*num_gaussians]

    #now we add the force for all gaussians. with num_gaussians terms.
    total_bias = np.zeros_like(x)
    for i in range(num_gaussians):
        total_bias += A[i] * np.exp(-(x-x0[i])**2/(2*sigma_x[i]**2))

    return total_bias

def get_total_bias_derivative(x, gaussian_param, num_gaussians = 20):
    """d(total bias)/dx, same layout and units as get_total_bias (kcal/mol/A)."""
    assert len(gaussian_param) == 3 * num_gaussians, "gaussian_param should be in A, x0, sigma_x, format."
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    sigma_x = gaussian_param[2*num_gaussians:3*num_gaussians]
    deriv = np.zeros_like(x, dtype=float)
    for i in range(num_gaussians):
        deriv += -A[i] * (x-x0[i]) / sigma_x[i]**2 * np.exp(-(x-x0[i])**2/(2*sigma_x[i]**2))
    return deriv
