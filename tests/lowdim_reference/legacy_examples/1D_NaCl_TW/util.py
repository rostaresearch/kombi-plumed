#util file for applying mfpt method on 1-d NaCl system.
#by TW 26th July 2023

import os
import numpy as np
from scipy.linalg import logm, expm
from mfpt_safety import checked_minimize as minimize, row_mfpt
import matplotlib.pyplot as plt
import sys 
import config

def gaussian(x, a, b, c): #self-defined gaussian function
        #same form as the bias applied in OpenMM (mfpt_NaCl.add_bias): a*exp(-(r-b)^2/(2c^2)).
        #was a*exp(-(x-b)^2/(2c)^2), a Gaussian sqrt(2) wider than the one applied.
        return a * np.exp(-(x - b)**2 / (2 * c**2))

def params_to_gaussians(params, num_gaussian):
    """
    the parameters are stored as [a(num_gaussian), b(num_gaussian), c(num_gaussian)].
    returns the per-Gaussian rows (a_i, b_i, c_i), shape [num_gaussian, 3].
    (reshape(-1, 3) of the stored vector reads (a0, a1, a2), (a3, a4, a5), ... instead.)
    """
    return np.asarray(params, dtype=float).reshape(3, num_gaussian).T

def state_centres(qspace):
    """
    np.digitize(x, qspace) returns i for qspace[i-1] <= x < qspace[i]; the centre of state i
    is qspace[i] - dq/2. the bias of state i is evaluated here (DHAM and optimiser alike).
    """
    return qspace - 0.5 * (qspace[1] - qspace[0])

def first_hit(values, threshold):
    """index of the first value >= threshold, or None."""
    hits = np.flatnonzero(np.asarray(values, dtype=float).ravel() >= threshold)
    return int(hits[0]) if hits.size else None

def first_passage_steps(prop_index, hit_index, steps_per_frame, propagation_step):
    """
    MD steps from the start of the run to the first recorded frame in the target.
    frame hit_index (0-based) of propagation prop_index (0-based) is recorded after
    (hit_index + 1) * steps_per_frame steps of that propagation.
    """
    return prop_index * propagation_step + (hit_index + 1) * steps_per_frame

def record_first_passage(path, time_tag, n_props, steps, reached, stepsize_ps):
    """
    appends one run to a csv: time_tag, propagations (or 0), first-passage steps (or the
    steps run, if the target was not reached), the same in ps, reached (1/0).
    """
    import csv, os
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['time_tag', 'n_propagations', 'steps', 'time_ps', 'reached'])
        writer.writerow([time_tag, n_props, int(steps), float(steps) * stepsize_ps, int(bool(reached))])

def is_periodic():
    return config.nonbonded_method in ('PME', 'CutoffPeriodic', 'Ewald', 'LJPME')

def nacl_distance(positions):
    """
    Na-Cl distance in A from positions in nm (atoms 0 and 1); minimum image when the
    system is periodic (config.box_nm, cubic). The raw difference of OpenMM's unwrapped
    positions is only the right distance while the pair is closer than half the box.
    """
    p = np.asarray(positions.value_in_unit(positions.unit) if hasattr(positions, 'unit') else positions, dtype=float)
    d = p[0] - p[1]
    if is_periodic():
        d = d - config.box_nm * np.round(d / config.box_nm)
    return float(np.linalg.norm(d)) * 10

def run_seeds(seed):
    """
    integrator and velocity seeds derived from one run seed, and how the optimiser and the
    initial bias are seeded. None (no seed): unseeded, as in the published runs.
    """
    if seed is None:
        return None
    a, b = (int(x) for x in np.random.SeedSequence(seed).generate_state(2) % 2000000000 + 1)
    return {'run_seed': int(seed), 'integrator_seed': a, 'velocity_seed': b,
            'optimiser': f'np.random.default_rng([{seed}, propagation index])',
            'initial_bias': f'np.random.default_rng([{seed}, 0, 1])'}

def optimiser_rng(prop_index):
    """generator for the optimiser's random starts in propagation prop_index."""
    return np.random.default_rng([config.seed, prop_index]) if config.seed is not None else np.random.default_rng()

def platform_properties():
    if config.deterministic_forces and config.platform is not None and config.platform.getName() == 'CUDA':
        return {'DeterministicForces': 'true'}
    return {}

def make_context(system, integrator):
    """Context on config.platform with the seeded integrator (if config.seed) and CUDA
    DeterministicForces (if config.deterministic_forces)."""
    import openmm as omm
    seeds = run_seeds(config.seed)
    if seeds:
        integrator.setRandomNumberSeed(seeds['integrator_seed'])
    if config.platform is None:
        return omm.Context(system, integrator)
    return omm.Context(system, integrator, config.platform, platform_properties())

def set_velocities(context, T):
    """velocities at T, seeded from config.seed if given."""
    seeds = run_seeds(config.seed)
    if seeds:
        context.setVelocitiesToTemperature(T, seeds['velocity_seed'])
    else:
        context.setVelocitiesToTemperature(T)

def write_run_info(path, context, extra=None):
    """seeds, platform, its properties and the OpenMM version, to replay the run."""
    import json
    import openmm as omm
    p = context.getPlatform()
    info = {'seeds': run_seeds(config.seed), 'openmm': omm.__version__, 'platform': p.getName(),
            'platform_properties': {k: p.getPropertyValue(context, k) for k in p.getPropertyNames()},
            'deterministic_forces_requested': config.deterministic_forces,
            'fine_every_steps': config.fine_every, 'ring_ps': config.ring_ps,
            'nonbonded_method': config.nonbonded_method, 'box_nm': config.box_nm,
            'target_distance_A': config.target_distance, 'T': config.T,
            'code_commit': os.environ.get('KOMBI_COMMIT')}
    info.update(extra or {})
    with open(path, 'w') as f:
        json.dump(info, f, indent=1)

class FineFrames:
    """
    all-atom frames every config.fine_every steps, written to an XTC file (nm, ps).
    keep = 0: each frame is written when added (KOMBI, MetaD: the whole run).
    keep > 0: only the last `keep` frames are held in memory; flush() writes them (unbiased: the
    last config.ring_ps before the first passage).
    """
    def __init__(self, path, keep=0, dt_ps=0.002):
        from collections import deque
        from mdtraj.formats import XTCTrajectoryFile
        self.f = XTCTrajectoryFile(path, 'w')
        self.keep, self.dt_ps = keep, dt_ps
        self.ring = deque(maxlen=keep) if keep else None
        self.last_step = None

    def add(self, step, positions_nm, box_nm):
        if step == self.last_step:
            return
        self.last_step = step
        frame = (int(step), np.asarray(positions_nm, dtype=np.float32), float(box_nm))
        if self.ring is not None:
            self.ring.append(frame)
        else:
            self._write([frame])

    def _write(self, frames):
        if not frames:
            return
        xyz = np.stack([f[1] for f in frames])
        box = np.stack([np.eye(3, dtype=np.float32) * f[2] for f in frames])
        steps = np.array([f[0] for f in frames], dtype=np.int32)
        self.f.write(xyz, time=(steps * self.dt_ps).astype(np.float32), step=steps, box=box)

    def flush(self):
        if self.ring is not None:
            self._write(list(self.ring))
            self.ring.clear()

    def close(self):
        self.flush()
        self.f.close()

def create_nacl_system(psf, params):
    """
    createSystem with an explicit nonbonded treatment, config.nonbonded_method.
    the published runs gave none, so OpenMM used NoCutoff without a periodic box.
    """
    import openmm.app as omm_app
    import openmm.unit as unit
    if is_periodic():
        assert config.box_nm, "a periodic nonbonded method needs config.box_nm"
        psf.setBox(config.box_nm * unit.nanometers, config.box_nm * unit.nanometers, config.box_nm * unit.nanometers)
    method = getattr(omm_app, config.nonbonded_method)
    return psf.createSystem(params,
                            nonbondedMethod=method,
                            nonbondedCutoff=1.0*unit.nanometers,
                            constraints=omm_app.HBonds)

def create_K_1D(fes, N=200, kT=0.5981):
    #create the K matrix for 1D model potential
    #K is a N*N matrix, representing the transition rate between states
    #The diagonal elements are the summation of the other elements in the same row, i.e. the overall outflow rate from state i
    #The off-diagonal elements are the transition rate from state i to state j (or from j to i???)
    
    #input:
    #fes: the free energy profile, a 1D array.

    K = np.zeros((N,N), dtype=np.float64) #, dtype=np.float64
    for i in range(N-1):
        K[i, i + 1] = np.exp((fes[i+1] - fes[i]) / 2 / kT)
        K[i + 1, i] = np.exp((fes[i] - fes[i+1]) / 2 / kT)
    for i in range(N):
        K[i, i] = 0
        K[i, i] = -np.sum(K[:, i])
    return K

def kemeny_constant_check(N, mfpt, peq):
    kemeny = np.zeros((N, 1))
    for i in range(N):
        for j in range(N):
            kemeny[i] = kemeny[i] + mfpt[i, j] * peq[j]
    print("Performing Kemeny constant check...")
    print("the min/max of the Kemeny constant is:", np.min(kemeny), np.max(kemeny))
    """
    if np.max(kemeny) - np.min(kemeny) > 1e-6:
        print("Kemeny constant check failed!")
        raise ValueError("Kemeny constant check failed!")"""
    return kemeny

#define a function calculating the mean first passage time
def mfpt_calc(peq, K):
    """
    peq is the probability distribution at equilibrium.
    K is the transition matrix.
    N is the number of states.
    """
    N = K.shape[0] #K is a square matrix.
    onevec = np.ones((N, 1)) #, dtype=np.float64
    Qinv = np.linalg.inv(peq.T * onevec - K.T)

    mfpt = np.zeros((N, N)) #, dtype=np.float64
    for j in range(N):
        for i in range(N):
            #to avoid devided by zero error:
            if peq[j] == 0:
                mfpt[i, j] = 0
            else:
                mfpt[i, j] = 1 / peq[j] * (Qinv[j, j] - Qinv[i, j])
    
    #result = kemeny_constant_check(N, mfpt, peq)
    return mfpt


def bias_K_1D(K, total_bias, kT=0.5981):
    """
    K is the unperturbed transition matrix.
    total_bias is the total biasing potential.
    kT is the thermal energy.
    This function returns the perturbed transition matrix K_biased.
    """
    N = K.shape[0]
    K_biased = np.zeros([N, N])#, #dtype=np.float64)

    for i in range(N-1):
        u_ij = total_bias[i+1] - total_bias[i]  # Calculate u_ij (Note: Indexing starts from 0)
        K_biased[i, i+1] = K[i, i+1] * np.exp(u_ij /(2*kT))  # Calculate K_biased
        K_biased[i+1, i] = K[i+1, i] * np.exp(-u_ij /(2*kT))

    for i in range(N):
        K_biased[i,i] = -np.sum(K_biased[:,i])
    return K_biased

def compute_free_energy(K, kT=0.5981):
    """
    K is the transition matrix
    kT is the thermal energy
    peq is the stationary distribution #note this was defined as pi in Simian's code.
    F is the free energy
    eigenvectors are the eigenvectors of K

    first we calculate the eigenvalues and eigenvectors of K
    then we use the eigenvalues to calculate the equilibrium distribution: peq.
    then we use the equilibrium distribution to calculate the free energy: F = -kT * ln(peq)
    """
    evalues, evectors = np.linalg.eig(K)

    #sort the eigenvalues and eigenvectors
    index = np.argsort(evalues) #sort the eigenvalues, the largest eigenvalue is at the end of the list
    evalues_sorted = evalues[index] #sort the eigenvalues based on index

    #calculate the equilibrium distribution
    peq = evectors[:, index[-1]].T #normalize the eigenvector
    peq = peq / np.sum(peq)
    peq = peq.real

    F = -kT * np.log(peq) #add a small number to avoid log(0)) # + 1e-16

    return [peq, F, evectors, evalues, evalues_sorted, index]

def compute_free_energy_power_method(K, kT=0.5981):
    """
    this use the power method to calculate the equilibrium distribution.
    num_iter is the number of iterations.
    """
    num_iter = 1000
    N = K.shape[0]
    peq = np.ones(N) / N #initialise the peq
    for i in range(num_iter):
        peq = np.dot(peq, K)
        peq = peq / np.sum(peq)
    F = -kT * np.log(peq)
    return [peq, F]

def try_and_optim(K=None, num_gaussian=10, start_state=0, end_state=0):
   #print("inside try_and_optim")
    """
    here we try different gaussian params 1000 times
    and use the best one (lowest mfpt) to local optimise the gaussian_params
    
    returns the best gaussian params
    """

    #we construct K from loading the fes in npy files.
    fes = np.load("Unbiased_Profile.npy")
    K = create_K_1D(fes, N=config.num_bins+1, kT=0.5981)

    best_mfpt = 1e12 #initialise the best mfpt np.inf
    for i in range(1000): 
        #print("in for loop")
        rng = np.random.default_rng()
        #we set a to be 1
        a = np.ones(num_gaussian)
        b = rng.uniform(1, 9, num_gaussian) #min/max of preloaded NaCl fes x-axis.
        c = rng.uniform(1, 5.0, num_gaussian) 
        
        qspace = np.linspace(2.0, 9, config.num_bins+1)
        total_bias = np.zeros_like(qspace)
        for j in range(num_gaussian):
            #print("adding params")
            total_bias += gaussian(qspace, a[j], b[j], c[j])
        
        K_biased = bias_K_1D(K, total_bias, kT=0.5981)
        peq = compute_free_energy(K_biased, kT=0.5981)[0]

        mfpts_biased = mfpt_calc(peq, K_biased)
        mfpt_biased = mfpts_biased[start_state, end_state]
        print(sum(peq))
        kemeny_constant_check(K.shape[0], mfpts_biased, peq)
        #print("random try:", i, "mfpt:", mfpt_biased)
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = (a, b, c)
            
    #print("best mfpt:", best_mfpt)
    #now we use the best params to local optimise the gaussian params

    def mfpt_helper(params, K, kT=0.5981, start_state = start_state, end_state = end_state):
        a = params[:num_gaussian]
        b = params[num_gaussian:2*num_gaussian]
        c = params[2*num_gaussian:]
        total_bias = np.zeros(50)
        for j in range(num_gaussian):
            total_bias += gaussian(np.arange(50), a[j], b[j], c[j])
        K_biased = bias_K_1D(K, total_bias, kT=0.5981)
        peq = compute_free_energy(K_biased, kT=0.5981)[0]
        mfpts_biased = mfpt_calc(peq, K_biased)
        mfpt_biased = mfpts_biased[start_state, end_state]
        return mfpt_biased

    res = minimize(mfpt_helper, 
                   best_params, 
                   args=(K,), 
                   method='Nelder-Mead', 
                   bounds= [(0, 1)]*10 + [(1, 9)]*10 + [(1.0, 5.0)]*10, #add bounds to the parameters
                   tol=1e-1)

    return res.x, best_params


def bias_M_1D(M, total_bias, kT=0.5981):
    """
    M is the unperturbed transition matrix, row-stochastic: M[i, j] = P(i -> j).
    total_bias is the total biasing potential.
    kT is the thermal energy.
    This function returns the perturbed transition matrix M_biased.
    """
    M = np.asarray(M, dtype=np.float64) #DHAM returns longdouble; the loop wrote into float64
    u = np.asarray(total_bias, dtype=float)
    #exp(-(u_j - u_i)/(2kT)); was np.exp(-u_ij / 2*kT), which is exp(-u_ij*kT/2).
    M_biased = M * np.exp(-(u[None, :] - u[:, None]) / (2 * kT))
    np.fill_diagonal(M_biased, np.diag(M))

    row_sum = M_biased.sum(axis=1, keepdims=True)
    M_biased = np.divide(M_biased, row_sum, out=np.zeros_like(M_biased), where=row_sum > 0)
    return M_biased.real

#below Markov_mfpt_calc is provided by Sam M.
def Markov_mfpt_calc(peq, M):
    """
    row convention: M[i, j] = P(i -> j), peq the stationary distribution.
    returns mfpt[i, j] = mean first passage time (in lag times) from i to j.
    """
    M = np.asarray(M, dtype=np.float64)
    peq = np.asarray(peq, dtype=np.float64)
    N = M.shape[0]
    onevec = np.ones((N, 1))
    Idn = np.diag(onevec[:, 0])
    A = (peq.reshape(-1, 1)) @ onevec.T #was peq.T @ onevec.T
    A = A.T
    Qinv = np.linalg.inv(Idn + A - M)
    term1 = np.diag(Qinv)[None, :] - Qinv + Idn
    with np.errstate(divide='ignore', invalid='ignore'):
        mfpt = term1 / peq[None, :]
    mfpt[peq[None, :] * term1 == 0] = 1000000000000
    #result = kemeny_constant_check(N, mfpt, peq)
    return mfpt


def try_and_optim_M(M, working_indices, num_gaussian=10, start_state=0, end_state=0, plot = False, rng = None):
    #print("inside try and optim_M")
    """
    here we try different gaussian params 1000 times
    and use the best one (lowest mfpt) to local optimise the gaussian_params
    
    returns the best gaussian params

    input:
    M: the working transition matrix, square, row-stochastic: M[i, j] = P(i -> j)
       (DHAM.run returns this orientation).
    working_indices: the indices of the working states.
    num_gaussian: number of gaussian functions to use.
    start_state: the starting state. note this has to be converted into the index space.
    end_state: the ending state. note this has to be converted into the index space.
    index_offset: the offset of the index space. e.g. if the truncated M (with shape [20, 20]) matrix starts from 13 to 33, then the index_offset is 13.
    """

    #first we convert the big index into "index to the working indices".

    #start_state_working_index = np.where(working_indices == start_state)[0][0] #convert start_state to the offset index space.
    #end_state_working_index = np.where(working_indices == end_state)[0][0] #convert end_state to the offset index space.
    start_state_working_index = np.argmin(np.abs(working_indices - start_state))
    end_state_working_index = np.argmin(np.abs(working_indices - end_state))
    print("optimizing to get g_param from start state:", start_state_working_index, "to end state:", end_state_working_index, "in working indices.")
    
    #now our M/working_indices could be incontinues. #N = M.shape[0]
    qspace = np.linspace(2.0, 9, config.num_bins+1) #hard coded for now.
    xs = state_centres(qspace) #the bias of state i is evaluated at its bin centre, as in DHAM.
    best_mfpt = np.inf
    best_params = None

    #one generator for all random starts; seeded by the driver (util.optimiser_rng) for replayable
    #runs. (was a fresh unseeded generator per try.)
    gen = rng if rng is not None else np.random.default_rng()
    for i in range(1000): 
        rng_i = gen
        #we set a to be 1
        a = np.ones(num_gaussian) #* 1.5
        b = rng_i.uniform(0, 10, num_gaussian) #fix this so it place gaussian at the working indices. it has to be in angstrom. because we need return these.
        #b = rng.uniform(b_min, b_max, num_gaussian)
        c = rng_i.uniform(0.3, 0.7, num_gaussian) 
        
        #we convert the working_indices to the qspace.

        total_bias = np.zeros_like(xs)
        for j in range(num_gaussian):
            total_bias += gaussian(xs, a[j], b[j], c[j])

        #now we need to convert the total_bias to the working_indices space.
        total_bias = total_bias[working_indices]
        
        M_biased = bias_M_1D(M, total_bias, kT=0.5981)
        #peq, F = compute_free_energy_power_method(M_biased, kT=0.5981)
        
        mfpt_biased = row_mfpt(M_biased, start_state_working_index, end_state_working_index)
        #print(peq)
        #kemeny_constant_check(M.shape[0], mfpts_biased, peq)
        if i % 100 == 0:
            print("random try:", i, "mfpt:", mfpt_biased)
            #kemeny_constant_check(M.shape[0], mfpts_biased, peq)
            #we plot the F.
            
        if np.isfinite(mfpt_biased) and mfpt_biased > 0 and best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = np.concatenate((a, b, c)) #we concatenate the params into a single array. in shape (30,)

    print("best mfpt:", best_mfpt)
    if False: 
        total_bias = np.zeros_like(qspace)
        for j in range(num_gaussian):
            total_bias += gaussian(qspace, best_params[j], best_params[j+num_gaussian], best_params[j+2*num_gaussian])
        working_bias = total_bias[working_indices]
        x = qspace[working_indices]
        plt.plot(qspace[working_indices], working_bias, label="working bias")
        unb_bins, unb_profile = np.load("Unbiased_Profile.npy")
        #we take first quarter.
        unb_bins = unb_bins[:len(unb_bins)//4]
        unb_profile = unb_profile[:len(unb_profile)//4]
        plt.plot(unb_bins, unb_profile, label="unbiased F")
        #plot the total_bias
        plt.plot(qspace, total_bias, label="total bias", alpha=0.3)
        plt.xlim(2.0, 9)
        plt.legend()
        plt.savefig(f"./bias_plots_fes_best_param_prop{i}.png")
        plt.close()
    
    #now we use the best params to local optimise the gaussian params

    def mfpt_helper(params, M, start_state = start_state, end_state = end_state, kT=0.5981, working_indices=working_indices):
        a = params[:num_gaussian]
        b = params[num_gaussian:2*num_gaussian]
        c = params[2*num_gaussian:]
        total_bias = np.zeros_like(xs)
        for j in range(num_gaussian):
            total_bias += gaussian(xs, a[j], b[j], c[j])

        #now we need to convert the total_bias to the working_indices space.
        total_bias = total_bias[working_indices]

        M_biased = bias_M_1D(M, total_bias, kT=0.5981)
        #peq, F = compute_free_energy_power_method(M_biased, kT=0.5981)
        mfpt_biased = row_mfpt(M_biased, start_state_working_index, end_state_working_index)

        return mfpt_biased

    res = minimize(mfpt_helper, #minimize comes from scipy.
                   best_params, #gaussian params
                   args=(M,
                         start_state_working_index, 
                         end_state_working_index,
                         working_indices), 
                   #method='Nelder-Mead', 
                   method='L-BFGS-B',
                   bounds= [(0.1, 2)]*10 + [(0,10)]*10 + [(0.3, 1.5)]*10, #add bounds to the parameters
                   tol=1e-3)

    if plot:
        a = res.x[:num_gaussian]
        b = res.x[num_gaussian:2*num_gaussian]
        c = res.x[2*num_gaussian:]

        total_bias = np.zeros_like(qspace[working_indices])
        for j in range(num_gaussian):
            total_bias += gaussian(qspace[working_indices], a[j], b[j], c[j])
        
        M_biased = bias_M_1D(M, total_bias, kT=0.5981)

        total_bias_A = np.zeros_like(qspace)
        for j in range(num_gaussian):
            total_bias_A += gaussian(qspace, a[j], b[j], c[j])

        plt.title("Scipy optimized F with ground truth")
        plt.plot(qspace[working_indices], F, label = "reconstructed biased M FES")
        plt.plot(qspace, total_bias_A, label="total bias in A space.", linestyle="--", linewidth=0.5)
        plt.plot(qspace[working_indices], total_bias, label="total bias in working indices", linewidth=1)
        unb_bins, unb_profile = np.load("Unbiased_Profile.npy")
        plt.plot(unb_bins, unb_profile, label="unbiased FES (ground truth)")
        plt.plot(qspace[working_indices[start_state_working_index]], F[start_state_working_index], "o", label="start state")
        plt.plot(qspace[working_indices[end_state_working_index]], F[end_state_working_index], "x", label="end state")
        plt.legend()
        plt.show()

    return res.x    #, best_params
