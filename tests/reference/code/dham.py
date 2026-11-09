#original code from github: https://github.com/rostaresearch/enhanced-sampling-workshop-2022/blob/main/Day1/src/dham.py
#modified by TW on 28th July 2023
#note that in this code we presume the bias is 10 gaussian functions added together.
#returns the Markov Matrix, free energy surface probed by DHAM.
#
#Ras rerun (Oct 2026):
# - global_gaussian_params must be per-Gaussian (a, b, c) triples in kcal/mol and A
#   (the legacy driver passed a scrambled reshape of [a, b, c]);
# - util.gaussian is the applied form a*exp(-(x-b)^2/(2c^2));
# - count_transitions and build_MM are vectorised with the same arithmetic as the loops
#   (tests/test_equivalence.py); counts can be passed in per propagation so that a decision
#   does not recount every earlier propagation;
# - build_MM sets MM[i,j] = 0 when no propagation has counts out of state i (sump1 == 0) instead
#   of dividing by zero; the number of such entries is reported in self.diagnostics;
# - bootstrap_error/align/rmsd (unused) and matplotlib are removed.

import numpy as np
from util import gaussian
import config
from MSM import *

def count_transitions_one(b_row, numbins, lagtime):
    """
    transition counts of one trajectory row at the given lag, as the inner loop of
    count_transitions: pairs (b[i-lag], b[i]) for i = lag..n-1; a pair with an index outside
    0..numbins-1 is skipped (the loop caught the IndexError).
    """
    b_row = np.asarray(b_row)
    i0 = b_row[:-lagtime] if lagtime > 0 else b_row
    i1 = b_row[lagtime:]
    ok = (i0 >= 0) & (i0 < numbins) & (i1 >= 0) & (i1 < numbins)
    C = np.zeros((numbins, numbins), dtype=np.int64)
    np.add.at(C, (i0[ok], i1[ok]), 1)
    return C

def count_transitions(b, numbins, lagtime, endpt=None, Ntr=None):
    """
    note the b is a 2D array,
     row represents the trajectory, column represents the time.
    Ntr: optional precomputed per-row counts, shape (rows, numbins, numbins).
    """
    if endpt is not None and endpt is not b:
        raise NotImplementedError("endpt different from b is not used by the Ras driver")
    if Ntr is None:
        Ntr = np.stack([count_transitions_one(b[k], numbins, lagtime) for k in range(b.shape[0])])
    sumtr = np.sum(Ntr, axis=0)
    trvec = np.sum(Ntr, axis=2)
    print("Directed counts in DHAM")
    return sumtr.real, trvec

class DHAM:

    def __init__(self, global_gaussian_params, num_bins):
        #gaussian_params comes in shape [prop_index + 1, num_gaussian, 3], per-Gaussian (a, b, c)
        self.global_gaussian_params = global_gaussian_params
        self.KbT = 0.001987204259 * 300  # energy unit: kcal/mol
        self.epsilon = 0.00001
        self.data = None
        self.vel = None
        self.datlength = None
        self.k_val = None
        self.constr_val = None
        self.qspace = None
        self.num_bins = num_bins
        self.lagtime = 1
        self.Ntr = None
        self.diagnostics = {}
        return

    def setup(self, CV, T, prop_index, time_tag, Ntr=None):
        self.data = CV
        self.KbT = 0.001987204259 * T
        self.prop_index = prop_index
        self.time_tag = time_tag
        self.Ntr = Ntr
        return

    def bias_on_grid(self):
        """total bias of every propagation on self.qspace, shape (props, len(qspace)), kcal/mol"""
        U = np.zeros((self.global_gaussian_params.shape[0], len(self.qspace)))
        for k in range(U.shape[0]):
            u = np.zeros_like(self.qspace)
            for g in range(self.global_gaussian_params.shape[1]):
                a,b,c = self.global_gaussian_params[k, g, :]
                u += gaussian(self.qspace, a, b, c)
            U[k] = u
        return U

    def build_MM(self, sumtr, trvec, biased=False):
        if not biased:
            raise NotImplementedError
        #for each chunk of trajectory k, the total bias u_k; then
        #MM[i,j] = sumtr[i,j] / sum_k trvec[k,i] exp(-(u_k[j]-u_k[i])/2kT), rows normalised.
        #the k-loop accumulates in the same order as the original triple loop.
        U = self.bias_on_grid()
        n = sumtr.shape[0]
        sump1 = np.zeros(sumtr.shape, dtype=np.float64)
        for k in range(trvec.shape[0]):
            w = trvec[k, :n]
            rows = w > 0
            if not np.any(rows):
                continue
            u = U[k, :n]
            term = w[rows][:, None] * np.exp(-(u[None, :] - u[rows][:, None]) / (2*self.KbT))
            sump1[rows, :] += term
        has = sumtr > 0
        empty = has & (sump1 == 0)
        self.diagnostics["zero_sump_with_counts"] = int(np.sum(empty))
        MM = np.zeros(shape=sumtr.shape, dtype=np.longdouble)
        ok = has & ~empty
        MM[ok] = sumtr[ok] / sump1[ok]
        row_sum = np.sum(MM, axis=1)
        pos = row_sum > 0
        MM[pos, :] = MM[pos, :] / row_sum[pos][:, None]
        MM[~pos, :] = 0
        return MM

    def run(self, plot=False, adjust=True, biased=False, conversion=2E-13, use_dynamic_bins=False):
        """

        :param plot: ignored (no plotting in the rerun)
        :param adjust:
        :param biased:
        :param conversion: from timestep to seconds
        :return: qspace, free energy (kcal/mol), M = MM.T (column-stochastic)
        """
        #given the data, we create dynamic qspace based on config.DHAM_num_bins
        #and discretize the data into bins.
        if use_dynamic_bins:
            raise NotImplementedError("dynamic bins are not used by the Ras driver")
        self.qspace = config.qspace
        print("using static Binning with {0:d} bins".format(config.qspace_num_bins))
        assert len(self.qspace) == self.num_bins, "DHAM evaluates the bias at qspace[bin], so len(qspace) must equal num_bins"

        b = None if self.Ntr is not None else np.digitize(self.data[:, :], self.qspace)
        sumtr, trvec = count_transitions(b, self.num_bins, self.lagtime, Ntr=self.Ntr)

        #diagnostic: the optimiser tilts only nearest neighbours, so counts should be (near) tridiagonal
        off = np.array(sumtr, dtype=float); np.fill_diagonal(off, 0)
        ii, jj = np.indices(off.shape)
        n_off = off.sum()
        self.diagnostics["frac_offdiag_nonadjacent"] = float(off[np.abs(ii - jj) > 1].sum() / n_off) if n_off > 0 else 0.0
        self.diagnostics["n_transitions"] = float(np.sum(sumtr))

        MM = self.build_MM(sumtr, trvec, biased)
        msm = MSM()
        msm.M = MM.T
        msm._compute_peq_fes_M()
        print("sum of peq in dham reconstruction: ", sum(msm.peq))
        return self.qspace, msm.free_energy, MM.T # fes in unit kcal/mol.
