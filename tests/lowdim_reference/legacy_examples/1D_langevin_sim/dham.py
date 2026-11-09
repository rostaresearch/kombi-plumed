#original code from github: https://github.com/rostaresearch/enhanced-sampling-workshop-2022/blob/main/Day1/src/dham.py
#modified by TW on 28th July 2023
#note that in this code we presume the bias is 10 gaussian functions added together.
#returns the Markov Matrix, free energy surface probed by DHAM. 

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.linalg import eig
from scipy.optimize import minimize
from util import gaussian, state_centres


def rmsd(offset, a, b):
    return np.sqrt(np.mean(np.square((a + offset) - b)))

def align(query, ref):
    offset = -10.0
    res = minimize(rmsd, offset, args=(query, ref))
    print(res.x[0], res.fun)
    return res.x[0]

def count_transitions(b, numbins, lagtime, endpt=None):
    """
    note the b is a 2D array, 
     row represents the trajectory, column represents the time.
    """
    if endpt is None:
        endpt = b
    Ntr = np.zeros(shape=(b.shape[0], numbins, numbins), dtype=np.int64)  # number of transitions
    for k in range(b.shape[0]):
        for i in range(lagtime, b.shape[1]):
            try:
                Ntr[k, b[k, i - lagtime], endpt[k, i]] += 1
            except IndexError:
                continue
    sumtr = np.sum(Ntr, axis=0)
    trvec = np.sum(Ntr, axis=2)
    # Directed evidence only: never invent reverse transitions by averaging C with C.T.
    # anti = 0.5 * (sumtr - np.transpose(sumtr))
    # print("Degree of symmetry:",
    #       (np.linalg.norm(sym) - np.linalg.norm(anti)) / (np.linalg.norm(sym) + np.linalg.norm(anti)))
    return sumtr.real, trvec

class DHAM:
    KbT = 0.001987204259 * 300  # energy unit: kcal/mol
    epsilon = 0.00001
    data = None
    vel = None
    datlength = None
    k_val = None
    constr_val = None
    qspace = None
    num_bins = 150
    lagtime = 1

    def __init__(self, gaussian_params, num_bins):
        #gaussian_params comes in shape [prop_index + 1, num_gaussian, 3]
        num_gaussian = gaussian_params.shape[1]
        self.gaussian_params = gaussian_params
        self.x = np.linspace(0, 2*np.pi, self.num_bins)
        return

    def setup(self, CV, T, prop_index, time_tag):
        self.data = CV #in shape [prop_index + 1, frames, 1]
        self.KbT = 0.001987204259 * T
        self.prop_index = prop_index #this tell us the current propagation index.
        self.time_tag = time_tag
        return

    def build_MM(self, sumtr, trvec, biased=False):
        """
        sumtr[i, j]: directed transition counts i -> j; trvec[k, i]: counts out of i in chunk k.
        returns the unbiased, row-stochastic MM[i, j] = P(i -> j):
            MM[i, j] = sumtr[i, j] / sum_k trvec[k, i] exp(-(u_k[j] - u_k[i]) / (2kT))
        gaussian_params[k] holds the per-Gaussian rows (a, b, c) of chunk k (kcal/mol, nm).
        """
        if not biased:
            raise NotImplementedError
        xs = state_centres(self.qspace)
        U = np.zeros((self.gaussian_params.shape[0], len(xs)))
        for k in range(self.gaussian_params.shape[0]):
            for g in range(self.gaussian_params.shape[1]):
                a, b, c = self.gaussian_params[k, g, :]
                U[k] += gaussian(xs, a, b, c)
        n = sumtr.shape[0]
        U = U[:trvec.shape[0], :n]
        #sump1[i, j] = sum_k trvec[k, i] exp(-(u_k[j] - u_k[i]) / (2kT))
        sump1 = np.einsum('ki,kij->ij', trvec, np.exp(-(U[:, None, :] - U[:, :, None]) / (2*self.KbT)))
        MM = np.zeros(shape=sumtr.shape, dtype=np.longdouble)
        ok = (sumtr > 0) & (sump1 > 0)
        MM[ok] = sumtr[ok] / sump1[ok]
        row_sum = MM.sum(axis=1, keepdims=True)
        MM = np.divide(MM, row_sum, out=np.zeros_like(MM), where=row_sum > 0)
        return MM

    def run(self, plot=False, adjust=True, biased=False, conversion=2E-13):
        """

        :param plot:
        :param adjust:
        :param biased:
        :param conversion: from timestep to seconds
        :return:
        """
        qspace = np.linspace(0, 2*np.pi, self.num_bins + 1) #hard coded for now.
        self.qspace = qspace
        b = np.digitize(self.data[:, :], qspace)
        sumtr, trvec = count_transitions(b, self.num_bins, self.lagtime)

        MM = self.build_MM(sumtr, trvec, biased)

        from util import compute_free_energy, compute_free_energy_power_method
        peq, mU2,_,_,_,_ = compute_free_energy(MM.T.astype(np.float64))
        #peq, mU2 = compute_free_energy_power_method(MM)
        #print(peq)
        print("sum of peq in dham reconstruction: ", sum(peq))

        if False:
            #unb_bins, unb_profile = np.load("Unbiased_Profile.npy")
            #plot the unbiased profile from 0 to 2pi nm.
            #plt.plot(unb_bins, unb_profile, label="ground truth")
            x = np.linspace(0, 2*np.pi, self.num_bins)
            plt.figure()
            plt.plot(x, mU2, label=" fes from reconstructed M by DHAMsym")
            plt.title("Lagtime={0:d} Nbins={1:d}".format(self.lagtime, self.num_bins))
            plt.xlim(0, 2*np.pi)
            plt.savefig(f"./test_dham_{self.prop_index}.png")
            plt.close()
        #MM is row-stochastic, MM[i, j] = P(i -> j), the orientation try_and_optim_M expects.
        #(this returned MM.T before, which the optimiser then treated as row-stochastic.)
        return mU2, MM # mU2 is in unit kcal/mol.

    def bootstrap_error(self, size, iter=100, plotall=False, save=None):
        full = self.run(plot=False)
        results = []
        data = np.copy(self.data)
        for _ in range(iter):
            idx = np.random.randint(data.shape[0], size=size)
            self.data = data[idx, :]
            try:
                results.append(self.run(plot=False, adjust=False))
            except ValueError:
                print(idx)
        r = np.array(results).astype(np.float_)
        r = r[~np.isnan(r).any(axis=(1, 2))]
        r = r[~np.isinf(r).any(axis=(1, 2))]
        if plotall:
            f, a = plt.subplots()
            for i in range(r.shape[0]):
                a.plot(r[i, 0], r[i, 1])
       #     plt.show()
        # interpolation
        newU = np.empty(shape=r[:, 1, :].shape, dtype=np.float_)
        for i in range(r.shape[0]):
            newU[i, :] = np.interp(full[0], r[i, 0], r[i, 1])
            # realign
            offset = align(newU[i, :], full[1])
            newU[i, :] += offset
        stderr = np.std(newU, axis=0)
        f, a = plt.subplots()
        a.plot(full[0], full[1])
        a.fill_between(full[0], full[1] - stderr, full[1] + stderr, alpha=0.2)
     #   plt.title("lagtime={0:d} bins={1:d}".format(self.lagtime, self.numbins))
        if save is None:
     #       plt.show()
            print("saved is none")
        else:
     #       plt.savefig(save)
             print("save is none")
        self.data = data
        return
