"""Verbatim copies of the legacy loops (legacy_examples/1D_PMg) that the rerun vectorised, used as
references in test_equivalence.py. Only the Gaussian is the corrected 2c^2 form, so that the
comparison isolates the vectorisation."""
import numpy as np
from scipy.linalg import inv


def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))


def count_transitions(b, numbins, lagtime, endpt=None):
    """dham.py:24-47"""
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
    return sumtr.real, trvec


def build_MM(sumtr, trvec, global_gaussian_params, qspace, KbT):
    """dham.py:73-106 (biased branch)"""
    MM = np.zeros(shape=sumtr.shape, dtype=np.longdouble)
    for i in range(sumtr.shape[0]):
        for j in range(sumtr.shape[1]):
            if sumtr[i, j] > 0:
                sump1 = 0.0
                for k in range(trvec.shape[0]):
                    u = np.zeros_like(qspace)
                    for g in range(global_gaussian_params.shape[1]):
                        a, b, c = global_gaussian_params[k, g, :]
                        u += gaussian(qspace, a, b, c)
                    if trvec[k, i] > 0:
                        sump1 += trvec[k, i] * np.exp(-(u[j] - u[i]) / (2*KbT))
                MM[i, j] = sumtr[i, j] / sump1
            else:
                MM[i, j] = 0
    for i in range(MM.shape[0]):
        row_sum = np.sum(MM[i, :])
        if row_sum > 0:
            MM[i, :] = MM[i, :] / row_sum
        else:
            MM[i, :] = 0
    return MM


def bias_M_direct(M, bias, kBT=0.5981):
    """MSM.py:383-426, method='direct_bias', dim=1 (with _is_adjacent: |i-j| == 1)"""
    n = M.shape[0]
    M_biased = np.zeros_like(M)
    for i in range(n):
        for j in range(i, n):
            if abs(i - j) == 1:
                u_ij = bias[j] - bias[i]
                M_biased[i, j] = M[i, j] * np.exp(u_ij / (2 * kBT))
                M_biased[j, i] = M[j, i] * np.exp(-u_ij / (2 * kBT))
    for i in range(n):
        M_biased[i, i] = M[i, i]
    for i in range(n):
        col_sum = np.sum(M_biased[:, i])
        if col_sum > 0:
            M_biased[:, i] = M_biased[:, i] / col_sum
        else:
            M_biased[:, i] = 0
    return M_biased


def mfpt_matrix_diag(M, peq, time_step):
    """MSM.py:216-238, method='diag'"""
    n = M.shape[0]
    onevec = np.ones((n, 1))
    I = np.diag(onevec[:, 0])
    A = (peq.reshape(-1, 1)) @ onevec.T
    A = A.T
    Qinv = inv(A + I - M.T)
    mfpts = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            term1 = Qinv[j, j] - Qinv[i, j] + I[i, j]
            if peq[j] * term1 == 0:
                mfpts[i, j] = np.inf
            else:
                mfpts[i, j] = 1 / peq[j] * term1
    return mfpts * time_step
