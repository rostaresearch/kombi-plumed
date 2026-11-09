"""Directed-count policy and orientation tests, independent of reversible chains."""
import inspect
import numpy as np
from dham import DHAM, count_transitions


def test_counts_keep_observed_direction():
    C, n = count_transitions(np.array([[0, 1, 1, 2]]), 3, 1)
    np.testing.assert_array_equal(C, [[0, 1, 0], [0, 1, 1], [0, 0, 0]])
    np.testing.assert_array_equal(n, [[1, 2, 0]])


def test_symmetrization_option_does_not_exist():
    assert "use_symmetry" not in inspect.signature(count_transitions).parameters
    assert "use_symmetry" not in inspect.signature(DHAM.run).parameters


def test_known_nonreversible_model_is_recovered():
    P = np.array([[.7, .3, 0], [0, .8, .2], [.4, 0, .6]])
    d = DHAM(np.array([[[.8, 2., .7]]]), 3)
    d.qspace = np.array([1., 2., 3.])
    U = d.bias_on_grid()[0]
    Pb = P*np.exp(-(U[None, :]-U[:, None])/(2*d.KbT))
    Pb /= Pb.sum(1, keepdims=True)
    n = np.array([200., 300., 100.])
    recovered = d.build_MM(n[:, None]*Pb, n[None, :], biased=True)
    np.testing.assert_allclose(recovered, P, atol=1e-12, rtol=1e-12)
