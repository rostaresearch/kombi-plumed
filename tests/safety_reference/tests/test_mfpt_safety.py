"""Pure numerical regression tests and actual low-dimensional driver wiring."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/one_d'))
import mfpt_safety as safety


class TestSafety(unittest.TestCase):
    def test_closed_class_with_transient_predecessors(self):
        P=np.array([[.2,.8,0,0],[0,0,1,0],[0,0,.8,.2],[0,0,.3,.7]])
        self.assertAlmostEqual(safety.row_mfpt(P,3,3,.1), .25)
        self.assertTrue(np.isinf(safety.row_mfpt(P,0,0)))

    def test_target_and_traps(self):
        P=np.array([[0,.5,.5],[0,1,0],[0,0,1]])
        self.assertTrue(np.isinf(safety.row_mfpt(P,0,1)))
        self.assertEqual(safety.row_mfpt(P,1,1),1)
        self.assertEqual(safety.row_mfpt(P,1,1,recurrence=False),0)

    def test_direction_not_reversed_or_symmetrized(self):
        P=np.array([[.5,.5,0],[0,.5,.5],[.5,0,.5]])
        self.assertEqual(safety.row_mfpt(P,0,2),4)
        self.assertEqual(safety.row_mfpt(P,2,0),2)

    def test_tiny_edge_preserved(self):
        self.assertAlmostEqual(safety.row_mfpt(np.array([[1.,1e-40],[0,1]]),0,1)/1e40,1)

    def test_independent_first_step_equations(self):
        rng=np.random.default_rng(982)
        for n in [3,6]:
            P=rng.uniform(.1,1,(n,n));P/=P.sum(1)[:,None]
            for target in range(n):
                keep=np.arange(n)!=target
                t=np.linalg.solve(np.eye(n-1)-P[np.ix_(keep,keep)],np.ones(n-1))
                for s,value in zip(np.flatnonzero(keep),t):
                    self.assertAlmostEqual(safety.row_mfpt(P,s,target),value,places=11)
                ret=1+P[target,keep]@t
                self.assertAlmostEqual(safety.row_mfpt(P,target,target),ret,places=11)

    def test_reject_bad_model(self):
        for P in [np.ones((2,2)),np.array([[np.nan,0],[0,1]]),np.array([[1.1,-.1],[0,1]])]:
            with self.assertRaises(ValueError): safety.row_mfpt(P,0,1)

    def test_all_infinite_stops_before_local_optimizer(self):
        with patch.object(safety,'_scipy_minimize') as local:
            with self.assertRaises(RuntimeError):
                safety.checked_minimize(lambda x:np.inf,[0.],bounds=[(-1,1)])
            local.assert_not_called()

    def test_bad_local_parameter_falls_back(self):
        for value in [np.nan,2.]:
            result=SimpleNamespace(x=np.array([value]),message='test',success=True)
            with patch.object(safety,'_scipy_minimize',return_value=result):
                r=safety.checked_minimize(lambda x:1+x[0]**2,[0.],bounds=[(-1,1)])
            self.assertEqual(r.x[0],0)
            self.assertEqual(r.fun,1)
            self.assertFalse(r.success)

    def test_bad_local_objective_falls_back(self):
        result=SimpleNamespace(x=np.array([.5]),message='test',success=True)
        with patch.object(safety,'_scipy_minimize',return_value=result):
            r=safety.checked_minimize(lambda x:1 if x[0]==0 else np.inf,[0.],bounds=[(-1,1)])
        self.assertEqual(r.x[0],0)

    def test_finite_local_result_kept(self):
        result=SimpleNamespace(x=np.array([.5]),message='test',success=True)
        with patch.object(safety,'_scipy_minimize',return_value=result):
            r=safety.checked_minimize(lambda x:1+x[0]**2,[0.],bounds=[(-1,1)])
        self.assertEqual(r.x[0],.5)
        self.assertEqual(r.fun,1.25)
