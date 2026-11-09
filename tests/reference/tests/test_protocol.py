import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
import openmm as mm
from openmm import unit
import config
from dham import DHAM,count_transitions_one
from estimator import counts,decide
from geometry import add_bias,set_bias,read_cv
from util import mean_return_time,mfpt_state_reduction,optimiser_total_bias

class ProtocolTests(unittest.TestCase):
    def test_directed_counts(self):
        C=count_transitions_one(np.array([0,1,1,2]),3,1)
        np.testing.assert_array_equal(C,[[0,1,0],[0,1,1],[0,0,0]])

    def test_no_cross_window_pairs(self):
        a=counts([1,1,1]);b=counts([3,3,3])
        self.assertEqual((a+b).sum(),4)
        self.assertEqual(np.count_nonzero(a+b-np.diag(np.diag(a+b))),0)

    def test_nonreversible_dham_recovery(self):
        P=np.array([[.7,.3,0],[0,.8,.2],[.4,0,.6]])
        d=DHAM(np.array([[[.8,2.,.7]]]),3);d.qspace=np.array([1.,2.,3.])
        u=d.bias_on_grid()[0];pb=P*np.exp(-(u[None,:]-u[:,None])/(2*d.KbT));pb/=pb.sum(1)[:,None]
        visits=np.array([200,300,100])
        np.testing.assert_allclose(d.build_MM(visits[:,None]*pb,visits[None,:],True),P,rtol=1e-12,atol=1e-12)

    def test_return_transient_predecessor(self):
        P=np.array([[0,1,0],[0,.8,.2],[0,.3,.7]])
        self.assertAlmostEqual(mean_return_time(P.T,1,1),1/.6)
        self.assertTrue(np.isinf(mean_return_time(P.T,0,1)))

    def test_trap_is_infinite(self):
        P=np.array([[0,.5,.5],[0,1,0],[0,0,1.]])
        self.assertTrue(np.isinf(mfpt_state_reduction(P.T,0,1,1)))

    def test_directed_mfpt_independent_linear_solve(self):
        P=np.array([[.7,.3,0],[.1,.6,.3],[.4,.1,.5]])
        for target in range(3):
            keep=[i for i in range(3) if i!=target]
            expected=np.linalg.solve(np.eye(2)-P[np.ix_(keep,keep)],np.ones(2))
            for n,start in enumerate(keep):self.assertAlmostEqual(mfpt_state_reduction(P.T,start,target,1),expected[n])

    def test_protocol_settings(self):
        self.assertEqual((config.num_gaussian,config.propagation_step,config.dham_lag_steps,config.n_random_starts),(10,10000,1,1000))
        self.assertEqual(config.amp_bounds,(.1,5.));self.assertEqual(config.width_bounds,(.7,2.))

    def test_optimizer_finite_bounds(self):
        # A genuinely visited connected birth-death chain, not artificial edges.
        cv=np.tile([2.,2.,2.2,2.2,2.,2.],200)
        p=np.r_[np.full(10,.01),np.full(10,2),np.ones(10)]
        new,info=decide([counts(cv)],[p],cv[-1],91234,0)
        self.assertTrue(np.isfinite(info['optimised_mfpt_ns']))
        self.assertTrue(np.all((new[:10]>=.1)&(new[:10]<=5)))
        self.assertTrue(np.all((new[20:]>=.7)&(new[20:]<=2)))

    def test_ten_hill_force_and_unit_conversion(self):
        system=mm.System()
        for _ in range(4):system.addParticle(12)
        system.setDefaultPeriodicBoxVectors(mm.Vec3(4,0,0),mm.Vec3(0,4,0),mm.Vec3(0,0,4))
        add_bias(system,[0,1],[2,3]);integ=mm.VerletIntegrator(.002)
        ctx=mm.Context(system,integ,mm.Platform.getPlatformByName('Reference'))
        x=np.array([[3.8,.1,.1],[3.8,.2,.1],[.2,.1,.1],[.2,.2,.1]])
        ctx.setPositions(x);p=np.r_[np.linspace(.1,5,10),np.linspace(3,5,10),np.linspace(.7,2,10)]
        set_bias(ctx,p);r,v=read_cv(ctx)
        self.assertAlmostEqual(r,4.)
        self.assertAlmostEqual(v,optimiser_total_bias(np.array([r]),p)[0])
        force=ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        for atom in range(4):
            for axis in range(3):
                energy=[]
                for sign in [-1,1]:
                    y=x.copy();y[atom,axis]+=sign*1e-6;ctx.setPositions(y)
                    energy.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                self.assertAlmostEqual(force[atom,axis],-(energy[1]-energy[0])/2e-6,places=5)
        np.testing.assert_allclose(force.sum(0),0,atol=1e-10)

if __name__=='__main__':unittest.main()
