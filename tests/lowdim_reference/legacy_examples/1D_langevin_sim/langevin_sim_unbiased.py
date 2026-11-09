#this is a langevin simulator in OPENMM.
# we put a particle in a box and simulate it with Langevin dynamics.
# the external force is defined using a function digitizing the phi/psi fes of dialanine.

import numpy as np
import matplotlib.pyplot as plt

import time

from tqdm import tqdm

import openmm
from openmm import unit
from openmm.app.topology import Topology
from openmm.app.element import Element

from openmm.unit import Quantity
from openmm import Vec3

import config
import csv
from util import *

#first we initialize the system.
# topology

#for amp applied on fes. note the gaussian parameters for fes is normalized.
if __name__ == "__main__":
    elem = Element(0, "X", "X", 1.0)
    top = Topology()
    top.addChain()
    top.addResidue("xxx", top._chains[0])
    top.addAtom("X", elem, top._chains[0]._residues[0])
    mass = 12.0 * unit.amu
    #the same target as KOMBI and MetaD (was end_state_unbiased = 3.9 nm, check commented out).
    end_state_xyz = config.end_state.value_in_unit_system(openmm.unit.md_unit_system)[0]
    

    x = np.linspace(0, 2*np.pi, config.num_bins)

    #starting point as [1.29,-1.29,0.0]
    for i_sim in range(config.num_sim):
        time_tag = time.strftime("%Y%m%d-%H%M%S")
        system = openmm.System()
        system.addParticle(mass)

        #potential setup
        #first we load the gaussians from the file.
        # params comes in A, x0, y0, sigma_x, sigma_y format.

        #gaussian_param = np.loadtxt("./params/gaussian_fes_param.txt") 

        system, fes = apply_fes(system = system, 
                        particle_idx=0, 
                        gaussian_param = None, 
                        pbc = config.pbc, 
                        name = "FES", 
                        amp=config.amp, 
                        mode = config.fes_mode,
                        plot = True)
        y_pot = openmm.CustomExternalForce("1e3 * y^2") # very large force constant in y
        y_pot.addParticle(0)
        z_pot = openmm.CustomExternalForce("1e3 * z^2") # very large force constant in z
        z_pot.addParticle(0)
        system.addForce(z_pot) #on z, large barrier
        system.addForce(y_pot)
        #pbc section
        if config.pbc:
            a = unit.Quantity((2*np.pi*unit.nanometers, 0*unit.nanometers, 0*unit.nanometers))
            b = unit.Quantity((0*unit.nanometers, 2*np.pi*unit.nanometers, 0*unit.nanometers))
            c = unit.Quantity((0*unit.nanometers, 0*unit.nanometers, 1*unit.nanometers)) # atom not moving in z so we set it to 1 nm
            system.setDefaultPeriodicBoxVectors(a,b,c)


        #integrator
        integrator = openmm.LangevinIntegrator(300*unit.kelvin, 
                                            1.0/unit.picoseconds, 
                                            config.stepsize) #2 fs (the comment said 0.2 ps).

        #before run, last check pbc:
        #print(system.getDefaultPeriodicBoxVectors())

        #CUDA
        platform = config.platform #CUDA if available

        #run the simulation
        simulation = openmm.app.Simulation(top, system, integrator, platform)
        simulation.context.setPositions(config.start_state)
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

        simulation.minimizeEnergy()
        if config.pbc:
            simulation.context.setPeriodicBoxVectors(a,b,c)

        pos_traj = np.zeros([int(config.sim_steps_unbiased/config.dcdfreq), 3]) #note config.sim_steps represents the maximum number of steps here in the unbiased case.
        reach = None
        n_steps = 0

        file_handle = open(f'trajectory/unbias/{time_tag}_unbias_traj.dcd', 'wb')
        dcd_file = openmm.app.DCDFile(file_handle, top, dt = config.stepsize) #was stepsize_unbias (0.2 ps) for 2 fs steps
        #the x coordinate is checked every config.fpt_stride steps; the run stops at the first
        #passage (within target_radius of end_state) or after sim_steps_unbiased steps.
        while n_steps < config.sim_steps_unbiased:
            simulation.step(config.fpt_stride)
            n_steps += config.fpt_stride
            state = simulation.context.getState(getPositions=True, enforcePeriodicBox=config.pbc)
            cur_pos = state.getPositions(asNumpy=True)[0].value_in_unit(unit.nanometers)
            if n_steps % config.dcdfreq == 0:
                i = n_steps // config.dcdfreq - 1
                pos_traj[i,:] = cur_pos
                dcd_file.writeModel(state.getPositions(asNumpy=True))
            if abs(cur_pos[0] - end_state_xyz[0]) < config.target_radius:
                print("reached end state")
                reach = True
                break
        file_handle.close()

        stepsize_ps = config.stepsize.value_in_unit(unit.picoseconds)
        print(f"first passage steps: {n_steps}, reached: {reach is not None}")
        record_first_passage("first_passage_unbiased.csv", time_tag, 0, n_steps, reach is not None, stepsize_ps)
        pos_traj = pos_traj[:n_steps // config.dcdfreq]

        #save the traj and plot it.
        pos_traj = np.array(pos_traj)
        np.savetxt(f"visited_states/{time_tag}_langevin_2D_unbias_traj.txt", pos_traj)

        #here we plot.
        
    print("all done")
