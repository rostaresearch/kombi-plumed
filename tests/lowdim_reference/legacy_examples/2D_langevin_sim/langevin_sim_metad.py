#this is a langevin simulator in OPENMM.
# we put a particle in a box and simulate it with Langevin dynamics.
# the external force is defined using a function digitizing the phi/psi fes of dialanine.

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from tqdm import tqdm

import openmm
from openmm import unit
from openmm import Vec3
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.app.metadynamics import BiasVariable, Metadynamics
from openmm.unit import Quantity

from util import *
import config
import csv

##############################################################
#### INITIALIZATION ####

if __name__ == "__main__":

    pbc = config.pbc
    time_tag = time.strftime("%Y%m%d-%H%M%S")

    #metaD parameters
    npoints = 101
    meta_freq = 5000
    meta_height = config.metad_height_kJ #kJ/mol (was amp/2 in kcal/mol)

    #
    elem = Element(0, "X", "X", 1.0)
    top = Topology()
    top.addChain()
    top.addResidue("xxx", top._chains[0])
    top.addAtom("X", elem, top._chains[0]._residues[0])

    mass = 12.0 * unit.amu
    #starting point as [1.29,-1.29,0.0]

    for i_sim in range(config.num_sim):
        #print the config.
        print(f"config: {config.__dict__}")
        system = openmm.System()
        system.addParticle(mass)

        ##############################################################
        ####  potential setup ####
        #first we load the gaussians from the file.
        # params comes in A, x0, y0, sigma_x, sigma_y format.

        #gaussian_param = np.loadtxt("./fes_digitize_gauss_params_15.txt") 
        #n_gaussians = int(len(gaussian_param)/5)

        system, fes = apply_fes(system = system, 
                                particle_idx=0, 
                                gaussian_param = None, 
                                pbc = config.pbc, 
                                name = "FES", 
                                amp=config.amp, 
                                mode = config.fes_mode,
                                plot = True)
        z_pot = openmm.CustomExternalForce("1e3 * z^2") # very large force constant in z
        z_pot.addParticle(0)
        system.addForce(z_pot) #on z, large barrier
        #pbc section
        if pbc:
            a = unit.Quantity((2*np.pi*unit.nanometers, 0*unit.nanometers, 0*unit.nanometers))
            b = unit.Quantity((0*unit.nanometers, 2*np.pi*unit.nanometers, 0*unit.nanometers))
            c = unit.Quantity((0*unit.nanometers, 0*unit.nanometers, 1*unit.nanometers)) # atom not moving in z so we set it to 1 nm
            system.setDefaultPeriodicBoxVectors(a,b,c)

        ##############################################################
        #### SIMULATION ####

        #we initialize the Metadynamics object
        #create a folder to store the aux files.
        aux_file_path = os.path.join("./langevin_approach/MetaD_auxfiles/" + time_tag)

        if not os.path.exists(aux_file_path):
            os.makedirs(aux_file_path)

        #apply the rmsd restraint as BiasVariable
        target_pos = config.end_state

        """
        rmsd_cv = openmm.RMSDForce(target_pos, [0])
        rmsd_cv.setReferencePositions(target_pos)
        print(rmsd_cv.getParticles())
        print(rmsd_cv.getReferencePositions())
        custom_force = openmm.CustomCVForce("k*RMSD")
        custom_force.addGlobalParameter("k", 1.0)
        custom_force.addCollectiveVariable("RMSD", rmsd_cv)

        rmsd_Bias_var = BiasVariable(custom_force, 1.0, 10, 1, False)
        """
        #CVs: x and y, on a grid that covers the whole domain including the walls.
        #(were (x-x0)^2 and (y-y0)^2 on a 2-10 nm^2 grid, which excluded both the start,
        #(16, 6.25), and the target, (0, 0).)
        if pbc:
            raise NotImplementedError #not implementedd for pbc yet.
        x_cv = openmm.CustomExternalForce("x")
        x_cv.addParticle(0)
        x_force = openmm.CustomCVForce("x_cv")
        x_force.addCollectiveVariable("x_cv", x_cv)

        y_cv = openmm.CustomExternalForce("y")
        y_cv.addParticle(0)
        y_force = openmm.CustomCVForce("y_cv")
        y_force.addCollectiveVariable("y_cv", y_cv)

        x_Bias_var = BiasVariable(x_force, -1.0, 2*np.pi+1.0, 0.2, config.pbc)
        y_Bias_var = BiasVariable(y_force, -1.0, 2*np.pi+1.0, 0.2, config.pbc)

        #check forces #the forces are correct. it is possible the BiasVariable is not compatible with ExternalForce class.
        #for force in system.getForces():
        #    print(force)

        metaD = Metadynamics(system=system,
                                variables=[x_Bias_var, y_Bias_var], #variables=[rmsd_Bias_var],
                                temperature=300*unit.kelvin,
                                biasFactor=5,
                                height=meta_height*unit.kilojoules_per_mole,
                                frequency=meta_freq, #1000
                                saveFrequency=meta_freq,
                                biasDir=aux_file_path,)

        platform = config.platform #CUDA if available

        #integrator
        integrator = openmm.LangevinIntegrator(300*unit.kelvin, 
                                            1.0/unit.picoseconds, 
                                            0.002*unit.picoseconds)

        #run the simulation
        simulation = openmm.app.Simulation(top, system, integrator, platform)
        simulation.context.setPositions(config.start_state)
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

        #minimize the energy
        s = time.time()
        print("minimizing energy")

        for _ in range(50):
            openmm.LocalEnergyMinimizer.minimize(simulation.context, 1e-2, 1000)

        print("minimizing energy done, time: %.2f" % (time.time()-s))


        #pos_traj = np.zeros([int(config.sim_steps/config.dcdfreq), 3])

        #store fes in 2D way
        #fes = np.zeros([int(sim_steps/dcdfreq), 50, 50])
        potential_energy = []

        #file_handle = open(f'./trajectory/metaD/{time_tag}_metaD_traj.dcd', 'wb')
        #dcd_file = openmm.app.DCDFile(file_handle, top, dt = config.stepsize)

        #one DCD reporter (it was appended inside the loop below, one more at every iteration,
        #and the first-passage csv was commented out).
        simulation.reporters.append(openmm.app.DCDReporter(f'./trajectory/metaD/{time_tag}_metaD_traj.dcd', config.dcdfreq))

        #(x, y) checked every config.fpt_stride steps; stop at the first passage.
        stop_at_target = True
        end_state_xy = np.asarray(config.end_state.value_in_unit_system(openmm.unit.md_unit_system)[0])[:2]
        first_passage = None
        n_steps = 0
        while n_steps < config.sim_steps:
            metaD.step(simulation, config.fpt_stride) #this is metaD object step.
            n_steps += config.fpt_stride
            state = simulation.context.getState(getPositions=True, enforcePeriodicBox=pbc)
            cur_xy = state.getPositions(asNumpy=True)[0,:2].value_in_unit(unit.nanometers)
            if n_steps % config.dcdfreq == 0:
                potential_energy.append(simulation.context.getState(getEnergy=True).getPotentialEnergy())
            if first_passage is None and np.linalg.norm(cur_xy - end_state_xy) < config.target_radius:
                first_passage = n_steps
                if stop_at_target:
                    break

        stepsize_ps = config.stepsize.value_in_unit(unit.picoseconds)
        print(f"first passage steps: {first_passage}")
        record_first_passage("first_passage_metaD.csv", time_tag, 0,
                             first_passage if first_passage is not None else n_steps,
                             first_passage is not None, stepsize_ps)

        #zip traj, biasand save.
        #np.save(f"./visited_states/{time_tag}_metaD_pos_traj_.npy", np.array(pos_traj))
        np.save(f"./visited_states/{time_tag}_metaD_potential_energy.npy", np.array(potential_energy))

        #this is for plain MD.
        """
        pos_traj = np.zeros([sim_steps, 3])
        for i in range(sim_steps):
            simulation.step(1)
            state = simulation.context.getState(getPositions=True)
            pos_traj[i,:] = state.getPositions(asNumpy=True)[0,:]
        """

        #pos_traj = np.array(pos_traj)
        #end_state_xyz = config.end_state.value_in_unit_system(openmm.unit.md_unit_system)[0]

        """for index_d, d in enumerate(pos_traj):
            if np.linalg.norm(d - end_state_xyz) < 0.1:
                steps_to_endstate = index_d*config.dcdfreq
                break
            if index_d == len(pos_traj)-1:
                steps_to_endstate = index_d*config.dcdfreq
            
        with open("total_steps_metaD.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([steps_to_endstate])
"""

        ### VISUALIZATION ###

        """#we plot the pos_traj.
        plt.figure()
        #here we plot the fes.
        plt.imshow(fes, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=config.amp * 12/7 * 4.184, origin="lower")
        plt.colorbar()
        plt.xlabel("x")
        #plt.xlim([-1, 2*np.pi+1])
        #plt.ylim([-1, 2*np.pi+1])
        plt.ylabel("y")
        plt.title("FES mode = multiwell, pbc=False")

        #plot the trajectory
        plot_inteval = len(pos_traj)//1000 #plot 10000 points
        plt.scatter(pos_traj[::plot_inteval,0], pos_traj[::plot_inteval,1], s=3.5, alpha = 0.5, c="black")
        plt.savefig(f"./figs/metaD/{time_tag}_metaD_traj.png")
        plt.close()

"""



print("all done")
