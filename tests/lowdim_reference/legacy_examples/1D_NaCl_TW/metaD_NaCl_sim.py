#here we do metaD simulation for NaCl in water.
# the CV is the distance between Na and Cl.
# and log the first-passage time to config.target_distance (8 A; was 7 A).
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from tqdm import tqdm

import openmm as omm
from openmm import unit
from openmm import Vec3
import openmm.app as omm_app
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.app.metadynamics import BiasVariable, Metadynamics
from openmm.unit import Quantity

from util import *
import config
import csv

import mdtraj

def minimize(context):
    st = time.time()
    s = time.time()
    print("Setting up the simulation")

    # Minimizing step
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy = True)
    energy = state.getPotentialEnergy()

    for _ in range(50):
        omm.openmm.LocalEnergyMinimizer.minimize(context, 1, 20)
        state = context.getState(getEnergy = True)
        energy = state.getPotentialEnergy()

    print("Minimization done in", time.time() - s, "seconds")
    s = time.time()
    return context, energy



if __name__ == "__main__":
    psf_file = 'toppar/step3_input.psf' # Path #tpr #prmtop
    pdb_file = 'toppar/step3_input25A.pdb' # Path # gro # inpcrd

    meta_freq = 5000
    meta_height = 1.5 #half the size of 3 kcal/mol

    psf = omm_app.CharmmPsfFile(psf_file)
    pdb = omm_app.PDBFile(pdb_file)
    #forcefield = omm_app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    params = omm_app.CharmmParameterSet('toppar/toppar_water_ions.str') #we modified the combined LJ term between NaCl to have a -6.0kcal.mol at 2.5A

    for i_sim in range(config.num_sim):
        print(f"MetaD Simulation {i_sim} starting")
        time_tag = time.strftime("%Y%m%d-%H%M%S")
        system = create_nacl_system(psf, params) #explicit nonbondedMethod, see config.
        
        #metaD
        bias_bond = omm.CustomBondForce("r")
        bias_bond.setUsesPeriodicBoundaryConditions(is_periodic()) #minimum-image r with a periodic box
        bias_bond.addBond(0, 1)
        dist_cv = BiasVariable(bias_bond, 0.20, 1.0, 0.02, False) 
        #we define the plotting axis for metaD.freeenergy.
        # 2.0 to 10.0, with bin width of 0.2
        cv_space = np.linspace(0.20, 1.0, 41)
        
        aux_file_path = os.path.join("./MetaD_auxfiles/" + time_tag)
        if not os.path.exists(aux_file_path):
            os.makedirs(aux_file_path)
        metaD = Metadynamics(system=system, 
                            variables=[dist_cv], #define the cv
                            temperature=config.T*unit.kelvin,
                            biasFactor=5.0,
                            height=meta_height * 4.184 * unit.kilojoules_per_mole,
                            frequency=meta_freq,
                            saveFrequency=meta_freq,
                            biasDir=aux_file_path)
        
        platform = config.platform #CUDA if available
        integrator = omm.LangevinIntegrator(config.T*unit.kelvin, #Desired Integrator
                                            10/unit.picoseconds,
                                            config.stepsize)

        NaCl_dist = [[]] #initialise the NaCl distance list.
        #simulation object, note the context object is automatically created.
        seeds = run_seeds(config.seed)
        if seeds:
            integrator.setRandomNumberSeed(seeds['integrator_seed'])
        sim = omm.app.Simulation(pdb.topology, system, integrator, platform, platform_properties()) if platform else omm.app.Simulation(pdb.topology, system, integrator)
        sim.context.setPositions(pdb.positions)

        #minimize the energy
        context, energy = minimize(sim.context)
        sim.context = context
        set_velocities(context, config.T*unit.kelvin) #seeded if config.seed (the published runs started from zero velocities)
        #all atoms every config.fine_every steps for the whole run (off if 0); Na-Cl every fpt_stride steps
        os.makedirs("fine", exist_ok=True)
        fine = FineFrames(f"fine/{time_tag}_metad.xtc") if config.fine_every else None
        dist_fine = []

        #run the simulation
        file_handle = open(f'trajectories/metaD/{time_tag}_metaD_traj.dcd', 'wb')
        dcd_file = omm_app.DCDFile(file_handle, pdb.topology, dt = config.stepsize)

        #the distance is checked every config.fpt_stride steps (was: DCD frames every 500 steps,
        #with the crossing taken at the start of the frame interval); the run stops at the first
        #passage when stop_at_target is set.
        stop_at_target = True
        stepsize_ps = config.stepsize.value_in_unit(unit.picoseconds)
        first_passage = None
        n_chunks = int(config.sim_steps/config.fpt_stride)
        for i_chunk in tqdm(range(n_chunks)):
            metaD.step(sim, config.fpt_stride)
            state = context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True)
            pos_nm = pos.value_in_unit(unit.nanometers)
            d_now = nacl_distance(pos_nm) #A, minimum image
            dist_fine.append(d_now)
            n_now = (i_chunk + 1) * config.fpt_stride
            if fine is not None and n_now % config.fine_every == 0:
                fine.add(n_now, pos_nm, config.box_nm)
            if first_passage is None and d_now >= config.target_distance:
                first_passage = n_now
                if fine is not None:
                    fine.add(n_now, pos_nm, config.box_nm) #the arrival frame
                if stop_at_target:
                    break
            if ((i_chunk + 1) * config.fpt_stride) % config.dcdfreq == 0:
                dcd_file.writeModel(pos)
        file_handle.close()
        steps_run = (i_chunk + 1) * config.fpt_stride
        energy_CV = metaD.getFreeEnergy() #The values are in kJ/mole. The i’th position along an axis corresponds to minValue + i*(maxValue-minValue)/gridWidth.
        gridwidth = int(np.ceil(5*(1.0-0.2)/0.02)) #200.
        cv_space = np.linspace(2, 10, gridwidth)

        #create the fes metad folder is not exist.
        if not os.path.exists("fes_metaD"):
            os.makedirs("fes_metaD")

        #plot the free energy surface
        plt.figure()
        plt.plot(cv_space, energy_CV/4.184)
        plt.xlabel("NaCl distance (A)")
        plt.ylabel("Free energy (kcal/mol)")
        plt.savefig(f"./fes_metaD/{time_tag}_fes_metaD_{i_sim}.png")
        plt.close()

        print("first passage steps:", first_passage)
        record_first_passage("first_passage_metaD.csv", time_tag, 0,
                             first_passage if first_passage is not None else steps_run,
                             first_passage is not None, stepsize_ps)
        if fine is not None:
            fine.close()
        np.save(f"visited_state/{time_tag}_NaCl_metaD_dist_every{config.fpt_stride}steps.npy", np.asarray(dist_fine, dtype=np.float32))
        write_run_info(f"{time_tag}_run_info.json", context,
                       {'method': 'metad', 'first_passage_steps': first_passage, 'steps_run': steps_run})
    
    
    
    
    
    
    
