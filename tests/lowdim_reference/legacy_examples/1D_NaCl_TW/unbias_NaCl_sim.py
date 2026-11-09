#here we do super long NaCl unbiased run.
# and log the first-passage time to config.target_distance (8 A; was 7 A).
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import openmm.app as omm_app
import openmm as omm
import openmm.unit as unit
import mdtraj
from util import *
sys.path.append("..")
from openmm.app import *
#from openmm.unit import *
#from openmm.unit import *
from dham import DHAM
import csv
import os
import sys
from PIL import Image
import time
from tqdm import tqdm

import config

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

platform = config.platform #CUDA if available
psf_file = 'toppar/step3_input.psf' # Path #tpr #prmtop
pdb_file = 'toppar/step3_input25A.pdb' # Path # gro # inpcrd
T = config.T    # temperature in K (was 298.15; KOMBI and MetaD ran at 300 K)
fricCoef = 10   # friction coefficient in 1/ps
stepsize = 2    # MD integration step size in fs
dcdfreq = 5000  # save coordinates every 10 ps (was every 100 steps: ~2 GB per 20 ns run)
fpt_stride = config.fpt_stride # check the distance every fpt_stride steps
num_simulations = 20
sim_steps = 50000000 # at most 100 ns per run (the loop below had no limit)

if __name__ == "__main__":

    psf = omm_app.CharmmPsfFile(psf_file)
    pdb = omm_app.PDBFile(pdb_file)
    #forcefield = omm_app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    params = omm_app.CharmmParameterSet('toppar/toppar_water_ions.str') #we modified the combined LJ term between NaCl to have a -6.0kcal.mol at 2.5A

    for i_sim in tqdm(range(num_simulations)):
        
        system = create_nacl_system(psf, params) #explicit nonbondedMethod, see config.
        
        #### setup an OpenMM context
        integrator = omm.LangevinIntegrator(T*unit.kelvin, #Desired Integrator
                                            fricCoef/unit.picoseconds,
                                            stepsize*unit.femtoseconds)

        NaCl_dist = [] #initialise the NaCl distance list.
        time_tag = time.strftime("%Y%m%d-%H%M%S")
        context = make_context(system, integrator) #seeded integrator, DeterministicForces if set

        context, energy = minimize(context)
        set_velocities(context, T*unit.kelvin) #seeded if config.seed (the published runs started from zero velocities)
        #all atoms every config.fine_every steps, of which only the last config.ring_ps are kept and
        #written at the end of the run (off if fine_every is 0); Na-Cl every fpt_stride steps.
        os.makedirs("fine", exist_ok=True)
        ring = int(round(config.ring_ps / (stepsize / 1000 * config.fine_every))) if config.fine_every else 0
        fine = FineFrames(f"fine/{time_tag}_unbiased_last{config.ring_ps:g}ps.xtc", keep=ring) if config.fine_every else None
        dist_fine = []

        file_handle = open(f'trajectories/unbiased/{time_tag}_unbiased_traj.dcd', 'wb')
        dcd_file = omm_app.DCDFile(file_handle, pdb.topology, dt = stepsize*unit.femtoseconds) #was dt = 2 (read as 2 ps)

        #distance checked every fpt_stride steps; first passage = steps to the first check at or
        #beyond the target. (the previous loop ran to 7 A with no step limit.)
        n_steps = 0
        reached = False
        while n_steps < sim_steps:
            integrator.step(fpt_stride)
            n_steps += fpt_stride
            state = context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True)
            pos_nm = pos.value_in_unit(unit.nanometers)
            d = nacl_distance(pos_nm) / 10 #nm, minimum image if periodic
            dist_fine.append(d * 10)
            if n_steps % dcdfreq == 0:
                dcd_file.writeModel(pos)
                NaCl_dist.append(d)
            if fine is not None and n_steps % config.fine_every == 0:
                fine.add(n_steps, pos_nm, config.box_nm)
            if d * 10 >= config.target_distance:
                reached = True
                if fine is not None:
                    fine.add(n_steps, pos_nm, config.box_nm) #the arrival frame
                break

        file_handle.close()

        record_first_passage("first_passage_unbiased.csv", time_tag, 0, n_steps, reached, stepsize / 1000)
        if fine is not None:
            fine.close() #writes the ring: the last config.ring_ps before the first passage
        np.save(f"visited_state/{time_tag}_NaCl_unbias_dist_every{fpt_stride}steps.npy", np.asarray(dist_fine, dtype=np.float32))
        write_run_info(f"{time_tag}_run_info.json", context,
                       {'method': 'unbiased', 'first_passage_steps': n_steps if reached else None,
                        'steps_run': n_steps, 'ring_frames': ring})
        #save the NaCl_dist
        np.savetxt(f"visited_state/{time_tag}_NaCl_unbias_traj.txt", NaCl_dist)

    
    
    
    
    
    
    
