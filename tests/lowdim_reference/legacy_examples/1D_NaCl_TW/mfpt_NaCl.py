#Here we explore the NaCl FES energy landscape on Na-Cl distance. place random bias.
# the workflow:
# propagate the system for a number of steps.
#    we have to start from loading the molecule, FF
#    then we put bias, log the bias form in txt file. (random guess around starting position if first propagation)
#    then we propagate the system for a number of steps.
#    use DHAM, feed in the NaCl distance, bias params for current propagation, get free energy landscape
#    use the partial free energy landscape to generate next bias params.
#    repeat.
## import required packages
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

psf_file = 'toppar/step3_input.psf' # Path #tpr #prmtop
pdb_file = 'toppar/step3_input25A.pdb' # Path # gro # inpcrd
fricCoef = 10   # friction coefficient in 1/ps
max_propagation = 150

fig, ax = plt.subplots()


def random_initial_bias(initial_position):
    #returns random a,b,c for 10 gaussian functions. around the initial position
    # initial position is in Anstrom
    #kcal/mol and A, the same units as the optimised parameters (add_bias converts to kJ/mol, nm).
    #(propagation 0 was saved in kJ/mol and nm, which DHAM then read as kcal/mol and A.)
    rng = np.random.default_rng([config.seed, 0, 1]) if config.seed is not None else np.random.default_rng()
    #a = np.ones(10)
    a = np.ones(10) * 0.01
    b = rng.uniform(initial_position-0.5, initial_position+0.5, 10) #min/max of preloaded NaCl fes x-axis.
    c = rng.uniform(1, 5.0, 10)
    return np.concatenate((a,b,c), axis=None)
    
def DHAM_it(CV, gaussian_params, T=300, lagtime=2, num_bins=None, prop_index = None, time_tag = None):
    """
    intput:
    CV: the collective variable we are interested in. Na-Cl distance.
    gaussian_params: the parameters of bias potential. (in our case the 10-gaussian params)
     format: (a,b,c)
    T: temperature 300

    output:
    the Markov Matrix
    Free energy surface probed by DHAM.
    """
    d = DHAM(gaussian_params, num_bins=num_bins)
    d.setup(CV, T, prop_index=prop_index, time_tag=time_tag)

    d.lagtime = lagtime
    d.num_bins = num_bins #num of bins, arbitrary.
    results = d.run(biased = True, plot=False) #result is [mU2, MM]
    return results

def propagate(context, prop_index, NaCl_dist, time_tag, steps=config.propagation_step, dcdfreq=config.dcdfreq_mfpt,  platform=None, stepsize=config.stepsize, num_bins=config.num_bins, reach = None):
    """
    propagate the system for a number of steps.
    we have to start from loading the molecule, FF
    then we define global params for bias (random guess around starting position if first propagation)
    
    each time, we generate new bias
    and propagate the system for a number of steps.
    use the partial free energy landscape to generate next bias params.
    repeat.
    """
    
    file_handle = open(f"trajectories/explore/{time_tag}_NaCl_exploring_traj_{prop_index}.dcd", 'bw')
    dcd_file = omm_app.dcdfile.DCDFile(file_handle, psf.topology, dt = stepsize)

    dist = []
    for _ in tqdm(range(int(steps/dcdfreq))):
        integrator.step(dcdfreq) #advance dcd freq steps and stop to record.
        state = context.getState(getPositions=True)
        dist.append(nacl_distance(state.getPositions(asNumpy=True).value_in_unit(unit.nanometers))) #in angstrom; minimum image if periodic
        run_step = prop_index * steps + (_ + 1) * dcdfreq #MD steps since the start of the run
        if fine is not None and run_step % config.fine_every == 0:
            fine.add(run_step, state.getPositions(asNumpy=True).value_in_unit(unit.nanometers), config.box_nm)
        if _ % config.dcdfreq == 0:
            dcd_file.writeModel(state.getPositions(asNumpy=True))
    file_handle.close()

    gaussian_params = np.zeros([prop_index+1, config.num_gaussian, 3])
    for i in range(prop_index+1):
        #stored as [a(10), b(10), c(10)]; DHAM needs the per-Gaussian rows (a_i, b_i, c_i).
        gaussian_params[i,:,:] = params_to_gaussians(np.loadtxt(f"gaussian_params/{time_tag}_gaussian_params_{i}.txt"), config.num_gaussian)
        print(f"gaussian_params for propagation {i} loaded.")


    #now we have the trajectory, we can calculate the Markov matrix.
    #top = mdtraj.load_psf(psf_file)
    #traj = mdtraj.load_dcd(f"trajectories/explore/{time_tag}_NaCl_exploring_traj_{prop_index}.dcd", top=top)
    #dist = mdtraj.compute_distances(traj, [[0, 1]]) *10 #unit in A #get distance over the traj (in this propagation)
    #print("this is prop index", prop_index, "this is raw dist: ", dist)
    np.savetxt(f"distance/{time_tag}_NaCl_exploring_traj_{prop_index}.txt", dist) #save the distance to a txt file. 
    #print(f"Inside the propagate function, lenght of dist is: {len(dist)}")

    #first frame of this propagation at or beyond the target (frame index, 0-based).
    #(this loop had no break, so it kept the last visit, and multiplied by dcdfreq.)
    hit = first_hit(dist, config.target_distance) #note here is in angstrom.
    if hit is not None:
        reach = hit

    #we concatenate the new dist to the old dist.
    # NaCl_dist is a list of renewed dist.
    combined_dist = np.concatenate((NaCl_dist[-1], dist), axis=None)
    NaCl_dist.append(combined_dist)


    #plot it.
    #plt.plot(combined_dist)
    #plt.show()
    F_M, MM = DHAM_it(combined_dist.reshape(prop_index+1, -1, 1), 
                      gaussian_params, 
                      T=300, 
                      lagtime=1, 
                      num_bins=num_bins, 
                      time_tag = time_tag, 
                      prop_index = prop_index)

    cur_pos = combined_dist[-1] #the last position of the traj. not our cur_pos is the CV distance.
    
    return cur_pos, NaCl_dist, MM, reach, F_M #return the CV traj and the MM.

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

def add_bias(system, gaussian_params):
    """
    gaussian params: np.array([a,b,c])
    system: the openmm system object.
    """
    num_gaussian = gaussian_params.shape[0]//3
    a = gaussian_params[:num_gaussian] * 4.184 #kcal/mol to kJ/mol
    b = gaussian_params[num_gaussian:2*num_gaussian] / 10 #A to nm
    c = gaussian_params[2*num_gaussian:] / 10
    potential = ' + '.join(f'a{i}* exp(-(r-b{i})^2/(2*c{i}^2))' for i in range(num_gaussian)) #in openmm the energy terms in kJ/mol

    print(potential)
    custom_bias = omm.CustomBondForce(potential)
    
    for i in range(num_gaussian):
        custom_bias.addGlobalParameter(f'a{i}', a[i]) #unit is kj/mol. a is in kcal/mol. so we multiply by 4.184
        custom_bias.addGlobalParameter(f'b{i}', b[i]) #in openmm distance unit is nm. b is in A. so we divide by 10.
        custom_bias.addGlobalParameter(f'c{i}', c[i]) # same to b.
    
    custom_bias.setUsesPeriodicBoundaryConditions(is_periodic()) #minimum-image r with a periodic box
    custom_bias.addBond(0, 1)
    system.addForce(custom_bias)
    return system

def get_working_MM(M):
    zero_rows = np.where(~M.any(axis=1))[0]
    zero_cols = np.where(~M.any(axis=0))[0]

    keep_indices = np.setdiff1d(range(M.shape[0]), np.union1d(zero_rows, zero_cols))
    M_work = M[np.ix_(keep_indices, keep_indices)]
    return M_work, keep_indices

def get_closest_state(qspace, target_state, working_indices):
    """
    usesage: qspace = np.linspace(2.0, 9, 150+1)
    target_state = 7 #find the closest state to 7A.
    """

    working_states = qspace[working_indices] #the NaCl distance of the working states.
    closest_state = working_states[np.argmin(np.abs(working_states - target_state))]
    return closest_state


###############################################
# here we start the main python process:
# propagate -> acess the Markov Matrix -> biasing -> propagate ...
###############################################

if __name__ == "__main__":

    psf = omm_app.CharmmPsfFile(psf_file)
    pdb = omm_app.PDBFile(pdb_file)
    unb_bins, unb_profile = np.load("Unbiased_Profile.npy")
    unb_profile = 4.184 * unb_profile #convert to kJ/mol
    """with open("output_files/NaCl_solvated_system", 'r') as file_handle:
        xml = file_handle.read()
    system = omm.XmlSerializer.deserialize(xml)
    """
    #forcefield = omm_app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    params = omm_app.CharmmParameterSet('toppar/toppar_water_ions.str') #we modified the combined LJ term between NaCl to have a -6.0kcal.mol at 2.5A

    for i_sim in range(config.num_sim):
        system = create_nacl_system(psf, params) #explicit nonbondedMethod, see config.

        platform = config.platform #CUDA if available
        
        #### setup an OpenMM context
        integrator = omm.LangevinIntegrator(config.T*unit.kelvin, #Desired Integrator
                                            fricCoef/unit.picoseconds,
                                            config.stepsize)
        qspace = np.linspace(2.0, 9, config.num_bins+1) #hard coded for now.
        NaCl_dist = [[]] #initialise the NaCl distance list.
        time_tag = time.strftime("%Y%m%d-%H%M%S")
        reach = None
        i = 0
        total_steps = None  #recorder to count the total steps reaching target.
        #all atoms every config.fine_every steps for the whole run (off if 0)
        os.makedirs("fine", exist_ok=True)
        fine = FineFrames(f"fine/{time_tag}_kombi.xtc") if config.fine_every else None
        #for i in range(max_propagation):
        while reach is None:
            if i > max_propagation:
                print("max propagation reached.")
                break
            if i == 0:
                print("propagation number 0 STARTING.")
                gaussian_params = random_initial_bias(initial_position = 2.3) #kcal/mol and A, like all later propagations.
                #save the gaussian params to a txt file.
                np.savetxt(f"gaussian_params/{time_tag}_gaussian_params_{i}.txt", gaussian_params)
                
                biased_system = add_bias(system, gaussian_params)

                ## construct an OpenMM context
                context = make_context(biased_system, integrator) #seeded integrator, DeterministicForces if set
                context, energy = minimize(context)         #minimize the system
                set_velocities(context, config.T*unit.kelvin) #seeded if config.seed (the published runs started from zero velocities)

                ## MD run "propagation"
                cur_pos, NaCl_dist, M, reach, mU2= propagate(context, 
                                                prop_index=i,
                                                NaCl_dist = NaCl_dist,
                                                steps = config.propagation_step,
                                                dcdfreq = config.dcdfreq_mfpt,
                                                stepsize = config.stepsize,
                                                num_bins = config.num_bins,
                                                time_tag = time_tag,
                                                platform=platform, 
                                                reach = reach,
                                                )
                #print(M.shape)
                #finding the closest element in MM to the end point. 7A in np.linspace(2.0, 9, 150+1)
                #trim the zero rows and columns markov matrix to avoid 0 rows.
                #!!!do everything in index space. !!!
                cur_pos_index = np.digitize(cur_pos, qspace) #the big index on full markov matrix.

                working_MM, working_indices = get_working_MM(M) #we call working_index the small index. its part of the full markov matrix.
                final_index = np.digitize(config.target_distance, qspace) #the big index of the target (was 7 A while the run stopped at 8 A).
                farest_index = working_indices[np.argmin(np.abs(working_indices - final_index))] #get the closest to the final index in qspace.
                
                i += 1
                
            else:
                print(f"propagation number {i} STARTING.")
                #renew the gaussian params using returned MM.
                print("getting gaussian parameters")

                #here instead of the cur_pos_index, we use the average of the last 50 positions.
                #avg_last50 = np.mean(NaCl_dist[-1][-50:])
                #avg_last50_digitized = np.digitize(avg_last50, qspace)
                
                print("farest index is: ", farest_index)
                print("cur_pos_index is: ", cur_pos_index)

                gaussian_params = try_and_optim_M(working_MM, 
                                                working_indices = working_indices,
                                                num_gaussian=config.num_gaussian, 
                                                start_state=cur_pos_index, #this is in large index.
                                                end_state=farest_index, #this is in large index also.
                                                plot = False,
                                                rng = optimiser_rng(i), #seeded from config.seed if given
                                                )

                #save the gaussian params to a txt file.
                np.savetxt(f"gaussian_params/{time_tag}_gaussian_params_{i}.txt", gaussian_params)

                #we use context.setParameters() to update the bias potential.
                for j in range(10):
                    context.setParameter(f'a{j}', gaussian_params[j] * 4.184) #unit in openmm is kJ/mol, the a is fitted in kcal/mol, so we multiply by 4.184
                    context.setParameter(f'b{j}', gaussian_params[j+10] / 10) #unit openmm is nm, the b is fitted in A, so we divide by 10.
                    context.setParameter(f'c{j}', gaussian_params[j+20] / 10) #same to b.
                
                #we plot the total bias.
                test_gaussian_params = []
                for j in range(10):
                    test_gaussian_params.append(context.getParameter(f'a{j}')) #we keep energy in kj/mol in plot.
                    test_gaussian_params.append(context.getParameter(f'b{j}')*10) #b, c in A. 
                    test_gaussian_params.append(context.getParameter(f'c{j}')*10)

                test_total_bias = np.zeros_like(qspace)
                for n in range(len(test_gaussian_params)//3):
                    test_total_bias += gaussian(qspace, test_gaussian_params[3*n], test_gaussian_params[3*n+1], test_gaussian_params[3*n+2])
                
                plt.figure()
                plt.plot(unb_bins, unb_profile, label="ground truth (LJ unmodified FES)")
                mU2 = mU2 * 4.184 #convert to kJ/mol
                plt.plot(qspace[:config.num_bins], mU2, label="reconstructed M_FES by DHAMsym")
                plt.plot(qspace, test_total_bias, label = "total bias")
                #plot the cur_pos
                plt.axvline(x=cur_pos, color='r', linestyle='--', label="current position")
                plt.xlim(2.0, 9)
                plt.xlabel("NaCl distance (A)")
                plt.ylabel("Free energy (kJ/mol)")
                plt.legend()
                plt.savefig(f"bias_plots/{time_tag}_bias_{i}.png")
                plt.close()

                ## MD run "propagation"
                cur_pos, NaCl_dist, M, reach, mU2= propagate(context, 
                                                prop_index=i,
                                                NaCl_dist = NaCl_dist,
                                                steps = config.propagation_step,
                                                dcdfreq = config.dcdfreq_mfpt,
                                                stepsize = config.stepsize,
                                                num_bins = config.num_bins,
                                                time_tag = time_tag,
                                                platform=platform, 
                                                reach = reach,
                                                )
                print(M.shape)
                cur_pos_index = np.digitize(cur_pos, qspace) #the big index on full markov matrix.

                working_MM, working_indices = get_working_MM(M) #we call working_index the small index. its part of the full markov matrix.
                farest_index = working_indices[np.argmin(np.abs(working_indices - final_index))] #get the closest to the final index in qspace.
                
                i += 1
                

        #first passage: all earlier propagations plus the steps of the hitting propagation up to
        #and including the first frame in the target. (was i*propagation_step + reach, i.e. the
        #last propagation counted in full plus the index of the last visit.)
        stepsize_ps = config.stepsize.value_in_unit(unit.picoseconds)
        if reach is None:
            total_steps = i * config.propagation_step #not reached: steps run.
        else:
            total_steps = first_passage_steps(i - 1, reach, config.dcdfreq_mfpt, config.propagation_step)
        print("first passage steps: ", total_steps, "reached:", reach is not None)
        record_first_passage("first_passage_mfpt.csv", time_tag, i, total_steps, reach is not None, stepsize_ps)
        if fine is not None:
            fine.close()
        write_run_info(f"{time_tag}_run_info.json", context,
                       {'method': 'kombi', 'first_passage_steps': int(total_steps), 'reached': reach is not None,
                        'n_propagations': i})
        #save the NaCl_dist
        np.savetxt(f"visited_state/{time_tag}_NaCl_exploring_traj.txt", NaCl_dist[-1])

    
    
    
    
    
    
    
