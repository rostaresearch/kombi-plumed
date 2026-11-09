#this code analysis the traj.

import mdtraj
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from util import *
import config


import openmm
from openmm import unit
from openmm import Vec3
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.app.metadynamics import BiasVariable, Metadynamics
from openmm.unit import Quantity
def nm_to_angstrom(x):
    return x * 10  # 1 nm = 10 angstroms

#set figure global font size
plt.rcParams.update({'font.size': 16})

#first we get all traj

unbias_analysis = False
metaD_analysis = False
plot = False
plot_modified_fes = False
plot_metad = False
plot_metad_fes = False
plot_unbias = True

if plot_metad_fes:
    #we load the stored energy_cv.
    time_tag = '20231123-122721'
    energy_cv = np.load(f"./visited_states/{time_tag}_metaD_energy_CV.npy")
    
    upper = 2*np.pi
    lower = 0
    bin_width = 0.2

    #note our cv is defined as (x-x0)^2; x0 = 5.2
    #wrap it back to x
    
    gridwidth = int(np.ceil(5*(upper-lower)/bin_width)) #200.
    cv_space = np.linspace(lower, upper, gridwidth)

    #plot on cv_space.
    plt.figure()
    plt.plot(cv_space, energy_cv)
    plt.xlabel("CV space")
    plt.ylabel("Energy (kJ/mol)")
    plt.title("Energy as a function of CV space")
    plt.savefig(f"./figs/metaD/{time_tag}_metaD_energy_CV.png")
    plt.close()

if plot_metad:
    file_list = ['./trajectory/metaD/20231123-171538_metaD_traj.dcd',
                 './trajectory/metaD/20231123-171548_metaD_traj.dcd']
    
    #this chunk we get the fes. 
    ###############################
    elem = Element(0, "X", "X", 1.0)
    top = Topology()
    top.addChain()
    top.addResidue("xxx", top._chains[0])
    top.addAtom("X", elem, top._chains[0]._residues[0])

    mass = 12.0 * unit.amu
    #starting point as [1.29,-1.29,0.0]
    system = openmm.System()
    system.addParticle(mass)
    system, fes = apply_fes(system = system, 
                        particle_idx=0, 
                        gaussian_param = None, 
                        pbc = config.pbc, 
                        name = "FES", 
                        amp=config.amp, 
                        mode = config.fes_mode,
                        plot = True)
    x = np.linspace(0, 2*np.pi, config.num_bins)
    for file in file_list:
        #traj = mdtraj.load(file, top=top)
        #we load with a stride of 1000
        top = './trajectory/explore/20231101-140249_langevin_sim_explore_0.pdb'

        traj = mdtraj.load(file, top=top)
        pos = []
        for frame in traj:
            pos.append(frame.xyz[0, :])
        pos = np.array(pos).squeeze()[:,0] # we only take x-coor
        pos_digitized = np.digitize(pos, x)
        #we trim off anything beyond index 100
        pos_digitized = pos_digitized[pos_digitized < 100]
        #plot the traj
        plt.figure()
        plt.plot(x, fes, label="original fes")
        plt.xlabel("x-coor position (nm)")
        plt.ylabel("fes (kJ/mol)")
        plt.title("FES mode = multiwell, pbc=False")

        plt.scatter(x[pos_digitized], fes[pos_digitized], s=3.5, alpha = 0.5, c="black")
        plt.savefig(f"./figs/metaD/replot_{file.split('/')[-1].split('.')[0]}_traj.png")
        plt.close()


if plot_modified_fes:
    #we load uniased_profile
    unb_bins, unb_profile = np.load("Unbiased_Profile.npy")
    unb_bins = unb_bins[:len(unb_bins)//4]
    unb_profile = unb_profile[:len(unb_profile)//4]

    #the LJ potential is modified that e = -6kcal/mol, r = 2.5A
    #x = np.linspace(2.0, 9, 100)
    #we use unb_bins as x
    x = unb_bins
    LJ = (-6) * ((2.5/x)**12 - (2.5/x)**6)

    #plot the biased profile
    fig, ax = plt.subplots()
    ax.plot(x, unb_profile, label="unbiased")
    ax.plot(x, LJ, label="LJ")
    ax.plot(x, unb_profile + LJ, label="modified")
    ax.set_xlabel("Distance (A)")
    ax.set_ylabel("Free energy (kcal/mol)")
    ax.legend()
    plt.savefig("modified_fes.png")

if unbias_analysis:
    traj_path_list = []
    for file in os.listdir("trajectory/unbias/"):
        if file.endswith(".dcd"):
            if '20231128' in file.split('.')[0].split('_')[0]:
                traj_path_list.append(file)

    for traj_path in traj_path_list:

        print(f"Processing {traj_path}")
        traj = mdtraj.load(f"trajectory/unbias/{traj_path}", top="./trajectory/explore/20231101-133419_langevin_sim_explore_0.pdb")
        coor = traj.xyz[:,0,:2]
        np.savetxt(f"visited_states/{traj_path.split('.')[0]}.csv", coor, delimiter=",")


"""        NaCl_dist = mdtraj.compute_distances(traj, [[0, 1]]) #distance in nm
        NaCl_dist = np.array(NaCl_dist)

        #get the first time NaCl_dist > 7A
        for index_d, d in enumerate(NaCl_dist):
            if d > 7e-1:
                steps_to_7A = index_d
                break
            if index_d == len(NaCl_dist)-1:
                steps_to_7A = index_d
        
        with open("total_steps_unbiased.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([steps_to_7A * 100])
"""
if metaD_analysis:
    traj_path_list = []
    for file in os.listdir("trajectories/metaD/"):
        if file.endswith(".dcd"):
            traj_path_list.append(file)

    for traj_path in traj_path_list:
        traj = mdtraj.load(f"trajectories/metaD/{traj_path}", top="./toppar/step3_input25A.pdb")

        NaCl_dist = mdtraj.compute_distances(traj, [[0, 1]]) #distance in nm
        NaCl_dist = np.array(NaCl_dist)

        #get the first time NaCl_dist > 7A
        for index_d, d in enumerate(NaCl_dist):
            if d > 7e-1:
                steps_to_7A = index_d
                break
            if index_d == len(NaCl_dist)-1:
                steps_to_7A = index_d
        
        with open("total_steps_metaD.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([steps_to_7A * 100])

if plot:
    mfpt_data = np.genfromtxt("total_steps_mfpt.csv", delimiter=',')
    unbias_data = np.genfromtxt("total_steps_unbiased.csv", delimiter=',')
    metaD_data = np.genfromtxt("total_steps_metaD.csv", delimiter=',')

    #reshape it to 1D array
    mfpt_data = mfpt_data.reshape(-1)
    unbias_data = unbias_data.reshape(-1)
    metaD_data = metaD_data.reshape(-1)

    #one step is 0.002 ps. conver to ps.
    mfpt_data = mfpt_data * 0.002
    unbias_data = unbias_data * 0.002
    metaD_data = metaD_data * 0.002

    #plot the violin plot, show the data points as dots.
    """fig, ax = plt.subplots()
    ax.violinplot([mfpt_data, unbias_data, metaD_data])
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['MFPT', 'Unbiased', 'MetaD'])
    ax.set_ylabel("Time (ps)")
    ax.set_title("Time to reach destination")
    ax.set_yscale('log')
    plt.savefig("violin_plot_log.png")

    fig, ax = plt.subplots()
    ax.boxplot([mfpt_data, unbias_data, metaD_data])
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['MFPT', 'Unbiased', 'MetaD'])
    ax.set_ylabel("Time (ps)")
    ax.set_title("Time to reach destination")
    ax.set_yscale('log')
    plt.savefig("box_plot_log.png")"""

    fig, ax = plt.subplots()
    ax.boxplot([mfpt_data, unbias_data], positions=[1, 2])
    box3 = ax.boxplot(metaD_data, positions=[3], patch_artist=True)
    for box in box3['boxes']:
        box.set_linestyle('-')
    for whisker in box3['whiskers']:
        whisker.set_linestyle('-')
    for cap in box3['caps']:
        cap.set_linestyle('-')
    for median in box3['medians']:
        median.set_linestyle('-')

    # Setting x-ticks and labels
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['MFPT', 'Unbiased (Estimated)', 'MetaD'])

    # Adding the y-axis label, title and setting y-scale to log
    ax.set_ylabel("Time (ps)")
    ax.set_title("Time to reach destination")
    ax.set_yscale('log')

    # Adding legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='b', lw=1, linestyle='-'),
                    Line2D([0], [0], color='b', lw=1, linestyle='-')]
    #ax.legend(legend_elements, ['Completed'], loc='best')

    # Saving the figure
    plt.savefig("./figs/box_plot_log.png")
    plt.close()

if plot_unbias:
    strideint = 10
    file_path = "./trajectory/unbias/20240422-153858_unbias_traj.dcd"#20240328-113249_unbias_traj.dcd"
    traj = mdtraj.load(file_path, top="./trajectory/explore/20231101-140249_langevin_sim_explore_0.pdb",
                       stride=strideint)

    #this chunk we get the fes. 
    ###############################
    elem = Element(0, "X", "X", 1.0)
    top = Topology()
    top.addChain()
    top.addResidue("xxx", top._chains[0])
    top.addAtom("X", elem, top._chains[0]._residues[0])

    mass = 12.0 * unit.amu
    #starting point as [1.29,-1.29,0.0]
    system = openmm.System()
    system.addParticle(mass)
    system, fes = apply_fes(system = system, 
                        particle_idx=0, 
                        gaussian_param = None, 
                        pbc = config.pbc, 
                        name = "FES", 
                        amp=config.amp, 
                        mode = config.fes_mode,
                        plot = True)

    fes = fes/4.184 #convert to kcal/mol

    #plot
    pos = []
    for frame in traj:
        pos.append(frame.xyz[0, :])
    pos = np.array(pos).squeeze()


    for index, p in enumerate(pos):
        if np.linalg.norm(p[:2] - np.array([1.0, 1.0])) < 0.1:
            print(f"Reached [1.0, 1.0] at step {index}")
            break

    time = index * 0.002 * 500 * strideint # in ps.
    time = time/1e6 #in us

    #plot the traj
    x = np.linspace(0, 2*np.pi+1, 100)
    plt.figure()
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.2)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))

    plt.plot(x,fes)
    plt.xlabel("x (Å)")
    plt.ylabel("FES (kcal/mol)")
    plt.title(f"Time to reach target state is {time:.2f} μs")
    plt.xlim(1, 7)
    #plot the traj
    plt.scatter(pos[:,0], fes[np.digitize(pos[:,0], x)], s=3.5, alpha = 0.5, c="black")
    plt.plot(pos[-1,0], fes[np.digitize(pos[-1,0], x)], 'ro')
    plt.savefig(f"./figs/unbias/replot_{file_path.split('/')[-1].split('.')[0]}_traj.png")
    plt.close()
