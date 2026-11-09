#this code analysis the traj.

import mdtraj
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from util import *
import config


import openmm
from openmm import unit
from openmm import Vec3
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.app.metadynamics import BiasVariable, Metadynamics
from openmm.unit import Quantity


#first we get all traj

unbias_analysis = False
metaD_analysis = False
plot = False
plot_modified_fes = False
plot_metad = False
post_process_unbias = False
plot_unbias = True

def nm_to_angstrom(x):
    return x * 10  # 1 nm = 10 angstroms

if plot_metad:
    file_list = ['./trajectory/metaD/20231125-1621170_metaD_traj.dcd',
                 './trajectory/metaD/20231125-2005261_metaD_traj.dcd']
    
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

    for file in file_list:
        #traj = mdtraj.load(file, top=top)
        #we load with a stride of 1000
        top = './trajectory/explore/20231101-133419_langevin_sim_explore_0.pdb'

        traj = mdtraj.load(file, top=top)
        pos = []
        for frame in traj:
            pos.append(frame.xyz[0, :])
        pos = np.array(pos).squeeze()
        #plot the traj
        plt.figure()
        plt.imshow(fes, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=config.amp * 12/7 * 4.184, origin="lower")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(pos[:, 0], pos[:, 1], s=3.5, alpha = 0.5, c="black")
        plt.savefig(f"./figs/metaD/replot_{file.split('/')[-1].split('.')[0]}_traj.png")
        plt.close()

if post_process_unbias:
    file_list_all = os.listdir("./trajectory/unbias/")
    file_list = []
    for file in file_list_all:
        if "20231128" in file.split('.')[0].split('_')[0]:
            file_list.append(file)
    
    #we load dcd files and use mdtraj get the xyz coordinates.
    for file in file_list:
        traj = mdtraj.load(f"./trajectory/unbias/{file}", top="./trajectory/explore/20231101-133419_langevin_sim_explore_0.pdb")
        coor = traj.xyz[:,0,:] 
        #save coor in txt
        np.savetxt(f"./visited_states/{file.split('.')[0]}.txt", coor)


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
    ax.set_xticklabels(['MFPT', 'Unbiased (estimated)', 'MetaD'])

    # Adding the y-axis label, title and setting y-scale to log
    ax.set_ylabel("Time (ps)")
    ax.set_title("Time to reach destination")
    ax.set_yscale('log')

    # Saving the figure
    plt.savefig("./figs/box_plot_log.png")
    plt.close()

if plot_unbias:
    file_path = "./trajectory/unbias/20240322-100627_unbias_traj.dcd" #0240328-113222_unbias_traj.dcd" #20240329-221509_unbias_traj.dcd #"./trajectory/unbias/20240322-100627_unbias_traj.dcd"
    traj = mdtraj.load(file_path, top="./trajectory/explore/20231101-133419_langevin_sim_explore_0.pdb")

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

    #traj_tail = traj[-1000:]
    #plot
    pos = []
    for frame in traj:
        pos.append(frame.xyz[0, :])
    pos = np.array(pos).squeeze()

    for index, p in enumerate(pos):
        if np.linalg.norm(p[:2] - np.array([1.0, 1.0])) < 0.1:
            print(f"Reached [1.0, 1.0] at step {index}")
            break

    time = index * 0.002 * 500 # in ps.
    time = time/1e6 #in us

    #plot the traj
    plt.figure()
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.2)
    
    plt.imshow(fes, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], origin="lower")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))

    cbar=plt.colorbar()
    cbar.set_label("U (kcal/mol)")
    plt.xlabel("x (Å)")
    plt.xlim([0, 2*np.pi])
    plt.ylim([0, 2*np.pi])
    plt.ylabel("y (Å)")

    plt.title(f"Time to reach target state is {time:.2f} μs")
    plt.scatter(pos[::10, 0], pos[::10, 1], s=3.5, alpha = 0.5, c="black")
    #here we connect the last 100 points.
    plt.plot(pos[-400:,0], pos[-400:,1], 'b-', alpha = 0.5)
    plt.plot(pos[-1,0], pos[-1,1], 'ro', markersize=10)
    plt.savefig(f"./figs/unbias/replot_{file_path.split('/')[-1].split('.')[0]}_traj.png")
    plt.close()
