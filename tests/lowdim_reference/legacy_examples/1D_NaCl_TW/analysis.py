#this code analysis the traj.

import mdtraj
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

#set matplotlib font size
plt.rcParams.update({'font.size': 20})
#first we get all traj

unbias_analysis = False
metaD_analysis = False
plot = True
plot_modified_fes = False

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
    for file in os.listdir("trajectories/unbiased/"):
        if file.endswith(".dcd"):
            traj_path_list.append(file)

    for traj_path in traj_path_list:
        traj = mdtraj.load(f"trajectories/unbiased/{traj_path}", top="./toppar/step3_input25A.pdb")

        NaCl_dist = mdtraj.compute_distances(traj, [[0, 1]]) #distance in nm
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
    import seaborn as sns
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

    # Set a style
    #sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    # Box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=[mfpt_data, unbias_data, metaD_data], ax=ax)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['MSM-opt MD', 'Classic MD', 'MetaDynamics'])
    ax.set_ylabel("Time (ps)")
    ax.set_title("Time to unbind Na-Cl pair")
    ax.set_yscale('log')
    plt.savefig("box_plot_log.png", dpi=300)