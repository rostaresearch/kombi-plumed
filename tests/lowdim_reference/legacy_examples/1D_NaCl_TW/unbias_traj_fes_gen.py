#this takes in a unbiased traj, and use boltzmann inversion to get the free energy profile.

import numpy as np
import matplotlib.pyplot as plt
from util import *
from dham import DHAM
#import mdtraj

#traj_path = 'trajectories/unbiased/20231019-164823_unbiased_traj.dcd'

#traj = mdtraj.load(traj_path, top='./toppar/step3_input25A.pdb')

#we load the data fromo visited_state folder. note unit is in nm.
#dist = np.loadtxt('visited_state/20231019-164823_NaCl_unbias_traj.txt')
dist = np.loadtxt('visited_state/20231019-164036_NaCl_unbias_traj.txt')
dist = dist*10 #convert to angstrom

#create a histogram, bin from 0 to 7.5A, with 0.1A bin size.
hist, bin_edges = np.histogram(dist, bins=np.linspace(2,9, 100+1))
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2

def random_initial_bias(initial_position):
    #returns random a,b,c for 10 gaussian functions. around the initial position
    # initial position is in Anstrom
    rng = np.random.default_rng()
    #a = np.ones(10)
    a = np.ones(10) * 0 #convert to kJ/mol
    b = rng.uniform(initial_position-0.5, initial_position+0.5, 10) /10 #min/max of preloaded NaCl fes x-axis.
    c = rng.uniform(1, 5.0, 10) /10
    return np.concatenate((a,b,c), axis=None)
    
def DHAM_it(CV, gaussian_params, T=300, lagtime=2, numbins=100, prop_index = None):
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
    d = DHAM(gaussian_params)
    d.setup(CV, T, prop_index)

    d.lagtime = lagtime
    d.numbins = numbins #num of bins, arbitrary.
    results = d.run(biased = True, plot=False) #result is [mU2, MM]
    return results

gaussian_params = random_initial_bias(initial_position = 2.3)

#run DHAM
mU2, MM = DHAM_it(CV=dist.reshape(-1,1), gaussian_params=gaussian_params, T=300, lagtime=2, numbins=100, prop_index = None)


#we plot the mU2.
plt.plot(np.linspace(2,9, 100), mU2, label='DHAM fes')
#plt.xlabel('NaCl distance (A)')
#plt.ylabel('mU2 (kJ/mol)')
#plt.savefig('NaCl_mU2_from_unbias.png')
#plt.close()




"""plt.figure()
plt.plot(bin_centers, hist)
plt.xlabel('NaCl distance (A)')
plt.ylabel('fes (kcal/mol)')
plt.savefig('NaCl_dist_from_unbias.png')
plt.show()
plt.close()
"""

#boltzmann distribution

prob = hist/np.sum(hist)
kT = 0.0019872041 * 300 #kcal/mol
fes = -np.log(prob)*kT

#plot the free energy surface
plt.plot(bin_centers, fes, label = 'boltzmann fes')
plt.xlabel('NaCl distance (A)')
plt.ylabel('Free Energy (kcal/mol)')
plt.legend()
plt.savefig('NaCl_fes_from_unbias.png')
plt.show()