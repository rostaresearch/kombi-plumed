"""TYK2 adaptation of the corrected Ras A5/L20 hard-bin KOMBI campaign.

Energy kcal/mol, CV Angstrom, time in OpenMM units. Geometry is TYK2-specific;
the numerical production core and search settings are inherited from Ras.
"""
import numpy as np
from openmm import unit
T=300
stepsize=.002*unit.picoseconds
dcdfreq_mfpt=1
dham_lag_steps=1
propagation_step=10000
num_gaussian=10
amp_bounds=(.1,5.)
width_bounds=(.7,2.)
random_start_amp=.6
random_start_width=(.7,1.)
n_random_starts=1000
initial_amp=.01
qspace_num_bins=100
DHAM_num_bins=100
qspace_low=0.
qspace_high=40.
qspace=np.linspace(qspace_low,qspace_high,qspace_num_bins)
center_bounds=(qspace_low,qspace_high)
target_distance=40. # User-authorized model target; endpoint is independently native6
use_dynamic_bins=False
mfpt_method='state_reduction'
minimum_contact_A=10.
exit_persistence_ps=20
bias_check_tol=1e-3
bias_check_rtol=1e-4

protocol_revision="native6_target40_v2"
native_endpoint_A=6.
