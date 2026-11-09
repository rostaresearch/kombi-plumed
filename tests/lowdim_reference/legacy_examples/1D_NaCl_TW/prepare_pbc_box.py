#prepare the periodic start structure for the NaCl runs (config.nonbonded_method = 'PME').
#the CHARMM-GUI input (566 TIP3P, step3_input25A.pdb) has no box (CRYST1 0 0 0), and the published
#runs used it without one: a cube of water in vacuum. here we
#  1. put it in a cubic box sized for the water density, PME with a 1.0 nm cutoff;
#  2. NPT (300 K, 1 bar) for npt_ps, with the ion pair kept in contact (flat-bottom, r <= 2.8 A;
#     the published runs started at 2.57 A);
#  3. take the mean box length over the second half of the NPT run;
#  4. rescale to that box, then NVT (300 K) for nvt_ps with the same restraint;
#  5. write the structure with the ion pair at the box centre and every water molecule whole
#     inside the box, plus the box length.
#usage: python prepare_pbc_box.py <toppar dir> [npt_ps] [nvt_ps]
#writes <toppar dir>/step3_input25A_pbc.pdb, <toppar dir>/box_nm.txt and prepare_pbc_box.log
import os
import sys

import numpy as np
import openmm as omm
import openmm.app as omm_app
import openmm.unit as unit

inp = sys.argv[1] if len(sys.argv) > 1 else 'toppar'
npt_ps = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
nvt_ps = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
T, fric, dt = 300 * unit.kelvin, 10 / unit.picoseconds, 0.002 * unit.picoseconds
platform = omm.Platform.getPlatformByName(os.environ.get('OPENMM_PLATFORM', 'CUDA'))
log = open('prepare_pbc_box.log', 'w')

def say(*a):
    print(*a, flush=True)
    print(*a, file=log, flush=True)

psf = omm_app.CharmmPsfFile(os.path.join(inp, 'step3_input.psf'))
pdb = omm_app.PDBFile(os.path.join(inp, 'step3_input25A.pdb'))
params = omm_app.CharmmParameterSet(os.path.join(inp, 'toppar_water_ions.str'))
names = [a.name for a in psf.topology.atoms()]
assert names[0] == 'SOD' and names[1] == 'CLA'
n_water = sum(1 for n in names if n in ('OH2', 'O'))
mass_g = (n_water * 18.015 + 22.990 + 35.453) / 6.02214076e23
L0 = (mass_g / 0.99) ** (1 / 3) * 1e7 #nm, for 0.99 g/cm^3
say(f"{n_water} waters; initial box {L0:.4f} nm")

def system_for(L, barostat):
    psf.setBox(L * unit.nanometers, L * unit.nanometers, L * unit.nanometers)
    system = psf.createSystem(params, nonbondedMethod=omm_app.PME, nonbondedCutoff=1.0*unit.nanometers,
                              constraints=omm_app.HBonds)
    keep = omm.CustomBondForce('step(r-r0)*0.5*k*(r-r0)^2') #flat bottom: ions stay in contact
    keep.addGlobalParameter('r0', 0.28)
    keep.addGlobalParameter('k', 10000.0)
    keep.setUsesPeriodicBoundaryConditions(True)
    keep.addBond(0, 1)
    system.addForce(keep)
    if barostat:
        system.addForce(omm.MonteCarloBarostat(1 * unit.bar, T, 25))
    integ = omm.LangevinIntegrator(T, fric, dt)
    return omm.Context(system, integ, platform), integ

def min_image(d, L):
    return d - L * np.round(d / L)

#NPT
ctx, integ = system_for(L0, True)
ctx.setPositions(pdb.positions)
omm.LocalEnergyMinimizer.minimize(ctx, 10, 2000)
ctx.setVelocitiesToTemperature(T)
n_frames = int(npt_ps)
box = []
for k in range(n_frames):
    integ.step(500) #1 ps
    st = ctx.getState(getPositions=True)
    box.append(st.getPeriodicBoxVectors()[0][0].value_in_unit(unit.nanometers))
box = np.array(box)
L = float(box[n_frames // 2:].mean())
dens = mass_g / (L * 1e-7) ** 3
say(f"NPT {npt_ps:.0f} ps: box {box[0]:.4f} -> {box[-1]:.4f} nm; mean of 2nd half {L:.4f} +- "
    f"{box[n_frames // 2:].std():.4f} nm; density {dens:.4f} g/cm^3")

#rescale to L (molecule centres scaled; molecules kept rigid) and NVT
x = st.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
Lf = st.getPeriodicBoxVectors()[0][0].value_in_unit(unit.nanometers)
res_atoms = [[a.index for a in r.atoms()] for r in pdb.topology.residues()]
for idx in res_atoms:
    c = x[idx].mean(0)
    x[idx] += c * (L / Lf - 1)
ctx, integ = system_for(L, False)
ctx.setPositions(x)
omm.LocalEnergyMinimizer.minimize(ctx, 10, 2000)
ctx.setVelocitiesToTemperature(T)
integ.step(int(round(nvt_ps / 0.002)))
st = ctx.getState(getPositions=True, getEnergy=True)
x = st.getPositions(asNumpy=True).value_in_unit(unit.nanometers)

#ion pair at the box centre, Cl as the minimum image of Na, every water molecule whole in the box
shift = np.full(3, L / 2) - x[0]
x = x + shift
x[1] = x[0] + min_image(x[1] - x[0], L)
for idx in res_atoms:
    if idx[0] in (0, 1):
        continue
    ref = x[idx[0]]
    x[idx] -= L * np.floor(ref / L)
d = np.linalg.norm(x[0] - x[1]) * 10
say(f"NVT {nvt_ps:.0f} ps at L = {L:.4f} nm: potential energy "
    f"{st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol; Na-Cl {d:.2f} A")

top = pdb.topology
top.setPeriodicBoxVectors(((L, 0, 0), (0, L, 0), (0, 0, L)) * unit.nanometers)
with open(os.path.join(inp, 'step3_input25A_pbc.pdb'), 'w') as f:
    omm_app.PDBFile.writeFile(top, x * unit.nanometers, f)
with open(os.path.join(inp, 'box_nm.txt'), 'w') as f:
    f.write(f"{L:.6f}\n")
say(f"wrote {os.path.join(inp, 'step3_input25A_pbc.pdb')} and box_nm.txt")
