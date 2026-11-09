"""Regenerate native PLUMED potentials directly from the frozen OpenMM source.

Requires the test dependencies. The generated .dat files need only PLUMED.
Run from any directory: python examples/make_examples.py
"""
from pathlib import Path
import contextlib
import importlib.util
import io
import re
import sys
import numpy as np
import pytest
import openmm as mm

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('legacy_examples_for_export',ROOT/'tests/lowdim_reference/tests/test_legacy_drivers.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)

def potential_system(dim):
    with pytest.MonkeyPatch.context() as patch:
        mod=reference._load(patch,f'{dim}D_langevin_sim',num_gaussian=20,num_bins=100 if dim==1 else 20)
        system=mm.System();system.addParticle(12.)
        with contextlib.redirect_stdout(io.StringIO()):
            mod.util.apply_fes(system,0,amp=6,mode='multiwell',plot=False)
        # The original MD drivers also confine the unused Cartesian dimensions.
        for expr in (['1e3*y^2','1e3*z^2'] if dim==1 else ['1e3*z^2']):
            force=mm.CustomExternalForce(expr);force.addParticle(0);system.addForce(force)
    return system

def csv(values):return ','.join(format(float(x),'.17g') for x in values)

def write_potential(dim):
    terms=[]
    system=potential_system(dim)  # retain the owner of the SWIG force views
    for force in system.getForces():
        expr=force.getEnergyFunction()
        for i in range(force.getNumGlobalParameters()):
            name=force.getGlobalParameterName(i)
            value=force.getGlobalParameterDefaultValue(i)
            expr=re.sub(r'\b'+re.escape(name)+r'\b',f'({value:.17g})',expr)
        terms.append('('+expr.replace(' ','')+')')
    expr='('+'+'.join(terms)+')/4.184'
    (HERE/f'potential_{dim}d.dat').write_text(
        '# Generated from frozen October corrected util.apply_fes(amp=6, mode=multiwell).\n'
        '# Caller must set UNITS LENGTH=nm ENERGY=kcal/mol and supply one particle.\n'
        'p: POSITION ATOM=1 NOPBC\n'
        f'u: CUSTOM ARG=p.x,p.y,p.z VAR=x,y,z FUNC={expr} PERIODIC=NO\n'
        'physical: BIASVALUE ARG=u\n')

def write_model(dim):
    rng=np.random.default_rng(20261103+dim);g=20
    if dim==1:
        # Historical initializer used .01*4.184 as the amplitude parameter.
        p=np.r_[np.full(g,.01*4.184),rng.uniform(0,2*np.pi,g),rng.uniform(0,2*np.pi,g)]
        settings='ARG=p.x GRID_MODE=LEGACY_1D GRID_MIN=0 GRID_MAX=6.283185307179586 BINS=100 TARGET=5.2\n AMP_MIN=.1 AMP_MAX=1.2 SIGMA_MIN=.6 SIGMA_MAX=5\n RANDOM_AMP=.6 RANDOM_SIGMA_MIN=.6 RANDOM_SIGMA_MAX=2'
    else:
        p=np.r_[np.full(g,.01),rng.uniform(0,2*np.pi,g),rng.uniform(0,2*np.pi,g),rng.uniform(1,5,g),rng.uniform(1,5,g)]
        settings='ARG=p.x,p.y GRID_MODE=LEGACY_2D CLIP_GRID GRID_MIN=0,0 GRID_MAX=6.283185307179586,6.283185307179586 BINS=20,20 TARGET=1,1.5\n AMP_MIN=.1 AMP_MAX=2 SIGMA_MIN=.3 SIGMA_MAX=1.5\n RANDOM_AMP=1 RANDOM_SIGMA_MIN=.3 RANDOM_SIGMA_MAX=1.5'
    (HERE/f'model_{dim}d.dat').write_text(
        '# Run from examples/ with one particle; this file supplies physical AND adaptive forces.\n'
        '# Initial parameters are a documented deterministic example, not a historical walker seed.\n'
        'UNITS LENGTH=nm ENERGY=kcal/mol TIME=ps\nLOAD FILE=../build/Kombi.so\n'
        f'INCLUDE FILE=potential_{dim}d.dat\n'
        'k: KOMBI ...\n '+settings+'\n'
        ' KBT=.5961612777 TILT_KBT=.5981 NGAUSSIANS=20 PACE=5000 LAG=1 SAMPLE_STRIDE=1\n'
        ' STOP_AT_TARGET TARGET_RADIUS=.1\n'
        ' RANDOM_TRIALS=1000 MAX_ITER=15000 TOL=.0001 SEED=20261103\n'
        f' INITIAL_PARAMS={csv(p)}\n'
        f' STATE_FILE=model_{dim}d.state HISTORY_FILE=model_{dim}d.history CHECKPOINT_STRIDE=5000\n...\n'
        f'PRINT ARG=p.x,p.y,u,k.bias,k.mfpt,k.window,k.fpt FILE=model_{dim}d.COLVAR STRIDE=1 FMT=%24.17g\n')

def write_ras():
    rng=np.random.default_rng(20261103);g=10
    p=np.r_[np.full(g,.01),rng.uniform(3.5,3.9,g),rng.uniform(1,5,g)]
    (HERE/'ras_pi.dat').write_text(
        '# Ras Pi A5/L20 example. Atom IDs are ONLY for the original step3 topology.\n'
        '# Verify Mg=7800 and phosphate P=7841 against YOUR topology (PLUMED is 1-based).\n'
        '# Minimum-image distance follows the current 73-run periodic Ras campaign.\n'
        '# At 2 fs, PACE=10000 is 20 ps. The MD engine supplies the physical force field.\n'
        'UNITS LENGTH=A ENERGY=kcal/mol TIME=ps\nLOAD FILE=../build/Kombi.so\n'
        'd: DISTANCE ATOMS=7800,7841\n'
        'k: KOMBI ...\n ARG=d GRID_MODE=RAS GRID_MIN=1 GRID_MAX=8 BINS=100 TARGET=8\n'
        ' KBT=.5961612777 TILT_KBT=.5981 NGAUSSIANS=10 PACE=10000 LAG=1 SAMPLE_STRIDE=1\n'
        ' STOP_AT_TARGET\n'
        ' AMP_MIN=.1 AMP_MAX=5 SIGMA_MIN=.7 SIGMA_MAX=2\n'
        ' RANDOM_AMP=.6 RANDOM_SIGMA_MIN=.7 RANDOM_SIGMA_MAX=1\n'
        ' RANDOM_TRIALS=1000 MAX_ITER=15000 TOL=.000001 SEED=20261103\n'
        f' INITIAL_PARAMS={csv(p)}\n'
        ' STATE_FILE=ras_pi.state HISTORY_FILE=ras_pi.history CHECKPOINT_STRIDE=10000\n...\n'
        'PRINT ARG=d,k.bias,k.mfpt,k.window,k.fpt FILE=ras_pi.COLVAR STRIDE=1 FMT=%24.17g\n')

def write_replay():
    (HERE/'replay.dat').write_text('''# Synthetic trajectory replay for installation verification, not an MD benchmark.
LOAD FILE=../build/Kombi.so
d: DISTANCE ATOMS=1,2 NOPBC
k: KOMBI ...
 ARG=d GRID_MODE=RAS GRID_MIN=0 GRID_MAX=8 BINS=9 TARGET=8
 KBT=.5961612777 TILT_KBT=.5981 PACE=12 LAG=1 NGAUSSIANS=2
 AMP_MIN=.1 AMP_MAX=5 SIGMA_MIN=.7 SIGMA_MAX=2
 RANDOM_AMP=.6 RANDOM_SIGMA_MIN=.7 RANDOM_SIGMA_MAX=1
 RANDOM_TRIALS=30 MAX_ITER=80 SEED=123
 INITIAL_PARAMS=.1,.2,1.5,2.5,.7,1 CHECKPOINT_STRIDE=1
 STATE_FILE=replay.state HISTORY_FILE=replay.history
...
PRINT ARG=d,k.bias,k.mfpt,k.window,k.fpt FILE=replay.COLVAR STRIDE=1 FMT=%24.17g
''')
    points=[1.2]+[1.2,1.2,2.2,2.2,1.2,1.2,2.2,2.2,1.2,1.2,2.2,1.2]*3
    (HERE/'replay.xyz').write_text(''.join(f'2\n20 20 20\nH 0 0 0\nH {x} 0 0\n' for x in points))

if __name__=='__main__':
    for dim in (1,2):write_potential(dim);write_model(dim)
    write_ras();write_replay()
