"""Check delivered model-potential files against the original physical forces."""
from pathlib import Path
import importlib.util
import os
import subprocess
import numpy as np
import pytest
import openmm as mm
from openmm import unit
from test_plumed import PLUMED,LIB

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('make_examples',ROOT/'examples/make_examples.py')
examples=importlib.util.module_from_spec(spec);spec.loader.exec_module(examples)

@pytest.mark.parametrize('dim',[1,2])
def test_native_physical_potentials_match_original_openmm(tmp_path,dim):
    points=np.array([[.01,.1,0],[1.12,1.34,0],[2.,0,0],[5.,4.,0],[1.,1.5,0],
                     [3.14,3.14,0],[6.2,6.,0],[-.7,-.7,0],[7.,7.,0]])
    (tmp_path/'plumed.dat').write_text('UNITS LENGTH=nm ENERGY=kcal/mol\n'
        f'INCLUDE FILE={ROOT}/examples/potential_{dim}d.dat\n'
        'PRINT ARG=u FILE=COLVAR STRIDE=1 FMT=%24.17g\n')
    (tmp_path/'traj.xyz').write_text(''.join('1\n20 20 20\nH '+' '.join(map(str,p))+'\n' for p in points))
    r=subprocess.run([PLUMED,'driver','--ixyz','traj.xyz','--dump-forces','forces.xyz','--dump-forces-fmt','%.17g'],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    energy=np.loadtxt(tmp_path/'COLVAR')[:,1]
    lines=(tmp_path/'forces.xyz').read_text().splitlines()
    force=np.array([list(map(float,lines[i].split()[-3:])) for i in range(2,len(lines),3)])
    ctx=mm.Context(examples.potential_system(dim),mm.VerletIntegrator(.002),mm.Platform.getPlatformByName('Reference'))
    for k,point in enumerate(points):
        ctx.setPositions([point]);state=ctx.getState(getEnergy=True,getForces=True)
        assert energy[k]==pytest.approx(state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole),rel=4e-14,abs=4e-14)
        expected=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)[0]
        np.testing.assert_allclose(force[k],expected,rtol=5e-13,atol=5e-13)

@pytest.mark.parametrize('dim',[1,2])
def test_delivered_model_input_loads(tmp_path,dim):
    data=(ROOT/f'examples/model_{dim}d.dat').read_text().replace('../build/Kombi.so',str(LIB)).replace(f'FILE=potential_{dim}d.dat',f'FILE={ROOT}/examples/potential_{dim}d.dat')
    (tmp_path/'plumed.dat').write_text(data)
    point='2 0 0' if dim==1 else '5 4 0'
    (tmp_path/'traj.xyz').write_text(f'1\n20 20 20\nH {point}\n')
    r=subprocess.run([PLUMED,'driver','--ixyz','traj.xyz','--timestep','.002'],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    assert np.isfinite(np.loadtxt(tmp_path/f'model_{dim}d.COLVAR')).all()

def test_delivered_ras_input_parses_for_reference_atom_count(tmp_path):
    data=(ROOT/'examples/ras_pi.dat').read_text().replace('../build/Kombi.so',str(LIB))
    (tmp_path/'plumed.dat').write_text(data)
    r=subprocess.run([PLUMED,'driver','--parse-only','--natoms','7841'],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr

def test_delivered_replay_completes_three_adaptive_windows(tmp_path):
    (tmp_path/'plumed.dat').write_text((ROOT/'examples/replay.dat').read_text().replace('../build/Kombi.so',str(LIB)))
    r=subprocess.run([PLUMED,'driver','--ixyz',str(ROOT/'examples/replay.xyz'),'--timestep','.002'],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    assert (tmp_path/'replay.history').read_text().count('END_WINDOW')==3
    col=np.loadtxt(tmp_path/'replay.COLVAR')
    assert col.shape==(37,6) and np.isfinite(col).all()
