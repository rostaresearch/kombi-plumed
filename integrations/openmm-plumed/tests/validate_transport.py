"""Deterministic replay tests for the transport, independent of MD sampling."""
import argparse,ctypes,gc,json,os,sys
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import unit
ctypes.CDLL(os.environ['PLUMED_KERNEL'],mode=ctypes.RTLD_GLOBAL)
from openmmplumed import PlumedForce
ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--platform',default='CUDA');ap.add_argument('--output-root',type=Path,default=Path('runs/bridge'));args=ap.parse_args()
root=(args.output_root/args.name).resolve()
root.mkdir(parents=True,exist_ok=False);os.chdir(root)
values={};cases=[]
scripts={
 'distance':'d: DISTANCE ATOMS=11,121\nr: RESTRAINT ARG=d AT=0.7 KAPPA=5',
 'centroid':'a: COM ATOMS=11,12,13,14\nb: COM ATOMS=100,101,102\nd: DISTANCE ATOMS=a,b\nr: RESTRAINT ARG=d AT=0.7 KAPPA=5',
 'dynamic':'d: DISTANCE ATOMS=11,121\ne: DISTANCE ATOMS=61,71\nr: RESTRAINT ARG=d AT=0.7 KAPPA=5\nPRINT ARG=e STRIDE=7 FILE=extra',
 'empty':'c: CONSTANT VALUE=0.3\nr: RESTRAINT ARG=c AT=0.7 KAPPA=5'
}
for precision in (['single','mixed','double'] if args.platform=='CUDA' else ['double']):
 for case,script in scripts.items():
    folder=root/(precision+'_'+case);folder.mkdir();os.chdir(folder)
    system=mm.System()
    for i in range(128):system.addParticle(12+(i%4))
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3,0,0),mm.Vec3(.2,3,0),mm.Vec3(.1,.3,3))
    nb=mm.NonbondedForce();nb.setNonbondedMethod(nb.CutoffPeriodic);nb.setCutoffDistance(.8)
    for i in range(128):nb.addParticle(0,.1,0)
    nb.setForceGroup(0);system.addForce(nb)
    f=PlumedForce(script);f.setForceGroup(5);system.addForce(f)
    integ=mm.VerletIntegrator(.001)
    props={'Precision':precision,'DeterministicForces':'true'} if args.platform=='CUDA' else {}
    ctx=mm.Context(system,integ,mm.Platform.getPlatformByName(args.platform),props)
    rng=np.random.default_rng(91823);x=rng.uniform(.05,2.5,(128,3));v=np.zeros_like(x)
    ctx.setVelocities(v)
    for step in range(24):
        # Image shifts, a changing triclinic box, and atom reordering after setPositions.
        pos=x.copy();pos[10,0]+=.011*step;pos[120,1]-=.017*step
        pos[::3,0]+=3*(step%3-1)
        ctx.setPeriodicBoxVectors(mm.Vec3(3+.001*step,0,0),mm.Vec3(.2,3,0),mm.Vec3(.1,.3,3))
        ctx.setStepCount(step);ctx.setPositions(pos)
        s=ctx.getState(getEnergy=True,groups=1<<5)
        values[f'{precision}_{case}_{step}_energy_only']=np.array(s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        for repeat in range(2):
            s=ctx.getState(getForces=True,getEnergy=True,groups=1<<5)
            key=f'{precision}_{case}_{step}_{repeat}'
            values[key+'_force']=s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            values[key+'_energy']=np.array(s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        # Excluded force group must not advance PLUMED or add a bias force.
        s=ctx.getState(getForces=True,getEnergy=True,groups=1)
        assert np.max(abs(s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)))==0
    checkpoint=ctx.createCheckpoint();ctx.loadCheckpoint(checkpoint)
    s=ctx.getState(getForces=True,getEnergy=True,groups=1<<5)
    values[f'{precision}_{case}_checkpoint_force']=s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    cases.append(precision+'_'+case)
    del ctx,integ,system;gc.collect()
np.savez_compressed(root/'values.npz',**values)
(root/'result.json').write_text(json.dumps({'cases':cases,'arrays':len(values),'sparse':os.environ.get('OPENMM_PLUMED_SPARSE_CUDA','0')},indent=2)+'\n')
print('VALIDATED',len(cases),'cases',len(values),'arrays')
