"""TYK2 centroid force, in OpenMM nm/kJ units; KOMBI uses Angstrom/kcal."""
import numpy as np
import openmm as mm
from openmm import unit
import config

def minimum_image(d,box):
    f=d@np.linalg.inv(box)
    return (f-np.rint(f))@box

def add_bias(system,ligand,pocket):
    terms=[f'a{i}*exp(-0.5*((distance(g1,g2)-b{i})/c{i})^2)' for i in range(config.num_gaussian)]
    f=mm.CustomCentroidBondForce(2,'+'.join(terms)+'+cv_probe*distance(g1,g2)')
    f.addGlobalParameter('cv_probe',0)
    f.addEnergyParameterDerivative('cv_probe')
    for i in range(config.num_gaussian):
        for key,value in [('a',0),('b',0),('c',.1)]:f.addGlobalParameter(f'{key}{i}',value)
    f.addGroup(ligand,[1.]*len(ligand));f.addGroup(pocket,[1.]*len(pocket))
    f.addBond([0,1],[]);f.setUsesPeriodicBoundaryConditions(True);f.setForceGroup(30)
    system.addForce(f)
    return f

def set_bias(ctx,p):
    a,b,c=np.asarray(p).reshape(3,config.num_gaussian)
    for i in range(config.num_gaussian):
        for key,value in [('a',a[i]*4.184),('b',b[i]/10),('c',c[i]/10)]:
            ctx.setParameter(f'{key}{i}',float(value))

def read_cv(ctx):
    s=ctx.getState(getEnergy=True,getParameterDerivatives=True,groups={30})
    return float(s.getEnergyParameterDerivatives()['cv_probe'])*10, s.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)

def geometry(ctx,ligand,pocket,protein):
    state=ctx.getState(getPositions=True,getEnergy=True,enforcePeriodicBox=False)
    x=np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    box=np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
    for g in (ligand,pocket):
        if np.max(np.abs((x[g]-x[g[0]])@np.linalg.inv(box)))>.5:
            raise RuntimeError('Centroid group spans a periodic cell: inspect imaging before continuing')
    r=float(np.linalg.norm(minimum_image(x[ligand].mean(0)-x[pocket].mean(0),box)))*10
    d=minimum_image(x[ligand,None,:]-x[None,protein,:],box)
    closest=float(np.linalg.norm(d,axis=2).min())*10
    energy=state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    if not np.isfinite(x).all() or not np.isfinite(energy):raise RuntimeError('Non-finite MD')
    engine_r,_=read_cv(ctx)
    if abs(engine_r-r)>1e-4:raise RuntimeError('GPU CV and independent coordinate centroid disagree')
    return x,box,r,closest,energy

def force_guard(sim,bias,restore):
    """Full-system platform force vs independent Reference finite differences."""
    from util import optimiser_total_bias
    state=sim.context.getState(getPositions=True)
    x=state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    r,_=read_cv(sim.context)
    trial=np.r_[np.linspace(.2,5,10),r+np.linspace(-1,1,10),np.linspace(.7,2,10)]
    set_bias(sim.context,trial)
    r_gpu,e=read_cv(sim.context)
    assert abs(e-optimiser_total_bias(np.array([r_gpu]),trial)[0])<1e-3
    actual=sim.context.getState(getForces=True,groups={30}).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    refsys=mm.System()
    for _ in x:refsys.addParticle(1.)
    refsys.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    refsys.addForce(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(bias)))
    integ=mm.VerletIntegrator(.002)
    ctx=mm.Context(refsys,integ,mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(x);set_bias(ctx,trial)
    expected=ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    platform_error=float(np.max(abs(actual-expected)))
    assert platform_error<2e-3,platform_error
    errors=[]
    atoms=[int(bias.getGroupParameters(i)[0][0]) for i in range(2)]
    for atom in atoms:
        for axis in range(3):
            energies=[]
            for sign in [-1,1]:
                y=x.copy();y[atom,axis]+=sign*1e-5;ctx.setPositions(y)
                energies.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            errors.append(abs(expected[atom,axis]+(energies[1]-energies[0])/2e-5))
    assert max(errors)<1e-4,errors
    del ctx,integ
    set_bias(sim.context,restore)
    return dict(passed=True,max_finite_difference_error_kj_mol_nm=max(errors),platform_force_error_kj_mol_nm=platform_error)
