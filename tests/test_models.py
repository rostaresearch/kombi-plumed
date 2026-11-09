"""Corrected October 1D/2D hard-bin code: actual native mappings, models and forces."""
from pathlib import Path
import importlib.util
import os
import subprocess
import numpy as np
import pytest
from test_native_core import probe,flat
from test_plumed import LIB,PLUMED,values,gauss

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('legacy_regressions',ROOT/'tests/lowdim_reference/tests/test_legacy_drivers.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)

@pytest.fixture
def one(monkeypatch):
    return reference._load(monkeypatch,'1D_langevin_sim',num_gaussian=2,num_bins=5)

@pytest.fixture
def two(monkeypatch):
    return reference._load(monkeypatch,'2D_langevin_sim',num_gaussian=2,num_bins=4)

def grid_probe(dim,n,mode,points,clip=0):
    pts=np.c_[points,np.zeros(len(points))] if dim==1 else np.asarray(points)
    ny=1 if dim==1 else n
    out=probe('grid',f'{n} {dim} {ny} {mode} {clip} 0 {2*np.pi:.17g} 0 {2*np.pi:.17g} {len(pts)}\n'+flat(pts))
    ns=n*ny
    return np.c_[out[:ns],out[ns:2*ns]],out[2*ns:].astype(int)

def test_legacy_1d_centers_and_invalid_pairs_match_source(one):
    points=np.array([-.1,0,1.3,2.6,5.1,2*np.pi,7])
    grid,bins=grid_probe(1,5,3,points)
    edges=np.linspace(0,2*np.pi,6)
    np.testing.assert_allclose(grid[:,0],one.util.state_centres(edges)[:5],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(bins,np.minimum(np.digitize(points,edges),5))
    actual=probe('counts','5 1 7\n'+' '.join(map(str,bins))).reshape(5,5)
    expected,_=one.dham.count_transitions(np.digitize(points,edges)[None,:],5,1)
    np.testing.assert_array_equal(actual,expected)

def test_legacy_2d_order_centers_and_clipping_match_source(two):
    xy=np.array([[-1,-1],[0,0],[.5,2.5],[5,4],[7,8],[1,1.5]])
    grid,bins=grid_probe(2,4,2,xy,1)
    cx=two.util.state_centres(4)
    gx,gy=np.meshgrid(cx,cx)
    np.testing.assert_allclose(grid,np.c_[gx.ravel(),gy.ravel()],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(bins,two.util.digitize_xy(xy,4))

def test_normal_bin_centers_are_explicit_separate_mode():
    points=np.array([[.1,.1],[1.7,3.5],[6,6]])
    grid,bins=grid_probe(2,4,1,points)
    edges=np.linspace(0,2*np.pi,5);centers=(edges[:-1]+edges[1:])/2
    x,y=np.meshgrid(centers,centers)
    np.testing.assert_allclose(grid,np.c_[x.ravel(),y.ravel()])
    i=np.digitize(points[:,0],edges)-1;j=np.digitize(points[:,1],edges)-1
    np.testing.assert_array_equal(bins,i+4*j)

@pytest.mark.parametrize('mode,dim,n,low,high',[(0,1,100,1.,8.),(3,1,100,0.,2*np.pi),(2,2,20,0.,2*np.pi),(1,1,100,0.,2*np.pi)])
def test_digitization_at_exact_numpy_edges_and_adjacent_floats(mode,dim,n,low,high):
    edges=np.linspace(low,high,n+(mode in (1,3)))
    x=np.r_[np.nextafter(edges[1:-1],-np.inf),edges[1:-1],np.nextafter(edges[1:-1],np.inf)]
    xy=np.c_[x,np.full(len(x),low)]
    ny=n if dim==2 else 1
    out=probe('grid',f'{n} {dim} {ny} {mode} 0 {low:.17g} {high:.17g} {low:.17g} {high:.17g} {len(xy)}\n'+flat(xy))
    actual=out[2*n*ny:].astype(int)
    expected=np.digitize(x,edges)
    if mode==1:expected-=1
    expected=np.minimum(expected,n if mode==3 else n-1)
    if dim==2:expected+=n  # y=low is digitized to bin 1 in the legacy 2D convention.
    np.testing.assert_array_equal(actual,expected)

def test_2d_bias_layout_energy_and_both_force_components(two):
    import openmm as mm
    from openmm import unit
    params=np.array([.5,1.7,1.1,4.3,2.7,.8,.4,1.3,1.1,.6])
    points=np.array([[1.4,2.3],[4.1,1.2],[3.3,3.4]])
    actual=probe('bias2','2 3\n'+flat(params)+'\n'+flat(points)).reshape(3,3)
    system=mm.System();system.addParticle(1)
    two.util.apply_bias(system,0,params,num_gaussians=2)
    integ=mm.VerletIntegrator(.002);ctx=mm.Context(system,integ,mm.Platform.getPlatformByName('Reference'))
    for row,xy in zip(actual,points):
        ctx.setPositions([[*xy,0]])
        state=ctx.getState(getEnergy=True,getForces=True)
        assert row[0]==pytest.approx(state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole),rel=2e-14)
        force=state.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole/unit.nanometer)[0,:2]
        np.testing.assert_allclose(-row[1:],force,rtol=2e-14,atol=2e-14)

P1=np.array([.01,.02,2.,3.,.7,1.])
P2=np.array([.01,.02,1.,3.,2.,1.,.7,1.,.8,.9])

def input_grid(path,dim,pace=12,lag=1,mode=None):
    path.mkdir(parents=True,exist_ok=True)
    mode=mode or ('LEGACY_1D' if dim==1 else 'LEGACY_2D')
    params=P1 if dim==1 else P2
    n=5 if dim==1 else 4
    axes='p.x' if dim==1 else 'p.x,p.y'
    bounds='0' if dim==1 else '0,0'
    maximum=str(2*np.pi) if dim==1 else f'{2*np.pi},{2*np.pi}'
    bins=str(n) if dim==1 else f'{n},{n}'
    target='5.2' if dim==1 else '1,1.5'
    clip='CLIP_GRID' if mode=='LEGACY_2D' else ''
    (path/'plumed.dat').write_text(f'''UNITS LENGTH=nm ENERGY=kcal/mol TIME=ps
LOAD FILE={LIB}
p: POSITION ATOM=1 NOPBC
k: KOMBI ...
 ARG={axes} GRID_MODE={mode} {clip} GRID_MIN={bounds} GRID_MAX={maximum} BINS={bins} TARGET={target}
 KBT=.5961612777 TILT_KBT=.5981 PACE={pace} LAG={lag} NGAUSSIANS=2
 AMP_MIN=.1 AMP_MAX=2 SIGMA_MIN=.3 SIGMA_MAX=1.5
 RANDOM_AMP=1 RANDOM_SIGMA_MIN=.3 RANDOM_SIGMA_MAX=1.5 RANDOM_TRIALS=25 MAX_ITER=100 TOL=.0001
 INITIAL_PARAMS={values(params)} CHECKPOINT_STRIDE=1 SEED=5544
...
PRINT ARG={axes},k.bias,k.mfpt,k.window FILE=COLVAR STRIDE=1 FMT=%24.17g
''')

def run_points(path,points,start=0,restart=False):
    xyz=''.join(f'1\n20 20 20\nH {x:.17g} {y:.17g} 0\n' for x,y in points)
    (path/'traj.xyz').write_text(xyz)
    cmd=[PLUMED,'driver','--ixyz','traj.xyz','--timestep','.002','--initial-step',str(start),'--dump-forces','forces.xyz','--dump-forces-fmt','%.17g']
    if restart:cmd.append('--restart')
    r=subprocess.run(cmd,cwd=path,text=True,capture_output=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
    (path/f'run-{start}.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout[-3000:]+r.stderr[-3000:]

def get_history(path,n):
    lines=(path/'k.history').read_text().splitlines();entries=[];i=2
    while i<len(lines):
        d={'skipped':int(lines[i].split()[-1])};i+=1
        d['params']=np.fromstring(lines[i+1],sep=' ');i+=2
        d['bias']=np.fromstring(lines[i+1],sep=' ');i+=2
        assert lines[i]=='COUNTS';d['counts']=np.array([np.fromstring(s,sep=' ') for s in lines[i+1:i+1+n]]);i+=1+n
        tokens=lines[i].split();m=int(tokens[1]);d['start']=int(tokens[3]);d['target']=int(tokens[5])
        d['keep']=np.fromstring(lines[i+1],sep=' ',dtype=int);i+=2
        d['model']=np.array([np.fromstring(s,sep=' ') for s in lines[i+1:i+1+m]]);i+=1+m
        d['mfpt']=float(lines[i].split()[2]);i+=1
        d['next']=np.fromstring(lines[i+1],sep=' ');i+=2;assert lines[i]=='END_WINDOW';i+=1
        entries.append(d)
    return entries

@pytest.mark.parametrize('dim,lag',[(1,1),(1,3),(2,1),(2,3)])
def test_loaded_lowdim_counts_models_and_restart(tmp_path,dim,lag,one,two):
    if dim==1:
        samples=np.array([1.3,1.3,2.6,2.6,1.3,6.,6.,1.3,2.6,1.3,2.6,1.3])
        xy=np.c_[np.r_[1.3,np.tile(samples,3)],np.zeros(37)]
    else:
        # Deliberately asymmetric directed cycle with diagonal/multi-bin transitions.
        samples=np.array([[.5,.5],[.5,.5],[2.5,.5],[2.5,.5],[2.5,2.5],[2.5,2.5],
                          [.5,2.5],[.5,2.5],[.5,.5],[2.5,2.5],[.5,.5],[2.5,2.5]])
        xy=np.vstack(([[.5,.5]],np.tile(samples,(3,1))))
    full=tmp_path/'full';resumed=tmp_path/'resumed';input_grid(full,dim,lag=lag);input_grid(resumed,dim,lag=lag)
    run_points(full,xy);run_points(resumed,xy[:7]);run_points(resumed,xy[6:],6,True)
    assert (full/'k.state').read_bytes()==(resumed/'k.state').read_bytes()
    assert (full/'k.history').read_bytes()==(resumed/'k.history').read_bytes()
    n=5 if dim==1 else 16;records=get_history(full,n);assert len(records)==3
    cs=[];ps=[]
    for k,r in enumerate(records):
        window=xy[1+k*12:1+(k+1)*12]
        if dim==1:
            ids=np.digitize(window[:,0],np.linspace(0,2*np.pi,6));mods=one
        else:ids=two.util.digitize_xy(window,4);mods=two
        c,_=mods.dham.count_transitions(ids[None,:],n,lag)
        np.testing.assert_array_equal(c,r['counts']);cs.append(c)
        if dim==1:
            final=int(np.digitize(5.2,np.linspace(0,2*np.pi,6)))
            expected_start=np.argmin(abs(r['keep']-ids[-1]))
            expected_target=np.argmin(abs(r['keep']-final))
        else:
            most_visited=np.argmax(np.bincount(ids))
            expected_start=np.argmin(abs(r['keep']-most_visited))
            final=int(two.util.digitize_xy(np.array([[1.,1.5]]),4)[0])
            distances=(r['keep']%4-final%4)**2+(r['keep']//4-final//4)**2
            expected_target=np.argmin(distances)
            assert most_visited != ids[-1], 'fixture must distinguish modal and last-bin objectives'
        assert r['start']==expected_start
        assert r['target']==expected_target
        ps.append(r['params'].reshape(1+2*dim,2).T)
        if dim==1:
            d=one.dham.DHAM(np.array(ps),5);d.qspace=np.linspace(0,2*np.pi,6)
        else:
            d=two.dham.DHAM(np.array(ps))
        expected=d.build_MM(np.sum(cs,axis=0),np.array(cs).sum(2),True)
        np.testing.assert_allclose(r['model'],np.asarray(expected)[np.ix_(r['keep'],r['keep'])],rtol=3e-14,atol=2e-15)
        assert r['skipped']==12-lag-c.sum()
        assert np.isfinite(r['mfpt']) and r['mfpt']>0
    # Confirm first new bias only appears on the step AFTER each completed window.
    col=np.loadtxt(full/'COLVAR')
    for step,point in enumerate(xy):
        params=(P1 if dim==1 else P2) if step<=12 else records[(step-1)//12-1]['next']
        if dim==1:e=gauss([point[0]],params)[0][0]
        else:e=sum(two.dham.gaussian_2d(*point,*row) for row in params.reshape(5,2).T)
        assert col[step,dim+1]==pytest.approx(e,rel=3e-14,abs=1e-14)

def test_loaded_2d_cartesian_force_and_finite_difference(tmp_path):
    input_grid(tmp_path,2,pace=1000)
    xy=np.array([[1.4,2.3],[1.4+1e-5,2.3],[1.4-1e-5,2.3],[1.4,2.3+1e-5],[1.4,2.3-1e-5]])
    run_points(tmp_path,xy)
    data=np.loadtxt(tmp_path/'COLVAR');lines=(tmp_path/'forces.xyz').read_text().splitlines()
    force=np.array(list(map(float,lines[2].split()[-3:])))
    numeric=-np.array([data[1,3]-data[2,3],data[3,3]-data[4,3]])/(2e-5)*4.184
    np.testing.assert_allclose(force[:2],numeric,rtol=3e-9,atol=1e-11)
