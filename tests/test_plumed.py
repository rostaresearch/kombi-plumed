"""Real PLUMED driver: loading, forces, adaptive windows, checkpoints and guards."""
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PLUMED=os.environ.get('PLUMED','plumed')
LIB=Path(os.environ.get('KOMBI_LIBRARY',ROOT/'build/Kombi.so')).resolve()
INITIAL=np.array([.1,.2,1.5,2.5,.7,1.])

def values(v):return ','.join(format(float(x),'.17g') for x in v)

def inputs(folder,params=INITIAL,extra='',pace=12,lag=1,stride=1,units='',cv='d: DISTANCE ATOMS=1,2 NOPBC',scaleE=1,scaleX=1):
    folder.mkdir(exist_ok=True,parents=True)
    text=f'''{units}
LOAD FILE={LIB}
{cv}
k: KOMBI ...
 ARG=d GRID_MIN=0 GRID_MAX={8*scaleX} BINS=9 TARGET={8*scaleX}
 KBT=.5961612777 TILT_KBT=.5981
 PACE={pace} LAG={lag} SAMPLE_STRIDE={stride} NGAUSSIANS={len(params)//3}
 AMP_MIN={.1*scaleE} AMP_MAX={5*scaleE} SIGMA_MIN={.7*scaleX} SIGMA_MAX={2*scaleX}
 RANDOM_AMP={.6*scaleE} RANDOM_SIGMA_MIN={.7*scaleX} RANDOM_SIGMA_MAX={scaleX}
 RANDOM_TRIALS=30 MAX_ITER=80 SEED=123
 INITIAL_PARAMS={values(params)} CHECKPOINT_STRIDE=1 {extra}
...
PRINT ARG=d,k.bias,k.mfpt,k.window FILE=COLVAR STRIDE=1 FMT=%24.17g
'''
    (folder/'plumed.dat').write_text(text)

def run(folder,distances,start=0,restart=False,length='nm',ok=True):
    xyz=''.join(f'2\n20 20 20\nH 0 0 0\nH {x:.17g} 0 0\n' for x in distances)
    (folder/'traj.xyz').write_text(xyz)
    command=[PLUMED,'driver','--plumed','plumed.dat','--ixyz','traj.xyz','--length-units',length,'--timestep','.002','--initial-step',str(start),'--dump-forces','forces.xyz','--dump-forces-fmt','%.17g']
    if restart:command.append('--restart')
    result=subprocess.run(command,cwd=folder,capture_output=True,text=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
    (folder/f'plumed-{start}.log').write_text(result.stdout+result.stderr)
    if ok:assert result.returncode==0,result.stdout[-4000:]+result.stderr[-4000:]
    else:assert result.returncode!=0
    return result.stdout+result.stderr

def read_forces(folder):
    lines=(folder/'forces.xyz').read_text().splitlines()
    return np.array([[list(map(float,lines[k+j].split()[-3:])) for j in (2,3)] for k in range(0,len(lines),4)])

def gauss(x,p):
    a,b,s=p.reshape(3,-1);v=a*np.exp(-.5*((np.asarray(x)[:,None]-b)/s)**2)
    return v.sum(1),(-v*(np.asarray(x)[:,None]-b)/s**2).sum(1)

def history(folder):
    lines=(folder/'k.history').read_text().splitlines();out=[];i=2
    while i<len(lines):
        head=lines[i].split();assert head[0]=='WINDOW';d={'window':int(head[1]),'step':int(head[3])};i+=1
        assert lines[i]=='APPLIED';d['applied']=np.fromstring(lines[i+1],sep=' ');i+=2
        assert lines[i]=='BIAS_ON_GRID';d['bias']=np.fromstring(lines[i+1],sep=' ');i+=2
        assert lines[i]=='COUNTS';d['counts']=np.array([np.fromstring(x,sep=' ') for x in lines[i+1:i+10]]);i+=10
        w=lines[i].split();n=int(w[1]);d['start']=int(w[3]);d['target']=int(w[5]);d['keep']=np.fromstring(lines[i+1],sep=' ',dtype=int);i+=2
        assert lines[i]=='MODEL_ROW_ORIENTED';d['model']=np.array([np.fromstring(x,sep=' ') for x in lines[i+1:i+1+n]]);i+=1+n
        f=lines[i].split();d['random']=float(f[1]);d['mfpt']=float(f[2]);i+=1
        assert lines[i]=='NEXT';d['next']=np.fromstring(lines[i+1],sep=' ');i+=2
        assert lines[i]=='END_WINDOW';i+=1;out.append(d)
    return out

@pytest.fixture
def trajectory():
    return np.r_[1.2,np.tile([1.2,1.2,2.2,2.2,1.2,1.2,2.2,2.2,1.2,1.2,2.2,1.2],3)]

def test_loaded_action_ten_hill_forces(tmp_path):
    # Same ten-hill case as the project force test, converted kcal/A -> kJ/nm.
    params=np.r_[np.linspace(.1,5,10)*4.184,np.linspace(3,5,10)*.1,np.linspace(.7,2,10)*.1]
    inputs(tmp_path,params,pace=10000,scaleE=4.184,scaleX=.1)
    x=np.linspace(.2,.6,13);run(tmp_path,x)
    col=np.loadtxt(tmp_path/'COLVAR');energy,derivative=gauss(x,params)
    np.testing.assert_allclose(col[:,2],energy,rtol=2e-14,atol=1e-12)
    f=read_forces(tmp_path)
    np.testing.assert_allclose(f[:,1,0],-derivative,rtol=2e-13,atol=1e-10)
    np.testing.assert_allclose(f[:,0,0],derivative,rtol=2e-13,atol=1e-10)
    np.testing.assert_allclose(f.sum(1),0,atol=1e-12)
    # Independent central differences in the actual loaded PLUMED action.
    eps=1e-6;x0=.411
    fd=tmp_path/'finite';inputs(fd,params,pace=10000,scaleE=4.184,scaleX=.1);run(fd,[x0-eps,x0,x0+eps])
    e=np.loadtxt(fd/'COLVAR')[:,2]
    assert read_forces(fd)[1,1,0]==pytest.approx(-(e[2]-e[0])/(2*eps),rel=2e-9)

def test_active_unit_conversion(tmp_path):
    p=np.r_[np.linspace(.1,5,10),np.linspace(3,5,10),np.linspace(.7,2,10)]
    inputs(tmp_path,p,pace=10000,units='UNITS LENGTH=A ENERGY=kcal/mol TIME=ps')
    x=np.array([3.,4.,5.]);run(tmp_path,x,length='A')
    col=np.loadtxt(tmp_path/'COLVAR');energy,derivative=gauss(x,p)
    np.testing.assert_allclose(col[:,1],x,atol=1e-13)
    np.testing.assert_allclose(col[:,2],energy,rtol=3e-14)
    # Driver force length unit follows --length-units; its energy unit is kJ/mol.
    np.testing.assert_allclose(read_forces(tmp_path)[:,1,0],-derivative*4.184,rtol=3e-13,atol=1e-10)

@pytest.mark.parametrize('lag,stride',[(1,1),(3,1),(1,2)])
def test_adaptive_windows_and_next_step_bias(tmp_path,trajectory,lag,stride):
    inputs(tmp_path,lag=lag,stride=stride);run(tmp_path,trajectory)
    h=history(tmp_path);assert len(h)==3
    col=np.loadtxt(tmp_path/'COLVAR')
    for k,d in enumerate(h):
        sampled=trajectory[12*k+stride:12*(k+1)+1:stride]
        bins=np.digitize(sampled,np.linspace(0,8,9));expected=np.zeros((9,9))
        np.add.at(expected,(bins[:-lag],bins[lag:]),1)
        np.testing.assert_array_equal(d['counts'],expected)
        assert d['counts'].sum()==12//stride-lag
        assert 0<d['mfpt']<=d['random']
        assert np.isfinite(d['model']).all()
        np.testing.assert_allclose(d['model'].sum(1),1,atol=1e-14)
    for step,x in enumerate(trajectory):
        p=INITIAL if step<=12 else h[(step-1)//12-1]['next']
        expected=gauss([x],p)[0][0]
        assert col[step,2]==pytest.approx(expected,rel=2e-14)
    assert col[12,4]==0 and col[13,4]==1
    assert col[13,3]==pytest.approx(h[0]['mfpt'])

@pytest.mark.parametrize('split',[7,12,19,24])
def test_restart_equals_uninterrupted(tmp_path,trajectory,split):
    full=tmp_path/'full';resumed=tmp_path/'resumed';inputs(full);inputs(resumed)
    run(full,trajectory);run(resumed,trajectory[:split+1]);run(resumed,trajectory[split:],start=split,restart=True)
    assert (full/'k.state').read_bytes()==(resumed/'k.state').read_bytes()
    assert (full/'k.history').read_bytes()==(resumed/'k.history').read_bytes()
    a=np.loadtxt(full/'COLVAR');b=np.loadtxt(resumed/'COLVAR')
    b=np.delete(b,split+1,axis=0) # PRINT emits the replayed checkpoint step once more
    np.testing.assert_allclose(a,b,rtol=0,atol=0)

@pytest.mark.parametrize('what',['settings','corruption','gap','position','history','history_content','timestep'])
def test_restart_rejects_mismatch(tmp_path,trajectory,what):
    inputs(tmp_path);run(tmp_path,trajectory[:8]);start=7;tail=trajectory[7:9].copy()
    if what=='settings':
        p=tmp_path/'plumed.dat';p.write_text(p.read_text().replace('SEED=123','SEED=124'))
    elif what=='corruption':
        p=tmp_path/'k.state';p.write_text(p.read_text().replace('KOMBI_STATE_V1','KOMBI_STATE_V2'))
    elif what=='gap':start=9
    elif what=='position':tail[0]+=.01
    elif what=='history':
        with (tmp_path/'k.history').open('a') as f:f.write('END_WINDOW\n')
    elif what=='history_content':
        p=tmp_path/'k.history';p.write_text(p.read_text().replace('HISTORY_V1','HISTORY_V2'))
    elif what=='timestep':
        p=tmp_path/'plumed.dat';p.write_text('UNITS TIME=fs\n'+p.read_text())
    log=run(tmp_path,tail,start=start,restart=True,ok=False)
    assert 'KOMBI' in log

def test_nonadjacent_transition_fails_and_preserves_evidence(tmp_path):
    inputs(tmp_path,pace=4)
    log=run(tmp_path,[1.2,1.2,4.2,4.2,1.2],ok=False)
    assert 'non-neighbor transition' in log
    assert (tmp_path/'k.state.failed').exists()

def test_terminal_only_arrival_cannot_be_trimmed(tmp_path):
    inputs(tmp_path,pace=4)
    log=run(tmp_path,[1.2,1.2,1.2,1.2,2.2],ok=False)
    assert 'trimming would discard probability' in log

@pytest.mark.parametrize('x',[0.,7.999999,8.,-1.,float('nan')])
def test_grid_boundaries(tmp_path,x):
    inputs(tmp_path,cv='pos: POSITION ATOM=2 NOPBC\nd: COMBINE ARG=pos.x PERIODIC=NO')
    run(tmp_path,[x],ok=(0<=x<8))

def test_periodic_cv_rejected(tmp_path):
    inputs(tmp_path,cv='d: TORSION ATOMS=1,2,3,4')
    p=subprocess.run([PLUMED,'driver','--plumed','plumed.dat','--parse-only','--natoms','4'],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode!=0 and 'nonperiodic scalar' in p.stdout+p.stderr

def test_two_actions_have_independent_files_and_identical_models(tmp_path,trajectory):
    inputs(tmp_path)
    p=tmp_path/'plumed.dat';text=p.read_text();action=text[text.index('k: KOMBI'):text.index('PRINT ARG=')]
    p.write_text(text.replace('PRINT ARG=',action.replace('k: KOMBI','other: KOMBI')+'PRINT ARG='))
    run(tmp_path,trajectory)
    assert (tmp_path/'k.state').read_bytes()==(tmp_path/'other.state').read_bytes()
    assert (tmp_path/'k.history').read_bytes()==(tmp_path/'other.history').read_bytes()

def test_restart_next_step_without_replay(tmp_path,trajectory):
    full=tmp_path/'full';resumed=tmp_path/'resumed';inputs(full);inputs(resumed)
    run(full,trajectory);run(resumed,trajectory[:13]);run(resumed,trajectory[13:],start=13,restart=True)
    assert (full/'k.state').read_bytes()==(resumed/'k.state').read_bytes()
    np.testing.assert_array_equal(np.loadtxt(full/'COLVAR'),np.loadtxt(resumed/'COLVAR'))

@pytest.mark.parametrize('old,new,expected',[
    ('BINS=9','BINS=1','BINS must'),
    ('SAMPLE_STRIDE=1','SAMPLE_STRIDE=5','invalid STRIDE'),
    ('LAG=1','LAG=12','invalid STRIDE'),
    ('AMP_MAX=5','AMP_MAX=.05','invalid parameter bounds'),
    ('KBT=.5961612777','KBT=0','must be positive'),
    ('NGAUSSIANS=2','NGAUSSIANS=3','INITIAL_PARAMS'),
    ('SIGMA_MIN=0.7','SIGMA_MIN=0.8','random starts outside bounds'),
])
def test_invalid_settings(tmp_path,old,new,expected):
    inputs(tmp_path);p=tmp_path/'plumed.dat';text=p.read_text();assert old in text;p.write_text(text.replace(old,new,1))
    assert expected in run(tmp_path,[1.2],ok=False)
