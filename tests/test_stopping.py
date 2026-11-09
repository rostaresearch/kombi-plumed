"""Physical first hits stop on the hit step, independently of sampled model lag."""
import numpy as np
import pytest
from test_plumed import inputs,run
from test_models import input_grid,run_points

def print_fpt(path):
    f=path/'plumed.dat'
    f.write_text(f.read_text().replace('k.window FILE','k.window,k.fpt FILE'))

def test_ras_first_hit_beyond_grid_stops_and_restarts(tmp_path):
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    for path in (full,resumed):
        inputs(path,pace=100,extra='STOP_AT_TARGET');print_fpt(path)
    x=[7.4,7.5,8.1,7.7,8.3]
    run(full,x);run(resumed,x[:2]);run(resumed,x[1:],start=1,restart=True)
    data=np.loadtxt(full/'COLVAR')
    assert len(data)==3
    np.testing.assert_array_equal(data[:,-1],[-1,-1,.004])
    assert (full/'k.state').read_bytes()==(resumed/'k.state').read_bytes()
    assert (full/'k.history').read_bytes()==(resumed/'k.history').read_bytes()
    # Replaying the finished checkpoint stops again and preserves its state.
    saved=(full/'k.state').read_bytes()
    run(full,x[2:],start=2,restart=True)
    assert (full/'k.state').read_bytes()==saved

@pytest.mark.parametrize('dim',[1,2])
def test_lowdim_radius_uses_first_hit_between_sampled_frames(tmp_path,dim):
    input_grid(tmp_path,dim,pace=100)
    f=tmp_path/'plumed.dat'
    f.write_text(f.read_text().replace('CHECKPOINT_STRIDE=1','STOP_AT_TARGET TARGET_RADIUS=.1 SAMPLE_STRIDE=4 CHECKPOINT_STRIDE=1'))
    print_fpt(tmp_path)
    xy=np.array([[4.,0],[5.04,0],[5.16,0],[5.2,0]]) if dim==1 else np.array([[5.,4.],[1.2,1.5],[1.06,1.5],[1.,1.5]])
    run_points(tmp_path,xy)
    data=np.loadtxt(tmp_path/'COLVAR')
    assert len(data)==3
    np.testing.assert_array_equal(data[:,-1],[-1,-1,.004])

def test_start_at_target_has_zero_first_passage_time(tmp_path):
    inputs(tmp_path,pace=100,extra='STOP_AT_TARGET');print_fpt(tmp_path)
    run(tmp_path,[8.,8.1])
    data=np.atleast_2d(np.loadtxt(tmp_path/'COLVAR'))
    assert len(data)==1 and data[0,-1]==0
