"""Run actual adaptive 2D Langevin dynamics and verify the public plotting example."""
from pathlib import Path
import json
import gzip
import subprocess
import sys
import numpy as np
from test_plumed import LIB, PLUMED

ROOT = Path(__file__).resolve().parents[1]


def test_analytical_demo_dynamics_and_energy_plot(tmp_path):
    folder = tmp_path / 'demo'
    result = subprocess.run([sys.executable, str(ROOT / 'examples/analytical_2d/run.py'),
                             '--output', str(folder), '--library', str(LIB),
                             '--plumed', PLUMED, '--steps', '10001'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((folder / 'summary.json').read_text())
    col = np.loadtxt(folder / 'model_2d.COLVAR')
    assert np.isfinite(col).all()
    np.testing.assert_array_equal(col[0, 1:3], [5., 4.])
    assert np.ptp(col[:, 1]) > .01 and np.ptp(col[:, 2]) > .01
    if summary['target_reached']:
        assert np.linalg.norm(col[-1, 1:3] - [1., 1.5]) <= .1
        assert summary['biased_first_passage_ps'] == summary['elapsed_ps']
    else:
        assert summary['steps_completed'] == 10001
        assert summary['complete_kombi_windows'] == 2
    # Exercise the same compressed-data route used by the bundled public sample.
    raw = folder / 'model_2d.COLVAR'
    with gzip.open(folder / 'model_2d.COLVAR.gz', 'wb') as handle:
        handle.write(raw.read_bytes())
    raw.unlink()
    result = subprocess.run([sys.executable, str(ROOT / 'examples/analytical_2d/plot.py'), str(folder)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (folder / 'trajectory.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
