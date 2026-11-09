"""Compare two independent validate_transport.py runs exactly."""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('baseline', type=Path)
ap.add_argument('candidate', type=Path)
args = ap.parse_args()
with np.load(args.baseline / 'values.npz') as a, np.load(args.candidate / 'values.npz') as b:
    assert set(a.files) == set(b.files), 'Different comparison arrays'
    for key in a.files:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    print(f'PASS: {len(a.files)} arrays agree exactly')
