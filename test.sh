#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1
python=${PYTHON:-python3}
"$python" -c 'import numpy, scipy, pytest, mpmath, openmm, matplotlib, PIL, tqdm, mdtraj, numba'
mkdir -p validation
"$python" -m pytest tests -k 'not nacl' -q --junitxml=validation/tests.xml "$@"
