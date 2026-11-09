# Optional OpenMM–PLUMED CUDA bridge

The analytical demo and the native KOMBI action do not need this bridge. This optional integration targets the tested Linux OpenMM **8.1.2.dev-440a9c7** CUDA ABI, one GPU, and PLUMED 2.11-dev. Rebuild and validate for another OpenMM version.

## What changes

The original OpenMM `CustomCPPForce` path transfers all atoms' positions and bias forces between GPU and CPU. The sparse path requests PLUMED's active atom list, gathers those positions on the GPU, and uploads only their force contributions. It keeps the original host coordinate reconstruction and fixed-point force conversion, including periodic images and mixed precision. Atom reorderings and changing selections refresh the mapping.

The native KOMBI library, estimator, optimizer, bias expression, precision and update cadence do not change. The accelerated path is enabled only when `OPENMM_PLUMED_SPARSE_CUDA=1`. With the flag unset or `0`, the original bridge path is used. Enabling it on another platform or multiple GPUs fails explicitly.

The included bridge is based on OpenMM-PLUMED commit `295e1880a9b24ba18a12a1841f4559b118b4a256`, retaining the initial-step and stop-flag fixes used during validation. Core implementation files match the tested version; the source is supplied instead of an ABI-specific prebuilt binary. The upstream source's license notices remain in place.

## Build into a separate directory

Use an OpenMM installation that contains its development headers and CUDA plugin, plus PLUMED headers/libraries and CUDA headers at `/usr/local/cuda/include`. Set the following paths for your environment:

```bash
cmake -S integrations/openmm-plumed -B build-openmm-plumed \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENMM_DIR="$OPENMM_PREFIX" \
  -DPLUMED_INCLUDE_DIR="$PLUMED_PREFIX/include/plumed" \
  -DPLUMED_LIBRARY_DIR="$PLUMED_PREFIX/lib" \
  -DCMAKE_INSTALL_PREFIX="$PWD/install-openmm-plumed" \
  -DPLUMED_BUILD_PYTHON_WRAPPERS=OFF
cmake --build build-openmm-plumed -j2
cmake --install build-openmm-plumed
```

The Python examples require a compatible `openmmplumed` wrapper; the original validation reused one built with SWIG 4.1. A wrapper can also be built from `integrations/openmm-plumed/python` using that source tree's CMake configuration. Do not assume a wrapper from a different OpenMM ABI is compatible.

Select the new library only for a fresh process:

```bash
export LD_LIBRARY_PATH="$PWD/install-openmm-plumed/lib:$OPENMM_PREFIX/lib:$OPENMM_PREFIX/lib/plugins:$PLUMED_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PLUMED_KERNEL="$PLUMED_PREFIX/lib/libplumedKernel.so"
OPENMM_PLUMED_SPARSE_CUDA=1 python your_new_simulation.py
```

When loading the separate KOMBI action into the embedded PLUMED kernel, make the kernel symbols globally available before creating the `PlumedForce`:

```python
import ctypes
import os
ctypes.CDLL(os.environ['PLUMED_KERNEL'], mode=ctypes.RTLD_GLOBAL)
from openmmplumed import PlumedForce
```

KOMBI `LOAD` must point at the extension compiled against this same PLUMED kernel. These environment settings do not update running processes. This repository does not change existing campaign launchers or checkpoints.

## Transport regression

With a compatible Python wrapper available, run the standalone test against both paths. Each command creates a new directory:

```bash
OPENMM_PLUMED_SPARSE_CUDA=0 python integrations/openmm-plumed/tests/validate_transport.py --name dense
OPENMM_PLUMED_SPARSE_CUDA=1 python integrations/openmm-plumed/tests/validate_transport.py --name sparse
python integrations/openmm-plumed/tests/compare_transport.py runs/bridge/dense runs/bridge/sparse
ctest --test-dir build-openmm-plumed --output-on-failure
```

The transport fixture covers distance and centroid coordinates, empty and changing atom lists, three GPU precisions, periodic images, triclinic boxes, force-group exclusion, energy-only queries, repeated force queries, atom reordering and checkpoint reload. These CUDA tests are separate from the CPU-only GitHub Actions workflow.

## Recorded performance and scientific checks

On 5 November 2026, a 100,027-atom Ras system on an RTX A4000 gave the following mean short-benchmark rates:

| Configuration | ns/day |
|---|---:|
| Original PLUMED bridge, detailed logging | 5.05 |
| Sparse bridge, identical logging | 9.62 |
| Sparse bridge, selecting logged atoms before unit conversion | 11.06 |
| Python/OpenMM GPU bias, original logging | 9.94 |
| Python/OpenMM GPU bias, selected-array conversion | 11.40 |

Each condition had two 400-step timed repeats after 100 warmup steps. These intervals isolate per-step cost and contain no adaptation. They exclude setup, DCD and campaign management overhead. The Python control uses a fixed initial bias through OpenMM's GPU force; it is not a complete adaptive Python benchmark. The logging optimization was measured in a separate benchmark script, not applied to existing runs.

A separate **30,010-step / 60.02 ps** adaptive Ras regression, using three original 20 ps KOMBI windows and sparse Python diagnostics, improved from **171.20 to 80.91 seconds**. All 322 saved comparison arrays, the full per-step COLVAR file, and native state/history matched exactly. Six short Ras comparisons also matched exactly, as did 1,452 transport comparison arrays. Paired stochastic OpenMM/KOMBI restart tests passed within and at the boundary of a bias window; both upstream bridge tests passed.

Evidence: [timings](../validation/timings.csv), [short comparisons](../validation/benchmark_comparison.json), [adaptive and transport comparisons](../validation/validation_comparison.json), [CUDA transfer measurements](../validation/cuda_memory_transfers.json). Timings retain exploratory and final runs; the final optimized rows are named `verified_*`. Raw molecular inputs and trajectories are not distributed here, so these timing results require the original system to reproduce. No existing TYK2 run was modified or benchmarked with the new bridge. NPT/barostat production was not validated.
