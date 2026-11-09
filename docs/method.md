# KOMBI for PLUMED

Native C++ `KOMBI` action for the corrected, directed-count Ras Pi and hard-bin 1D/2D model-potential protocols. It supplies the adaptive energy and forces, reconstructs the unbiased transition model from constant-bias windows, and selects a replacement Gaussian bias by minimizing model first-passage time. RKHS is deferred.

This is a loadable PLUMED extension, with source and tests. It has not been merged into upstream PLUMED. See [validation.md](validation.md) for the builds and checks actually completed. Run shell commands in this document from the repository root.

## Build and test

Use Linux, a C/C++17 compiler, and a development installation of PLUMED with `LOAD`, `mklib`, and the usual setup, bias, colvar, function and generic modules. Build against the same PLUMED installation that will run the action:

```sh
bash build.sh
# Or select another installation explicitly:
PLUMED=/path/to/plumed bash build.sh
```

The output is `build/Kombi.so`. No Python is used by the running action. The native L-BFGS-B solver is included under `third_party/lbfgsb`, with its BSD license.

A quick installation check replays a supplied synthetic trajectory through three adaptive windows:

```sh
cd examples
plumed driver --plumed replay.dat --ixyz replay.xyz --timestep .002
```

It produces `replay.COLVAR`, `replay.state` and `replay.history`. This checks loading and adaptation; it is not an MD performance benchmark.

Python is needed only for verification and regenerating example inputs:

```sh
python3 -m pip install -r requirements-test.txt
bash test.sh
# With a particular Python or PLUMED:
PYTHON=/path/to/python PLUMED=/path/to/plumed bash test.sh
```

If several PLUMED installations coexist, set their library search paths as well as the executable. Confirm `plumed info --long-version` and `plumed info --include-dir` before building. A plugin compiled against another installation is not a portable binary.

## Supplied inputs

Run from `examples/`, or adjust `LOAD` and `INCLUDE` paths.

| Input | Protocol |
|---|---|
| `ras_pi.dat` | Ras A5/L20 settings: ten hills, amplitudes 0.1–5 kcal/mol, 20 ps windows at a 2 fs MD step, 100 fixed grid points over 1–8 Å. |
| `model_1d.dat` | Corrected October 1D driver: 100 bins, 20 hills, 5,000-step windows, target 5.2 nm, radius 0.1 nm. |
| `model_2d.dat` | Corrected October 2D driver: 20 × 20 grid, 20 hills, 5,000-step windows, target (1, 1.5) nm, radius 0.1 nm. |
| `potential_1d.dat`, `potential_2d.dat` | Physical multiwell potentials at amplitude 6, including the original walls, corrected 2D barrier, and confinement of unused Cartesian dimensions. |

The model inputs supply **both physical and adaptive forces** for one particle. Do not add their physical potential a second time in the MD engine. Their physical energies and Cartesian forces are checked against the frozen OpenMM source. The MD engine still supplies the integrator, mass, temperature and initial coordinates. Reference starts are (2, 0, 0) nm and (5, 4, 0) nm, with a 2 fs step and 300 K.

The Ras input supplies the adaptive force only. Its atom IDs, Mg 7800 and phosphate P 7841, are **specific to the original step3 topology** and use PLUMED's 1-based numbering. Verify them against the intended topology. The example uses the minimum-image periodic Mg–P distance. The engine must supply the verified periodic box, coordinates and physical force field. The initial ten hills have amplitude 0.01 kcal/mol, centres 3.7 ± 0.2 Å and widths 1–5 Å.

Initial Gaussian arrays in the supplied files are deterministic examples generated with the documented seed in `make_examples.py`. They are not claimed to reproduce a historical walker. The examples use 1,000 random candidates, followed by bounded L-BFGS-B; tolerance is 1e-6 for Ras and 1e-4 for the model potentials.

## Numerical conventions

All matrices inside the extension and in its history files are **row oriented**: `P[i][j]` means origin i to destination j. Reference Ras code uses the transposed convention externally; tests account for that explicitly.

For window k, its directed lag counts are Cᵏ and its actually applied bias is Uᵏ. The reconstruction is

```text
n_i^k = sum_j C_ij^k
W_ij  = sum_k C_ij^k / sum_k [n_i^k exp((U_i^k - U_j^k)/(2 KBT))]
P_ij  = W_ij / sum_j W_ij
P^V_ij ∝ P_ij exp((V_i - V_j)/(2 TILT_KBT))   [normalize each row]
```

Denominators and tilts use logarithms to avoid overflow. Observed positive edges are never deleted by a numerical threshold; unrepresentable edges cause a clear failure. Counts are directed and unsymmetrized, and never cross a bias-window boundary. There are no ghost paths, reverse-count additions or pseudocounts.

| `GRID_MODE` | Exact mapping and objective origin |
|---|---|
| `RAS` | `q=linspace(low,high,N)`, `digitize(x,q)` without subtracting one; bias at `q[state]`. Strict sampled domain `[low,high)`. Last sampled bin is the objective origin. Final model target uses `digitize(TARGET,q)-1`. Observed non-neighbor transitions are rejected, preserving the reference neighbor-only tilt's support. |
| `LEGACY_1D` | N+1 equally spaced edges; `digitize` without subtracting one. Bias at `edges[:N] - spacing/2`. Upper indices N and above are invalid and pairs involving them are skipped, as in the corrected driver. Negative finite x maps to state zero. Last sampled bin is the objective origin. |
| `LEGACY_2D` | N points on each axis; `digitize` without subtracting one; shifted centers `grid-spacing/2`. Flattening is `ix + Nx*iy`, x fastest. `CLIP_GRID` reproduces the driver's edge clipping. The most visited bin in the last window is the objective origin; smallest index wins ties. |
| `CENTERS` | An explicit additional conventional-bin option: N equal intervals, midpoint bias evaluation. Last sampled bin is the objective origin. This is not substituted for a historical protocol. |

Legacy targets are mapped to the nearest retained bin in integer grid space, with deterministic first-index ties. The 2D origin's projection onto the working chain follows the reference flattened-index rule. `CENTERS` uses coordinate distance. Each sampled bin's visits are checkpointed, so the 2D modal origin survives a restart.

The working chain retains states with both incoming and outgoing evidence. Trimming is refused if it would discard **any** outgoing positive probability from a retained state. This structural guard is stricter than the old floating-point tolerance check. A reachable closed trap makes unconditional MFPT infinite. Stable positive state reduction avoids cancellation at high barriers; first-return times use the relevant closed communicating class. No finite candidate means no new bias is applied.

The Gaussian bias is positive and **replaced** at each update, not accumulated. Parameter blocks are `[a, b, sigma]` in 1D and `[a, bx, by, sx, sy]` in 2D, with all values of each block contiguous. Sigma is the standard deviation in `exp(-distance²/(2 sigma²))`. Analytic CV derivatives propagate through PLUMED to atomic forces.

## Time, units and first passage

Energy, length/CV and time inputs use active PLUMED `UNITS`. `KBT` is explicit. Examples preserve the historical DHAM value 0.5961612777 kcal/mol and optimizer value `TILT_KBT=0.5981`; the latter otherwise defaults to `KBT`.

`PACE` is in MD steps. `SAMPLE_STRIDE` controls sampling; `LAG` is in sampled frames. Forces are evaluated every step; inherited bias `STRIDE` must remain 1. Step zero is the initial condition and is not counted. Each complete window contains exactly `PACE/SAMPLE_STRIDE` samples. A newly optimized bias first acts on the step **after** the window completes.

`k.mfpt` is the last selected model MFPT in active time units, using `MD timestep × SAMPLE_STRIDE × LAG`; it is -1 before the first update. It is not an unbiased experimental off-rate. When origin and model target coincide, optimization uses first-return time rather than the zero-time hitting convention.

With `STOP_AT_TARGET`, the action checks the physical target every MD step. In Ras mode it stops at `x >= TARGET`; otherwise it stops within Euclidean `TARGET_RADIUS` of `TARGET`. `k.fpt` reports the first detected hit time, or -1 before a hit. An initial configuration already in the target has FPT zero. These are biased-trajectory first-passage times. Terminal frames are checkpointed without fitting an incomplete window; Ras can stop cleanly on the first frame beyond its upper grid boundary. The MD engine must support PLUMED's stop request. Without this flag, `TARGET` is only an optimization target.

## Checkpoints and evidence

`STATE_FILE` and `HISTORY_FILE` default to `<label>.state` and `<label>.history`. Each full history record includes the applied parameters, bias on the grid, directed window counts, skipped-pair count, retained indices, reconstructed model, objective values, optimizer status and next parameters.

State stores accumulated counts and DHAM denominators, incomplete-window counts, visits and lag history, current/replay parameters, timestep and step, and a rolling history digest. It is saved at `CHECKPOINT_STRIDE` (default `PACE`), window boundaries, and engine checkpoint requests. FNV checksums detect accidental changes; they are not cryptographic signatures.

Restart with PLUMED's normal `RESTART` mechanism and a **matched MD checkpoint, KOMBI state and history** from the same point. Preserve that set externally; the plugin does not atomically archive the MD engine's files. It accepts replay of the saved step or continuation at the next step, and rejects gaps, changed settings, timestep mismatches, different replay CVs, and changed/unpaired history. Restore a matching history file when rolling MD back; the action does not silently truncate newer evidence. CV definitions, topology, units and engine state must also remain consistent.

Failures during an update save evidence separately as `.state.failed`, leaving the previous usable snapshot intact. A failed-evidence file is diagnostic, not a restart file. Multiple independent actions need separate file names.

## Scope and reproducibility

The implementation supports one or two nonperiodic scalar CVs, up to 4,096 total states. Dense storage is quadratic, and optimization cost grows with the visited chain. Periodic CVs, coupled walkers, RKHS, and automatic changes to the fixed grid are not implemented.

The explicit SplitMix64 proposal generator is reproducible within this port. It differs from NumPy's generator, so equal integer seeds do not imply equal proposals or trajectories. Native L-BFGS-B uses finite differences and retains the best finite bounded candidate encountered, including the random candidate. This finite-best safeguard can differ from the original script's final-iterate choice. The iteration status is recorded; a retained finite candidate need not mean local convergence. Numerical convention and force equivalence are tested; bitwise identity to historical Python optimization paths is not claimed.

The supplied inputs follow the corrected October hard-bin protocols. They do not silently substitute the earlier July 2D protocol with artificial target paths or the later experimental RKHS estimator. Full ensemble reproduction of unbinding times or acceleration factors requires new MD campaigns; the validation here does not establish those scientific results.

## Source provenance

Frozen original code/tests are preserved in `tests/reference`, `tests/lowdim_reference` and `tests/safety_reference`. `PROVENANCE.json` records their relative paths and SHA-256 hashes; private machine-specific locations are omitted. The Ras numerical core includes the later directed-trap and return-time fixes retained in the TYK2 adaptation; its molecular setup is not used for the Ras example. The actual Ras A5/L20 run settings are frozen in `tests/protocols/ras_a5_l20_run100.json`. The low-dimensional reference includes the corrected directed-count drivers and stable-MFPT updates. Its safety helper is byte-identical to the separately frozen numerical safety reference. Original regression files are not edited to fit the port.

The native solver is Stephen Becker's C translation of L-BFGS-B 3.0, commit `164463486a20a5185dd087c2636b6b278a297e4b`, with license preserved. Its reverse-communication solver is serialized within each process because the translation contains static work variables.

References: [PLUMED LOAD](https://www.plumed.org/doc-v2.10/user-doc/html/_l_o_a_d.html), [PLUMED mklib](https://www.plumed.org/doc-v2.10/user-doc/html/mklib.html), [L-BFGS-B-C](https://github.com/stephenbeckr/L-BFGS-B-C).
