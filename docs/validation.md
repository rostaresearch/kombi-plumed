# Validation

## Native KOMBI

The original source passed **147 tests on each of PLUMED 2.10.2 and PLUMED 2.11-dev** on 3 November 2026. The suite contains unchanged reference regressions and tests of the C++ implementation. Twenty-two NaCl-specific cases are deselected because they are outside this port's scope. The periodic Ras input was separately retested after its example update; native action source was unchanged.

The public package was rebuilt and tested on 9 November 2026 with both PLUMED 2.11-dev and PLUMED 2.10.2: **148 passed, 22 deselected on each build**. The additional test runs actual 2D Langevin dynamics with `pesmd`, verifies the initial position and adaptive windows, and checks plotted physical energies against the saved PLUMED values. The final compressed-sample plotting route was included in the 2.10.2 run. GitHub Actions independently builds and tests PLUMED 2.10.2 using the supplied environment.

The native suite checks:

- Directed lag counts, separate bias windows and DHAM reconstruction.
- Grid conventions, edge cases, 2D flattening and working-chain selection.
- MFPT/return times, traps, high barriers and independent numerical oracles.
- Gaussian parameter layout, energies, derivatives and atomic forces.
- The 1D/2D physical potentials against the frozen OpenMM reference.
- Adaptation timing, first target hit and initial-at-target behaviour.
- Checkpoint pairing, saved-step replay, within-window continuation and invalid-restart rejection.

A one-rank versus two-rank PLUMED replay produced byte-identical COLVAR, state and history files. This is a driver/action MPI check, not validation of every MD engine's domain decomposition or coupled walkers. The existing reference MSM test emits one expected `log(0)` warning for an unoccupied state; the native estimator does not use that free-energy calculation.

## Public analytical example

The documented seed reached the target at **21.70 ps**, after **10,850 integration steps and two complete adaptive windows**, in the checked PLUMED 2.11-dev build. These are biased unit-mass `pesmd` dynamics, not an unbiased rate estimate or a reproduction of historical OpenMM trajectories. Other builds and seeds can differ. The plot and summary show the actual observed result; the run also supports a clearly reported step cap.

The displayed energy units are kcal/mol. Plotting evaluates the bundled potential at every saved point and verifies it against PLUMED's reported physical potential. The surface panel is shifted by its displayed grid minimum; the energy trace is unshifted.

See [the demo summary](../validation/analytical-demo.json) and [the reference suite counts](../validation/test-counts.json).

## Optional OpenMM CUDA bridge

The bridge has a separate validation history, described in [openmm-bridge.md](openmm-bridge.md). Its opt-in sparse transport was checked against the original path using the unchanged KOMBI library, including full-system adaptive Ras dynamics and restarts. CUDA tests are not part of the CPU-only public CI job.

## Reproducibility boundaries

The C++ action, frozen numerical reference files, analytical potentials and established tests retain their original contents. Their hashes are recorded in `validation/original-source-sha256.json`. Public packaging adds documentation, the demo, an end-to-end test, a portable transport-test entry point and CI configuration. Machine-specific source locations are omitted from public provenance while content hashes remain.

These checks establish the tested numerical and integration behaviour. They do not establish full ensemble agreement, an unbiased off-rate, or acceleration factors for a molecular system. Native proposals differ from NumPy's generator, and the documented finite-best optimizer guard can differ from historical Python final-iterate selection.
