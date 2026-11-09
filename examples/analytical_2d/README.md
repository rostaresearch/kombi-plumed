# A 2D analytical example

From the repository root, after building the extension:

```bash
python examples/analytical_2d/run.py
python examples/analytical_2d/plot.py runs/analytical-2d
```

This runs genuine Langevin dynamics with adaptive KOMBI on the supplied multiwell potential. It does not replay a prespecified path. Both physical and adaptive forces are evaluated by PLUMED.

The checked trajectory is also included as compressed data. To regenerate the README figure without running dynamics:

```bash
python examples/analytical_2d/plot.py examples/analytical_2d/sample --output runs/sample-figure.png
```

With the default seed in the checked PLUMED 2.11-dev build, the target was reached at 21.70 ps after two complete updates. This is one illustrative trajectory, not an ensemble benchmark.

The runner reads the existing `examples/model_2d.dat` and `examples/potential_2d.dat`. It changes the library path, the requested seed, and the moving particle's index from 1 to 2: PLUMED's `pesmd` engine reserves atom 1 at the origin. It also adds an output flush. It preserves the original surface expression, hill parameters, 20 × 20 grid, 5,000-step update interval, 1,000 random trials, local optimizer, target and stopping rule.

The default cap is 50,000 steps. A target hit stops the run sooner. A capped run is reported as capped, with no invented first-passage time. Choose another output directory to run again; the runner refuses to overwrite an existing directory.

## Dynamics and units

The particle starts at (5, 4) nm. The target is (1, 1.5) nm with radius 0.1 nm. The timestep is 0.002 in the PLUMED base time unit (ps here), with `pesmd` friction parameter 1 and unit particle mass. This simulator differs from the historical OpenMM setup, so trajectories and first-passage times should not be compared as if their integrators and masses were identical.

`pesmd`'s `temperature` input is a thermal energy in the MD base energy unit, not kelvin. The runner supplies `0.5961612777 × 4.184` in kJ/mol, matching the unchanged KOMBI `KBT=0.5961612777` kcal/mol. The historical optimizer `TILT_KBT=0.5981` kcal/mol is also retained. Simulation inputs remain in their required native units; plots present energies in kcal/mol.

The surface panel shows physical potential energy relative to the minimum of the displayed grid. The energy trace uses the absolute physical potential and KOMBI bias in kcal/mol. The trajectory progresses from purple to yellow; dashed vertical lines indicate adaptive updates. The plotter checks its physical energy expression against every saved trajectory energy before creating the figure.

## Restart scope

`model_2d.state` and `model_2d.history` record KOMBI state. This small runner does not save a matching `pesmd` velocity/random-state checkpoint and does not offer restart. Start a fresh example run instead. Molecular simulations require an MD engine checkpoint paired with the KOMBI files.

For the algorithm and numerical conventions, see [the method reference](../../docs/method.md). The [PLUMED pesmd documentation](https://www.plumed.org/doc-v2.10/user-doc/html/pesmd.html) describes the simulator.
