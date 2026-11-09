# Licensing and attribution

Original KOMBI extension code, new integration changes, tests, examples and documentation in this repository are Copyright (c) 2026 Edina Rosta and contributors and licensed under **LGPL-3.0-or-later**, except where an existing notice specifies another license. See `LICENSE` (LGPL v3) and `COPYING` (GPL v3, incorporated by LGPL v3). This software is provided without warranty.

The following third-party notices and licenses are retained:

- `third_party/lbfgsb/`: Stephen Becker's C translation of L-BFGS-B 3.0, commit `164463486a20a5185dd087c2636b6b278a297e4b`, BSD 3-Clause. See its `LICENSE` and source notices. Upstream: https://github.com/stephenbeckr/L-BFGS-B-C.
- `integrations/openmm-plumed/`: based on the OpenMM-PLUMED bridge, commit `295e1880a9b24ba18a12a1841f4559b118b4a256`. Existing MIT-style Stanford/author notices remain in each source file; our additions are covered by the repository license. Upstream: https://github.com/openmm/openmm-plumed.
- `tests/reference/`, `tests/lowdim_reference/`, `tests/safety_reference/`: frozen Rosta-group reference implementations and regression tests. Numerical files are retained byte-for-byte as described in `PROVENANCE.json`; they are not an installation of the molecular simulation campaigns.
- PLUMED and OpenMM are external dependencies. Their code is not relicensed by this repository. PLUMED: https://www.plumed.org. OpenMM: https://openmm.org.

The repository is an external loadable PLUMED extension, not an upstream PLUMED release. It contains no experimental molecular trajectories or molecular force-field/topology files.
