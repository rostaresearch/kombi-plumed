#!/usr/bin/env python3
"""Run the corrected 2D analytical potential with PLUMED's built-in pesmd."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def prepare(folder, library, steps, seed):
    """Adapt only the atom index and file paths for pesmd's fixed origin atom."""
    folder.mkdir(parents=True, exist_ok=False)
    potential = (ROOT / 'examples/potential_2d.dat').read_text()
    assert 'POSITION ATOM=1 NOPBC' in potential
    potential = potential.replace('POSITION ATOM=1 NOPBC', 'POSITION ATOM=2 NOPBC')
    (folder / 'potential_2d.dat').write_text(potential)
    script = (ROOT / 'examples/model_2d.dat').read_text()
    script = script.replace('../build/Kombi.so', str(library))
    script = script.replace('SEED=20261103', f'SEED={seed}')
    script += '\nFLUSH STRIDE=1000\n'
    (folder / 'plumed.dat').write_text(script)
    # pesmd passes forces to a unit-mass integrator in PLUMED's MD base units.
    # The temperature argument is kBT in those units (kJ/mol), not kelvin.
    kbt_kcal = 0.5961612777
    control = (f'temperature {kbt_kcal * 4.184:.17g}\n'
               'tstep 0.002\nfriction 1\ndimension 2\nperiodic false\n'
               f'nstep {steps}\nipos 5 4\nidum {seed}\nplumed plumed.dat\n')
    (folder / 'pesmd.in').write_text(control)
    return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/analytical-2d')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--plumed', default=os.environ.get('PLUMED', 'plumed'))
    parser.add_argument('--library', type=Path,
                        default=Path(os.environ.get('KOMBI_LIBRARY', ROOT / 'build/Kombi.so')))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    library = args.library.resolve()
    if not library.is_file():
        parser.error(f'Build KOMBI first: missing {library}')
    if any(ch.isspace() for ch in str(library)):
        parser.error('Use a checkout/library path without whitespace for the PLUMED LOAD line')
    folder = args.output.resolve()
    control = prepare(folder, library, args.steps, args.seed)
    version = subprocess.run([args.plumed, 'info', '--long-version'],
                             check=True, capture_output=True, text=True).stdout.strip()
    started = time.perf_counter()
    with (folder / 'plumed.log').open('w') as log:
        result = subprocess.run([args.plumed, 'pesmd'], input=control, text=True,
                                cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'OMP_NUM_THREADS': '1'})
    if result.returncode:
        raise SystemExit(f'PLUMED failed; inspect {folder / "plumed.log"}')
    lines = (folder / 'model_2d.COLVAR').read_text().splitlines()
    fields = next(line.split()[2:] for line in lines if line.startswith('#! FIELDS'))
    values = [line for line in lines if line and not line.startswith('#')]
    last = dict(zip(fields, map(float, values[-1].split())))
    windows = (folder / 'model_2d.history').read_text().count('END_WINDOW')
    summary = {
        'plumed_version': version, 'engine': 'PLUMED pesmd', 'seed': args.seed,
        'steps_requested': args.steps, 'steps_completed': round(last['time'] / .002),
        'elapsed_ps': last['time'], 'complete_kombi_windows': windows,
        'target_reached': last['k.fpt'] >= 0,
        'biased_first_passage_ps': last['k.fpt'] if last['k.fpt'] >= 0 else None,
        'final_xy_nm': [last['p.x'], last['p.y']],
        'thermal_energy_kcal_mol': 0.5961612777,
        'wall_seconds': time.perf_counter() - started,
        'note': 'Unit-mass pesmd demonstration; not a reproduction of OpenMM kinetics.'}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Plot: python examples/analytical_2d/plot.py {folder}')


if __name__ == '__main__':
    main()
