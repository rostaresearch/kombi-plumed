#!/usr/bin/env python3
"""Plot the actual trajectory on the supplied analytical surface (kcal/mol)."""
import argparse
import json
import gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[2]


def surface(x, y):
    # Evaluate only the bundled, trusted analytical expression; no run-supplied code.
    line = next(s for s in (ROOT / 'examples/potential_2d.dat').read_text().splitlines()
                if s.startswith('u: CUSTOM'))
    expression = line.split('FUNC=', 1)[1].split(' PERIODIC=', 1)[0].replace('^', '**')
    return eval(expression, {'__builtins__': {}}, {'x': x, 'y': y, 'z': 0., 'exp': np.exp})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    folder = args.run.resolve()
    path = folder / 'model_2d.COLVAR'
    if not path.exists():
        path = folder / 'model_2d.COLVAR.gz'
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        fields = next(s.split()[2:] for s in handle if s.startswith('#! FIELDS'))
    a = np.loadtxt(path, ndmin=2)
    c = {name: a[:, i] for i, name in enumerate(fields)}
    xy = np.c_[c['p.x'], c['p.y']]
    # Independent check that the plotted potential is the actual physical energy.
    np.testing.assert_allclose(surface(xy[:, 0], xy[:, 1]), c['u'], atol=2e-10, rtol=2e-11)
    grid = np.linspace(0, 2*np.pi, 240)
    x, y = np.meshgrid(grid, grid)
    energy = surface(x, y)
    reference = float(energy.min())
    fig = plt.figure(figsize=(12, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[:, 0]); top = fig.add_subplot(gs[0, 1]); bottom = fig.add_subplot(gs[1, 1])
    bg = ax.contourf(x, y, energy-reference, levels=24, cmap='Greys', alpha=.8)
    fig.colorbar(bg, ax=ax, shrink=.8, label='Physical potential − grid minimum (kcal/mol)')
    sampled = np.unique(np.r_[np.arange(0, len(xy), max(1, len(xy)//12000)), len(xy)-1])
    points = xy[sampled]
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', linewidths=.8, alpha=.9)
    lc.set_array(c['time'][sampled][:-1]);ax.add_collection(lc)
    ax.scatter([5], [4], marker='o', s=70, facecolor='white', edgecolor='#172b40', label='Start (5, 4)')
    ax.add_patch(plt.Circle((1, 1.5), .1, color='#c94932', fill=False, lw=2))
    ax.scatter([1], [1.5], marker='*', s=110, color='#c94932', label='Target (1, 1.5)')
    ax.scatter(xy[-1, 0], xy[-1, 1], marker='s', s=45, color='#29a17c', edgecolor='white', label='Last frame')
    ax.set(xlim=(0, 2*np.pi), ylim=(0, 2*np.pi), xlabel='x (nm)', ylabel='y (nm)', title='KOMBI on the 2D analytical surface', aspect='equal')
    ax.legend(loc='upper left', fontsize=8, framealpha=.9)
    top.plot(c['time'], c['p.x'], lw=.8, color='#22678b', label='x')
    top.plot(c['time'], c['p.y'], lw=.8, color='#c46a3e', label='y')
    top.set(ylabel='Position (nm)', title='Trajectory and adaptive updates');top.legend(ncol=2, loc='upper right')
    bottom.plot(c['time'], c['u'], lw=.8, color='#4a5966', label='Physical potential')
    bottom.plot(c['time'], c['k.bias'], lw=.8, color='#287f66', label='KOMBI bias')
    bottom.set(xlabel='Simulation time (ps)', ylabel='Energy (kcal/mol)');bottom.legend(loc='best', fontsize=9)
    for axis in [top, bottom]:
        for t in c['time'][np.r_[False, np.diff(c['k.window']) > 0]]:
            axis.axvline(t, color='#85958f', alpha=.4, lw=.8, linestyle='--')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=.15)
    summary = json.loads((folder / 'summary.json').read_text())
    outcome = ('target reached' if summary['target_reached'] else 'step limit reached')
    fig.suptitle(f"{summary['complete_kombi_windows']} complete bias windows · {summary['elapsed_ps']:.2f} ps · {outcome}", fontsize=12)
    destination = args.output or folder / 'trajectory.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches='tight')
    print(destination)


if __name__ == '__main__':
    main()
