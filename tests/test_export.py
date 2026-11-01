"""export_results.py must copy summaries and write every validation curve exactly."""
import csv
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def cli(*args):
    r = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r


def test_export_matches_checkpoints(tmp_path):
    run, out = tmp_path / 'run', tmp_path / 'results'
    cli('run_imp.py', '--steps', '20', '--eval-every', '10', '--rounds', '2', '--device', 'cpu',
        '--no-progress', '--out', str(run))
    cli('run_controls.py', '--run', str(run), '--conditions', 'ticket,reinit,random_mask,random_both',
        '--device', 'cpu', '--no-progress')
    cli('export_results.py', '--run', str(run), '--out', str(out))

    assert (out / 'summary.csv').read_text() == (run / 'summary.csv').read_text()
    for c in ('ticket', 'reinit', 'random_mask', 'random_both'):
        assert (out / 'controls' / c / 'summary.csv').read_text() == (run / 'controls' / c / 'summary.csv').read_text()
    assert not list(out.rglob('*.pt'))

    with open(out / 'histories.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    # IMP levels 0-2 as 'ticket', 3 controls x levels 1-2; 2 evaluations each (steps 10, 20).
    expected = {('ticket', l) for l in (0, 1, 2)} | {(c, l) for c in ('reinit', 'random_mask', 'random_both') for l in (1, 2)}
    assert {(r['condition'], int(r['level'])) for r in rows} == expected
    assert len(rows) == 2 * len(expected)

    def curve(cond, level):
        return [(int(r['step']), float(r['val_loss']), float(r['val_accuracy']))
                for r in rows if r['condition'] == cond and int(r['level']) == level]

    for cond, level, path in [('ticket', 2, run / 'level_02.pt'),
                              ('reinit', 1, run / 'controls' / 'reinit' / 'level_01.pt')]:
        hist = torch.load(path, map_location='cpu', weights_only=True)['history']
        assert curve(cond, level) == [(h['step'], h['val_loss'], h['val_accuracy']) for h in hist]
