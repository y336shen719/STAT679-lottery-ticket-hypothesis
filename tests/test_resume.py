"""Resume after a crash must reproduce an uninterrupted run exactly.

Runs tiny CPU experiments through the real CLI (downloads MNIST to data/ if missing).
"""
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def run(out, *extra, rounds=3):
    cmd = [sys.executable, 'run_imp.py', '--steps', '20', '--eval-every', '10', '--rounds', str(rounds),
           '--device', 'cpu', '--no-progress', '--out', str(out), *extra]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def masks_of(path):
    return torch.load(path, map_location='cpu', weights_only=True)['masks']


def test_resume_matches_uninterrupted(tmp_path):
    out = tmp_path / 'run'
    r = run(out)
    assert r.returncode == 0, r.stderr
    full_summary = (out / 'summary.csv').read_text()
    full_masks = masks_of(out / 'level_03.pt')

    # Crash after level 1: later checkpoints and the summary are gone.
    for name in ('level_02.pt', 'level_03.pt', 'summary.csv'):
        (out / name).unlink()
    r = run(out)
    assert r.returncode == 0, r.stderr
    assert 'Resuming from level 2' in r.stdout
    assert (out / 'summary.csv').read_text() == full_summary
    resumed = masks_of(out / 'level_03.pt')
    assert all(torch.equal(full_masks[k], resumed[k]) for k in full_masks)

    # Crash while writing level 3: a truncated checkpoint is redone, not trusted.
    (out / 'level_03.pt').write_bytes(b'truncated')
    r = run(out)
    assert r.returncode == 0, r.stderr
    assert 'redoing level 3' in r.stdout
    assert (out / 'summary.csv').read_text() == full_summary

    # Already finished: nothing is retrained.
    r = run(out)
    assert r.returncode == 0, r.stderr
    assert 'already complete' in r.stdout


def test_config_mismatch_is_refused(tmp_path):
    out = tmp_path / 'run'
    assert run(out, rounds=0).returncode == 0
    r = run(out, '--lr', '0.05', rounds=0)
    assert r.returncode != 0
    assert 'different config' in r.stderr
