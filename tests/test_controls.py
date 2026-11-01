"""Control experiments: correct masks/inits, exact reproduction of the IMP run, resume."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from run_controls import make_condition, mask_seed, init_seed

REPO = Path(__file__).resolve().parents[1]


def cli(*args):
    r = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r


@pytest.fixture(scope='module')
def imp_run(tmp_path_factory):
    out = tmp_path_factory.mktemp('imp') / 'run'
    cli('run_imp.py', '--steps', '20', '--eval-every', '10', '--rounds', '2', '--optimizer', 'adam', '--lr', '1.2e-3',
        '--device', 'cpu', '--no-progress', '--out', str(out))
    return out


def controls(run, out, conditions='ticket,reinit,random_mask,random_both'):
    return cli('run_controls.py', '--run', str(run), '--conditions', conditions, '--device', 'cpu',
               '--no-progress', '--out', str(out))


def strip_level0(csv_text):
    lines = csv_text.splitlines()
    return [lines[0]] + [l for l in lines[1:] if not l.startswith('0,')]


def test_ticket_reproduces_imp_and_resume_works(imp_run, tmp_path):
    out = tmp_path / 'controls'
    r = controls(imp_run, out)
    assert 'matches IMP run: NO' not in r.stdout
    assert r.stdout.count('matches IMP run: YES') == 2
    # The ticket condition is the IMP run itself: identical summary rows (levels 1-2).
    assert (out / 'ticket' / 'summary.csv').read_text().splitlines() == strip_level0((imp_run / 'summary.csv').read_text())
    full = {c: (out / c / 'summary.csv').read_text() for c in ('reinit', 'random_mask', 'random_both')}
    # Controls really differ from the ticket.
    assert len(set(full.values()) | {(out / 'ticket' / 'summary.csv').read_text()}) == 4

    # Crash: a finished unit and the summary vanish, another is truncated -> redone identically.
    (out / 'reinit' / 'level_02.pt').unlink()
    (out / 'reinit' / 'summary.csv').unlink()
    (out / 'random_both' / 'level_01.pt').write_bytes(b'truncated')
    r = controls(imp_run, out)
    assert 'redoing' in r.stdout
    assert {c: (out / c / 'summary.csv').read_text() for c in full} == full


def test_condition_masks_and_inits(imp_run):
    ckpt = torch.load(imp_run / 'level_02.pt', map_location='cpu', weights_only=True)
    imp_masks, orig = ckpt['masks'], ckpt['original_state']
    seed, level = 42, 2
    ticket = make_condition('ticket', ckpt, 'lenet300100', seed, level)
    reinit = make_condition('reinit', ckpt, 'lenet300100', seed, level)
    rmask = make_condition('random_mask', ckpt, 'lenet300100', seed, level)
    rboth = make_condition('random_both', ckpt, 'lenet300100', seed, level)

    for k in imp_masks:
        assert torch.equal(ticket[1][k], imp_masks[k]) and torch.equal(reinit[1][k], imp_masks[k])
        # Random masks keep each layer's count but not the positions; the two random conditions share one.
        assert int(rmask[1][k].sum()) == int(imp_masks[k].sum())
        assert not torch.equal(rmask[1][k], imp_masks[k])
        assert torch.equal(rmask[1][k], rboth[1][k])
    for k, v in orig.items():
        assert torch.equal(ticket[0][k], v) and torch.equal(rmask[0][k], v)
        assert torch.equal(reinit[0][k], rboth[0][k])  # one shared fresh init
        if k.endswith('weight'):
            assert not torch.equal(reinit[0][k], v)
            # Same distribution (Xavier normal): similar std.
            assert abs(reinit[0][k].std() / v.std() - 1) < 0.05
        else:
            assert torch.all(reinit[0][k] == 0)
    # Deterministic and level-specific.
    assert torch.equal(make_condition('reinit', ckpt, 'lenet300100', seed, level)[0]['fc1.weight'], reinit[0]['fc1.weight'])
    assert init_seed(seed, 1) != init_seed(seed, 2) and mask_seed(seed, 1) != init_seed(seed, 1)


def test_different_imp_config_is_refused(imp_run, tmp_path):
    out = tmp_path / 'controls'
    controls(imp_run, out, conditions='ticket')
    (out / 'imp_config.json').write_text('{"seed": 1}')
    r = subprocess.run([sys.executable, 'run_controls.py', '--run', str(imp_run), '--conditions', 'ticket',
                        '--device', 'cpu', '--no-progress', '--out', str(out)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and 'different IMP config' in r.stderr
