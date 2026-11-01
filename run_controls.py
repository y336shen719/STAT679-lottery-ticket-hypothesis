"""Control experiments for a finished IMP run (2x2: mask source x initialization).

    condition     mask                        initialization
    ticket        IMP mask                    original   (= the IMP run itself; a correctness check)
    reinit        IMP mask                    fresh random
    random_mask   random, same per-layer size original
    random_both   random, same per-layer size fresh random

Every control at level L uses the IMP run's settings (config.json) and the same minibatch
order as the winning ticket at level L, so only the mask and/or the initialization differ.
Fresh weights and random masks come from fixed seeds per (IMP seed, level); `reinit` and
`random_both` share one fresh init, `random_mask` and `random_both` share one random mask.

Example: python run_controls.py --run runs/mnist_adam_seed42
Rerunning the same command after a crash skips the finished (condition, level) pairs.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from lth.data import loaders
from lth.models import make_model
from lth.pruning import random_masks_like, remaining_fraction
from lth.train import train_fixed_steps
from run_imp import atomic_write, write_summary

CONDITIONS = ('ticket', 'reinit', 'random_mask', 'random_both')
METRICS = ('best_step', 'best_val_loss', 'best_test_accuracy', 'end_test_accuracy')


def init_seed(seed, level):
    return 1_000_000 + 1000 * seed + level


def mask_seed(seed, level):
    return 2_000_000 + 1000 * seed + level


def load_level(run, level):
    return torch.load(run / f'level_{level:02d}.pt', map_location='cpu', weights_only=True)


def make_condition(condition, ckpt, model_name, seed, level):
    """Return (initial state_dict, masks) for one control at one level."""
    masks, state = ckpt['masks'], ckpt['original_state']
    if condition in ('random_mask', 'random_both'):
        masks = random_masks_like(masks, mask_seed(seed, level))
    if condition in ('reinit', 'random_both'):
        torch.manual_seed(init_seed(seed, level))
        state = {k: v.detach().clone() for k, v in make_model(model_name).state_dict().items()}
    return state, masks


def read_summary(path):
    with open(path, newline='') as f:
        return {int(r['level']): r for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='Finished run_imp.py output folder')
    p.add_argument('--conditions', default='reinit,random_mask,random_both',
                   help=f'Comma-separated subset of {",".join(CONDITIONS)}')
    p.add_argument('--levels', default='all', help='Comma-separated levels (>= 1), or "all"')
    p.add_argument('--device', default='auto')
    p.add_argument('--no-progress', action='store_true')
    p.add_argument('--out', default=None, help='Default: <run>/controls')
    args = p.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / 'config.json').read_text())
    conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]
    if not conditions or any(c not in CONDITIONS for c in conditions):
        p.error(f'--conditions must be a subset of {CONDITIONS}')
    available = sorted(int(f.stem.split('_')[1]) for f in run.glob('level_*.pt'))
    levels = [l for l in available if l >= 1] if args.levels == 'all' else [int(x) for x in args.levels.split(',')]
    missing = [l for l in levels if l < 1 or l not in available or l - 1 not in available]
    if missing:
        p.error(f'levels {missing} need level_XX.pt and the previous level in {run} (level 0 has no pruned mask)')

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    out = Path(args.out) if args.out else run / 'controls'
    out.mkdir(parents=True, exist_ok=True)
    # Results in `out` are only comparable if they come from the same IMP settings.
    cfg_path = out / 'imp_config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        p.error(f'{out} holds controls for a different IMP config. Use another --out.')
    atomic_write(cfg_path, lambda f: f.write(json.dumps(cfg, indent=2).encode()))

    progress = not args.no_progress
    if progress:
        from tqdm import tqdm
    log_file = (out / 'train.log').open('a', encoding='utf-8')

    def log(msg):
        line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
        (tqdm.write if progress else print)(line)
        log_file.write(line + '\n'); log_file.flush()

    train_loader, val_loader, test_loader = loaders(cfg['dataset'], cfg['batch_size'], cfg['seed'],
                                                    split_seed=cfg['split_seed'], device=device)
    imp_rows = read_summary(run / 'summary.csv')
    units = [(c, l) for c in conditions for l in levels]  # condition-major: finish one curve first
    log(f'Controls for {run} on {device}: conditions={conditions}, levels={levels}')

    outer = tqdm(units, desc='controls', position=0, dynamic_ncols=True) if progress else units
    for condition, level in outer:
        path = out / condition / f'level_{level:02d}.pt'
        if path.exists():
            try:
                torch.load(path, map_location='cpu', weights_only=True)
                continue  # already done
            except Exception as e:
                log(f'WARNING: cannot read {path} ({e}); redoing it')
        path.parent.mkdir(exist_ok=True)
        state, masks = make_condition(condition, load_level(run, level), cfg['model'], cfg['seed'], level)
        model = make_model(cfg['model'])
        model.load_state_dict(state)
        # Same minibatch order as the winning ticket at this level.
        train_loader.generator.set_state(load_level(run, level - 1)['train_rng_state'])
        frac = remaining_fraction(masks)
        m = train_fixed_steps(model, train_loader, val_loader, test_loader, masks,
                              steps=cfg['steps'], lr=cfg['lr'], optimizer_name=cfg['optimizer'],
                              eval_every=cfg['eval_every'], device=device, progress=progress,
                              desc=f'{condition} level {level:02d} ({frac:.1%} weights)')
        row = {'level': level, 'weights_remaining': frac, **{k: m[k] for k in METRICS}}
        unit = {'metrics': row, 'history': m['history'], 'condition': condition,
                'init_seed': init_seed(cfg['seed'], level), 'mask_seed': mask_seed(cfg['seed'], level)}
        atomic_write(path, lambda f: torch.save(unit, f))
        rows = [torch.load(f, map_location='cpu', weights_only=True)['metrics']
                for f in sorted((out / condition).glob('level_*.pt'))]
        write_summary(out / condition / 'summary.csv', rows)
        msg = ', '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items())
        if condition == 'ticket':
            same = all(str(row[k]) == imp_rows[level][k] for k in METRICS)
            msg += f' | matches IMP run: {"YES" if same else "NO"}'
        log(f'{condition} level {level:02d} done: {msg}')
    log('All requested controls complete.')
    log_file.close()


if __name__ == '__main__':
    main()
