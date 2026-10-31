"""Run a checkpointed iterative magnitude pruning experiment.

Example: python run_imp.py --dataset mnist --steps 100 --rounds 2 --out runs/smoke

Rerunning the same command after a crash resumes after the last completed
level. Each level_XX.pt stores the data-shuffle RNG state, so a resumed run
gives the same results as an uninterrupted one.
"""
import argparse
import csv
import io
import json
import os
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from lth.data import loaders
from lth.models import make_model
from lth.pruning import initial_masks, prune_by_magnitude, remaining_fraction, apply_masks
from lth.train import train_fixed_steps

# Arguments that may change between a run and its resume without changing results.
# `rounds` may grow, which extends a finished run with more pruning levels.
RESUME_SAFE_ARGS = {'out', 'device', 'no_progress', 'rounds'}


def atomic_write(path, write_fn):
    """Write via a temp file + fsync + rename, so a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_summary(path, rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=rows[0].keys(), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    atomic_write(path, lambda f: f.write(buf.getvalue().encode()))


def check_config(out, args, parser):
    """Refuse to mix results from different settings in one output folder."""
    path = out / 'config.json'
    current = vars(args)
    if path.exists():
        old = json.loads(path.read_text())
        diffs = {k: (old.get(k), current.get(k)) for k in sorted(set(old) | set(current))
                 if k not in RESUME_SAFE_ARGS and old.get(k) != current.get(k)}
        if diffs:
            lines = ', '.join(f'{k}: saved={a!r} now={b!r}' for k, (a, b) in diffs.items())
            parser.error(f'{out} holds a run with a different config ({lines}). Use another --out.')
    elif any(out.glob('level_*.pt')):
        parser.error(f'{out} has checkpoints but no config.json; cannot verify settings. Use another --out.')


def load_completed(out, rounds, log):
    """Load consecutive finished levels 0, 1, ... and stop at the first missing/unreadable one."""
    done = []
    for level in range(rounds + 1):
        path = out / f'level_{level:02d}.pt'
        if not path.exists():
            break
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as e:  # e.g. truncated by a crash; this level is simply redone
            log(f'WARNING: cannot read {path.name} ({e}); redoing level {level}')
            break
        if 'train_rng_state' not in ckpt:
            raise SystemExit(f'{path} was written by an older run_imp.py without resume info; use another --out.')
        done.append(ckpt)
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['mnist', 'fashion_mnist', 'cifar10'], default='mnist')
    p.add_argument('--model', choices=['lenet300100', 'conv2'], default='lenet300100')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-seed', type=int, default=0, help='Fixes the 55k/5k train/val split across all runs')
    p.add_argument('--rounds', type=int, default=15, help='Pruning rounds, excluding dense level 0')
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--batch-size', type=int, default=60)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    p.add_argument('--pruning', choices=['layerwise', 'global'], default='layerwise')
    p.add_argument('--global-rate', type=float, default=0.2)
    p.add_argument('--device', default='auto')
    p.add_argument('--no-progress', action='store_true', help='Disable progress bars (logs still printed)')
    p.add_argument('--out', default='runs/mnist_seed42')
    args = p.parse_args()
    if args.model == 'lenet300100' and args.dataset == 'cifar10':
        p.error('Use Conv-2 for CIFAR-10, which is not implemented in this starter.')
    if args.model == 'conv2':
        p.error('Conv-2 is a reserved module, not yet implemented.')
    if args.steps < 1 or args.eval_every < 1 or args.rounds < 0:
        p.error('steps/eval-every must be positive and rounds nonnegative')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    check_config(out, args, p)
    atomic_write(out / 'config.json', lambda f: f.write(json.dumps(vars(args), indent=2).encode()))

    progress = not args.no_progress
    if progress:
        from tqdm import tqdm
    log_file = (out / 'train.log').open('a', encoding='utf-8')

    def log(msg):
        line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
        if progress:
            tqdm.write(line)  # prints above the bars without breaking them
        else:
            print(line, flush=True)
        log_file.write(line + '\n'); log_file.flush()

    train_loader, val_loader, test_loader = loaders(args.dataset, args.batch_size, args.seed, split_seed=args.split_seed)
    model = make_model(args.model)
    rates = {'fc1.weight': 0.2, 'fc2.weight': 0.2, 'fc3.weight': 0.1} if args.pruning == 'layerwise' else {'global': args.global_rate}

    done = load_completed(out, args.rounds, log)
    rows = [c['metrics'] for c in done]
    start = len(done)
    if done:
        last = done[-1]
        original, masks = last['original_state'], last['masks']
        model.load_state_dict(last['end_state'])  # next mask is pruned from these weights
        train_loader.generator.set_state(last['train_rng_state'])
        write_summary(out / 'summary.csv', rows)
        log(f'Resuming from level {start} ({start} of {args.rounds + 1} levels already done)')
    else:
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        masks = initial_masks(model)
        log(f'Starting new run in {out} on {device}: {json.dumps(vars(args))}')
    del done

    levels = range(start, args.rounds + 1)
    outer = tqdm(levels, total=args.rounds + 1, initial=start, desc='IMP levels', position=0,
                 dynamic_ncols=True) if progress else levels
    for level in outer:
        if level:
            masks = prune_by_magnitude(model, masks, rates, strategy=args.pruning)
            model.load_state_dict(original)
            apply_masks(model, masks)
        frac = remaining_fraction(masks)
        metrics = train_fixed_steps(model, train_loader, val_loader, test_loader, masks,
                    steps=args.steps, lr=args.lr, optimizer_name=args.optimizer,
                    eval_every=args.eval_every, device=device,
                    progress=progress, desc=f'level {level:02d} ({frac:.1%} weights)')
        row = {'level': level, 'weights_remaining': frac,
               'best_step': metrics['best_step'], 'best_val_loss': metrics['best_val_loss'],
               'best_test_accuracy': metrics['best_test_accuracy'],
               'end_test_accuracy': metrics['end_test_accuracy']}
        rows.append(row)
        ckpt = {'original_state': original, 'masks': masks,
                'best_state': metrics['best_state'], 'end_state': metrics['end_state'],
                'metrics': row, 'history': metrics['history'],
                'train_rng_state': train_loader.generator.get_state()}
        atomic_write(out / f'level_{level:02d}.pt', lambda f: torch.save(ckpt, f))
        write_summary(out / 'summary.csv', rows)
        log(f'level {level:02d} done: ' + ', '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                                                 for k, v in row.items()))
    if start > args.rounds:
        log(f'All {args.rounds + 1} levels already complete; nothing to do.')
    log_file.close()


if __name__ == '__main__':
    main()
