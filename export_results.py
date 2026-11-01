"""Copy the shareable results of an IMP run (+ its controls) from runs/ into results/.

Writes small text files only (no .pt checkpoints), so results/ can be committed and
analysis.ipynb can redraw every figure without the checkpoints:
    config.json, summary.csv, train.log           IMP run
    controls/<condition>/summary.csv, train.log  control runs (if any)
    histories.csv                                 validation curves of every training run:
        condition, level, weights_remaining, step, val_loss, val_accuracy
        (condition 'ticket' = the IMP run itself; the controls' 'ticket' check is not repeated)

Example: python export_results.py --run runs/mnist_adam_seed42
"""
import argparse
import csv
import shutil
from pathlib import Path

import torch

HISTORY_FIELDS = ['condition', 'level', 'weights_remaining', 'step', 'val_loss', 'val_accuracy']


def history_rows(condition, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    m = ckpt['metrics']
    return [{'condition': condition, 'level': m['level'], 'weights_remaining': m['weights_remaining'], **h}
            for h in ckpt['history']]


def export(run, out):
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'summary.csv', 'train.log'):
        shutil.copy2(run / name, out / name)
    rows = []
    for path in sorted(run.glob('level_*.pt')):
        rows += history_rows('ticket', path)
    controls = run / 'controls'
    if controls.is_dir():
        (out / 'controls').mkdir(exist_ok=True)
        for name in ('imp_config.json', 'train.log'):
            if (controls / name).exists():
                shutil.copy2(controls / name, out / 'controls' / name)
        for cond_dir in sorted(p for p in controls.iterdir() if p.is_dir()):
            if (cond_dir / 'summary.csv').exists():
                (out / 'controls' / cond_dir.name).mkdir(exist_ok=True)
                shutil.copy2(cond_dir / 'summary.csv', out / 'controls' / cond_dir.name / 'summary.csv')
            if cond_dir.name != 'ticket':  # the ticket check duplicates the IMP run's curves
                for path in sorted(cond_dir.glob('level_*.pt')):
                    rows += history_rows(cond_dir.name, path)
    with open(out / 'histories.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='run_imp.py output folder')
    p.add_argument('--out', default=None, help='Default: results/<run folder name>')
    args = p.parse_args()
    out = Path(args.out) if args.out else Path('results') / Path(args.run).name
    rows = export(args.run, out)
    runs = len({(r['condition'], r['level']) for r in rows})
    print(f'Exported {args.run} -> {out}: {runs} training runs, {len(rows)} history rows')


if __name__ == '__main__':
    main()
