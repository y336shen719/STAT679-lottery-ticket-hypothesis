"""Run a checkpointed iterative magnitude pruning experiment.

Example: python run_imp.py --dataset mnist --steps 100 --rounds 2 --out runs/smoke
"""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from lth.data import loaders
from lth.models import make_model
from lth.pruning import initial_masks, prune_by_magnitude, remaining_fraction, apply_masks
from lth.train import train_fixed_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['mnist', 'fashion_mnist', 'cifar10'], default='mnist')
    p.add_argument('--model', choices=['lenet300100', 'conv2'], default='lenet300100')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rounds', type=int, default=15, help='Pruning rounds, excluding dense level 0')
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--batch-size', type=int, default=60)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    p.add_argument('--pruning', choices=['layerwise', 'global'], default='layerwise')
    p.add_argument('--global-rate', type=float, default=0.2)
    p.add_argument('--device', default='auto')
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
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    train_loader, val_loader, test_loader = loaders(args.dataset, args.batch_size, args.seed)
    model = make_model(args.model)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    masks = initial_masks(model)
    rates = {'fc1.weight': 0.2, 'fc2.weight': 0.2, 'fc3.weight': 0.1} if args.pruning == 'layerwise' else {'global': args.global_rate}
    rows = []
    for level in range(args.rounds + 1):
        if level:
            masks = prune_by_magnitude(model, masks, rates, strategy=args.pruning)
            model.load_state_dict(original)
            apply_masks(model, masks)
        metrics = train_fixed_steps(model, train_loader, val_loader, test_loader, masks,
                    steps=args.steps, lr=args.lr, optimizer_name=args.optimizer,
                    eval_every=args.eval_every, device=device)
        row = {'level': level, 'weights_remaining': remaining_fraction(masks),
               'best_step': metrics['best_step'], 'best_val_loss': metrics['best_val_loss'],
               'best_test_accuracy': metrics['best_test_accuracy'],
               'end_test_accuracy': metrics['end_test_accuracy']}
        rows.append(row)
        torch.save({'original_state': original, 'masks': masks,
                    'best_state': metrics['best_state'], 'end_state': metrics['end_state'],
                    'metrics': row, 'history': metrics['history']}, out / f'level_{level:02d}.pt')
        with (out / 'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        print(row, flush=True)


if __name__ == '__main__':
    main()
