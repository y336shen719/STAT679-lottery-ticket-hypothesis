# CS679 Lottery Ticket Hypothesis — PyTorch starter

Research starter based on the algorithm in Frankle & Carbin (2019) and the
Google Research TensorFlow 1.x *reimplementation*:
https://github.com/google-research/lottery-ticket-hypothesis

**Not** a line-by-line port and **not yet** a full paper reproduction.
The code in this repo is newly written; original repository is Apache-2.0.
See its `mnist_fc/constants.py`, `foundations/model_fc.py`,
`foundations/model_base.py`, `foundations/pruning.py`, and
`foundations/experiment.py` for relevant source-level comparisons.

## Setup

Use a clean Python 3.10–3.12 environment and install compatible PyTorch / torchvision:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m pytest -q
```

## Try a small training run

```bash
python run_imp.py --dataset mnist --steps 100 --rounds 2 --out runs/smoke
```

## Reference-style MNIST run (potentially expensive)

```bash
python run_imp.py --dataset mnist --steps 50000 --rounds 15 --seed 11 --out runs/mnist_seed11
```

Use `--dataset fashion_mnist` for the proposal's controlled extension.
An optional alternative is `--pruning global --global-rate 0.2`.

At every level, output includes both minimum-validation-loss checkpoint
and fixed-budget endpoint TEST accuracies. Keep test metrics for reporting
only, not hyperparameter selection. The next mask is formed from the
**end-of-budget** training weights (not best validation checkpoint).
Original weights are restored after each pruning round; optimizer state
starts fresh. Saved `.pt` files contain masks, initial state, and best/end
weights; `summary.csv` is regenerated after each level.

## Caveats / next pull requests

- Exact TF reimplementation config: 784→300→100→10, ReLU hidden units,
  Xavier-normal kernels, zero biases, SGD lr=.1, 50k steps, batch 60,
  per-round prune 20% / 20% / 10% for the three layers.
- Reimplementation's MNIST preprocessing/data split and validation cadence
  need further matching for an exact reproduction. Here 55k/5k split is seeded
  and a default eval interval is 100 steps.
- TF code uses magnitude *cutoffs* (extra ties may be pruned); starter
  uses deterministic exact counts, which usually differ only at ties.
- Validation is checked repeatedly; test evaluated twice per level for
  reporting, with no test-based selection.
- Conv-2 for CIFAR-10 is **not yet implemented**. Paper-specific architecture,
  training schedule, and preprocessing must be agreed before coding.
- The 2×2 controls, convergence-speed curves and multi-seed aggregation
  are **not yet implemented**. `random_masks_like` supplies one primitive.
- Resume from an interrupted process is not yet implemented: existing
  checkpoints prevent loss of output, but rerunning starts at level 0.
- Full bitwise reproducibility between CPU/CUDA and versions is not guaranteed.

## Division of modules

- `lth/models.py`: model definition
- `lth/data.py`: datasets + fixed seeded split
- `lth/pruning.py`: masks, IMP magnitude pruning, random masks
- `lth/train.py`: fixed-step training and evaluation
- `run_imp.py`: experiments + level checkpoints
- `tests/`: basic correctness checks

Keep large datasets and checkpoints outside GitHub (`.gitignore`).
