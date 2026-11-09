# CS679 / AMATH649 — Lottery Ticket Hypothesis

This project studies the Lottery Ticket Hypothesis (LTH) with PyTorch using
Iterative Magnitude Pruning (IMP). The completed experiments evaluate
LeNet-300-100 on MNIST and Fashion-MNIST, with seeds **1, 2, and 42** and
**Levels 0–15** (a dense baseline followed by 15 pruning rounds).

The implementation is informed by Frankle & Carbin (2019) and the
[Google Research TensorFlow reimplementation](https://github.com/google-research/lottery-ticket-hypothesis).
It is an independent implementation and a focused reproduction, not a complete
reproduction of every architecture or experiment in the paper.

## Project structure

| Path | Purpose |
|---|---|
| `lth/models.py` | Model definitions, including LeNet-300-100 |
| `lth/data.py` | Dataset loading, preprocessing, and seeded train/validation split |
| `lth/pruning.py` | Magnitude pruning, binary masks, and random masks |
| `lth/train.py` | Fixed-step training, validation, and test evaluation |
| `run_imp.py` | IMP training and per-level checkpoints; resume after completed levels |
| `run_controls.py` | Training controls for mask choice and initialization |
| `export_results.py` | Export training artifacts into analysis-ready result tables |
| `analysis.ipynb` | Main experimental analysis and Extensions 1–4 |
| `extension_analysis/` | Reusable mask, random-reference, prediction, sparsity, and plotting functions |
| `tests/` | Training/pruning/export tests and extension-analysis tests |
| `runs/` | Run configurations and checkpoints, where available |
| `results/` | Main experiment exports and figures |
| `results/extensions/` | Saved extension executions, tables, figures, and prediction caches |
| `data/` | Local dataset cache; ignored by Git |

## Environment

Use Python 3.10–3.12 with compatible PyTorch and torchvision builds for your
CPU or CUDA environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install the notebook interface separately:
pip install jupyterlab
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Completed main experiments

The recorded formal runs use:

- **Architecture:** 784 → 300 → 100 → 10, with ReLU hidden layers.
- **Initialization:** Xavier-normal weights and zero biases.
- **Training:** Adam, learning rate 0.0012, batch size 60, 50,000 steps per level.
- **Data:** 55,000 training and 5,000 validation examples, using split seed 0;
  10,000 official test examples per dataset. Image values are divided by 255.
- **Evaluation:** validation every 100 steps; report both the minimum-validation-loss
  checkpoint (`best_state`) and the fixed-budget endpoint (`end_state`).
- **Pruning:** remove 20% of surviving FC1/FC2 weights and 10% of surviving FC3
  weights each round, with integer rounding. Biases are not pruned.

Each new mask is selected by magnitude from the previous level's **end-of-budget
weights**. Surviving weights and biases are reset to the original initialization,
then trained with a fresh optimizer. Pruning removes individual connections;
it does not change the layer dimensions. Level 15 retains **9,537 / 266,200
prunable weights (about 3.58%)**.

The 2×2 design compares IMP versus random masks and original versus fresh random
initialization. The main notebook reports accuracy versus sparsity, early-stopping
and end-of-budget results, convergence speed, and multi-seed summaries. Test
metrics are for reporting, not checkpoint or hyperparameter selection.

## Extensions 1–4

The extensions reuse the **first formal experiment's saved checkpoints and
analysis artifacts**. The notebook reads saved extension executions; opening or
running these sections does not retrain models, repeat IMP, or repeat inference.

| Extension | Analysis | Interpretation |
|---|---|---|
| 1 — Cross-seed mask stability | Overall and per-layer Jaccard for seed pairs (1,2), (1,42), and (2,42) | Raw coordinate overlap decreases with sparsity; density and hidden-neuron permutations limit its interpretation. |
| 2 — Density-matched random reference | 2,000 Monte Carlo replicates matching exact per-layer surviving counts; observed overlap and excess Jaccard | Overall excess is larger on MNIST early in pruning. Reference intervals describe random-mask variation, not confidence intervals for the observed seed mean. |
| 3 — Prediction agreement | Saved `best_state` predictions on the same ordered test examples; agreement, shared correct/wrong, and Conditional Error Agreement (CEA) | Low positional overlap can coexist with high prediction agreement. Accuracy partly constrains agreement; CEA adds an error-conditioned view. |
| 4 — Per-layer sparsity | Surviving counts, sparsity, parameter shares, and rounding effects | The prescribed schedule determines layer counts; magnitude pruning determines which coordinates survive. This is not automated layer allocation. |

CEA is computed **for each seed pair before aggregation**:

```text
CEA = Shared Wrong / (1 − Shared Correct)
```

It is the fraction of examples with at least one model error on which both models
predict the exact same incorrect class. A zero denominator gives an undefined
value. CEA does not establish identical representations or decision boundaries.
Shaded ±1 sample SD bands describe variation across shared-seed pairs, not
confidence intervals from independent pair replications.

The notebook also includes an Excess Jaccard grouped bar chart, a CEA trajectory
chart across all 16 levels, and an independent **Overall Discussion — Extensions
1–4** section.

### Saved executions used by the notebook

All paths below are relative to `results/extensions/`:

| Extension | Execution directory | Main contents |
|---|---|---|
| 1 | `ext1_20261108_233054_662fcc/` | Artifact inventory; pairwise and summary mask Jaccard tables; figures |
| 2 | `ext2_20261109_091130_9715c8/` | Random-reference samples and summaries; comparison and excess-overlap figures |
| 3 | `ext3_20261109_143352_7e93af/` | 96 prediction caches; accuracy, agreement, and structure–function tables; figures |
| 4 | `ext4_20261109_151206_b7cf93/` | Detailed layer counts, sparsity summaries, allocation table, and figures |

CEA is derived in memory from the saved Extension 3 pair table. Its summary table
and chart are displayed in the notebook without modifying the original CSVs.
Execution metadata retains historical provenance; current read helpers check
required artifacts and table structure rather than enforcing historical source
hash equality.

## Viewing the analysis

Start Jupyter from the repository root so relative paths resolve correctly:

```bash
jupyter lab analysis.ipynb
```

Saved notebook outputs allow inspection without starting training. To regenerate
extension displays, run the Extension 1–4 cells in order. Required result folders
must be present; some read helpers also require the referenced checkpoint paths.
The extension plotting cells render from saved tables, and the added bar-chart
cell saves its PNG/PDF only if those files do not already exist.

**Running the entire notebook also executes the original main-analysis cells,
which regenerate images under `results/figures/`.** To avoid overwriting those
images, run only the extension sections. If plotting modules change while a
notebook kernel is active, restart the kernel before rerunning the extension cells.
Reload the notebook from disk after external edits to avoid saving an older open
copy over the updated file.

## Optional: reproduce training

These commands start training; they are **not required to read the completed
extension analyses**. Use a new output directory for a separate experiment.

Small smoke run:

```bash
python run_imp.py --dataset mnist --steps 100 --rounds 2 --out runs/smoke
```

Formal-style run for one dataset and seed:

```bash
python run_imp.py --dataset mnist --optimizer adam --lr 0.0012 \
  --steps 50000 --rounds 15 --seed 1 --split-seed 0 \
  --batch-size 60 --eval-every 100 --out runs/mnist_adam_seed1_reproduction
python run_controls.py --run runs/mnist_adam_seed1_reproduction
python export_results.py --run runs/mnist_adam_seed1_reproduction
```

Repeat with the other seeds and `--dataset fashion_mnist` as needed. The CLI's
default optimizer is SGD with learning rate 0.1; specify Adam and 0.0012 to match
the recorded formal configuration. Dataset loading may download missing data.

Rerunning an IMP command with matching settings resumes after the last completed
level; it does not resume partway through an unfinished level. IMP `.pt` files
store `masks`, `original_state`, `best_state`, `end_state`, metrics, training
history, and `train_rng_state`. Control checkpoints have a different format and
should not be assumed to contain trained model states for prediction analysis.

## Tests and limitations

```bash
python -m pytest -q
# Extension-analysis tests only:
python -m pytest -q tests/test_extension_*.py
```

The conclusions are limited to the evaluated architecture, datasets, and three
training seeds. Pairwise comparisons share runs and are not independent
replications. No formal significance test or causal claim about dataset difficulty
is implied by the extension figures. Exact CPU/CUDA and cross-version bitwise
reproducibility is not guaranteed.

The original TensorFlow implementation uses magnitude cutoffs, whereas this
implementation prunes exact rounded counts; behavior can differ at ties.
Conv-2/CIFAR-10 experiments and global-versus-layerwise comparisons are outside
this completed analysis.

Datasets, `runs/`, and `.pt`/`.pth` files are ignored by default. Some historical
checkpoints may already be tracked; confirm the required artifacts exist locally
rather than assuming every clone or checkout contains them.
