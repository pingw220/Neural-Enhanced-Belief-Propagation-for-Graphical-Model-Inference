# Neural-Enhanced-Belief-Propagation-for-Graphical-Model-Inference

Stage 1 (exact enumeration + classical BP) is preserved, and Stage 2 adds Neural-Enhanced BP with a residual MLP correction to the classical message logit update.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
python scripts/run_stage1_baselines.py --config configs/stage1.yaml
python scripts/train_neural_bp.py --config configs/neuralbp_train.yaml
python scripts/eval_neural_bp.py --config configs/neuralbp_eval.yaml
```

By default, training also updates `results/stage2_train/latest/model_latest.pt`, and `configs/neuralbp_eval.yaml` points to it.
You can still override with an explicit checkpoint:

```bash
python scripts/eval_neural_bp.py --config configs/neuralbp_eval.yaml --checkpoint results/stage2_train/<timestamp>/model_epoch4.pt
```

## Project Layout

```text
src/nebpmrf/
  graphs/
  inference/
  neural/
  data/
  experiments/
  utils/
scripts/
configs/
results/
```

## Stage 1 Baselines

`python scripts/run_stage1_baselines.py --config configs/stage1.yaml`

Outputs are JSON files under `results/stage1/<timestamp>/`:
- `exact_*.json`
- `bp_*.json`
- `compare_*.json`

## Stage 2 Training

`python scripts/train_neural_bp.py --config configs/neuralbp_train.yaml`

Outputs are stored in `results/stage2_train/<timestamp>/`:
- `model_epochX.pt`
- `config.yaml`
- `history.json`

Neural update per directed edge uses:
- Features: `[sum_in_excl, u_i, logpsi00, logpsi01, logpsi10, logpsi11]`
- Residual update: `l_next = (1-damping)*l + damping*(l_classic + delta_theta(features))`
- Stable classical update via `torch.logsumexp`
- Message logit clamp to `[-20, 20]`

## Stage 2 Evaluation

`python scripts/eval_neural_bp.py --config configs/neuralbp_eval.yaml --checkpoint <path_to_ckpt>`

Outputs are stored in `results/stage2_eval/<timestamp>/metrics.json` and include:
- Average L1 per beta
- Average CE and KL per beta
- Optional neural convergence curve (mean max-delta per iteration)

## Sanity Checks

### 1) Tree exactness check (classical BP)
On tree graphs, BP should match exact marginals up to numerical precision.

```bash
python - <<'PY'
import numpy as np
from nebpmrf.graphs.generators import build_edges
from nebpmrf.graphs.mrf import make_model
from nebpmrf.graphs.potentials import make_phi_bias, make_psi_prefer_equal
from nebpmrf.inference.exact_enum import exact_marginals_bruteforce
from nebpmrf.inference.bp_classical import bp_sum_product

n=6
edges=build_edges('tree', n=n, seed=0)
model=make_model(n, edges, make_phi_bias(1.0,0,0), make_psi_prefer_equal(0.8))
_, exact = exact_marginals_bruteforce(model)
bp, _, _ = bp_sum_product(model, max_iters=50, tol=1e-12, damping=0.0)
err = np.max(np.abs(np.array(exact)-np.array(bp)))
print('max_abs_error=', err)
PY
```

### 2) Neural BP (untrained) no-crash check
Run one short training epoch on CPU:

```bash
python scripts/train_neural_bp.py --config configs/neuralbp_train.yaml
```

### 3) Overfit check (tiny set)
Set in `configs/neuralbp_train.yaml`:
- `num_train_samples: 20`
- `epochs: 20`

Then rerun training and verify from logs/history:
- train loss decreases
- train L1 decreases
- neural train-set L1 can beat classical on that tiny set

### 4) Generalization check
Train on `graph_types: [loopy1]`, then evaluate with:
- `graph_types: [loopy1]` (in-distribution)
- `graph_types: [loopy2]` (transfer)

using `scripts/eval_neural_bp.py`.
