# Refactor Notes (Stage 1 -> Stage 2)

## Part A: Original code inspection

Original Stage 1 files in this repo:
- Graph generators: `src/graphs.py`
- MRF definition: `src/model.py`
- Potentials: `src/potentials.py`
- Exact enumeration: `src/exact_inference.py`
- Classical BP: `src/bp_sum_product.py`
- Marginal comparison: `src/compare.py`
- Evaluation loop + JSON saving: `src/experiments_v1.py`, `src/utils_io.py`

Original `src/` listing (before adding `nebpmrf` package):
```text
src/
  bp_sum_product.py
  compare.py
  exact_inference.py
  experiments_v1.py
  graphs.py
  model.py
  potentials.py
  utils_io.py
  legacy/
```

## New module mapping

- `src/graphs.py` -> `src/nebpmrf/graphs/generators.py`
- `src/model.py` -> `src/nebpmrf/graphs/mrf.py`
- `src/potentials.py` -> `src/nebpmrf/graphs/potentials.py`
- `src/exact_inference.py` -> `src/nebpmrf/inference/exact_enum.py`
- `src/bp_sum_product.py` -> `src/nebpmrf/inference/bp_classical.py`
- `src/compare.py` -> `src/nebpmrf/inference/metrics.py` (Stage 1-compatible metric retained)
- `src/experiments_v1.py` + `src/utils_io.py` -> `src/nebpmrf/experiments/evaluate.py`, `src/nebpmrf/utils/io.py`

New CLI entrypoints:
- `scripts/run_stage1_baselines.py`
- `scripts/train_neural_bp.py`
- `scripts/eval_neural_bp.py`

## Assumptions / simplifications

- Stage 1 algorithm behavior is preserved in the copied core functions (`exact_marginals_bruteforce`, `bp_sum_product`).
- Training dataset uses fixed `n` and requires `graph_types` with equal edge count for batching.
  - Example valid: `[loopy1]`
  - Example invalid (for batched training): `[tree, loopy1]` because edge counts differ.
- No dataset files are saved; graphs are generated on-the-fly in `SyntheticMRFDataset`.
