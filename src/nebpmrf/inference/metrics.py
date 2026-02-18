from typing import Any, Dict

import numpy as np


def compare_marginals_lists(exact: list, approx: list) -> Dict[str, Any]:
    """
    Stage 1-compatible L1 metric on binary distributions [P0,P1].
    """
    if len(exact) != len(approx):
        raise ValueError(f"Length mismatch: len(exact)={len(exact)} vs len(approx)={len(approx)}")

    errs = []
    for e, a in zip(exact, approx):
        err = abs(e[0] - a[0]) + abs(e[1] - a[1])
        errs.append(err)

    arr = np.asarray(errs, dtype=np.float64)
    return {
        "mean_l1_error": float(np.mean(arr)) if arr.size else 0.0,
        "max_l1_error": float(np.max(arr)) if arr.size else 0.0,
    }


def compare_node_marginals(exact_p1: np.ndarray, approx_p1: np.ndarray) -> Dict[str, Any]:
    if exact_p1.shape != approx_p1.shape:
        raise ValueError(f"Shape mismatch: exact={exact_p1.shape}, approx={approx_p1.shape}")
    l1_per_node = np.abs(exact_p1 - approx_p1)
    return {
        "mean_l1_error": float(np.mean(l1_per_node)),
        "max_l1_error": float(np.max(l1_per_node)),
    }


def binary_cross_entropy(exact_p1: np.ndarray, pred_p1: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(exact_p1, eps, 1.0 - eps)
    q = np.clip(pred_p1, eps, 1.0 - eps)
    ce = -(p * np.log(q) + (1.0 - p) * np.log(1.0 - q))
    return float(np.mean(ce))


def binary_kl(exact_p1: np.ndarray, pred_p1: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(exact_p1, eps, 1.0 - eps)
    q = np.clip(pred_p1, eps, 1.0 - eps)
    kl = p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
    return float(np.mean(kl))
