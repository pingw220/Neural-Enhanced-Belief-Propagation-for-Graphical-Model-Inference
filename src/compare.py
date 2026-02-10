# src/compare.py
from typing import Any, Dict, List


def l1_dist_2(p: List[float], q: List[float]) -> float:
    """
    L1 distance between two binary distributions p and q.
    p, q are length-2 lists: [P(x=0), P(x=1)].
    """
    if len(p) != 2 or len(q) != 2:
        raise ValueError(f"Expected length-2 distributions, got len(p)={len(p)}, len(q)={len(q)}")
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def compare_marginals(exact: List[List[float]], approx: List[List[float]]) -> Dict[str, Any]:
    """
    Compare node marginals from exact inference vs an approximate method (e.g., BP).

    Args:
        exact:  list of length n, exact[i] = [P0, P1]
        approx: list of length n, approx[i] = [P0, P1]

    Returns:
        dict with:
          - mean_l1_error
          - max_l1_error
          - per_node: mapping "i" -> {"exact":..., "approx":..., "l1_error":...}
    """
    if len(exact) != len(approx):
        raise ValueError(f"Length mismatch: len(exact)={len(exact)} vs len(approx)={len(approx)}")

    n = len(exact)
    per_node: Dict[str, Any] = {}
    max_err = 0.0
    sum_err = 0.0

    for i in range(n):
        e = exact[i]
        a = approx[i]
        err = l1_dist_2(e, a)

        per_node[str(i)] = {
            "exact": e,
            "approx": a,
            "l1_error": err,
        }

        if err > max_err:
            max_err = err
        sum_err += err

    return {
        "mean_l1_error": sum_err / n if n > 0 else 0.0,
        "max_l1_error": max_err,
        "per_node": per_node,
    }
