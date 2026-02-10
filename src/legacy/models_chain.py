import math
from typing import List, Tuple

def chain_edges(n: int) -> List[Tuple[int, int]]:
    """Undirected chain edges: (0,1), (1,2), ..., (n-2,n-1)."""
    return [(i, i + 1) for i in range(n - 1)]

def psi_equal(beta: float) -> List[List[float]]:
    """
    Pairwise potential matrix for binary states {0,1}.
    psi[xi][xj] = exp(beta) if xi==xj else 1
    """
    e = math.exp(beta)
    return [
        [e, 1.0],
        [1.0, e],
    ]

def phi_bias(i: int, xi: int, h: float, bias_node: int = 0, favor_value: int = 0) -> float:
    """
    Unary potential phi_i(xi).
    Only applies to one node (bias_node). Favor favor_value with strength h.
    """
    if i != bias_node:
        return 1.0
    return math.exp(h) if xi == favor_value else 1.0

def normalize(vec: List[float]) -> List[float]:
    s = vec[0] + vec[1]
    if s <= 0:
        raise ValueError("Cannot normalize a non-positive-sum vector.")
    return [vec[0] / s, vec[1] / s]
