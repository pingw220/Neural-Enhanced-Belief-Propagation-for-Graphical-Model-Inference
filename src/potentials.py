import math
from typing import Callable

# Unary: bias one node toward one value
def make_phi_bias(h: float, bias_node: int = 0, favor_value: int = 0) -> Callable[[int, int], float]:
    """
    phi(i, xi) = exp(h) if (i==bias_node and xi==favor_value) else 1
    """
    def phi(i: int, xi: int) -> float:
        if i == bias_node and xi == favor_value:
            return math.exp(h)
        return 1.0
    return phi

# Pairwise: Ising-like "prefer equal" potential
def make_psi_prefer_equal(beta: float) -> Callable[[int, int, int, int], float]:
    """
    psi(i,j,xi,xj) = exp(beta) if xi==xj else 1
    """
    e = math.exp(beta)
    def psi(i: int, j: int, xi: int, xj: int) -> float:
        return e if xi == xj else 1.0
    return psi
