from itertools import product
from typing import Dict, List, Tuple

from model import GraphModel

def exact_marginals_bruteforce(model: GraphModel) -> Tuple[float, List[List[float]]]:
    """
    Exact by enumeration. Works for small n (e.g., n<=20-ish is already too big).
    Returns (Z, marginals[i]=[P0,P1])
    """
    n = model.n
    marg = [[0.0, 0.0] for _ in range(n)]
    Z = 0.0

    # pre-normalize edges to (min,max) to avoid double counting
    edges = [(a, b) for a, b in model.edges]

    for x in product([0, 1], repeat=n):
        w = 1.0
        # unary
        for i in range(n):
            w *= model.phi(i, x[i])
        # pairwise
        for a, b in edges:
            w *= model.psi(a, b, x[a], x[b])

        Z += w
        for i in range(n):
            marg[i][x[i]] += w

    for i in range(n):
        marg[i][0] /= Z
        marg[i][1] /= Z

    return Z, marg
