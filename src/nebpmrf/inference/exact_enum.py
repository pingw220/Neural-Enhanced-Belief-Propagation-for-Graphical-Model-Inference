from itertools import product
from typing import List, Tuple

from nebpmrf.graphs.mrf import GraphModel


def exact_marginals_bruteforce(model: GraphModel) -> Tuple[float, List[List[float]]]:
    """
    Exact by enumeration. Works for small n.
    Returns (Z, marginals[i]=[P0,P1]).
    """
    n = model.n
    marg = [[0.0, 0.0] for _ in range(n)]
    z_part = 0.0

    edges = [(a, b) for a, b in model.edges]

    for x in product([0, 1], repeat=n):
        weight = 1.0
        for i in range(n):
            weight *= model.phi(i, x[i])
        for a, b in edges:
            weight *= model.psi(a, b, x[a], x[b])

        z_part += weight
        for i in range(n):
            marg[i][x[i]] += weight

    for i in range(n):
        marg[i][0] /= z_part
        marg[i][1] /= z_part

    return z_part, marg
