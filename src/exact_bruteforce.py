import math
from itertools import product
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

def unnormalized_weight(x: Tuple[int, ...], edges: List[Tuple[int, int]], psi: List[List[float]]) -> float:
    """Compute \tilde{mu}(x) = prod_{(i,j) in E} psi(x_i, x_j)."""
    w = 1.0
    for i, j in edges:
        w *= psi[x[i]][x[j]]
    return w

def exact_marginals_bruteforce(n: int, beta: float):
    """
    Enumerate all 2^n assignments, compute exact node marginals.
    Returns: (Z, marginals) where marginals[i] = [P(x_i=0), P(x_i=1)]
    """
    edges = chain_edges(n)
    psi = psi_equal(beta)

    # accumulate unnormalized counts for marginals
    marg = [[0.0, 0.0] for _ in range(n)]
    Z = 0.0

    for x in product([0, 1], repeat=n):
        w = unnormalized_weight(x, edges, psi)
        Z += w
        for i in range(n):
            marg[i][x[i]] += w

    # normalize
    for i in range(n):
        marg[i][0] /= Z
        marg[i][1] /= Z

    return Z, marg

if __name__ == "__main__":
    n = 6
    beta = 0.8  # try: 0.2, 0.8, 1.5
    Z, marg = exact_marginals_bruteforce(n, beta)

    print(f"n={n}, beta={beta}")
    print(f"Partition function Z = {Z:.6f}\n")
    for i in range(n):
        p0, p1 = marg[i]
        print(f"node {i}: P(x=0)={p0:.6f}, P(x=1)={p1:.6f}, sum={p0+p1:.6f}")
