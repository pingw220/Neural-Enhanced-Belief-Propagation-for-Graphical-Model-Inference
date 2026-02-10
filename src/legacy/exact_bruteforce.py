from itertools import product
from typing import List, Tuple, Dict, Any

from models_chain import chain_edges, psi_equal, phi_bias
from utils_io import save_json, make_run_id

def unnormalized_weight(
    x: Tuple[int, ...],
    edges: List[Tuple[int, int]],
    psi: List[List[float]],
    h: float,
    bias_node: int = 0,
    favor_value: int = 0,
) -> float:
    """Compute unnormalized score: prod_i phi_i(x_i) * prod_(i,j) psi_ij(x_i,x_j)."""
    w = 1.0

    # unary terms
    for i in range(len(x)):
        w *= phi_bias(i, x[i], h=h, bias_node=bias_node, favor_value=favor_value)

    # pairwise terms
    for i, j in edges:
        w *= psi[x[i]][x[j]]

    return w

def exact_marginals_bruteforce(
    n: int,
    beta: float,
    h: float = 0.0,
    bias_node: int = 0,
    favor_value: int = 0,
):
    """
    Enumerate all 2^n assignments, compute exact node marginals.
    Returns: (Z, marginals) where marginals[i] = [P(x_i=0), P(x_i=1)]
    """
    edges = chain_edges(n)
    psi = psi_equal(beta)

    marg = [[0.0, 0.0] for _ in range(n)]
    Z = 0.0

    for x in product([0, 1], repeat=n):
        w = unnormalized_weight(
            x, edges, psi,
            h=h, bias_node=bias_node, favor_value=favor_value
        )
        Z += w
        for i in range(n):
            marg[i][x[i]] += w

    for i in range(n):
        marg[i][0] /= Z
        marg[i][1] /= Z

    return Z, marg

def save_exact_results(
    n: int, beta: float, h: float, bias_node: int, favor_value: int,
    Z: float, marginals: List[List[float]]
) -> str:
    run_id = make_run_id()
    out_path = f"../results/exact_chain_n{n}_beta{beta}_h{h}_bias{bias_node}_favor{favor_value}_{run_id}.json"

    payload: Dict[str, Any] = {
        "method": "exact_bruteforce",
        "graph": "chain",
        "n": n,
        "beta": beta,
        "h": h,
        "bias_node": bias_node,
        "favor_value": favor_value,
        "partition_Z": Z,
        "marginals": {str(i): marginals[i] for i in range(n)},
    }
    save_json(payload, out_path)
    return out_path

if __name__ == "__main__":
    n = 6
    beta = 0.8
    h = 1.0
    bias_node = 0
    favor_value = 0

    Z, marg = exact_marginals_bruteforce(n, beta, h=h, bias_node=bias_node, favor_value=favor_value)

    print(f"EXACT (bruteforce) marginals on chain: n={n}, beta={beta}, h={h} (bias node {bias_node} favor {favor_value})")
    print(f"Partition function Z = {Z:.6f}\n")
    for i in range(n):
        p0, p1 = marg[i]
        print(f"node {i}: P(x=0)={p0:.6f}, P(x=1)={p1:.6f}, sum={p0+p1:.6f}")

    out_path = save_exact_results(n, beta, h, bias_node, favor_value, Z, marg)
    print(f"\nSaved: {out_path}")
