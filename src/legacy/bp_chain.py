from typing import List, Dict, Any

from models_chain import psi_equal, phi_bias, normalize
from utils_io import save_json, make_run_id

def bp_chain_marginals(
    n: int,
    beta: float,
    h: float,
    bias_node: int = 0,
    favor_value: int = 0,
) -> List[List[float]]:
    """
    Sum-product BP on a chain via forward/backward messages.
    Computes exact marginals on a chain (tree).
    """
    psi = psi_equal(beta)

    # forward messages m_{i->i+1}(x_{i+1})
    m_fwd = [None] * n  # type: ignore
    incoming_left = [0.5, 0.5]  # conceptual m_{-1->0}

    for i in range(n - 1):
        out = [0.0, 0.0]
        for xj in [0, 1]:
            s = 0.0
            for xi in [0, 1]:
                phi_i = phi_bias(i, xi, h=h, bias_node=bias_node, favor_value=favor_value)
                s += phi_i * psi[xi][xj] * incoming_left[xi]
            out[xj] = s
        out = normalize(out)
        m_fwd[i] = out
        incoming_left = out

    # backward messages m_{i->i-1}(x_{i-1})
    m_bwd = [None] * n  # type: ignore
    incoming_right = [0.5, 0.5]  # conceptual m_{n->n-1}

    for i in range(n - 1, 0, -1):
        out = [0.0, 0.0]
        for xj in [0, 1]:
            s = 0.0
            for xi in [0, 1]:
                phi_i = phi_bias(i, xi, h=h, bias_node=bias_node, favor_value=favor_value)
                s += phi_i * psi[xi][xj] * incoming_right[xi]
            out[xj] = s
        out = normalize(out)
        m_bwd[i] = out
        incoming_right = out

    # node beliefs
    beliefs: List[List[float]] = []
    for i in range(n):
        b = [0.0, 0.0]
        for xi in [0, 1]:
            val = phi_bias(i, xi, h=h, bias_node=bias_node, favor_value=favor_value)
            if i - 1 >= 0:
                val *= m_fwd[i - 1][xi]
            if i + 1 < n:
                val *= m_bwd[i + 1][xi]
            b[xi] = val
        b = normalize(b)
        beliefs.append(b)

    return beliefs

def save_bp_results(
    n: int, beta: float, h: float, bias_node: int, favor_value: int,
    beliefs: List[List[float]]
) -> str:
    run_id = make_run_id()
    out_path = f"../results/bp_chain_n{n}_beta{beta}_h{h}_bias{bias_node}_favor{favor_value}_{run_id}.json"

    payload: Dict[str, Any] = {
        "method": "bp_chain_forward_backward",
        "graph": "chain",
        "n": n,
        "beta": beta,
        "h": h,
        "bias_node": bias_node,
        "favor_value": favor_value,
        "marginals": {str(i): beliefs[i] for i in range(n)},
    }
    save_json(payload, out_path)
    return out_path

if __name__ == "__main__":
    n = 6
    beta = 0.8
    h = 1.0
    bias_node = 0
    favor_value = 0

    beliefs = bp_chain_marginals(n, beta, h, bias_node=bias_node, favor_value=favor_value)

    print(f"BP marginals on chain: n={n}, beta={beta}, h={h} (bias node {bias_node} favor {favor_value})")
    for i, b in enumerate(beliefs):
        print(f"node {i}: P(x=0)={b[0]:.6f}, P(x=1)={b[1]:.6f}, sum={(b[0]+b[1]):.6f}")

    out_path = save_bp_results(n, beta, h, bias_node, favor_value, beliefs)
    print(f"\nSaved: {out_path}")
