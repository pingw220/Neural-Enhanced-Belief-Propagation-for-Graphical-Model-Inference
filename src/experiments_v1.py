from typing import Dict, Any, List, Tuple

from model import make_model
from potentials import make_phi_bias, make_psi_prefer_equal
from graphs import make_chain_edges, make_random_tree_edges, add_one_cycle_edge
from exact_inference import exact_marginals_bruteforce
from bp_sum_product import bp_sum_product
from compare import compare_marginals
from utils_io import save_json, make_run_id

def build_edges(graph_type: str, n: int, seed: int) -> List[Tuple[int, int]]:
    if graph_type == "chain":
        return make_chain_edges(n)
    if graph_type == "tree":
        return make_random_tree_edges(n, seed=seed)
    if graph_type == "loopy":
        base = make_random_tree_edges(n, seed=seed)
        return add_one_cycle_edge(base, n=n, seed=seed + 999)
    raise ValueError("graph_type must be one of: chain, tree, loopy")

def run_one(
    graph_type: str,
    n: int,
    beta: float,
    h: float,
    seed: int,
    bias_node: int,
    favor_value: int,
    bp_max_iters: int,
    bp_tol: float,
    bp_damping: float,
) -> None:
    edges = build_edges(graph_type, n, seed)

    phi = make_phi_bias(h=h, bias_node=bias_node, favor_value=favor_value)
    psi = make_psi_prefer_equal(beta=beta)
    model = make_model(n=n, edges=edges, phi=phi, psi=psi)

    # exact
    Z, exact_marg = exact_marginals_bruteforce(model)

    # bp
    bp_marg, _, deltas = bp_sum_product(
        model,
        max_iters=bp_max_iters,
        tol=bp_tol,
        damping=bp_damping,
    )

    comp = compare_marginals(exact_marg, bp_marg)

    run_id = make_run_id()
    tag = f"{graph_type}_n{n}_beta{beta}_h{h}_seed{seed}_bias{bias_node}_favor{favor_value}_{run_id}"

    # save exact
    save_json({
        "method": "exact_bruteforce",
        "graph": graph_type,
        "n": n,
        "beta": beta,
        "h": h,
        "seed": seed,
        "bias_node": bias_node,
        "favor_value": favor_value,
        "edges": edges,
        "partition_Z": Z,
        "marginals": {str(i): exact_marg[i] for i in range(n)},
    }, f"../results/exact_{tag}.json")

    # save bp
    save_json({
        "method": "bp_sum_product",
        "graph": graph_type,
        "n": n,
        "beta": beta,
        "h": h,
        "seed": seed,
        "bias_node": bias_node,
        "favor_value": favor_value,
        "edges": edges,
        "bp_max_iters": bp_max_iters,
        "bp_tol": bp_tol,
        "bp_damping": bp_damping,
        "convergence_deltas": deltas,
        "marginals": {str(i): bp_marg[i] for i in range(n)},
    }, f"../results/bp_{tag}.json")

    # save compare
    save_json({
        "method": "compare_exact_vs_bp",
        "graph": graph_type,
        "n": n,
        "beta": beta,
        "h": h,
        "seed": seed,
        "bias_node": bias_node,
        "favor_value": favor_value,
        "bp_max_iters": bp_max_iters,
        "bp_tol": bp_tol,
        "bp_damping": bp_damping,
        **comp,
    }, f"../results/compare_{tag}.json")

    print(f"[DONE] {tag}")
    print("  mean L1 error:", comp["mean_l1_error"])
    print("  max  L1 error:", comp["max_l1_error"])
    print()

if __name__ == "__main__":
    # Default: reproduce your current setting
    graph_type = "loopy"   # change to: "tree" or "loopy"
    n = 6
    beta = 1.5
    h = 1.0
    seed = 0
    bias_node = 0
    favor_value = 0

    # BP settings
    bp_max_iters = 50
    bp_tol = 1e-9
    bp_damping = 0.3   # for loopy you might set 0.2~0.5

    run_one(
        graph_type=graph_type,
        n=n,
        beta=beta,
        h=h,
        seed=seed,
        bias_node=bias_node,
        favor_value=favor_value,
        bp_max_iters=bp_max_iters,
        bp_tol=bp_tol,
        bp_damping=bp_damping,
    )
