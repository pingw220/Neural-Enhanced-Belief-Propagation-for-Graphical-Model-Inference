import math
from typing import Any, Dict, List

import numpy as np
import torch

from nebpmrf.graphs.generators import build_edges
from nebpmrf.graphs.mrf import make_model
from nebpmrf.graphs.potentials import make_phi_bias, make_psi_prefer_equal
from nebpmrf.inference.bp_classical import bp_sum_product
from nebpmrf.inference.exact_enum import exact_marginals_bruteforce
from nebpmrf.inference.metrics import (
    binary_cross_entropy,
    binary_kl,
    compare_marginals_lists,
)
from nebpmrf.neural.neural_bp import NeuralBPResidual
from nebpmrf.utils.io import make_timestamp, save_json


def run_stage1_baselines(cfg: Dict[str, Any]) -> str:
    out_root = cfg.get("output_root", "results/stage1")
    run_dir = f"{out_root}/{make_timestamp()}"

    runs = cfg["runs"]
    for run in runs:
        graph_type = run["graph_type"]
        n = int(run["n"])
        beta = float(run["beta"])
        h = float(run["h"])
        seed = int(run["seed"])
        bias_node = int(run["bias_node"])
        favor_value = int(run["favor_value"])
        bp_max_iters = int(run["bp_max_iters"])
        bp_tol = float(run["bp_tol"])
        bp_damping = float(run["bp_damping"])

        edges = build_edges(graph_type=graph_type, n=n, seed=seed)
        phi = make_phi_bias(h=h, bias_node=bias_node, favor_value=favor_value)
        psi = make_psi_prefer_equal(beta=beta)
        model = make_model(n=n, edges=edges, phi=phi, psi=psi)

        z_part, exact_marg = exact_marginals_bruteforce(model)
        bp_marg, _, deltas = bp_sum_product(
            model=model,
            max_iters=bp_max_iters,
            tol=bp_tol,
            damping=bp_damping,
        )

        comp = compare_marginals_lists(exact=exact_marg, approx=bp_marg)

        tag = (
            f"{graph_type}_n{n}_beta{beta}_h{h}_seed{seed}_"
            f"bias{bias_node}_favor{favor_value}"
        )

        save_json(
            {
                "method": "exact_bruteforce",
                "graph": graph_type,
                "n": n,
                "beta": beta,
                "h": h,
                "seed": seed,
                "bias_node": bias_node,
                "favor_value": favor_value,
                "edges": edges,
                "partition_Z": z_part,
                "marginals": {str(i): exact_marg[i] for i in range(n)},
            },
            f"{run_dir}/exact_{tag}.json",
        )

        save_json(
            {
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
            },
            f"{run_dir}/bp_{tag}.json",
        )

        save_json(
            {
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
            },
            f"{run_dir}/compare_{tag}.json",
        )

    return run_dir


def _build_single_instance(
    graph_type: str,
    n: int,
    beta: float,
    h: float,
    seed: int,
    bias_node: int,
    favor_value: int,
) -> Dict[str, Any]:
    edges = build_edges(graph_type=graph_type, n=n, seed=seed)

    phi = make_phi_bias(h=h, bias_node=bias_node, favor_value=favor_value)
    psi = make_psi_prefer_equal(beta=beta)
    model = make_model(n=n, edges=edges, phi=phi, psi=psi)

    _, exact_marg = exact_marginals_bruteforce(model)
    p_exact = np.asarray([m[1] for m in exact_marg], dtype=np.float64)

    u_node = np.zeros(n, dtype=np.float32)
    u_node[bias_node] = h if favor_value == 1 else -h

    log_psi_one = np.asarray([[beta, 0.0], [0.0, beta]], dtype=np.float32)
    log_psi_und = np.stack([log_psi_one for _ in edges], axis=0)
    edge_index_und = np.asarray(edges, dtype=np.int64).T

    return {
        "edges": edges,
        "model": model,
        "p_exact": p_exact,
        "u_node": u_node,
        "log_psi_und": log_psi_und,
        "edge_index_und": edge_index_und,
    }


def evaluate_neural_bp(
    model: NeuralBPResidual,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    n = int(cfg["n"])
    h = float(cfg["h"])
    bias_node = int(cfg["bias_node"])
    favor_value = int(cfg["favor_value"])
    num_graphs = int(cfg["num_test_graphs"])
    graph_types = cfg["graph_types"]
    betas = cfg["beta_values"]

    bp_max_iters = int(cfg["bp_max_iters"])
    bp_tol = float(cfg["bp_tol"])
    bp_damping = float(cfg["bp_damping"])

    t_iters = int(cfg["T"])
    neural_damping = float(cfg["damping"])

    summary: Dict[str, Any] = {
        "config": cfg,
        "results": [],
    }

    model.eval()

    with torch.no_grad():
        for graph_type in graph_types:
            for beta in betas:
                cls_l1s: List[float] = []
                neu_l1s: List[float] = []
                cls_ce: List[float] = []
                neu_ce: List[float] = []
                cls_kl: List[float] = []
                neu_kl: List[float] = []
                curves: List[List[float]] = []

                for k in range(num_graphs):
                    seed = int(cfg.get("seed", 0)) + 10000 * k + int(round(beta * 1000))
                    instance = _build_single_instance(
                        graph_type=graph_type,
                        n=n,
                        beta=float(beta),
                        h=h,
                        seed=seed,
                        bias_node=bias_node,
                        favor_value=favor_value,
                    )

                    exact = instance["p_exact"]

                    bp_belief, _, _ = bp_sum_product(
                        model=instance["model"],
                        max_iters=bp_max_iters,
                        tol=bp_tol,
                        damping=bp_damping,
                    )
                    classical = np.asarray([x[1] for x in bp_belief], dtype=np.float64)

                    u_node = torch.tensor(instance["u_node"], dtype=torch.float32, device=device).unsqueeze(0)
                    log_psi_und = torch.tensor(instance["log_psi_und"], dtype=torch.float32, device=device).unsqueeze(0)
                    edge_index_und = torch.tensor(instance["edge_index_und"], dtype=torch.long, device=device).unsqueeze(0)

                    p_hat, curve = model(
                        u_node=u_node,
                        log_psi_und=log_psi_und,
                        edge_index_und=edge_index_und,
                        num_iters=t_iters,
                        damping=neural_damping,
                        return_curve=True,
                    )
                    neural = p_hat.squeeze(0).cpu().numpy().astype(np.float64)

                    cls_l1s.append(float(np.mean(np.abs(exact - classical))))
                    neu_l1s.append(float(np.mean(np.abs(exact - neural))))
                    cls_ce.append(binary_cross_entropy(exact, classical))
                    neu_ce.append(binary_cross_entropy(exact, neural))
                    cls_kl.append(binary_kl(exact, classical))
                    neu_kl.append(binary_kl(exact, neural))
                    if curve is not None and len(curve) > 0:
                        curves.append(curve[0])

                mean_curve = []
                if curves:
                    min_len = min(len(c) for c in curves)
                    arr = np.asarray([c[:min_len] for c in curves], dtype=np.float64)
                    mean_curve = arr.mean(axis=0).tolist()

                summary["results"].append(
                    {
                        "graph_type": graph_type,
                        "beta": float(beta),
                        "classical": {
                            "avg_l1": float(np.mean(cls_l1s)),
                            "avg_ce": float(np.mean(cls_ce)),
                            "avg_kl": float(np.mean(cls_kl)),
                        },
                        "neural": {
                            "avg_l1": float(np.mean(neu_l1s)),
                            "avg_ce": float(np.mean(neu_ce)),
                            "avg_kl": float(np.mean(neu_kl)),
                        },
                        "neural_convergence_curve": mean_curve,
                    }
                )

    return summary
