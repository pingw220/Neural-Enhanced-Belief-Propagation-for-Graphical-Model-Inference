import math
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from nebpmrf.graphs.generators import build_edges
from nebpmrf.graphs.mrf import make_model
from nebpmrf.graphs.potentials import make_phi_bias, make_psi_prefer_equal
from nebpmrf.inference.exact_enum import exact_marginals_bruteforce


@dataclass
class SyntheticConfig:
    n: int
    graph_types: List[str]
    beta_min: float
    beta_max: float
    h: float
    bias_node: int
    favor_value: int


class SyntheticMRFDataset(Dataset):
    """
    On-the-fly synthetic MRF dataset with fixed n and fixed E per batch.
    This implementation supports variable topology but requires graph_types with
    the same edge count for batch collation.
    """

    def __init__(
        self,
        num_samples: int,
        cfg: SyntheticConfig,
        seed: int = 0,
    ) -> None:
        self.num_samples = int(num_samples)
        self.cfg = cfg
        self.seed = int(seed)
        self._check_edge_count_compatibility()

    def _check_edge_count_compatibility(self) -> None:
        n = self.cfg.n
        counts = {}
        for g in self.cfg.graph_types:
            if g == "chain":
                counts[g] = n - 1
            elif g == "tree":
                counts[g] = n - 1
            elif g == "loopy1":
                counts[g] = n
            elif g == "loopy2":
                counts[g] = n + 1
            else:
                raise ValueError(f"Unknown graph type: {g}")
        if len(set(counts.values())) != 1:
            raise ValueError(
                "For batched training, graph_types must share the same edge count. "
                f"Got counts: {counts}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        graph_type = rng.choice(self.cfg.graph_types)
        beta = rng.uniform(self.cfg.beta_min, self.cfg.beta_max)
        edge_seed = self.seed * 1000003 + idx

        edges = build_edges(graph_type=graph_type, n=self.cfg.n, seed=edge_seed)
        phi = make_phi_bias(h=self.cfg.h, bias_node=self.cfg.bias_node, favor_value=self.cfg.favor_value)
        psi = make_psi_prefer_equal(beta=beta)
        model = make_model(n=self.cfg.n, edges=edges, phi=phi, psi=psi)

        _, marg = exact_marginals_bruteforce(model)
        p_exact = np.asarray([m[1] for m in marg], dtype=np.float32)

        # Unary logit u_i = log phi_i(1) - log phi_i(0)
        u_node = np.zeros(self.cfg.n, dtype=np.float32)
        if self.cfg.favor_value == 1:
            u_node[self.cfg.bias_node] = float(self.cfg.h)
        else:
            u_node[self.cfg.bias_node] = float(-self.cfg.h)

        # Pairwise log potential table per undirected edge.
        # Stage 1 potential: log psi = beta if xi==xj else 0.
        log_psi_one = np.asarray([[beta, 0.0], [0.0, beta]], dtype=np.float32)
        log_psi_und = np.stack([log_psi_one for _ in edges], axis=0)

        edge_index_und = np.asarray(edges, dtype=np.int64).T  # [2, E_und]

        return {
            "u_node": torch.from_numpy(u_node),
            "log_psi_und": torch.from_numpy(log_psi_und),
            "edge_index_und": torch.from_numpy(edge_index_und),
            "p_exact": torch.from_numpy(p_exact),
            "beta": torch.tensor(beta, dtype=torch.float32),
            "graph_type": graph_type,
        }


def collate_synthetic_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    out["u_node"] = torch.stack([x["u_node"] for x in items], dim=0)
    out["log_psi_und"] = torch.stack([x["log_psi_und"] for x in items], dim=0)
    out["edge_index_und"] = torch.stack([x["edge_index_und"] for x in items], dim=0)
    out["p_exact"] = torch.stack([x["p_exact"] for x in items], dim=0)
    out["beta"] = torch.stack([x["beta"] for x in items], dim=0)
    out["graph_type"] = [x["graph_type"] for x in items]
    return out
