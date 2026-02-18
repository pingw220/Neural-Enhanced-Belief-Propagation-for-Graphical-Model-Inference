from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

State = int
PhiFn = Callable[[int, State], float]
PsiFn = Callable[[int, int, State, State], float]


@dataclass(frozen=True)
class GraphModel:
    """
    Pairwise binary MRF model:
      mu(x) ∝ prod_i phi(i, x_i) * prod_(i,j) psi(i,j,x_i,x_j)
    """

    n: int
    edges: List[Tuple[int, int]]
    phi: PhiFn
    psi: PsiFn
    neighbors: Dict[int, List[int]]


def build_neighbors(n: int, edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in edges:
        if a == b:
            raise ValueError("Self-loop not supported.")
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"Edge {(a, b)} out of range for n={n}.")
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs


def make_model(n: int, edges: List[Tuple[int, int]], phi: PhiFn, psi: PsiFn) -> GraphModel:
    nbrs = build_neighbors(n, edges)
    return GraphModel(n=n, edges=edges, phi=phi, psi=psi, neighbors=nbrs)
