import random
from typing import List, Tuple


def make_chain_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def make_star_edges(n: int, center: int = 0) -> List[Tuple[int, int]]:
    return [(center, i) for i in range(n) if i != center]


def make_random_tree_edges(n: int, seed: int = 0) -> List[Tuple[int, int]]:
    """Simple random tree: for each node i>=1 connect to a random previous node."""
    rng = random.Random(seed)
    edges: List[Tuple[int, int]] = []
    for i in range(1, n):
        j = rng.randrange(0, i)
        a, b = (j, i) if j < i else (i, j)
        edges.append((a, b))
    return edges


def _add_new_edge(edges: List[Tuple[int, int]], n: int, seed: int) -> Tuple[int, int]:
    rng = random.Random(seed)
    existing = {(min(a, b), max(a, b)) for a, b in edges}
    for _ in range(10000):
        u = rng.randrange(0, n)
        v = rng.randrange(0, n)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing:
            continue
        return (a, b)
    raise RuntimeError("Failed to find a new edge to add (graph too dense?).")


def add_one_cycle_edge(edges: List[Tuple[int, int]], n: int, seed: int = 0) -> List[Tuple[int, int]]:
    """Add one extra undirected edge not already present."""
    return edges + [_add_new_edge(edges, n=n, seed=seed)]


def add_second_cycle_edge(edges: List[Tuple[int, int]], n: int, seed: int = 0) -> List[Tuple[int, int]]:
    """Add another extra undirected edge not already present."""
    first = add_one_cycle_edge(edges, n=n, seed=seed)
    return add_one_cycle_edge(first, n=n, seed=seed + 101)


def build_edges(graph_type: str, n: int, seed: int) -> List[Tuple[int, int]]:
    if graph_type == "chain":
        return make_chain_edges(n)
    if graph_type == "tree":
        return make_random_tree_edges(n, seed=seed)
    if graph_type in ("loopy", "loopy1"):
        base = make_random_tree_edges(n, seed=seed)
        return add_one_cycle_edge(base, n=n, seed=seed + 999)
    if graph_type == "loopy2":
        base = make_random_tree_edges(n, seed=seed)
        return add_second_cycle_edge(base, n=n, seed=seed + 999)
    raise ValueError("graph_type must be one of: chain, tree, loopy1, loopy2")
