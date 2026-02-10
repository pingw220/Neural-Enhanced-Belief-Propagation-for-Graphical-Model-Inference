import random
from typing import List, Tuple

def make_chain_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]

def make_star_edges(n: int, center: int = 0) -> List[Tuple[int, int]]:
    return [(center, i) for i in range(n) if i != center]

def make_random_tree_edges(n: int, seed: int = 0) -> List[Tuple[int, int]]:
    """
    Simple random tree: for each node i>=1 connect to a random previous node.
    Produces a valid tree with n-1 edges.
    """
    rng = random.Random(seed)
    edges: List[Tuple[int, int]] = []
    for i in range(1, n):
        j = rng.randrange(0, i)
        edges.append((j, i))
    return edges

def add_one_cycle_edge(edges: List[Tuple[int, int]], n: int, seed: int = 0) -> List[Tuple[int, int]]:
    """
    Add a single extra edge (u,v) not already present, to create one cycle (loopy graph).
    """
    rng = random.Random(seed)
    existing = set((min(a,b), max(a,b)) for a,b in edges)
    # try random pairs until find a new one
    for _ in range(10000):
        u = rng.randrange(0, n)
        v = rng.randrange(0, n)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing:
            continue
        return edges + [(a, b)]
    raise RuntimeError("Failed to find a new edge to add (graph too dense?).")
