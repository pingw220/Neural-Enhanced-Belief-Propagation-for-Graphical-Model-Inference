from typing import Dict, List, Tuple

from nebpmrf.graphs.mrf import GraphModel


def normalize_2(vec: List[float]) -> List[float]:
    s_val = vec[0] + vec[1]
    if s_val <= 0:
        raise ValueError("Cannot normalize a non-positive-sum vector.")
    return [vec[0] / s_val, vec[1] / s_val]


def bp_sum_product(
    model: GraphModel,
    max_iters: int = 50,
    tol: float = 1e-9,
    damping: float = 0.0,
) -> Tuple[List[List[float]], Dict[Tuple[int, int], List[float]], List[float]]:
    """
    General sum-product BP for binary pairwise MRF.
    Returns beliefs, directed messages, and convergence deltas.
    """
    n = model.n
    nbrs = model.neighbors

    messages: Dict[Tuple[int, int], List[float]] = {}
    for i in range(n):
        for j in nbrs[i]:
            messages[(i, j)] = [0.5, 0.5]

    deltas: List[float] = []

    for _ in range(max_iters):
        max_delta = 0.0
        new_messages = dict(messages)

        for i in range(n):
            for j in nbrs[i]:
                out = [0.0, 0.0]
                for xj in (0, 1):
                    s_val = 0.0
                    for xi in (0, 1):
                        val = model.phi(i, xi) * model.psi(i, j, xi, xj)
                        for k in nbrs[i]:
                            if k == j:
                                continue
                            val *= messages[(k, i)][xi]
                        s_val += val
                    out[xj] = s_val

                out = normalize_2(out)

                if damping > 0.0:
                    old = messages[(i, j)]
                    out = [
                        (1.0 - damping) * out[0] + damping * old[0],
                        (1.0 - damping) * out[1] + damping * old[1],
                    ]
                    out = normalize_2(out)

                old = messages[(i, j)]
                delta = max(abs(out[0] - old[0]), abs(out[1] - old[1]))
                if delta > max_delta:
                    max_delta = delta

                new_messages[(i, j)] = out

        messages = new_messages
        deltas.append(max_delta)
        if max_delta < tol:
            break

    beliefs: List[List[float]] = []
    for i in range(n):
        b = [0.0, 0.0]
        for xi in (0, 1):
            val = model.phi(i, xi)
            for k in nbrs[i]:
                val *= messages[(k, i)][xi]
            b[xi] = val
        beliefs.append(normalize_2(b))

    return beliefs, messages, deltas
