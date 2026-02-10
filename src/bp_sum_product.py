from typing import Dict, List, Tuple

from model import GraphModel

def normalize_2(vec: List[float]) -> List[float]:
    s = vec[0] + vec[1]
    if s <= 0:
        raise ValueError("Cannot normalize a non-positive-sum vector.")
    return [vec[0] / s, vec[1] / s]

def bp_sum_product(
    model: GraphModel,
    max_iters: int = 50,
    tol: float = 1e-9,
    damping: float = 0.0,
) -> Tuple[List[List[float]], Dict[Tuple[int, int], List[float]], List[float]]:
    """
    General sum-product BP for binary pairwise MRF.
    Works for tree (exact) and loopy (approx).
    Returns:
      beliefs: list of [P0,P1] per node
      messages: dict (i,j)->[m(x_j=0), m(x_j=1)]
      deltas: per-iteration max change (for convergence plot)
    """
    n = model.n
    nbrs = model.neighbors

    # initialize all directed messages to uniform
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
                # compute m_{i->j}(x_j)
                out = [0.0, 0.0]
                for xj in (0, 1):
                    s = 0.0
                    for xi in (0, 1):
                        val = model.phi(i, xi) * model.psi(i, j, xi, xj)
                        # product of incoming messages to i excluding from j
                        for k in nbrs[i]:
                            if k == j:
                                continue
                            val *= messages[(k, i)][xi]
                        s += val
                    out[xj] = s

                out = normalize_2(out)

                # damping: m_new = (1-d)*out + d*old
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

    # compute beliefs
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
