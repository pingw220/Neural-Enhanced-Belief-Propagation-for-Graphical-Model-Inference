from typing import Dict

import torch


def build_directed_edges(edge_index_und: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Args:
        edge_index_und: [2, E_und] with undirected edges.
    Returns:
        src, dst, und_id, rev for directed edges (length E_dir=2*E_und).
    """
    if edge_index_und.dim() != 2 or edge_index_und.shape[0] != 2:
        raise ValueError(f"edge_index_und must have shape [2, E], got {tuple(edge_index_und.shape)}")

    e_und = edge_index_und.shape[1]
    src = torch.empty(2 * e_und, dtype=torch.long, device=edge_index_und.device)
    dst = torch.empty(2 * e_und, dtype=torch.long, device=edge_index_und.device)
    und_id = torch.empty(2 * e_und, dtype=torch.long, device=edge_index_und.device)
    rev = torch.empty(2 * e_und, dtype=torch.long, device=edge_index_und.device)

    for e in range(e_und):
        i = int(edge_index_und[0, e].item())
        j = int(edge_index_und[1, e].item())
        e0 = 2 * e
        e1 = 2 * e + 1
        src[e0], dst[e0] = i, j
        src[e1], dst[e1] = j, i
        und_id[e0], und_id[e1] = e, e
        rev[e0], rev[e1] = e1, e0

    return {"src": src, "dst": dst, "und_id": und_id, "rev": rev}


def build_directed_edges_batched(edge_index_und: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Args:
        edge_index_und: [B, 2, E_und]
    Returns:
        src: [B, E_dir]
        dst: [B, E_dir]
        und_id: [E_dir] (same for all batch entries)
        rev: [E_dir] (same for all batch entries)
    """
    if edge_index_und.dim() != 3 or edge_index_und.shape[1] != 2:
        raise ValueError(f"edge_index_und must have shape [B,2,E], got {tuple(edge_index_und.shape)}")

    bsz = edge_index_und.shape[0]
    e_und = edge_index_und.shape[2]
    e_dir = 2 * e_und

    src = torch.empty((bsz, e_dir), dtype=torch.long, device=edge_index_und.device)
    dst = torch.empty((bsz, e_dir), dtype=torch.long, device=edge_index_und.device)

    for e in range(e_und):
        e0 = 2 * e
        e1 = 2 * e + 1
        src[:, e0] = edge_index_und[:, 0, e]
        dst[:, e0] = edge_index_und[:, 1, e]
        src[:, e1] = edge_index_und[:, 1, e]
        dst[:, e1] = edge_index_und[:, 0, e]

    und_id = torch.arange(e_und, device=edge_index_und.device, dtype=torch.long).repeat_interleave(2)
    rev = torch.empty(e_dir, dtype=torch.long, device=edge_index_und.device)
    for e in range(e_und):
        e0 = 2 * e
        e1 = 2 * e + 1
        rev[e0] = e1
        rev[e1] = e0

    return {"src": src, "dst": dst, "und_id": und_id, "rev": rev}
