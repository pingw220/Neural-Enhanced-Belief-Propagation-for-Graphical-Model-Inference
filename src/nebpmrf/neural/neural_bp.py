from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from nebpmrf.neural.features import build_directed_edges_batched


class NeuralBPResidual(nn.Module):
    def __init__(self, hidden: int = 32, logit_clip: float = 20.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.logit_clip = float(logit_clip)

    def _incoming_sums(
        self,
        l_msg: torch.Tensor,
        dst: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """
        Args:
            l_msg: [B, E_dir]
            dst: [B, E_dir]
        Returns:
            incoming_sum: [B, n]
        """
        bsz = l_msg.shape[0]
        incoming_sum = torch.zeros((bsz, n), dtype=l_msg.dtype, device=l_msg.device)
        for b in range(bsz):
            for node in range(n):
                idx = torch.nonzero(dst[b] == node, as_tuple=False).squeeze(-1)
                if idx.numel() > 0:
                    incoming_sum[b, node] = l_msg[b, idx].sum()
        return incoming_sum

    def _classical_logit_update(
        self,
        l_msg: torch.Tensor,
        u_node: torch.Tensor,
        log_psi_und: torch.Tensor,
        edge_index_und: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            l_msg: [B, E_dir]
            u_node: [B, n]
            log_psi_und: [B, E_und, 2, 2]
            edge_index_und: [B, 2, E_und]
        Returns:
            l_classic: [B, E_dir]
            features: [B, E_dir, 6]
            src: [B, E_dir]
            dst: [B, E_dir]
        """
        mapping = build_directed_edges_batched(edge_index_und)
        src = mapping["src"]
        dst = mapping["dst"]
        und_id = mapping["und_id"]
        rev = mapping["rev"]

        n = u_node.shape[1]
        incoming_sum = self._incoming_sums(l_msg=l_msg, dst=dst, n=n)

        # sum_{k in N(i)\{j}} l_{k->i} = incoming_sum[i] - l_{j->i}
        sum_in_excl = torch.gather(incoming_sum, 1, src) - l_msg[:, rev]

        u_i = torch.gather(u_node, 1, src)
        pair = log_psi_und[:, und_id, :, :]  # [B,E_dir,2,2]
        logpsi00 = pair[:, :, 0, 0]
        logpsi01 = pair[:, :, 0, 1]
        logpsi10 = pair[:, :, 1, 0]
        logpsi11 = pair[:, :, 1, 1]

        a_val = u_i + sum_in_excl
        s0 = torch.logsumexp(torch.stack([logpsi00, logpsi10 + a_val], dim=-1), dim=-1)
        s1 = torch.logsumexp(torch.stack([logpsi01, logpsi11 + a_val], dim=-1), dim=-1)
        l_classic = s1 - s0

        features = torch.stack([sum_in_excl, u_i, logpsi00, logpsi01, logpsi10, logpsi11], dim=-1)
        return l_classic, features, src, dst

    def _belief_p1_from_messages(
        self,
        l_msg: torch.Tensor,
        u_node: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        n = u_node.shape[1]
        incoming_sum = self._incoming_sums(l_msg=l_msg, dst=dst, n=n)
        belief_logit = u_node + incoming_sum
        return torch.sigmoid(belief_logit)

    def forward(
        self,
        u_node: torch.Tensor,
        log_psi_und: torch.Tensor,
        edge_index_und: torch.Tensor,
        num_iters: int,
        damping: float,
        return_curve: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[List[float]]]]:
        """
        Args:
            u_node: [B, n]
            log_psi_und: [B, E_und, 2, 2]
            edge_index_und: [B, 2, E_und]
        Returns:
            p_hat: [B, n]
        """
        if u_node.dim() != 2:
            raise ValueError("u_node must have shape [B,n]")
        if log_psi_und.dim() != 4:
            raise ValueError("log_psi_und must have shape [B,E,2,2]")
        if edge_index_und.dim() != 3:
            raise ValueError("edge_index_und must have shape [B,2,E]")

        bsz = u_node.shape[0]
        e_dir = 2 * edge_index_und.shape[2]

        l_msg = torch.zeros((bsz, e_dir), dtype=u_node.dtype, device=u_node.device)
        curves: List[List[float]] = [[] for _ in range(bsz)]

        final_dst = None
        for _ in range(num_iters):
            l_classic, features, _, dst = self._classical_logit_update(
                l_msg=l_msg,
                u_node=u_node,
                log_psi_und=log_psi_und,
                edge_index_und=edge_index_und,
            )
            final_dst = dst

            delta = self.mlp(features.view(-1, 6)).view(bsz, e_dir)
            target = l_classic + delta
            l_next = (1.0 - damping) * l_msg + damping * target
            l_next = torch.clamp(l_next, -self.logit_clip, self.logit_clip)

            if return_curve:
                diff = torch.max(torch.abs(l_next - l_msg), dim=1).values
                for b in range(bsz):
                    curves[b].append(float(diff[b].item()))

            l_msg = l_next

        if final_dst is None:
            mapping = build_directed_edges_batched(edge_index_und)
            final_dst = mapping["dst"]

        p_hat = self._belief_p1_from_messages(l_msg=l_msg, u_node=u_node, dst=final_dst)
        return p_hat, (curves if return_curve else None)
