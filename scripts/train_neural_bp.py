#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nebpmrf.data.dataset import SyntheticConfig, SyntheticMRFDataset, collate_synthetic_batch
from nebpmrf.neural.neural_bp import NeuralBPResidual
from nebpmrf.utils.io import ensure_dir, load_yaml, make_timestamp, save_json, save_yaml
from nebpmrf.utils.seed import set_seed


def train(cfg: Dict) -> str:
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg.get("device", "cpu"))

    synth_cfg = SyntheticConfig(
        n=int(cfg["n"]),
        graph_types=list(cfg["graph_types"]),
        beta_min=float(cfg["beta_min"]),
        beta_max=float(cfg["beta_max"]),
        h=float(cfg["h"]),
        bias_node=int(cfg["bias_node"]),
        favor_value=int(cfg["favor_value"]),
    )

    dataset = SyntheticMRFDataset(
        num_samples=int(cfg["num_train_samples"]),
        cfg=synth_cfg,
        seed=int(cfg["seed"]),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_synthetic_batch,
    )

    model = NeuralBPResidual(hidden=int(cfg["hidden"])).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    out_root = cfg.get("output_root", "results/stage2_train")
    run_dir = os.path.join(out_root, make_timestamp())
    ensure_dir(run_dir)
    save_yaml(cfg, os.path.join(run_dir, "config.yaml"))
    latest_dir = os.path.join(out_root, "latest")
    ensure_dir(latest_dir)

    epochs = int(cfg["epochs"])
    t_iters = int(cfg["T"])
    damping = float(cfg["damping"])
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 1))

    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_l1 = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            u_node = batch["u_node"].to(device)
            log_psi_und = batch["log_psi_und"].to(device)
            edge_index_und = batch["edge_index_und"].to(device)
            p_exact = batch["p_exact"].to(device)

            p_hat, _ = model(
                u_node=u_node,
                log_psi_und=log_psi_und,
                edge_index_und=edge_index_und,
                num_iters=t_iters,
                damping=damping,
                return_curve=False,
            )

            loss = F.binary_cross_entropy(p_hat, p_exact)
            l1 = torch.mean(torch.abs(p_hat - p_exact))

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += float(loss.item())
            running_l1 += float(l1.item())
            num_batches += 1

            if step % log_every == 0 or step == 1 or step == len(loader):
                pbar.set_postfix({"loss": f"{loss.item():.6f}", "l1": f"{l1.item():.6f}"})

        epoch_loss = running_loss / max(1, num_batches)
        epoch_l1 = running_l1 / max(1, num_batches)
        history.append({"epoch": epoch, "loss": epoch_loss, "l1": epoch_l1})
        print(f"[train] epoch={epoch} loss={epoch_loss:.6f} l1={epoch_l1:.6f}")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(run_dir, f"model_epoch{epoch}.pt")
            ckpt_obj = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "train_config": cfg,
            }
            torch.save(
                ckpt_obj,
                ckpt_path,
            )
            torch.save(ckpt_obj, os.path.join(latest_dir, "model_latest.pt"))
            print(f"[train] saved checkpoint: {ckpt_path}")

    save_json({"history": history}, os.path.join(run_dir, "history.json"))
    print(f"Training outputs saved to: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neural-Enhanced BP.")
    parser.add_argument("--config", type=str, default="configs/neuralbp_train.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
