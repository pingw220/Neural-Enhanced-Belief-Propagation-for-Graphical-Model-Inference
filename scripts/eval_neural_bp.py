#!/usr/bin/env python3
import argparse
import os

import torch

from nebpmrf.experiments.evaluate import evaluate_neural_bp
from nebpmrf.neural.neural_bp import NeuralBPResidual
from nebpmrf.utils.io import ensure_dir, load_yaml, make_timestamp, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Neural BP vs classical BP.")
    parser.add_argument("--config", type=str, default="configs/neuralbp_eval.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg.get("device", "cpu"))

    ckpt_path = args.checkpoint if args.checkpoint else cfg.get("checkpoint", "")
    if not ckpt_path:
        raise ValueError("Checkpoint is required via --checkpoint or config key 'checkpoint'.")

    ckpt = torch.load(ckpt_path, map_location=device)
    hidden = int(cfg.get("hidden", ckpt.get("train_config", {}).get("hidden", 32)))

    model = NeuralBPResidual(hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state"])

    metrics = evaluate_neural_bp(model=model, cfg=cfg, device=device)
    metrics["checkpoint"] = ckpt_path

    out_root = cfg.get("output_root", "results/stage2_eval")
    out_dir = os.path.join(out_root, make_timestamp())
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "metrics.json")
    save_json(metrics, out_path)

    print(f"Saved evaluation metrics to: {out_path}")
    for item in metrics["results"]:
        gt = item["graph_type"]
        beta = item["beta"]
        c_l1 = item["classical"]["avg_l1"]
        n_l1 = item["neural"]["avg_l1"]
        print(f"[eval] graph={gt} beta={beta:.3f} classical_l1={c_l1:.6f} neural_l1={n_l1:.6f}")


if __name__ == "__main__":
    main()
