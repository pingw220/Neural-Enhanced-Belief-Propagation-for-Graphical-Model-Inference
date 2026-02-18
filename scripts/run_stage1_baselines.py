#!/usr/bin/env python3
import argparse

from nebpmrf.experiments.evaluate import run_stage1_baselines
from nebpmrf.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 exact + classical BP baselines.")
    parser.add_argument("--config", type=str, default="configs/stage1.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_dir = run_stage1_baselines(cfg)
    print(f"Saved Stage 1 baseline results to: {run_dir}")


if __name__ == "__main__":
    main()
