import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def resolve_metrics_file(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"metrics file not found: {path}")
        return path

    files = sorted(Path("results/stage2_eval").glob("*/metrics.json"))
    if not files:
        raise FileNotFoundError("No metrics.json found under results/stage2_eval/")
    return files[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from Stage 2 metrics.")
    parser.add_argument("--metrics", type=str, default=None, help="Optional path to metrics.json")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory to save figures")
    args = parser.parse_args()

    metrics_file = resolve_metrics_file(args.metrics)
    print("Using metrics file:", metrics_file)

    with open(metrics_file, encoding="utf-8") as f:
        data = json.load(f)

    rows = data["results"]
    if not rows:
        raise ValueError("No rows found in metrics file.")

    print("Number of rows:", len(rows))
    print("First row keys:", rows[0].keys())
    print("First row:", rows[0])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    betas = sorted({float(r["beta"]) for r in rows})
    graphs = ["loopy1", "loopy2"]

    def pick_metric(row, method, metric_name_candidates):
        method_block = row[method]
        for name in metric_name_candidates:
            if name in method_block:
                return method_block[name]
        raise KeyError(
            f"None of {metric_name_candidates} found in row[{method}] keys {list(method_block.keys())}"
        )

    def find_row(graph, beta):
        for r in rows:
            if r["graph_type"] == graph and abs(float(r["beta"]) - beta) < 1e-12:
                return r
        raise ValueError(f"No row found for graph={graph}, beta={beta}")

    # -----------------------------
    # Figure 1: avg_l1 vs beta
    # -----------------------------
    for graph in graphs:
        classical_vals = []
        neural_vals = []

        for beta in betas:
            row = find_row(graph, beta)
            classical_vals.append(pick_metric(row, "classical", ["avg_l1", "l1"]))
            neural_vals.append(pick_metric(row, "neural", ["avg_l1", "l1"]))

        plt.figure()
        plt.plot(betas, classical_vals, marker="o", label="Classical BP")
        plt.plot(betas, neural_vals, marker="o", label="Neural BP")
        plt.xlabel("Interaction strength β")
        plt.ylabel("Average marginal L1 error")
        plt.title(graph)
        plt.legend()
        plt.savefig(out_dir / f"{graph}_l1.png", dpi=300, bbox_inches="tight")
        plt.close()

    # -----------------------------
    # Optional Figure 2: cross-entropy vs beta
    # -----------------------------
    for graph in graphs:
        classical_vals = []
        neural_vals = []

        for beta in betas:
            row = find_row(graph, beta)
            classical_vals.append(
                pick_metric(row, "classical", ["avg_ce", "cross_entropy", "ce"])
            )
            neural_vals.append(
                pick_metric(row, "neural", ["avg_ce", "cross_entropy", "ce"])
            )

        plt.figure()
        plt.plot(betas, classical_vals, marker="o", label="Classical BP")
        plt.plot(betas, neural_vals, marker="o", label="Neural BP")
        plt.xlabel("Interaction strength β")
        plt.ylabel("Cross entropy")
        plt.title(graph)
        plt.legend()
        plt.savefig(out_dir / f"{graph}_ce.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("Saved:")
    for name in ["loopy1_l1.png", "loopy2_l1.png", "loopy1_ce.png", "loopy2_ce.png"]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
    
