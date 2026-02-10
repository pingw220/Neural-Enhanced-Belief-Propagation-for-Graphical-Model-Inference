import glob
import json
import os
from typing import Dict, Any

from utils_io import save_json

def load_latest(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return files[-1]

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # change these if you change n/beta/h defaults
    exact_path = load_latest("../results/exact_chain_n6_beta0.8_h1.0_bias0_favor0_*.json")
    bp_path = load_latest("../results/bp_chain_n6_beta0.8_h1.0_bias0_favor0_*.json")

    exact = read_json(exact_path)
    bp = read_json(bp_path)

    n = int(exact["n"])
    errs = {}
    max_err = 0.0
    sum_err = 0.0

    for i in range(n):
        e0, e1 = exact["marginals"][str(i)]
        b0, b1 = bp["marginals"][str(i)]
        # L1 distance between 2-d distributions
        err = abs(e0 - b0) + abs(e1 - b1)
        errs[str(i)] = {
            "exact": [e0, e1],
            "bp": [b0, b1],
            "l1_error": err,
        }
        max_err = max(max_err, err)
        sum_err += err

    payload = {
        "graph": "chain",
        "n": n,
        "beta": exact["beta"],
        "h": exact["h"],
        "bias_node": exact["bias_node"],
        "favor_value": exact["favor_value"],
        "exact_file": os.path.basename(exact_path),
        "bp_file": os.path.basename(bp_path),
        "per_node": errs,
        "mean_l1_error": sum_err / n,
        "max_l1_error": max_err,
    }

    out_path = "../results/compare_chain_n6_beta0.8_h1.0.json"
    save_json(payload, out_path)

    print("Wrote:", out_path)
    print("mean L1 error:", payload["mean_l1_error"])
    print("max  L1 error:", payload["max_l1_error"])

if __name__ == "__main__":
    main()
