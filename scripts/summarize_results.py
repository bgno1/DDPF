from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ddpf.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rows = []
    for (method, shot), group in df.groupby(["method", "shot"]):
        rows.append(
            {
                "method": method,
                "shot": int(shot),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=0),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=0),
            }
        )
    ensure_dir(Path(args.output).parent)
    pd.DataFrame(rows).sort_values(["shot", "method"]).to_csv(args.output, index=False)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
