from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddpf.data import aligned_malimg, eval_transform
from ddpf.fewshot import sample_support_indices
from ddpf.utils import load_yaml, save_json, set_seed


def parse_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--shots", default="1,5,10,20")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="results/support_plan_r20.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["fewshot"]["seed"]))
    transform = eval_transform(int(cfg["image"]["input_size"]))
    train, query, classes = aligned_malimg(
        cfg["paths"]["malimg_train"],
        cfg["paths"]["malimg_validation"],
        transform,
        cfg["class_alignment"]["aliases"],
    )
    tasks = []
    for shot in parse_list(args.shots):
        for repeat in range(args.repeats):
            seed = int(cfg["fewshot"]["seed"]) + shot * 1000 + repeat
            support_indices = sample_support_indices(train, shot, seed)
            tasks.append(
                {
                    "shot": shot,
                    "repeat": repeat,
                    "seed": seed,
                    "support_indices": support_indices,
                    "support_paths": [str(Path(train.samples[i][0]).resolve()) for i in support_indices],
                    "support_labels": [int(train.targets[i]) for i in support_indices],
                    "query_indices": list(range(len(query))),
                    "query_paths": [str(Path(p).resolve()) for p, _ in query.samples],
                    "query_labels": [int(y) for y in query.targets],
                }
            )
    save_json({"class_names": classes, "tasks": tasks}, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
