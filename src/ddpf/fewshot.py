from __future__ import annotations

from collections import defaultdict

import numpy as np


def sample_support_indices(dataset, shot: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset.targets):
        by_class[int(label)].append(index)
    indices: list[int] = []
    for class_id in sorted(by_class):
        pool = by_class[class_id]
        if len(pool) < shot:
            raise ValueError(f"Class {class_id} has {len(pool)} samples, need {shot}.")
        indices.extend(int(i) for i in rng.choice(pool, size=shot, replace=False))
    return indices
