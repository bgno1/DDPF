from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def l2(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def metrics(y_true, y_pred) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def prototypes(features: torch.Tensor, labels: torch.Tensor, indices: list[int], num_classes: int) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long)
    support_x = features[idx]
    support_y = labels[idx]
    protos = []
    for class_id in range(num_classes):
        xs = support_x[support_y == class_id]
        if xs.numel() == 0:
            raise ValueError(f"No support sample for class {class_id}.")
        protos.append(F.normalize(xs.mean(dim=0), dim=0))
    return torch.stack(protos)


def proto_predict(query: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    return (query @ proto.T).argmax(dim=1)


def one_nn_predict(support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
    return support_y[(query_x @ support_x.T).argmax(dim=1)]


def dual_scores(q_i: torch.Tensor, p_i: torch.Tensor, q_m: torch.Tensor, p_m: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1 - alpha) * (q_i @ p_i.T) + alpha * (q_m @ p_m.T)


def ddpf_predict(q_i: torch.Tensor, p_i: torch.Tensor, q_m: torch.Tensor, p_m: torch.Tensor, alpha: float) -> torch.Tensor:
    return dual_scores(q_i, p_i, q_m, p_m, alpha).argmax(dim=1)


def inter_class_margin(p_i: torch.Tensor, p_m: torch.Tensor, alpha: float) -> float:
    score = (1 - alpha) * (p_i @ p_i.T) + alpha * (p_m @ p_m.T)
    score = score.masked_fill(torch.eye(score.size(0), dtype=torch.bool), float("-inf"))
    return float((1 - score.max(dim=1).values).mean())


def support_loo_score(
    f_i: torch.Tensor,
    f_m: torch.Tensor,
    labels: torch.Tensor,
    support_indices: list[int],
    num_classes: int,
    alpha: float,
) -> dict[str, float]:
    y_true, y_pred, margins = [], [], []
    support_labels = labels[torch.as_tensor(support_indices)]
    for held_pos, held_idx in enumerate(support_indices):
        train_idx = [idx for pos, idx in enumerate(support_indices) if pos != held_pos]
        p_i = prototypes(f_i, labels, train_idx, num_classes)
        p_m = prototypes(f_m, labels, train_idx, num_classes)
        score = dual_scores(f_i[held_idx].unsqueeze(0), p_i, f_m[held_idx].unsqueeze(0), p_m, alpha).squeeze(0)
        top = torch.sort(score, descending=True).values
        margins.append(float(top[0] - top[1]))
        y_true.append(int(support_labels[held_pos]))
        y_pred.append(int(score.argmax()))
    out = metrics(y_true, y_pred)
    out["margin"] = float(sum(margins) / len(margins))
    return out


def select_alpha(
    f_i: torch.Tensor,
    f_m: torch.Tensor,
    labels: torch.Tensor,
    support_indices: list[int],
    num_classes: int,
    shot: int,
    alphas: list[float],
) -> tuple[float, str, float, list[dict[str, Any]]]:
    rows = []
    if shot == 1:
        p_i = prototypes(f_i, labels, support_indices, num_classes)
        p_m = prototypes(f_m, labels, support_indices, num_classes)
        for alpha in alphas:
            rows.append({"alpha": alpha, "margin": inter_class_margin(p_i, p_m, alpha)})
        rows.sort(key=lambda r: (-r["margin"], r["alpha"]))
        return rows[0]["alpha"], "support_inter_class_margin", rows[0]["margin"], rows
    for alpha in alphas:
        score = support_loo_score(f_i, f_m, labels, support_indices, num_classes, alpha)
        rows.append({"alpha": alpha, "macro_f1": score["macro_f1"], "margin": score["margin"]})
    rows.sort(key=lambda r: (-r["macro_f1"], -r["margin"], r["alpha"]))
    return rows[0]["alpha"], "support_loo_macro_f1", rows[0]["macro_f1"], rows
