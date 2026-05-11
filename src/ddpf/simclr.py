from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    logits = (z @ z.T) / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = torch.cat(
        [
            torch.arange(batch, 2 * batch, device=z.device),
            torch.arange(0, batch, device=z.device),
        ]
    )
    return F.cross_entropy(logits, targets)
