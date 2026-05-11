from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


class ResNet18Encoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


def _clean_state_dict(state: dict) -> dict:
    return {
        key.removeprefix("module.").removeprefix("backbone."): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }


def build_encoder(kind: str, checkpoint: str | Path | None = None) -> ResNet18Encoder:
    if kind == "imagenet":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif kind == "malsim":
        if checkpoint is None:
            raise ValueError("MalSim encoder requires a checkpoint.")
        model = models.resnet18(weights=None)
    elif kind == "random":
        model = models.resnet18(weights=None)
    else:
        raise ValueError(f"Unknown encoder kind: {kind}")
    model.fc = nn.Identity()
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("backbone") or ckpt.get("encoder_state_dict") or ckpt.get("encoder") or ckpt
        model.load_state_dict(_clean_state_dict(state), strict=False)
    encoder = ResNet18Encoder(model)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


class SimCLRModel(nn.Module):
    def __init__(self, init: str = "imagenet", projection_dim: int = 128) -> None:
        super().__init__()
        if init == "imagenet":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif init == "random":
            backbone = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unknown init: {init}")
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
