from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pil_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def eval_transform(input_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def simclr_transform(input_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.03, 0.03))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TwoViewTransform:
    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, image: Image.Image):
        return self.transform(image), self.transform(image)


class RemappedImageFolder(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.ImageFolder, indices: list[int], label_map: dict[int, int], class_names: list[str]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map
        self.classes = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.targets = [label_map[int(dataset.targets[i])] for i in indices]
        self.samples = [(dataset.samples[i][0], label_map[int(dataset.samples[i][1])]) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[self.indices[index]]
        return image, self.label_map[int(label)]


def imagefolder(root: str | Path, transform: Callable | None = None) -> datasets.ImageFolder:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {root}")
    return datasets.ImageFolder(root=root, transform=transform, loader=pil_rgb)


def aligned_malimg(train_root: str | Path, val_root: str | Path, transform: Callable, aliases: dict[str, str]):
    train = imagefolder(train_root, transform)
    val = imagefolder(val_root, transform)
    train_classes = list(train.classes)
    val_alias = [aliases.get(name, name) for name in val.classes]
    if set(train_classes) != set(val_alias):
        raise ValueError(f"Class mismatch after alias: train={sorted(train_classes)}, val={sorted(val_alias)}")
    final_to_idx = {name: i for i, name in enumerate(train_classes)}
    train_map = {train.class_to_idx[name]: final_to_idx[name] for name in train_classes}
    val_map = {val.class_to_idx[old]: final_to_idx[new] for old, new in zip(val.classes, val_alias, strict=True)}
    train_ds = RemappedImageFolder(train, list(range(len(train))), train_map, train_classes)
    val_ds = RemappedImageFolder(val, list(range(len(val))), val_map, train_classes)
    return train_ds, val_ds, train_classes


class SimCLRImageFolder(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, input_size: int) -> None:
        self.dataset = imagefolder(root, TwoViewTransform(simclr_transform(input_size)))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        (x1, x2), _ = self.dataset[index]
        return x1, x2
