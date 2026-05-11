from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import models
from tqdm import tqdm

from ddpf.data import aligned_malimg, eval_transform
from ddpf.methods import (
    ddpf_predict,
    l2,
    metrics,
    one_nn_predict,
    proto_predict,
    prototypes,
    select_alpha,
)
from ddpf.models import build_encoder
from ddpf.utils import device, ensure_dir, load_json, load_yaml, set_seed


def extract(encoder, dataset, batch_size: int, dev: torch.device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=dev.type == "cuda")
    xs, ys = [], []
    encoder.eval()
    with torch.no_grad():
        for image, label in tqdm(loader, desc="extract"):
            xs.append(l2(encoder(image.to(dev))).cpu())
            ys.append(label.cpu())
    return torch.cat(xs), torch.cat(ys)


def train_linear(support_x, support_y, query_x, num_classes: int, dev: torch.device):
    head = nn.Linear(support_x.size(1), num_classes).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(support_x, support_y), batch_size=min(64, len(support_y)), shuffle=True)
    for _ in range(100):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x), y)
            loss.backward()
            opt.step()
    with torch.no_grad():
        return head(query_x.to(dev)).argmax(1).cpu()


def freeze_bn(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def finetune_predict(kind, checkpoint, train_ds, query_ds, support, query, num_classes, dev):
    if kind == "ImgNet-FT":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif kind == "MalSim-FT":
        model = models.resnet18(weights=None)
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("backbone") or ckpt.get("encoder_state_dict") or ckpt.get("encoder")
        if state is None:
            raise ValueError(f"Unsupported checkpoint: {checkpoint}")
        model.load_state_dict(state, strict=False)
    else:
        raise ValueError(kind)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(dev)
    loader = DataLoader(Subset(train_ds, support), batch_size=min(32, len(support)), shuffle=True, num_workers=2, pin_memory=dev.type == "cuda")
    qloader = DataLoader(Subset(query_ds, query), batch_size=128, shuffle=False, num_workers=2, pin_memory=dev.type == "cuda")
    backbone, head = [], []
    for name, param in model.named_parameters():
        (head if name.startswith("fc.") else backbone).append(param)
    opt = torch.optim.AdamW([{"params": backbone, "lr": 1e-5}, {"params": head, "lr": 1e-3}], weight_decay=1e-4)
    for _ in range(50):
        model.train()
        freeze_bn(model)
        for image, label in loader:
            image, label = image.to(dev), label.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(image), label)
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for image, _ in qloader:
            preds.append(model(image.to(dev)).argmax(1).cpu())
    return torch.cat(preds)


def evaluate_method(method, task, feats, labels, query_labels, num_classes, alphas, dev, train_ds, query_ds, checkpoint):
    support = task["support_indices"]
    query = task["query_indices"]
    y_support = labels["train"][support]
    y_query = query_labels[query]
    f_i_s, f_i_q = feats["ImgNet_train"], feats["ImgNet_query"][query]
    f_m_s, f_m_q = feats["MalSim_train"], feats["MalSim_query"][query]
    selected_alpha = None

    if method == "ImgNet-1NN":
        pred = one_nn_predict(f_i_s[support], y_support, f_i_q)
    elif method == "MalSim-1NN":
        pred = one_nn_predict(f_m_s[support], y_support, f_m_q)
    elif method == "ImgNet-Proto":
        pred = proto_predict(f_i_q, prototypes(f_i_s, labels["train"], support, num_classes))
    elif method == "MalSim-Proto":
        pred = proto_predict(f_m_q, prototypes(f_m_s, labels["train"], support, num_classes))
    elif method == "ImgNet-LP":
        pred = train_linear(f_i_s[support], y_support, f_i_q, num_classes, dev)
    elif method == "MalSim-LP":
        pred = train_linear(f_m_s[support], y_support, f_m_q, num_classes, dev)
    elif method == "DDPF-Oracle":
        selected_alpha = 0.5
        p_i = prototypes(f_i_s, labels["train"], support, num_classes)
        p_m = prototypes(f_m_s, labels["train"], support, num_classes)
        pred = ddpf_predict(f_i_q, p_i, f_m_q, p_m, selected_alpha)
    elif method == "DDPF-Adp":
        selected_alpha, _, _, _ = select_alpha(f_i_s, f_m_s, labels["train"], support, num_classes, task["shot"], alphas)
        p_i = prototypes(f_i_s, labels["train"], support, num_classes)
        p_m = prototypes(f_m_s, labels["train"], support, num_classes)
        pred = ddpf_predict(f_i_q, p_i, f_m_q, p_m, selected_alpha)
    elif method in {"ImgNet-FT", "MalSim-FT"}:
        pred = finetune_predict(method, checkpoint, train_ds, query_ds, support, query, num_classes, dev)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return {"selected_alpha": selected_alpha, **metrics(y_query.numpy(), pred.numpy())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--support-plan", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--malsim-checkpoint", required=True)
    parser.add_argument("--output", default="results/fewshot_results.csv")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["fewshot"]["seed"]))
    dev = device()
    transform = eval_transform(int(cfg["image"]["input_size"]))
    train, query, classes = aligned_malimg(cfg["paths"]["malimg_train"], cfg["paths"]["malimg_validation"], transform, cfg["class_alignment"]["aliases"])
    enc_i = build_encoder("imagenet").to(dev)
    enc_m = build_encoder("malsim", args.malsim_checkpoint).to(dev)
    batch_size = int(cfg["fewshot"]["batch_size"])
    f_i_train, y_train = extract(enc_i, train, batch_size, dev)
    f_i_query, y_query = extract(enc_i, query, batch_size, dev)
    f_m_train, _ = extract(enc_m, train, batch_size, dev)
    f_m_query, _ = extract(enc_m, query, batch_size, dev)
    feats = {"ImgNet_train": f_i_train, "ImgNet_query": f_i_query, "MalSim_train": f_m_train, "MalSim_query": f_m_query}
    labels = {"train": y_train}
    plan = load_json(args.support_plan)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []
    for method in methods:
        for task in plan["tasks"]:
            out = evaluate_method(method, task, feats, labels, y_query, len(classes), cfg["ddpf"]["alpha_candidates"], dev, train, query, args.malsim_checkpoint)
            rows.append({"method": method, "shot": task["shot"], "repeat": task["repeat"], "seed": task["seed"], **out})
    ensure_dir(Path(args.output).parent)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
