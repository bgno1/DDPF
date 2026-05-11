from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpf.data import SimCLRImageFolder
from ddpf.models import SimCLRModel
from ddpf.simclr import nt_xent
from ddpf.utils import device, ensure_dir, load_yaml, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default="checkpoints/malsim_resnet18_ep20.pth")
    parser.add_argument("--log-csv", default="results/malsim_pretrain_log.csv")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["fewshot"]["seed"]))
    dev = device()
    dataset = SimCLRImageFolder(cfg["paths"]["maldeb_all"], int(cfg["image"]["input_size"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=int(cfg["fewshot"]["num_workers"]), pin_memory=dev.type == "cuda")
    model = SimCLRModel(init="imagenet").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    rows = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        losses = []
        model.train()
        for x1, x2 in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"):
            x1, x2 = x1.to(dev, non_blocking=True), x2.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev.type, enabled=dev.type == "cuda"):
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = nt_xent(z1, z2, args.temperature)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        rows.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "epoch_time_sec": time.time() - start})
    ensure_dir(Path(args.output).parent)
    torch.save(
        {
            "backbone": model.encoder.state_dict(),
            "projection_head": model.projector.state_dict(),
            "epochs": args.epochs,
            "final_loss": rows[-1]["train_loss"],
        },
        args.output,
    )
    ensure_dir(Path(args.log_csv).parent)
    pd.DataFrame(rows).to_csv(args.log_csv, index=False)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
