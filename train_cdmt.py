# train_cdmt.py

import argparse
import os
from typing import Dict, Any

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from cdmt_model import CDMTConfig, CDMT
from cdmt_dataset import MemeDataset, build_label_map


def collate_fn(batch):
    # Simple stack collate (all already padded)
    images = torch.stack([b["images"] for b in batch], dim=0)
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    cds: torch.Tensor,
    dis_probs: torch.Tensor = None,
) -> Dict[str, Any]:
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    acc = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds, average="macro")

    mean_cds = cds.mean().item()

    # simple "agreement" metric: correct + low CDS
    cds_np = cds.cpu().numpy()
    threshold = np.percentile(cds_np, 80)  # example
    mask_agree = (preds == labels_np) & (cds_np <= threshold)
    haar = mask_agree.mean().item()

    metrics = {
        "acc": acc,
        "f1": f1,
        "mean_cds": mean_cds,
        "haar": haar,
    }

    if dis_probs is not None:
        # treat top 20% dis_probs as predicted disagreement
        dis_np = dis_probs.cpu().numpy()
        k = max(1, int(0.2 * len(dis_np)))
        idx_sorted = np.argsort(-dis_np)
        flagged = idx_sorted[:k]

        err = (preds != labels_np).astype(float)
        fp_at_k = err[flagged].mean().item()
        # simple AUROC over (cds vs err) as rough measure
        try:
            auroc_dis = roc_auc_score(err, dis_np)
        except ValueError:
            auroc_dis = float("nan")

        metrics["fp_at_20pct"] = fp_at_k
        metrics["auroc_dis"] = auroc_dis

    return metrics


def train_epoch(
    model: CDMT,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    tau_dis: float,
) -> Dict[str, float]:
    model.train()
    losses = []

    for batch in tqdm(loader, desc="Train", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)

        out = model(batch, tau_dis=tau_dis)
        loss = out["loss_total"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

    return {"loss": float(np.mean(losses))}


def eval_epoch(
    model: CDMT,
    loader: DataLoader,
    device: torch.device,
    tau_dis: float,
) -> Dict[str, Any]:
    model.eval()
    all_logits = []
    all_labels = []
    all_cds = []
    all_dis_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            for k in batch:
                batch[k] = batch[k].to(device)

            out = model(batch, tau_dis=tau_dis)
            all_logits.append(out["logits"].cpu())
            all_labels.append(batch["labels"].cpu())
            all_cds.append(out["cds"].cpu())
            if out["dis_probs"] is not None:
                all_dis_probs.append(out["dis_probs"].cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    cds = torch.cat(all_cds, dim=0)
    dis_probs = torch.cat(all_dis_probs, dim=0) if all_dis_probs else None

    metrics = compute_metrics(logits, labels, cds, dis_probs)
    return metrics


def estimate_tau_dis(
    model: CDMT,
    loader: DataLoader,
    device: torch.device,
    percentile: float = 90.0,
) -> float:
    model.eval()
    all_cds = []

    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            out = model(batch, tau_dis=None)
            all_cds.append(out["cds"].cpu())

    cds = torch.cat(all_cds, dim=0).numpy()
    tau = float(np.percentile(cds, percentile))
    return tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=False, default=None)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    label_map = build_label_map(args.labels)

    train_dataset = MemeDataset(
        csv_path=args.train_csv,
        image_root=args.image_root,
        label_map=label_map,
        max_length=48,
    )

    if args.val_csv is not None:
        val_dataset = MemeDataset(
            csv_path=args.val_csv,
            image_root=args.image_root,
            label_map=label_map,
            max_length=48,
        )
    else:
        # simple split
        n_total = len(train_dataset)
        n_val = int(0.1 * n_total)
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(
            train_dataset, [n_train, n_val]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    cfg = CDMTConfig(num_labels=len(args.labels))
    model = CDMT(cfg).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Estimate tau_dis once on validation set
    print("Estimating disagreement threshold tau_dis ...")
    tau_dis = estimate_tau_dis(model, val_loader, device, percentile=90.0)
    print(f"Using tau_dis = {tau_dis:.4f}")

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_stats = train_epoch(model, train_loader, optimizer, device, tau_dis)
        val_stats = eval_epoch(model, val_loader, device, tau_dis)

        scheduler.step()

        print(f"Train loss: {train_stats['loss']:.4f}")
        print(
            f"Val acc: {val_stats['acc']:.4f}, "
            f"F1: {val_stats['f1']:.4f}, "
            f"mean-CDS: {val_stats['mean_cds']:.4f}, "
            f"HAAR: {val_stats['haar']:.4f}, "
            f"AUROC_dis: {val_stats.get('auroc_dis', float('nan')):.4f}"
        )

        if val_stats["f1"] > best_f1:
            best_f1 = val_stats["f1"]
            ckpt_path = os.path.join(args.output_dir, f"cdmt_best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "label_map": label_map,
                    "tau_dis": tau_dis,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
