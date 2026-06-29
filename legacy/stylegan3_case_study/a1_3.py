#!/usr/bin/env python3
"""
a1_3: Utility evaluation for the a1 ResNet18 downstream branch.

Evaluates checkpoints from:
  /work/hdd/bcga/priv-bench/downstream_models/resnet18_a1/checkpoints

Default checkpoint:
  --checkpoint-kind best_val_macro_f1

Metrics match a_3:
  micro label accuracy, micro/macro F1, macro balanced accuracy,
  mAP/AUPRC, macro AUROC, BCE, exact match, majority baseline,
  bootstrap SE and 95% CI.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.resnet import BasicBlock
from tqdm import tqdm

from opacus.validators import ModuleValidator


ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/resnet18_a1")
CKPT_DIR = ROOT / "checkpoints"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "utility"
IMG_DIR = Path("/u/jliu80/priv-bench/data/celeba/img_align_celeba")
MODELS = ["A_real_40", "B_synth_40", "C_dp_40"]


def safe_basicblock_forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    out = self.relu(out)
    return out


BasicBlock.forward = safe_basicblock_forward


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class Records(Dataset):
    def __init__(self, records, image_size):
        self.items = list(records.items())
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, y = self.items[idx]
        image = Image.open(IMG_DIR / fname).convert("RGB")
        return self.tf(image), torch.tensor(y, dtype=torch.float32)


def build_model(is_dp):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 40)
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return ModuleValidator.fix(model) if is_dp else model


def ckpt_path(name, kind):
    if kind == "canonical":
        return CKPT_DIR / f"resnet18_{name}.pth"
    return CKPT_DIR / f"resnet18_{name}_{kind}.pth"


def load_model(name, kind, device):
    model = build_model(name == "C_dp_40")
    path = ckpt_path(name, kind)
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), str(path)


def predict(model, loader, device):
    ys, ps, losses = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="predict", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            ys.append(y.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
            losses.append(F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(dim=1).cpu().numpy())
    return np.vstack(ys).astype(int), np.vstack(ps), np.concatenate(losses)


def macro_auroc(y, p):
    vals = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if len(np.unique(y[:, j])) == 2]
    return float(np.mean(vals))


def macro_auprc(y, p):
    vals = [average_precision_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if y[:, j].sum() > 0]
    return float(np.mean(vals))


def macro_bal_acc(y, pred):
    vals = [balanced_accuracy_score(y[:, j], pred[:, j]) for j in range(y.shape[1]) if len(np.unique(y[:, j])) == 2]
    return float(np.mean(vals))


def metrics(y, p, losses):
    pred = (p >= 0.5).astype(int)
    return {
        "bce": float(losses.mean()),
        "micro_label_acc": float((pred == y).mean()),
        "micro_f1": float(f1_score(y.reshape(-1), pred.reshape(-1), zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_balanced_acc": macro_bal_acc(y, pred),
        "map_macro_auprc": macro_auprc(y, p),
        "macro_auroc": macro_auroc(y, p),
        "exact_match_acc": float((pred == y).all(axis=1).mean()),
        "true_positive_rate": float(y.mean()),
        "pred_positive_rate": float(pred.mean()),
    }


def with_bootstrap(y, p, losses, n_boot, seed):
    base = metrics(y, p, losses)
    if n_boot <= 0:
        return {k: {"value": v, "se": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")} for k, v in base.items()}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in base}
    for _ in tqdm(range(n_boot), desc="bootstrap", leave=False):
        idx = rng.integers(0, len(y), len(y))
        row = metrics(y[idx], p[idx], losses[idx])
        for k, v in row.items():
            if np.isfinite(v):
                boot[k].append(v)
    out = {}
    for k, v in base.items():
        vals = np.asarray(boot[k], dtype=float)
        out[k] = {
            "value": v,
            "se": float(vals.std(ddof=1)),
            "ci95_low": float(np.percentile(vals, 2.5)),
            "ci95_high": float(np.percentile(vals, 97.5)),
        }
    return out


def majority_baseline(y):
    pred = np.tile((y.mean(axis=0) >= 0.5).astype(int), (len(y), 1))
    return {
        "majority_micro_label_acc": float((pred == y).mean()),
        "majority_micro_f1": float(f1_score(y.reshape(-1), pred.reshape(-1), zero_division=0)),
        "majority_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "majority_macro_balanced_acc": macro_bal_acc(y, pred),
        "majority_exact_match_acc": float((pred == y).all(axis=1).mean()),
    }


def flat_row(name, summary, ckpt):
    row = {"model": name, "checkpoint": ckpt}
    for k, s in summary.items():
        row[k] = s["value"]
        row[f"{k}_se"] = s["se"]
        row[f"{k}_ci95_low"] = s["ci95_low"]
        row[f"{k}_ci95_high"] = s["ci95_high"]
    return row


def write_csv(path, rows):
    keys = ["model", "checkpoint"] + sorted(k for k in rows[0] if k not in {"model", "checkpoint"})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-kind", choices=["best_val_macro_f1", "best_val_loss", "last", "canonical"], default="best_val_macro_f1")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_json(META_DIR / "nonmember_records.json")
    loader = DataLoader(
        Records(records, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    results, rows, baseline = {}, [], None
    for name in MODELS:
        print("=" * 70)
        print(f"Utility evaluation: {name}")
        model, ckpt = load_model(name, args.checkpoint_kind, device)
        print(f"Checkpoint: {ckpt}")
        y, p, loss = predict(model, loader, device)
        summary = with_bootstrap(y, p, loss, args.bootstrap, args.seed)
        if baseline is None:
            baseline = majority_baseline(y)
        results[name] = {"checkpoint": ckpt, "metrics": summary}
        rows.append(flat_row(name, summary, ckpt))
        print(
            f"{name}: micro_acc={summary['micro_label_acc']['value']:.4f} "
            f"macro_f1={summary['macro_f1']['value']:.4f} "
            f"micro_f1={summary['micro_f1']['value']:.4f} "
            f"macro_bal_acc={summary['macro_balanced_acc']['value']:.4f} "
            f"mAP/AUPRC={summary['map_macro_auprc']['value']:.4f} "
            f"macro_AUROC={summary['macro_auroc']['value']:.4f} "
            f"BCE={summary['bce']['value']:.4f}"
        )

    tag = args.checkpoint_kind
    save_json(OUT_DIR / f"a1_3_resnet18_utility_results_{tag}.json", {"models": results, "majority_baseline": baseline})
    write_csv(OUT_DIR / f"a1_3_resnet18_utility_summary_{tag}.csv", rows)
    print("=" * 70)
    print("Majority baseline:")
    for k, v in baseline.items():
        print(f"{k}: {v:.4f}")
    print(f"Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
