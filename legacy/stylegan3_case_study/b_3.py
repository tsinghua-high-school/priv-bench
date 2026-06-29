#!/usr/bin/env python3
"""
b_3: Utility evaluation for Swin-T downstream models.

Fixed protocol:
  - Evaluate A/B/C checkpoints from downstream_models/swim_transformer/checkpoints.
  - Test set is metadata/nonmember_records.json from b_2.
  - Threshold is fixed at 0.5.
  - Report the same metrics as a_3, with bootstrap SE/95% CI.
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
from torchvision import transforms
from torchvision.models import swin_t
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from opacus.validators import ModuleValidator


ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/swim_transformer")
CKPT_DIR = ROOT / "checkpoints"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "utility"
IMG_DIR = Path("/u/jliu80/priv-bench/data/celeba/img_align_celeba")
MODELS = ["A_real_40", "B_synth_40", "C_dp_40"]


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
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, y = self.items[idx]
        x = Image.open(IMG_DIR / fname).convert("RGB")
        return self.tf(x), torch.tensor(y, dtype=torch.float32)


def build_model(is_dp):
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, 40)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head.")
    return ModuleValidator.fix(model) if is_dp else model


def load_model(name, device):
    is_dp = name == "C_dp_40"
    model = build_model(is_dp)
    state = torch.load(CKPT_DIR / f"swin_t_{name}.pth", map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def predict(model, loader, device):
    ys, ps, losses = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="predict", leave=False):
            x, y = x.to(device), y.to(device)
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
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in base}
    for _ in tqdm(range(n_boot), desc="bootstrap", leave=False):
        idx = rng.integers(0, len(y), len(y))
        m = metrics(y[idx], p[idx], losses[idx])
        for k, v in m.items():
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


def write_csv(path, rows):
    keys = ["model"] + sorted(k for k in rows[0] if k != "model")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flat_row(name, summary):
    row = {"model": name}
    for k, s in summary.items():
        row[k] = s["value"]
        row[f"{k}_se"] = s["se"]
        row[f"{k}_ci95_low"] = s["ci95_low"]
        row[f"{k}_ci95_high"] = s["ci95_high"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_json(META_DIR / "nonmember_records.json")
    loader = DataLoader(Records(records, args.image_size), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    results, rows, baseline = {}, [], None
    for name in MODELS:
        print("=" * 70)
        print(f"Utility evaluation: {name}")
        y, p, loss = predict(load_model(name, device), loader, device)
        summary = with_bootstrap(y, p, loss, args.bootstrap, args.seed)
        if baseline is None:
            baseline = majority_baseline(y)
        results[name] = summary
        rows.append(flat_row(name, summary))
        print(
            f"{name}: micro_acc={summary['micro_label_acc']['value']:.4f} "
            f"macro_f1={summary['macro_f1']['value']:.4f} "
            f"micro_f1={summary['micro_f1']['value']:.4f} "
            f"macro_bal_acc={summary['macro_balanced_acc']['value']:.4f} "
            f"mAP/AUPRC={summary['map_macro_auprc']['value']:.4f} "
            f"macro_AUROC={summary['macro_auroc']['value']:.4f} "
            f"BCE={summary['bce']['value']:.4f}"
        )

    save_json(OUT_DIR / "b_3_swin_utility_results.json", {"models": results, "majority_baseline": baseline})
    write_csv(OUT_DIR / "b_3_swin_utility_summary.csv", rows)
    print("=" * 70)
    print("Majority baseline:")
    for k, v in baseline.items():
        print(f"{k}: {v:.4f}")
    print(f"Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
