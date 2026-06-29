#!/usr/bin/env python3
"""
b1_6: Compact no-gradient white-box MIA for the b1 Swin-T branch.

This revised version avoids flattening large raw Swin activation maps. It uses
compact target-model signals only:
  - penultimate feature vector
  - output probabilities
  - true labels
  - per-sample BCE loss

Reads b1_2 outputs:
  /work/hdd/bcga/priv-bench/downstream_models/swin_transformer_b1

Default target checkpoint:
  --checkpoint-kind best_val_macro_f1
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import swin_t
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

try:
    from opacus.validators import ModuleValidator
except ImportError:
    ModuleValidator = None


ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/swin_transformer_b1")
CKPT_DIR = ROOT / "checkpoints"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "attacks" / "whitebox_nograd_compact"
MEMBER_IMG_DIR = Path("/work/hdd/bcga/priv-bench/datasets/stylegan3/paired_real_train_10k")
NONMEMBER_IMG_DIR = Path("/u/jliu80/priv-bench/data/celeba/img_align_celeba")
MODELS = ["A_real_40", "B_synth_40", "C_dp_40"]
FEATURES = 40


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class MIADataset(Dataset):
    def __init__(self, members, nonmembers, image_size):
        self.samples = []
        for fname, y in members.items():
            self.samples.append((MEMBER_IMG_DIR / fname, y, 1))
        for fname, y in nonmembers.items():
            self.samples.append((NONMEMBER_IMG_DIR / fname, y, 0))
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y, m = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.tf(image), torch.tensor(y, dtype=torch.float32), int(m)


def build_model(is_dp):
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, FEATURES)
    if is_dp:
        if ModuleValidator is None:
            raise ImportError("opacus is required to load C_dp_40.")
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("head.")
        model = ModuleValidator.fix(model)
    return model


def ckpt_path(name, kind):
    if kind == "canonical":
        return CKPT_DIR / f"swin_t_{name}.pth"
    return CKPT_DIR / f"swin_t_{name}_{kind}.pth"


def load_model(name, kind, device):
    model = build_model(name == "C_dp_40")
    path = ckpt_path(name, kind)
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), str(path)


def swin_features_and_logits(model, x):
    z = model.features(x)
    z = model.norm(z)
    if hasattr(model, "permute") and callable(model.permute):
        z = model.permute(z)
    else:
        z = z.permute(0, 3, 1, 2)
    z = model.avgpool(z)
    feat = model.flatten(z)
    logits = model.head(feat)
    return feat, logits


class CompactAttackModel(nn.Module):
    def __init__(self, n_classes=FEATURES, dropout=0.2):
        super().__init__()
        self.feat_encoder = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU())
        self.prob_encoder = nn.Sequential(nn.Linear(n_classes, 64), nn.ReLU(), nn.Dropout(dropout))
        self.label_encoder = nn.Sequential(nn.Linear(n_classes, 64), nn.ReLU(), nn.Dropout(dropout))
        self.loss_encoder = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Dropout(dropout))
        self.final = nn.Sequential(
            nn.Linear(128 + 64 + 64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, feat, probs, labels, loss):
        parts = [
            self.feat_encoder(feat.float()),
            self.prob_encoder(probs.float()),
            self.label_encoder(labels.float()),
            self.loss_encoder(loss.float()),
        ]
        return self.final(torch.cat(parts, dim=1))


def attack_features(target_model, x, labels):
    with torch.no_grad():
        feat, logits = swin_features_and_logits(target_model, x)
        probs = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1, keepdim=True)
    return feat, probs, loss


def warmup(attack_model, target_model, device, image_size):
    attack_model.eval()
    target_model.eval()
    x = torch.randn(2, 3, image_size, image_size, device=device)
    y = torch.randint(0, 2, (2, FEATURES), device=device).float()
    feat, probs, loss = attack_features(target_model, x, y)
    with torch.no_grad():
        attack_model(feat, probs, y, loss)


def save_roc(y, score, auc, name, tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Compact no-gradient white-box MIA: {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / f"roc_whitebox_nograd_compact_{name}_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_attack(name, dataset, args, device):
    print("=" * 70)
    print(f"Compact no-gradient white-box MIA: {name}")
    target_model, ckpt = load_model(name, args.checkpoint_kind, device)
    print(f"Checkpoint: {ckpt}")
    print("Signals: penultimate_feature + output_probs + true_labels + per_sample_bce")

    y = np.array([m for _, _, m in dataset.samples])
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    attack_model = CompactAttackModel(dropout=args.dropout).to(device)
    warmup(attack_model, target_model, device, args.image_size)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        attack_model.train()
        losses = []
        for x, labels, is_member in tqdm(train_loader, desc=f"{name} epoch {epoch}/{args.epochs}", leave=False):
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            target = is_member.float().unsqueeze(1).to(device)
            feat, probs, per_sample_loss = attack_features(target_model, x, labels)
            pred = attack_model(feat, probs, labels, per_sample_loss)
            loss = F.binary_cross_entropy_with_logits(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[{name}] epoch {epoch:03d}/{args.epochs} attack_loss={np.mean(losses):.4f}")

    attack_model.eval()
    scores, labels_all = [], []
    with torch.no_grad():
        for x, labels, is_member in tqdm(test_loader, desc=f"{name} eval", leave=False):
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            feat, probs, per_sample_loss = attack_features(target_model, x, labels)
            scores.extend(torch.sigmoid(attack_model(feat, probs, labels, per_sample_loss)).cpu().numpy().ravel())
            labels_all.extend(is_member.numpy())

    scores = np.asarray(scores)
    labels_all = np.asarray(labels_all)
    auc = roc_auc_score(labels_all, scores)
    acc = accuracy_score(labels_all, (scores >= 0.5).astype(int))
    tag = args.checkpoint_kind
    save_roc(labels_all, scores, auc, name, tag)
    result = {
        "model": name,
        "auc": float(auc),
        "accuracy": float(acc),
        "attack": "whitebox_nograd_compact",
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint": ckpt,
        "signals": ["penultimate_feature", "output_probs", "true_labels", "per_sample_bce"],
    }
    save_json(OUT_DIR / f"result_whitebox_nograd_compact_{name}_{tag}.json", result)
    print(f"{name}: AUC={auc:.4f} ACC={acc:.4f}")
    return result


def write_csv(path, rows):
    keys = sorted({k for row in rows for k in row if not isinstance(row[k], list)})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k in keys})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-kind", choices=["best_val_macro_f1", "best_val_loss", "last", "canonical"], default="best_val_macro_f1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    members = load_json(META_DIR / "member_records.json")
    nonmembers = load_json(META_DIR / "nonmember_records.json")
    dataset = MIADataset(members, nonmembers, args.image_size)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_attack(name, dataset, args, device) for name in MODELS]
    tag = args.checkpoint_kind
    save_json(OUT_DIR / f"summary_whitebox_nograd_compact_{tag}.json", results)
    write_csv(OUT_DIR / f"summary_whitebox_nograd_compact_{tag}.csv", results)


if __name__ == "__main__":
    main()

