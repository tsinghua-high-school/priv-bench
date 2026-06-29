#!/usr/bin/env python3
"""
b_4: Random-forest label-informed black-box MIA for Swin-T A/B/C.

Reads outputs from b_2:
  checkpoints: /work/hdd/bcga/priv-bench/downstream_models/swim_transformer/checkpoints
  metadata:    /work/hdd/bcga/priv-bench/downstream_models/swim_transformer/metadata
"""

import argparse
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import swin_t
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from opacus.validators import ModuleValidator


ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/swim_transformer")
CKPT_DIR = ROOT / "checkpoints"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "attacks" / "rf_blackbox"
MEMBER_IMG_DIR = Path("/work/hdd/bcga/priv-bench/datasets/stylegan3/paired_real_train_10k")
NONMEMBER_IMG_DIR = Path("/u/jliu80/priv-bench/data/celeba/img_align_celeba")
MODELS = ["A_real_40", "B_synth_40", "C_dp_40"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


class MIADataset(Dataset):
    def __init__(self, members, nonmembers, image_size):
        self.samples = []
        for fname, y in members.items():
            self.samples.append((MEMBER_IMG_DIR / fname, y, 1))
        for fname, y in nonmembers.items():
            self.samples.append((NONMEMBER_IMG_DIR / fname, y, 0))
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y, m = self.samples[idx]
        x = Image.open(path).convert("RGB")
        return self.tf(x), torch.tensor(y, dtype=torch.float32), int(m)


def build_model(is_dp):
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, 40)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head.")
    return ModuleValidator.fix(model) if is_dp else model


def load_model(name, device):
    model = build_model(name == "C_dp_40")
    state = torch.load(CKPT_DIR / f"swin_t_{name}.pth", map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def add_stats(parts, x):
    parts.extend([x.mean(1, keepdim=True), x.std(1, unbiased=False, keepdim=True), x.min(1, keepdim=True).values, x.max(1, keepdim=True).values, x.median(1, keepdim=True).values])


def features_from_logits(logits, labels):
    eps = 1e-12
    probs = torch.sigmoid(logits)
    labels = labels.float()
    true_conf = labels * probs + (1 - labels) * (1 - probs)
    entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    margin = torch.abs(probs - 0.5)
    correct = ((probs >= 0.5).float() == labels).float()

    parts = [probs, true_conf, entropy, bce, margin, correct]
    for x in [true_conf, entropy, bce, margin, correct]:
        add_stats(parts, x)
    parts.extend([bce.mean(1, keepdim=True), entropy.mean(1, keepdim=True), correct.sum(1, keepdim=True), true_conf.min(1, keepdim=True).values, bce.max(1, keepdim=True).values])
    return torch.cat(parts, dim=1).cpu().numpy().astype(np.float32)


def extract_features(model, loader, device):
    xs, ys = [], []
    with torch.no_grad():
        for x, labels, is_member in tqdm(loader, desc="extract", leave=False):
            x, labels = x.to(device), labels.to(device)
            xs.append(features_from_logits(model(x), labels))
            ys.append(np.asarray(is_member, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def save_roc(y, score, auc, model_name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"RF black-box MIA: {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / f"roc_rf_blackbox_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_attack(model_name, dataset, args, device):
    print("=" * 70)
    print(f"RF black-box MIA: {model_name}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    x, y = extract_features(load_model(model_name, device), loader, device)
    print(f"features={x.shape[1]} samples={len(y)} member_rate={y.mean():.4f}")

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    clf = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    clf.fit(x_tr, y_tr)
    score = clf.predict_proba(x_te)[:, 1]
    pred = (score >= 0.5).astype(int)
    auc = roc_auc_score(y_te, score)
    acc = accuracy_score(y_te, pred)
    save_roc(y_te, score, auc, model_name)

    result = {"model": model_name, "auc": float(auc), "accuracy": float(acc), "num_features": int(x.shape[1]), "num_samples": int(len(y))}
    with open(OUT_DIR / f"result_rf_blackbox_{model_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"{model_name}: AUC={auc:.4f} ACC={acc:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    members = load_json(META_DIR / "member_records.json")
    nonmembers = load_json(META_DIR / "nonmember_records.json")
    dataset = MIADataset(members, nonmembers, args.image_size)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_attack(name, dataset, args, device) for name in MODELS]
    with open(OUT_DIR / "summary_rf_blackbox.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("Summary")
    for r in results:
        print(f"{r['model']:<12} AUC={r['auc']:.4f} ACC={r['accuracy']:.4f} features={r['num_features']}")


if __name__ == "__main__":
    main()
