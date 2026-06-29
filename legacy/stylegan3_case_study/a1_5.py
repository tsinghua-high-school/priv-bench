#!/usr/bin/env python3
"""
a1_5: Shadow-model black-box MIA for the a1 ResNet18 branch.

Shadow models:
  - ResNet18, ImageNet initialization, full fine-tuning on auxiliary official-val splits.
  - Shadow train split = shadow members.
  - Shadow holdout split = shadow nonmembers.

Target evaluation:
  - Target checkpoints and member/nonmember records come from a1_2 outputs.
  - The attack classifier is trained only on shadow-model outputs, not on target
    member/nonmember labels.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.resnet import BasicBlock
from tqdm import tqdm

try:
    from opacus.validators import ModuleValidator
except ImportError:
    ModuleValidator = None


STYLEGAN_META = Path("/work/hdd/bcga/priv-bench/datasets/stylegan3/metadata")
ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/resnet18_a1")
CKPT_DIR = ROOT / "checkpoints"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "attacks" / "shadow_blackbox"
OFFICIAL_IMG_DIR = Path("/u/jliu80/priv-bench/data/celeba/img_align_celeba")
MEMBER_IMG_DIR = Path("/work/hdd/bcga/priv-bench/datasets/stylegan3/paired_real_train_10k")
MODELS = ["A_real_40", "B_synth_40", "C_dp_40"]
FEATURES = 40


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


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class Records(Dataset):
    def __init__(self, records, img_dir, image_size):
        self.items = list(records.items()) if isinstance(records, dict) else list(records)
        self.img_dir = Path(img_dir)
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, y = self.items[idx]
        image = Image.open(self.img_dir / fname).convert("RGB")
        return self.tf(image), torch.tensor(y, dtype=torch.float32)


class TargetMIADataset(Dataset):
    def __init__(self, members, nonmembers, image_size):
        self.samples = [(MEMBER_IMG_DIR / f, y, 1) for f, y in members.items()]
        self.samples += [(OFFICIAL_IMG_DIR / f, y, 0) for f, y in nonmembers.items()]
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y, m = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.tf(image), torch.tensor(y, dtype=torch.float32), int(m)


def build_resnet(pretrained=False, is_dp=False):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, FEATURES)
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    if is_dp:
        if ModuleValidator is None:
            raise ImportError("opacus is required to load C_dp_40.")
        model = ModuleValidator.fix(model)
    return model


def ckpt_path(name, kind):
    if kind == "canonical":
        return CKPT_DIR / f"resnet18_{name}.pth"
    return CKPT_DIR / f"resnet18_{name}_{kind}.pth"


def load_target(name, checkpoint_kind, device):
    model = build_resnet(pretrained=False, is_dp=(name == "C_dp_40"))
    path = ckpt_path(name, checkpoint_kind)
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), str(path)


def train_shadow(model, records, args, device):
    loader = DataLoader(
        Records(records, OFFICIAL_IMG_DIR, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.shadow_epochs + 1):
        losses = []
        for x, y in tqdm(loader, desc=f"shadow train {epoch}/{args.shadow_epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"shadow epoch {epoch}: loss={np.mean(losses):.4f}")
    return model.eval()


def add_stats(parts, x):
    parts += [
        x.mean(1, keepdim=True),
        x.std(1, unbiased=False, keepdim=True),
        x.min(1, keepdim=True).values,
        x.max(1, keepdim=True).values,
        x.median(1, keepdim=True).values,
    ]


def mia_features(logits, labels):
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
    parts += [
        bce.mean(1, keepdim=True),
        entropy.mean(1, keepdim=True),
        correct.sum(1, keepdim=True),
        true_conf.min(1, keepdim=True).values,
        bce.max(1, keepdim=True).values,
    ]
    return torch.cat(parts, 1).cpu().numpy().astype(np.float32)


def extract_shadow_features(model, member_records, nonmember_records, args, device):
    xs, ys = [], []
    for records, m in [(member_records, 1), (nonmember_records, 0)]:
        loader = DataLoader(
            Records(records, OFFICIAL_IMG_DIR, args.image_size),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        with torch.no_grad():
            for x, labels in tqdm(loader, desc=f"shadow extract m={m}", leave=False):
                x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                xs.append(mia_features(model(x), labels))
                ys.append(np.full(len(labels), m, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def extract_target_features(model, dataset, args, device):
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    xs, ys = [], []
    with torch.no_grad():
        for x, labels, m in tqdm(loader, desc="target extract", leave=False):
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            xs.append(mia_features(model(x), labels))
            ys.append(np.asarray(m, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def shadow_splits(records, shadow_id, args):
    rng = random.Random(args.seed + shadow_id)
    items = list(records.items())
    rng.shuffle(items)
    need = args.shadow_member_size + args.shadow_nonmember_size
    if len(items) < need:
        raise ValueError(f"Need {need} official-val samples for each shadow, got {len(items)}.")
    return dict(items[:args.shadow_member_size]), dict(items[args.shadow_member_size:need])


def save_roc(y, score, auc, name, tag):
    fpr, tpr, _ = roc_curve(y, score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Shadow black-box MIA: {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / f"roc_shadow_blackbox_{name}_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-kind", choices=["best_val_macro_f1", "best_val_loss", "last", "canonical"], default="best_val_macro_f1")
    parser.add_argument("--num-shadows", type=int, default=4)
    parser.add_argument("--shadow-member-size", type=int, default=2500)
    parser.add_argument("--shadow-nonmember-size", type=int, default=2500)
    parser.add_argument("--shadow-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    official_val = load_json(STYLEGAN_META / "official_val_records_40.json")

    x_train, y_train = [], []
    for sid in range(args.num_shadows):
        print("=" * 70)
        print(f"Training shadow ResNet18 {sid + 1}/{args.num_shadows}")
        shadow_mem, shadow_non = shadow_splits(official_val, sid, args)
        shadow = train_shadow(build_resnet(pretrained=True), shadow_mem, args, device)
        x_s, y_s = extract_shadow_features(shadow, shadow_mem, shadow_non, args, device)
        x_train.append(x_s)
        y_train.append(y_s)
        del shadow
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
    attack = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    attack.fit(x_train, y_train)
    print(f"Attack train features={x_train.shape[1]} samples={len(y_train)}")

    target_ds = TargetMIADataset(
        load_json(META_DIR / "member_records.json"),
        load_json(META_DIR / "nonmember_records.json"),
        args.image_size,
    )
    rows = []
    tag = args.checkpoint_kind
    for name in MODELS:
        print("=" * 70)
        print(f"Evaluating target {name}")
        target, ckpt = load_target(name, args.checkpoint_kind, device)
        print(f"Checkpoint: {ckpt}")
        x_t, y_t = extract_target_features(target, target_ds, args, device)
        score = attack.predict_proba(x_t)[:, 1]
        auc = roc_auc_score(y_t, score)
        acc = accuracy_score(y_t, (score >= 0.5).astype(int))
        save_roc(y_t, score, auc, name, tag)
        row = {
            "model": name,
            "auc": float(auc),
            "accuracy": float(acc),
            "attack": "shadow_blackbox_rf",
            "checkpoint_kind": args.checkpoint_kind,
            "checkpoint": ckpt,
            "num_shadow_models": args.num_shadows,
            "num_features": int(x_t.shape[1]),
        }
        rows.append(row)
        save_json(OUT_DIR / f"result_shadow_blackbox_{name}_{tag}.json", row)
        print(f"{name}: AUC={auc:.4f} ACC={acc:.4f}")
        del target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(OUT_DIR / f"summary_shadow_blackbox_{tag}.json", rows)
    with open(OUT_DIR / f"summary_shadow_blackbox_{tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
