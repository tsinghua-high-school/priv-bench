#!/usr/bin/env python3
"""
a1_10: Offline LiRA-style MIA for the a1 ResNet18 branch.

Protocol:
  - Train K OUT shadow ResNet18 models on official-val subsets.
  - Each shadow model uses 10,000 official-val images, matching the target
    downstream training-set size.
  - For every target audit sample, collect its score under every OUT shadow
    model and fit a per-sample OUT Gaussian.
  - Query the target model once; membership score is the negative log
    likelihood of the observed score under the OUT distribution.

Default target checkpoints are the a1_2 validation-selected checkpoints:
  /work/hdd/bcga/priv-bench/downstream_models/resnet18_a1/checkpoints/
    resnet18_{A_real_40,B_synth_40,C_dp_40}_best_val_macro_f1.pth

Score:
  score(x, y) = -BCEWithLogits(target(x), y), so higher means the model is
  more confident on the true multilabel target.
"""

import argparse
import csv
import json
import math
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
OUT_DIR = ROOT / "attacks" / "lira_offline"
SHADOW_DIR = OUT_DIR / "shadow_checkpoints"
SCORE_DIR = OUT_DIR / "score_cache"
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
    def __init__(self, records, img_dir, image_size, train=False, hflip=False):
        self.items = list(records.items()) if isinstance(records, dict) else list(records)
        ops = [transforms.Resize((image_size, image_size))]
        if train and hflip:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        self.tf = transforms.Compose(ops)
        self.img_dir = Path(img_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, labels = self.items[idx]
        image = Image.open(self.img_dir / fname).convert("RGB")
        return self.tf(image), torch.tensor(labels, dtype=torch.float32), fname


class TargetAuditDataset(Dataset):
    def __init__(self, members, nonmembers, image_size):
        self.items = [(MEMBER_IMG_DIR / f, f, y, 1) for f, y in members.items()]
        self.items += [(OFFICIAL_IMG_DIR / f, f, y, 0) for f, y in nonmembers.items()]
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, fname, labels, member = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.tf(image), torch.tensor(labels, dtype=torch.float32), int(member), fname


def build_resnet(pretrained):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, FEATURES)
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model


def ckpt_path(name, kind):
    if kind == "canonical":
        return CKPT_DIR / f"resnet18_{name}.pth"
    return CKPT_DIR / f"resnet18_{name}_{kind}.pth"


def load_target(name, checkpoint_kind, device):
    model = build_resnet(pretrained=False)
    if name == "C_dp_40":
        if ModuleValidator is None:
            raise ImportError("opacus is required to load C_dp_40.")
        model = ModuleValidator.fix(model)
        BasicBlock.forward = safe_basicblock_forward
    path = ckpt_path(name, checkpoint_kind)
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), str(path)


def shadow_path(sid):
    return SHADOW_DIR / f"shadow_resnet18_out_{sid:03d}.pth"


def train_shadow(sid, official_val, args, device):
    path = shadow_path(sid)
    if path.exists() and not args.retrain_shadows:
        print(f"shadow {sid:03d}: checkpoint exists, skip")
        return

    rng = random.Random(args.seed + sid)
    items = list(official_val.items())
    rng.shuffle(items)
    if len(items) < args.shadow_train_size:
        raise ValueError(f"Need {args.shadow_train_size} official-val records, got {len(items)}.")
    train_records = dict(items[: args.shadow_train_size])

    model = build_resnet(pretrained=True).to(device).train()
    loader = DataLoader(
        Records(train_records, OFFICIAL_IMG_DIR, args.image_size, train=True, hflip=args.shadow_hflip),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.shadow_epochs + 1):
        losses = []
        for x, y, _ in tqdm(loader, desc=f"shadow {sid:03d} epoch {epoch}/{args.shadow_epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"shadow_id": sid, "epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan")}
        history.append(row)
        print(f"shadow {sid:03d} epoch {epoch}: loss={row['train_loss']:.4f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    save_json(SHADOW_DIR / f"shadow_resnet18_out_{sid:03d}_history.json", history)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def lira_score_from_logits(logits, labels):
    per_attr = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return -per_attr.mean(1)


def score_dataset(model, dataset, args, device):
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    scores, labels, names = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="score dataset", leave=False):
            if len(batch) == 3:
                x, y, fname = batch
                m = None
            else:
                x, y, m, fname = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            scores.append(lira_score_from_logits(model(x), y).cpu().numpy().astype(np.float32))
            if m is not None:
                labels.append(np.asarray(m, dtype=np.int64))
            names.extend(list(fname))
    scores = np.concatenate(scores)
    labels = np.concatenate(labels) if labels else None
    return scores, labels, names


def load_shadow_model(path, device):
    model = build_resnet(pretrained=False)
    state = torch.load(path, map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def train_shadows(args, device):
    official_val = load_json(STYLEGAN_META / "official_val_records_40.json")
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    for sid in range(args.shadow_start, args.shadow_end):
        print("=" * 70)
        print(f"OUT shadow model {sid + 1}/{args.num_shadows}")
        train_shadow(sid, official_val, args, device)


def build_target_dataset(args):
    return TargetAuditDataset(
        load_json(META_DIR / "member_records.json"),
        load_json(META_DIR / "nonmember_records.json"),
        args.image_size,
    )


def score_shadows(args, device):
    dataset = build_target_dataset(args)
    all_scores, used = [], []
    for sid in range(args.shadow_start, args.shadow_end):
        ckpt = shadow_path(sid)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing shadow checkpoint: {ckpt}")
        cache = SCORE_DIR / f"shadow_scores_out_{sid:03d}.npz"
        if cache.exists() and not args.rescore:
            z = np.load(cache, allow_pickle=True)
            all_scores.append(z["scores"].astype(np.float32))
            used.append(sid)
            print(f"shadow {sid:03d}: reused scores")
            continue
        print("=" * 70)
        print(f"Scoring audit samples with OUT shadow {sid:03d}")
        model = load_shadow_model(ckpt, device)
        scores, labels, names = score_dataset(model, dataset, args, device)
        SCORE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, scores=scores, labels=labels, names=np.asarray(names, dtype=object), shadow_id=sid)
        all_scores.append(scores)
        used.append(sid)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    matrix = np.stack(all_scores, axis=0)
    summary = SCORE_DIR / f"shadow_out_score_matrix_{args.shadow_start:03d}_{args.shadow_end:03d}.npz"
    np.savez_compressed(summary, scores=matrix, shadow_ids=np.asarray(used), labels=labels, names=np.asarray(names, dtype=object))
    print(f"Saved shadow OUT score matrix: {summary} shape={matrix.shape}")
    return summary


def load_shadow_matrix(args):
    hits = sorted(SCORE_DIR.glob("shadow_scores_out_*.npz"))
    if len(hits) < args.num_shadows and args.require_all_shadows:
        raise RuntimeError(f"Found {len(hits)} shadow score files, expected {args.num_shadows}.")
    if not hits:
        raise FileNotFoundError(f"No shadow score files found under {SCORE_DIR}")
    rows, ids, labels, names = [], [], None, None
    for path in hits[: args.num_shadows]:
        z = np.load(path, allow_pickle=True)
        rows.append(z["scores"].astype(np.float32))
        ids.append(int(z["shadow_id"]) if "shadow_id" in z else int(path.stem.split("_")[-1]))
        labels = z["labels"].astype(np.int64)
        names = z["names"]
    matrix = np.stack(rows, axis=0)
    print(f"Loaded OUT shadow scores: shape={matrix.shape}, shadows={len(ids)}")
    return matrix, labels, names, ids


def tpr_at_fpr(labels, scores, fpr_level):
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = np.where(fpr <= fpr_level)[0]
    return float(tpr[ok].max()) if len(ok) else 0.0


def save_roc(labels, scores, auc, name, tag):
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Offline LiRA MIA: {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / f"roc_lira_offline_{name}_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_targets(args, device):
    shadow_scores, labels, names, shadow_ids = load_shadow_matrix(args)
    mu = np.median(shadow_scores, axis=0) if args.center == "median" else shadow_scores.mean(axis=0)
    sigma = shadow_scores.std(axis=0, ddof=1)
    sigma = np.maximum(sigma, args.sigma_floor)

    dataset = build_target_dataset(args)
    rows = []
    for name in MODELS:
        print("=" * 70)
        print(f"Offline LiRA target: {name}")
        target, ckpt = load_target(name, args.checkpoint_kind, device)
        obs, y_true, obs_names = score_dataset(target, dataset, args, device)
        if not np.array_equal(y_true, labels):
            raise RuntimeError("Target labels and shadow-score labels are not aligned.")
        if not np.array_equal(np.asarray(obs_names), np.asarray(names)):
            raise RuntimeError("Target names and shadow-score names are not aligned.")
        z = (obs - mu) / sigma
        lira = 0.5 * (z ** 2) + np.log(sigma)
        if args.one_sided:
            lira = np.where(obs >= mu, lira, -lira)
        auc = roc_auc_score(y_true, lira)
        acc = accuracy_score(y_true, (lira >= args.threshold).astype(int))
        row = {
            "model": name,
            "attack": "offline_lira_neg_bce_gaussian_out",
            "auc": float(auc),
            "accuracy_at_threshold": float(acc),
            "threshold": args.threshold,
            "tpr_at_fpr_0.1pct": tpr_at_fpr(y_true, lira, 0.001),
            "tpr_at_fpr_1pct": tpr_at_fpr(y_true, lira, 0.01),
            "tpr_at_fpr_5pct": tpr_at_fpr(y_true, lira, 0.05),
            "num_shadows": int(shadow_scores.shape[0]),
            "shadow_train_size": args.shadow_train_size,
            "shadow_epochs": args.shadow_epochs,
            "score": "negative_mean_bce",
            "lira_statistic": "negative_out_log_likelihood",
            "center": args.center,
            "one_sided": args.one_sided,
            "checkpoint_kind": args.checkpoint_kind,
            "checkpoint": ckpt,
            "shadow_ids": shadow_ids,
        }
        rows.append(row)
        tag = args.checkpoint_kind
        save_json(OUT_DIR / f"result_lira_offline_{name}_{tag}.json", row)
        np.savez_compressed(
            OUT_DIR / f"scores_lira_offline_{name}_{tag}.npz",
            lira_scores=lira.astype(np.float32),
            z_scores=z.astype(np.float32),
            target_scores=obs.astype(np.float32),
            out_center=mu.astype(np.float32),
            out_std=sigma.astype(np.float32),
            labels=y_true.astype(np.int64),
            names=np.asarray(obs_names, dtype=object),
        )
        save_roc(y_true, lira, auc, name, tag)
        print(
            f"{name}: AUC={auc:.4f} ACC@{args.threshold:g}={acc:.4f} "
            f"TPR@1%FPR={row['tpr_at_fpr_1pct']:.4f} "
            f"TPR@0.1%FPR={row['tpr_at_fpr_0.1pct']:.4f}"
        )
        del target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(OUT_DIR / f"summary_lira_offline_{args.checkpoint_kind}.json", rows)
    with open(OUT_DIR / f"summary_lira_offline_{args.checkpoint_kind}.csv", "w", newline="") as f:
        keys = [k for k in rows[0] if k != "shadow_ids"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["all", "train-shadows", "score-shadows", "eval"], default="all")
    p.add_argument("--checkpoint-kind", choices=["best_val_macro_f1", "best_val_loss", "last", "canonical"], default="best_val_macro_f1")
    p.add_argument("--num-shadows", type=int, default=128)
    p.add_argument("--shadow-start", type=int, default=0)
    p.add_argument("--shadow-end", type=int, default=-1)
    p.add_argument("--shadow-train-size", type=int, default=10000)
    p.add_argument("--shadow-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--shadow-hflip", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sigma-floor", type=float, default=1e-3)
    p.add_argument("--center", choices=["median", "mean"], default="median")
    p.add_argument("--one-sided", action="store_true", help="Only reward scores above the OUT center; default matches two-sided OUT likelihood.")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--retrain-shadows", action="store_true")
    p.add_argument("--rescore", action="store_true")
    p.add_argument("--require-all-shadows", action="store_true")
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--device", default="")
    args = p.parse_args()
    if args.shadow_end < 0:
        args.shadow_end = args.num_shadows
    if not (0 <= args.shadow_start < args.shadow_end <= args.num_shadows):
        raise ValueError("--shadow-start/--shadow-end must define a valid range inside --num-shadows.")

    seed_all(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    config = vars(args).copy()
    config["root"] = str(ROOT)
    config["protocol"] = "Offline LiRA: per-sample OUT Gaussian from official-val shadow models, target score = -mean BCE, attack score = -log p_OUT."
    save_json(OUT_DIR / "a1_10_lira_offline_config.json", config)

    if args.mode in ["all", "train-shadows"]:
        train_shadows(args, device)
    if args.mode in ["all", "score-shadows"]:
        score_shadows(args, device)
    if args.mode in ["all", "eval"]:
        evaluate_targets(args, device)


if __name__ == "__main__":
    main()

