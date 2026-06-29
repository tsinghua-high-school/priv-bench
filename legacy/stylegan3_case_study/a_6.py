#!/usr/bin/env python3
"""
a_6: No-gradient white-box MIA for ResNet18 A/B/C.

Reads outputs from a_2:
  checkpoints: /work/hdd/bcga/priv-bench/downstream_models/resnet18/checkpoints
  metadata:    /work/hdd/bcga/priv-bench/downstream_models/resnet18/metadata

Candidate set:
  member    = member_records.json, images from paired_real_train_10k
  nonmember = nonmember_records.json, images from official CelebA

This is a supervised white-box attack model. It uses target-model activations,
logits, true labels, and per-sample BCE loss, but it does not use gradients.
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
from torchvision import models, transforms
from tqdm import tqdm

try:
    from opacus.validators import ModuleValidator
except ImportError:
    ModuleValidator = None


ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/resnet18")
CKPT_DIR = ROOT / "checkpoints"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "attacks" / "whitebox_nograd"
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
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y, m = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.tf(image), torch.tensor(y, dtype=torch.float32), int(m)


def build_model(is_dp):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, FEATURES)
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    if is_dp:
        if ModuleValidator is None:
            raise ImportError("opacus is required to load the C_dp_40 checkpoint architecture.")
        model = ModuleValidator.fix(model)
    return model


def load_model(name, device):
    model = build_model(name == "C_dp_40")
    state = torch.load(CKPT_DIR / f"resnet18_{name}.pth", map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def register_hooks(model):
    acts, handles = [], []

    def hook(_, __, output):
        acts.append(output.detach())

    for layer in [model.layer4[0].conv2, model.layer4[1].conv1, model.layer4[1].conv2, model.fc]:
        handles.append(layer.register_forward_hook(hook))
    return acts, handles


class MetaEncoder(nn.Module):
    def __init__(self, n_act=3, n_classes=FEATURES):
        super().__init__()
        self.act_encoders = nn.ModuleList(
            [nn.Sequential(nn.Flatten(), nn.LazyLinear(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64)) for _ in range(n_act)]
        )
        self.logit_encoder = nn.Sequential(nn.LazyLinear(64), nn.ReLU(), nn.Dropout(0.2))
        self.label_encoder = nn.Sequential(nn.Linear(n_classes, 64), nn.ReLU(), nn.Dropout(0.2))
        self.loss_encoder = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Dropout(0.2))
        self.final = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, acts, labels, loss):
        act_feats = [enc(a) for enc, a in zip(self.act_encoders, acts[:-1])]
        parts = act_feats + [self.logit_encoder(acts[-1]), self.label_encoder(labels.float()), self.loss_encoder(loss)]
        return self.final(torch.cat(parts, dim=1))


def warmup(encoder, target_model, device, image_size):
    encoder.eval()
    target_model.eval()
    x = torch.randn(2, 3, image_size, image_size, device=device)
    y = torch.randint(0, 2, (2, FEATURES), device=device).float()
    acts, handles = register_hooks(target_model)
    with torch.no_grad():
        logits = target_model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(dim=1, keepdim=True)
        encoder(acts, y, loss)
    for handle in handles:
        handle.remove()


def batch_attack_features(target_model, x, labels):
    acts, handles = register_hooks(target_model)
    with torch.no_grad():
        logits = target_model(x)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1, keepdim=True)
    for handle in handles:
        handle.remove()
    return acts, loss


def save_roc(y, score, auc, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"No-gradient white-box MIA: {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / f"roc_whitebox_nograd_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_attack(name, dataset, args, device):
    print("=" * 70)
    print(f"No-gradient white-box MIA: {name}")
    target_model = load_model(name, device)
    y = np.array([m for _, _, m in dataset.samples])
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=args.test_size, stratify=y, random_state=args.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    encoder = MetaEncoder().to(device)
    warmup(encoder, target_model, device, args.image_size)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        losses = []
        for x, labels, is_member in tqdm(train_loader, desc=f"{name} epoch {epoch}/{args.epochs}", leave=False):
            x, labels = x.to(device), labels.to(device)
            target = is_member.float().unsqueeze(1).to(device)
            acts, per_sample_loss = batch_attack_features(target_model, x, labels)
            pred = encoder(acts, labels, per_sample_loss)
            loss = F.binary_cross_entropy_with_logits(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[{name}] epoch {epoch:03d}/{args.epochs} attack_loss={np.mean(losses):.4f}")

    encoder.eval()
    scores, labels_all = [], []
    with torch.no_grad():
        for x, labels, is_member in tqdm(test_loader, desc=f"{name} eval", leave=False):
            x, labels = x.to(device), labels.to(device)
            acts, per_sample_loss = batch_attack_features(target_model, x, labels)
            scores.extend(torch.sigmoid(encoder(acts, labels, per_sample_loss)).cpu().numpy().ravel())
            labels_all.extend(is_member.numpy())

    auc = roc_auc_score(labels_all, scores)
    acc = accuracy_score(labels_all, (np.asarray(scores) >= 0.5).astype(int))
    save_roc(labels_all, scores, auc, name)
    result = {"model": name, "auc": float(auc), "accuracy": float(acc), "attack": "whitebox_nograd_metaencoder"}
    save_json(OUT_DIR / f"result_whitebox_nograd_{name}.json", result)
    print(f"{name}: AUC={auc:.4f} ACC={acc:.4f}")
    return result


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    members = load_json(META_DIR / "member_records.json")
    nonmembers = load_json(META_DIR / "nonmember_records.json")
    dataset = MIADataset(members, nonmembers, args.image_size)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_attack(name, dataset, args, device) for name in MODELS]
    save_json(OUT_DIR / "summary_whitebox_nograd.json", results)
    write_csv(OUT_DIR / "summary_whitebox_nograd.csv", results)


if __name__ == "__main__":
    main()
