#!/usr/bin/env python3
"""
a1_7: Paper-style full white-box MIA for the a1 ResNet18 branch.

This keeps the Nasr-style implementation from a_7:
  - balanced member/nonmember attack batches
  - activation components
  - selected per-sample gradient components
  - output-probability, label, and loss components
  - encoder-style final attack classifier
  - best attack checkpoint selected by held-out attack-test AUC

Reads a1_2 outputs:
  /work/hdd/bcga/priv-bench/downstream_models/resnet18_a1

Default target checkpoint:
  --checkpoint-kind best_val_macro_f1
"""

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.models.resnet import BasicBlock
from tqdm import tqdm

try:
    from opacus.validators import ModuleValidator
except ImportError:
    ModuleValidator = None


ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/resnet18_a1")
CKPT_DIR = ROOT / "checkpoints"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "attacks" / "whitebox_full"
MEMBER_IMG_DIR = Path("/work/hdd/bcga/priv-bench/datasets/stylegan3/paired_real_train_10k")
NONMEMBER_IMG_DIR = Path("/u/jliu80/priv-bench/data/celeba/img_align_celeba")
MODELS = ["A_real_40", "B_synth_40", "C_dp_40"]
DP_MODELS = {"C_dp_40"}
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class MIADataset(Dataset):
    def __init__(self, members, nonmembers, image_size):
        self.samples = [(MEMBER_IMG_DIR / f, y, 1) for f, y in members.items()]
        self.samples += [(NONMEMBER_IMG_DIR / f, y, 0) for f, y in nonmembers.items()]
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


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size, seed):
        if batch_size % 2:
            raise ValueError("--batch-size must be even.")
        self.member = [i for i, y in enumerate(labels) if int(y) == 1]
        self.nonmember = [i for i, y in enumerate(labels) if int(y) == 0]
        self.half = batch_size // 2
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        mem, non = self.member[:], self.nonmember[:]
        rng.shuffle(mem)
        rng.shuffle(non)
        for b in range(min(len(mem), len(non)) // self.half):
            batch = mem[b * self.half : (b + 1) * self.half] + non[b * self.half : (b + 1) * self.half]
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return min(len(self.member), len(self.nonmember)) // self.half


def split_indices(dataset, train_fraction, seed):
    member = [i for i, s in enumerate(dataset.samples) if s[2] == 1]
    nonmember = [i for i, s in enumerate(dataset.samples) if s[2] == 0]
    rng = random.Random(seed)
    rng.shuffle(member)
    rng.shuffle(nonmember)
    n_m, n_n = int(len(member) * train_fraction), int(len(nonmember) * train_fraction)
    train = member[:n_m] + nonmember[:n_n]
    test = member[n_m:] + nonmember[n_n:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def subset_labels(subset):
    base = subset.dataset
    return [int(base.samples[i][2]) for i in subset.indices]


def build_model(is_dp):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, FEATURES)
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    if is_dp:
        if ModuleValidator is None:
            raise ImportError("opacus is required to load C_dp_40.")
        model = ModuleValidator.fix(model)
    return model


def ckpt_path(name, kind):
    if kind == "canonical":
        return CKPT_DIR / f"resnet18_{name}.pth"
    return CKPT_DIR / f"resnet18_{name}_{kind}.pth"


def load_model(name, kind, device):
    model = build_model(name in DP_MODELS)
    path = ckpt_path(name, kind)
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(True)
    return model.to(device).eval(), str(path)


def resolve_module(root, dotted):
    cur = root
    for part in dotted.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


ACTIVATION_LAYERS = ["layer3.1.conv2", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"]
GRADIENT_PARAMS = ["layer4.0.conv2.weight", "layer4.1.conv1.weight", "layer4.1.conv2.weight", "fc.weight", "fc.bias"]


@dataclass
class AttackFeatures:
    activations: List[torch.Tensor]
    output_probs: torch.Tensor
    labels: torch.Tensor
    losses: torch.Tensor
    gradients: List[torch.Tensor]


class WhiteBoxFeatureExtractor:
    def __init__(self, model, activation_layers, gradient_params, max_grad_elements):
        self.model = model
        self.activation_layers = activation_layers
        named_params = dict(model.named_parameters())
        self.gradient_names = [n for n in gradient_params if n in named_params]
        self.gradient_params = [named_params[n] for n in self.gradient_names]
        self.max_grad_elements = max_grad_elements
        self.cache = {}
        if len(self.gradient_names) != len(gradient_params):
            missing = sorted(set(gradient_params) - set(self.gradient_names))
            print(f"Warning: missing gradient params skipped: {missing}")

    def activation_hooks(self):
        activations, handles = [], []

        def hook(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            activations.append(output.detach())

        for name in self.activation_layers:
            handles.append(resolve_module(self.model, name).register_forward_hook(hook))
        return activations, handles

    def downsample(self, name, grad):
        flat = grad.detach().flatten()
        if self.max_grad_elements <= 0 or flat.numel() <= self.max_grad_elements:
            return flat
        key = (name, flat.numel(), self.max_grad_elements, flat.device)
        if key not in self.cache:
            self.cache[key] = torch.linspace(0, flat.numel() - 1, self.max_grad_elements, device=flat.device).long()
        return flat.index_select(0, self.cache[key])

    def per_sample_gradients(self, x, labels):
        buckets = [[] for _ in self.gradient_params]
        for i in range(x.size(0)):
            self.model.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                self.model(x[i : i + 1]),
                labels[i : i + 1],
                reduction="mean",
            )
            grads = torch.autograd.grad(loss, self.gradient_params, retain_graph=False, create_graph=False)
            for j, grad in enumerate(grads):
                buckets[j].append(self.downsample(self.gradient_names[j], grad))
        self.model.zero_grad(set_to_none=True)
        return [torch.stack(v, dim=0) for v in buckets]

    def extract(self, x, labels):
        self.model.eval()
        with torch.no_grad():
            activations, handles = self.activation_hooks()
            logits = self.model(x)
            for h in handles:
                h.remove()
            losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1, keepdim=True)
            probs = torch.sigmoid(logits)
        gradients = self.per_sample_gradients(x, labels)
        return AttackFeatures(activations, probs.detach(), labels.detach(), losses.detach(), gradients)


class TwoLayerFC(nn.Module):
    def __init__(self, in_features=None, hidden=128, out=64, dropout=0.2):
        super().__init__()
        first = nn.Linear(in_features, hidden) if in_features is not None else nn.LazyLinear(hidden)
        self.net = nn.Sequential(first, nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out), nn.ReLU())

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return self.net(x.float())


class GradientComponent(nn.Module):
    def __init__(self, conv_kernels=1000, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, conv_kernels, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(conv_kernels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, grad):
        grad = grad.float()
        grad = grad / grad.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        return self.net(grad.unsqueeze(1))


class NasrWhiteBoxAttackModel(nn.Module):
    def __init__(self, n_classes, n_activations, n_gradients, dropout=0.2):
        super().__init__()
        self.output_component = TwoLayerFC(n_classes, dropout=dropout)
        self.label_component = TwoLayerFC(n_classes, dropout=dropout)
        self.loss_component = TwoLayerFC(1, dropout=dropout)
        self.activation_components = nn.ModuleList([TwoLayerFC(None, dropout=dropout) for _ in range(n_activations)])
        self.gradient_components = nn.ModuleList([GradientComponent(dropout=dropout) for _ in range(n_gradients)])
        self.encoder = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, f):
        parts = [
            self.output_component(f.output_probs),
            self.label_component(f.labels),
            self.loss_component(f.losses),
        ]
        parts += [m(a) for m, a in zip(self.activation_components, f.activations)]
        parts += [m(g) for m, g in zip(self.gradient_components, f.gradients)]
        return self.encoder(torch.cat(parts, dim=1))


def evaluate(attack_model, extractor, loader, device):
    attack_model.eval()
    scores, labels = [], []
    for x, y, m in tqdm(loader, desc="MIA evaluation", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        features = extractor.extract(x, y)
        with torch.no_grad():
            scores.extend(torch.sigmoid(attack_model(features)).cpu().numpy().reshape(-1))
        labels.extend(np.asarray(m).reshape(-1).tolist())
    return np.asarray(labels), np.asarray(scores)


def save_roc(labels, scores, auc, model_name, tag):
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"Paper-style full white-box MIA: {model_name}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / f"roc_whitebox_full_{model_name}_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_one(model_name, dataset, args, device):
    print("=" * 70)
    print(f"Paper-style full white-box MIA: {model_name}")
    target_model, ckpt = load_model(model_name, args.checkpoint_kind, device)
    extractor = WhiteBoxFeatureExtractor(target_model, ACTIVATION_LAYERS, GRADIENT_PARAMS, args.max_grad_elements)
    print("Checkpoint:", ckpt)
    print("Activation layers:", ACTIVATION_LAYERS)
    print("Gradient params:", extractor.gradient_names)

    train_idx, test_idx = split_indices(dataset, args.attack_train_fraction, args.seed)
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
    train_loader = DataLoader(
        train_set,
        batch_sampler=BalancedBatchSampler(subset_labels(train_set), args.batch_size, args.seed),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    attack_model = NasrWhiteBoxAttackModel(FEATURES, len(ACTIVATION_LAYERS), len(extractor.gradient_names), args.dropout).to(device)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=args.lr)
    best = {"auc": -math.inf, "epoch": -1, "accuracy": 0.0}
    tag = args.checkpoint_kind

    for epoch in range(1, args.epochs + 1):
        attack_model.train()
        losses = []
        for x, y, m in tqdm(train_loader, desc=f"{model_name} train {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            m = m.float().view(-1, 1).to(device, non_blocking=True)
            pred = attack_model(extractor.extract(x, y))
            loss = F.binary_cross_entropy_with_logits(pred, m)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        labels, scores = evaluate(attack_model, extractor, test_loader, device)
        auc = roc_auc_score(labels, scores)
        acc = accuracy_score(labels, (scores >= 0.5).astype(int))
        print(f"[{model_name}] epoch {epoch:03d}/{args.epochs} loss={np.mean(losses):.4f} auc={auc:.4f} acc={acc:.4f}")
        if auc > best["auc"]:
            best = {
                "model": model_name,
                "auc": float(auc),
                "accuracy": float(acc),
                "epoch": epoch,
                "attack": "paper_style_whitebox_full",
                "checkpoint_kind": args.checkpoint_kind,
                "checkpoint": ckpt,
                "activation_layers": ACTIVATION_LAYERS,
                "gradient_params": extractor.gradient_names,
            }
            torch.save(
                {
                    "model": model_name,
                    "epoch": epoch,
                    "auc": auc,
                    "attack_model": attack_model.state_dict(),
                    "activation_layers": ACTIVATION_LAYERS,
                    "gradient_params": extractor.gradient_names,
                    "checkpoint_kind": args.checkpoint_kind,
                    "checkpoint": ckpt,
                    "args": vars(args),
                },
                OUT_DIR / f"best_attack_{model_name}_{tag}.pt",
            )
            save_roc(labels, scores, auc, model_name, tag)
            save_json(OUT_DIR / f"result_whitebox_full_{model_name}_{tag}.json", best)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"{model_name}: best AUC={best['auc']:.4f} at epoch={best['epoch']}")
    return best


def write_csv(path, rows):
    keys = sorted({k for row in rows for k in row if not isinstance(row[k], list)})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k in keys})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-kind", choices=["best_val_macro_f1", "best_val_loss", "last", "canonical"], default="best_val_macro_f1")
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--attack-train-fraction", type=float, default=0.8)
    p.add_argument("--max-grad-elements", type=int, default=16384)
    p.add_argument("--seed", type=int, default=20260622)
    p.add_argument("--device", default="")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = MIADataset(
        load_json(META_DIR / "member_records.json"),
        load_json(META_DIR / "nonmember_records.json"),
        args.image_size,
    )
    results = [run_one(model_name, dataset, args, device) for model_name in args.models]
    tag = args.checkpoint_kind
    model_tag = "_".join(args.models)
    save_json(OUT_DIR / f"summary_whitebox_full_{tag}_{model_tag}.json", results)
    write_csv(OUT_DIR / f"summary_whitebox_full_{tag}_{model_tag}.csv", results)


if __name__ == "__main__":
    main()
