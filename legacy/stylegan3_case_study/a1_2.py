#!/usr/bin/env python3
"""
a1_2: Train ResNet18 downstream models with validation-selected checkpoints.

This branch is intentionally separate from a_2:
  - a_2  kept last-epoch checkpoints and is useful as overfitting sensitivity.
  - a1_2 saves best_val_loss, best_val_macro_f1, and last checkpoints.

Models:
  A_real_40  : ImageNet-pretrained ResNet18 full fine-tuning on real member images
  B_synth_40 : ImageNet-pretrained ResNet18 full fine-tuning on synthetic images
  C_dp_40    : ImageNet-pretrained ResNet18 full fine-tuning on real member images with DP-SGD

Official val is used only for checkpoint selection and monitoring.
Final utility belongs in a1_3.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.resnet import BasicBlock
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


STYLEGAN3_DIR = Path("/work/hdd/bcga/priv-bench/datasets/stylegan3")
STYLEGAN3_META = STYLEGAN3_DIR / "metadata"
REAL_DIR = STYLEGAN3_DIR / "paired_real_train_10k"
SYNTH_DIR = STYLEGAN3_DIR / "paired_synthetic_train_10k"
OFFICIAL_IMG_DIR = Path("/u/jliu80/priv-bench/data/celeba/img_align_celeba")

OUT_ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/resnet18_a1")
CKPT_DIR = OUT_ROOT / "checkpoints"
META_DIR = OUT_ROOT / "metadata"
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


def patch_resnet_no_inplace_residual(model):
    BasicBlock.forward = safe_basicblock_forward
    return model


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


class MultiLabelDataset(Dataset):
    def __init__(self, records_or_json, img_dir, image_size):
        if isinstance(records_or_json, (str, Path)):
            self.records = list(load_json(Path(records_or_json)).items())
        else:
            self.records = list(records_or_json)
        self.img_dir = Path(img_dir)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fname, labels = self.records[idx]
        image = Image.open(self.img_dir / fname).convert("RGB")
        return self.transform(image), torch.tensor(labels, dtype=torch.float32)


def build_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, FEATURES)
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return patch_resnet_no_inplace_residual(model)


def clean_state_dict(model):
    raw = model._module.state_dict() if hasattr(model, "_module") else model.state_dict()
    return {k.replace("_module.", "").replace("module.", ""): v for k, v in raw.items()}


def save_model_state(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean_state_dict(model), path)


def finite_module_params(model):
    state = clean_state_dict(model)
    return [k for k, v in state.items() if not torch.isfinite(v).all()]


def finite_trainable_grads(model):
    for p in model.parameters():
        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def monitor(model, loader, criterion, device):
    model.eval()
    losses, accs, f1s = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                continue
            pred = (torch.sigmoid(logits) >= 0.5).float()
            y_true = y.cpu().numpy().astype(np.int64)
            y_pred = pred.cpu().numpy().astype(np.int64)
            losses.append(float(loss.detach().cpu()))
            accs.append(float((y_true == y_pred).mean()))
            f1s.append(float(f1_score(y_true, y_pred, average="macro", zero_division=0)))
    if not losses:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(accs)), float(np.mean(f1s))


def train_one(name, dataset, val_loader, args, is_dp=False):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 70)
    print(f"Training {name}: ImageNet-pretrained ResNet18 full fine-tuning, DP={is_dp}")

    model = build_resnet18()
    if is_dp:
        model = ModuleValidator.fix(model)
        model = patch_resnet_no_inplace_residual(model)
        errors = ModuleValidator.validate(model, strict=False)
        print(f"ModuleValidator warnings/errors count: {len(errors)}")
    model = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    privacy_engine = None
    if is_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=args.dp_noise_multiplier,
            max_grad_norm=args.dp_max_grad_norm,
            poisson_sampling=args.dp_poisson_sampling,
        )

    best_loss, best_f1 = float("inf"), -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        skipped_bad = skipped_grad = 0
        for x, y in tqdm(loader, desc=f"{name} epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                skipped_bad += 1
                continue
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                skipped_bad += 1
                continue
            loss.backward()
            if not finite_trainable_grads(model):
                optimizer.zero_grad(set_to_none=True)
                skipped_grad += 1
                continue
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_loss, val_acc, val_f1 = monitor(model, val_loader, criterion, device)
        bad_params = finite_module_params(model)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": val_loss,
            "val_micro_acc_monitor": val_acc,
            "val_macro_f1_monitor": val_f1,
            "used_batches": len(losses),
            "skipped_nonfinite_batches": skipped_bad,
            "skipped_nonfinite_grad_batches": skipped_grad,
            "bad_param_count": len(bad_params),
            "batch_size": args.batch_size,
            "dp": is_dp,
        }
        history.append(row)
        print(
            f"[{name}] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
            f"val_micro_acc={val_acc:.4f} val_macro_f1={val_f1:.4f} "
            f"used={len(losses)} skipped_bad={skipped_bad} skipped_grad={skipped_grad} "
            f"bad_params={len(bad_params)}"
        )
        if bad_params:
            raise RuntimeError(f"{name} has non-finite parameters after epoch {epoch}: {bad_params[:5]}")
        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss = val_loss
            save_model_state(model, CKPT_DIR / f"resnet18_{name}_best_val_loss.pth")
        if np.isfinite(val_f1) and val_f1 > best_f1:
            best_f1 = val_f1
            save_model_state(model, CKPT_DIR / f"resnet18_{name}_best_val_macro_f1.pth")

    save_model_state(model, CKPT_DIR / f"resnet18_{name}_last.pth")
    save_model_state(model, CKPT_DIR / f"resnet18_{name}.pth")
    save_json(META_DIR / f"history_resnet18_{name}.json", history)

    if is_dp:
        epsilon = privacy_engine.get_epsilon(delta=args.dp_delta)
        save_json(
            META_DIR / "dp_metadata_resnet18_C_dp_40.json",
            {
                "epsilon": float(epsilon),
                "delta": args.dp_delta,
                "noise_multiplier": args.dp_noise_multiplier,
                "max_grad_norm": args.dp_max_grad_norm,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "poisson_sampling": args.dp_poisson_sampling,
                "protocol": "ImageNet-pretrained ResNet18 full-model DP-SGD fine-tuning; NaN/Inf guarded",
            },
        )
        print(f"[{name}] DP epsilon={epsilon:.4f} delta={args.dp_delta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--models", nargs="+", default=["A_real_40", "B_synth_40", "C_dp_40"], choices=["A_real_40", "B_synth_40", "C_dp_40"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--nonmember-size", type=int, default=10000)
    parser.add_argument("--dp-noise-multiplier", type=float, default=1.0)
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-poisson-sampling", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    real_json = STYLEGAN3_META / "paired_real_train_10k_labels_40.json"
    synth_json = STYLEGAN3_META / "paired_synthetic_train_10k_labels_40.json"
    val_json = STYLEGAN3_META / "official_val_records_40.json"
    test_json = STYLEGAN3_META / "official_test_records_40.json"
    for path in [real_json, synth_json, val_json, test_json, REAL_DIR, SYNTH_DIR, OFFICIAL_IMG_DIR]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    real_ds = MultiLabelDataset(real_json, REAL_DIR, args.image_size)
    synth_ds = MultiLabelDataset(synth_json, SYNTH_DIR, args.image_size)
    val_ds = MultiLabelDataset(val_json, OFFICIAL_IMG_DIR, args.image_size)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_records = list(load_json(test_json).items())
    rng = random.Random(args.seed)
    rng.shuffle(test_records)
    save_json(META_DIR / "member_records.json", dict(real_ds.records))
    save_json(META_DIR / "synth_train_records.json", dict(synth_ds.records))
    save_json(META_DIR / "nonmember_records.json", dict(test_records[: args.nonmember_size]))
    save_json(
        META_DIR / "a1_2_resnet18_train_config.json",
        {
            "architecture": "torchvision.models.resnet18",
            "output_root": str(OUT_ROOT),
            "protocol": "ImageNet-pretrained ResNet18 full fine-tuning; C uses full-model DP-SGD; save best/last checkpoints",
            "A_real_40": "trained on paired real member images",
            "B_synth_40": "trained on paired synthetic images",
            "C_dp_40": "trained on paired real member images with DP-SGD",
            "validation": "official val for checkpoint selection and monitoring only",
            "checkpoint_policy": "save last, best_val_loss, best_val_macro_f1; canonical resnet18_NAME.pth is last",
            "nonmember_records": "fixed official-test sample for later attacks",
            "models": args.models,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "dp_poisson_sampling": args.dp_poisson_sampling,
        },
    )

    jobs = {
        "A_real_40": (real_ds, False),
        "B_synth_40": (synth_ds, False),
        "C_dp_40": (real_ds, True),
    }
    for name in args.models:
        dataset, is_dp = jobs[name]
        train_one(name, dataset, val_loader, args, is_dp=is_dp)


if __name__ == "__main__":
    main()
