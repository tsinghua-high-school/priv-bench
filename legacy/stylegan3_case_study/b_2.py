"""
b_2: Train Swin-T downstream models, fixed head-only protocol.

All three models use the same pretrained frozen Swin-T backbone and train only
the 40-dim classification head:
  A_real_40  : head trained on paired real member images
  B_synth_40 : head trained on paired synthetic images
  C_dp_40    : head trained on paired real member images with DP-SGD

Official val is used only for training-time monitoring. Final utility belongs
in b_3. A fixed official-test subset is saved as nonmember_records.json for
later MIA scripts, but official test is not evaluated here.
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
from torchvision import transforms
from torchvision.models import Swin_T_Weights, swin_t
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


STYLEGAN3_DIR = Path("/work/hdd/bcga/priv-bench/datasets/stylegan3")
STYLEGAN3_META = STYLEGAN3_DIR / "metadata"
REAL_DIR = STYLEGAN3_DIR / "paired_real_train_10k"
SYNTH_DIR = STYLEGAN3_DIR / "paired_synthetic_train_10k"
OFFICIAL_IMG_DIR = Path("/u/jliu80/priv-bench/data/celeba/img_align_celeba")

OUT_ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/swim_transformer")
CKPT_DIR = OUT_ROOT / "checkpoints"
META_DIR = OUT_ROOT / "metadata"

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


class MultiLabelDataset(Dataset):
    def __init__(self, records_or_json, img_dir, image_size):
        self.records = list(load_json(Path(records_or_json)).items()) if isinstance(records_or_json, (str, Path)) else list(records_or_json)
        self.img_dir = Path(img_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fname, labels = self.records[idx]
        image = Image.open(self.img_dir / fname).convert("RGB")
        return self.transform(image), torch.tensor(labels, dtype=torch.float32)


def build_head_only_swin():
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, FEATURES)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head.")
    return model


def save_model_state(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model._module.state_dict() if hasattr(model, "_module") else model.state_dict(), path)


def monitor(model, loader, criterion, device):
    model.eval()
    losses, accs, f1s = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            y_true = y.cpu().numpy().astype(np.int64)
            y_pred = pred.cpu().numpy().astype(np.int64)
            losses.append(float(criterion(logits, y).detach().cpu()))
            accs.append(float((y_true == y_pred).mean()))
            f1s.append(float(f1_score(y_true, y_pred, average="macro", zero_division=0)))
    return float(np.mean(losses)), float(np.mean(accs)), float(np.mean(f1s))


def train_one(name, dataset, val_loader, args, is_dp=False):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = args.dp_batch_size if is_dp else args.batch_size
    print("=" * 70)
    print(f"Training {name}: pretrained frozen Swin-T backbone, head-only, DP={is_dp}")

    model = build_head_only_swin()
    if is_dp:
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
    model = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_dp,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    privacy_engine = None
    if is_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=args.dp_noise_multiplier,
            max_grad_norm=args.dp_max_grad_norm,
        )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(loader, desc=f"{name} epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_loss, val_acc, val_f1 = monitor(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_micro_acc_monitor": val_acc,
            "val_macro_f1_monitor": val_f1,
            "batch_size": batch_size,
        }
        history.append(row)
        print(
            f"[{name}] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_micro_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

    save_model_state(model, CKPT_DIR / f"swin_t_{name}.pth")
    save_json(META_DIR / f"history_swin_t_{name}.json", history)

    if is_dp:
        epsilon = privacy_engine.get_epsilon(delta=args.dp_delta)
        save_json(
            META_DIR / "dp_metadata_swin_t_C_dp_40.json",
            {
                "epsilon": float(epsilon),
                "delta": args.dp_delta,
                "noise_multiplier": args.dp_noise_multiplier,
                "max_grad_norm": args.dp_max_grad_norm,
                "epochs": args.epochs,
                "batch_size": batch_size,
                "protocol": "pretrained frozen Swin-T backbone; DP-SGD on classification head only",
            },
        )
        print(f"[{name}] DP epsilon={epsilon:.4f} delta={args.dp_delta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dp-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--nonmember-size", type=int, default=10000)
    parser.add_argument("--dp-noise-multiplier", type=float, default=1.0)
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
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
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    test_records = list(load_json(test_json).items())
    rng = random.Random(args.seed)
    rng.shuffle(test_records)
    save_json(META_DIR / "member_records.json", dict(real_ds.records))
    save_json(META_DIR / "synth_train_records.json", dict(synth_ds.records))
    save_json(META_DIR / "nonmember_records.json", dict(test_records[: args.nonmember_size]))
    save_json(
        META_DIR / "b_2_swim_transformer_train_config.json",
        {
            "architecture": "torchvision.models.swin_t",
            "protocol": "pretrained ImageNet Swin-T frozen backbone; train 40-dim head for all A/B/C",
            "A_real_40": "head trained on paired real member images",
            "B_synth_40": "head trained on paired synthetic images",
            "C_dp_40": "head trained on paired real member images with DP-SGD",
            "validation": "official val, training-time monitoring only",
            "nonmember_records": "fixed official-test sample for later MIA",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dp_batch_size": args.dp_batch_size,
            "image_size": args.image_size,
            "seed": args.seed,
        },
    )

    train_one("A_real_40", real_ds, val_loader, args, is_dp=False)
    train_one("B_synth_40", synth_ds, val_loader, args, is_dp=False)
    train_one("C_dp_40", real_ds, val_loader, args, is_dp=True)


if __name__ == "__main__":
    main()
