#!/usr/bin/env python3
"""
b1_8: Flexible feature-based AIA for the b1 Swin-T branch.

Attack train = official CelebA val records.
Attack test  = fixed nonmember_records.json from b1_2.

Choose hidden sensitive attributes with:
  --sensitive-attrs Eyeglasses
  --sensitive-attrs Male,Young
  --sensitive-attrs 1,2
  --sensitive-attrs 7,8,10

Inputs:
  prior_only         : true labels except hidden sensitive attributes
  prob_only          : target model output probabilities except hidden attrs
  feature_only       : target model penultimate feature vector
  feature_plus_prior : penultimate feature vector + true non-sensitive labels

Default target checkpoint:
  --checkpoint-kind best_val_macro_f1
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import swin_t
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

try:
    from opacus.validators import ModuleValidator
except ImportError:
    ModuleValidator = None


STYLEGAN3_META = Path("/work/hdd/bcga/priv-bench/datasets/stylegan3/metadata")
ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/swin_transformer_b1")
CKPT_DIR = ROOT / "checkpoints"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "attacks" / "attribute_inference_flexible"
IMG_DIR = Path("/u/jliu80/priv-bench/data/celeba/img_align_celeba")
MODELS = ["A_real_40", "B_synth_40", "C_dp_40"]

ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry",
    "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses",
    "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
    "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face",
    "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns",
    "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat",
    "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]


def seed_all(seed):
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


def parse_attrs(text):
    out = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if item.isdigit():
            idx = int(item)
            if idx < 0 or idx >= len(ATTRS):
                raise ValueError(f"Attribute index out of range: {idx}")
        else:
            if item not in ATTRS:
                raise ValueError(f"Unknown attribute: {item}. Valid names: {ATTRS}")
            idx = ATTRS.index(item)
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError("No sensitive attributes were provided.")
    return out


class Records(Dataset):
    def __init__(self, records, image_size):
        self.items = list(records.items())
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
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
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, 40)
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


def extract(model, loader, device):
    ys, ps, fs = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="extract", leave=False):
            x = x.to(device, non_blocking=True)
            feat, logits = swin_features_and_logits(model, x)
            ys.append(y.numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
            fs.append(feat.cpu().numpy())
    return np.vstack(ys).astype(int), np.vstack(ps).astype(np.float32), np.vstack(fs).astype(np.float32)


def make_inputs(labels, probs, feats, sensitive_idx, mode):
    keep = [i for i in range(labels.shape[1]) if i not in sensitive_idx]
    prior = labels[:, keep].astype(np.float32)
    prob = probs[:, keep].astype(np.float32)
    if mode == "prior_only":
        return prior
    if mode == "prob_only":
        return prob
    if mode == "feature_only":
        return feats
    if mode == "feature_plus_prior":
        return np.concatenate([feats, prior], axis=1)
    raise ValueError(mode)


def train_attacker(kind, x, y, args):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=args.logreg_max_iter,
                solver="liblinear",
                class_weight="balanced",
                random_state=args.seed,
            ),
        ).fit(x, y)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        ).fit(x, y)
    raise ValueError(kind)


def safe_metrics(y, score):
    pred = (score >= 0.5).astype(int)
    out = {
        "auprc": float(average_precision_score(y, score)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y, score))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def bootstrap(y, score, n_boot, seed):
    base = safe_metrics(y, score)
    if n_boot <= 0:
        return {k: {"value": v, "se": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")} for k, v in base.items()}
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in base}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        row = safe_metrics(y[idx], score[idx])
        for k, v in row.items():
            samples[k].append(v)
    out = {}
    for k, v in base.items():
        arr = np.asarray(samples[k], dtype=float)
        out[k] = {
            "value": v,
            "se": float(np.nanstd(arr, ddof=1)),
            "ci95_low": float(np.nanpercentile(arr, 2.5)),
            "ci95_high": float(np.nanpercentile(arr, 97.5)),
        }
    return out


def flatten(row):
    out = {k: v for k, v in row.items() if k != "metrics"}
    for k, stat in row["metrics"].items():
        out[k] = stat["value"]
        out[f"{k}_se"] = stat["se"]
        out[f"{k}_ci95_low"] = stat["ci95_low"]
        out[f"{k}_ci95_high"] = stat["ci95_high"]
    return out


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_model(name, train_loader, test_loader, sensitive_idx, args, device):
    print("=" * 70)
    print(f"Flexible AIA: {name}")
    print("Hidden attrs:", [(i, ATTRS[i]) for i in sensitive_idx])
    model, ckpt = load_model(name, args.checkpoint_kind, device)
    print(f"Checkpoint: {ckpt}")
    y_train_all, p_train, f_train = extract(model, train_loader, device)
    y_test_all, p_test, f_test = extract(model, test_loader, device)

    rows = []
    for mode in args.modes.split(","):
        mode = mode.strip()
        x_train = make_inputs(y_train_all, p_train, f_train, sensitive_idx, mode)
        x_test = make_inputs(y_test_all, p_test, f_test, sensitive_idx, mode)
        for attr_idx in sensitive_idx:
            y_train = y_train_all[:, attr_idx]
            y_test = y_test_all[:, attr_idx]
            majority_score = np.full(len(y_test), float(y_train.mean()))
            for attack in args.attacks.split(","):
                attack = attack.strip()
                clf = train_attacker(attack, x_train, y_train, args)
                score = clf.predict_proba(x_test)[:, 1]
                metrics = bootstrap(y_test, score, args.bootstrap, args.seed + attr_idx)
                majority = safe_metrics(y_test, majority_score)
                row = {
                    "target_model": name,
                    "checkpoint_kind": args.checkpoint_kind,
                    "checkpoint": ckpt,
                    "hidden_attr_index": attr_idx,
                    "hidden_attr_name": ATTRS[attr_idx],
                    "hidden_attr_set": ",".join(ATTRS[i] for i in sensitive_idx),
                    "mode": mode,
                    "attack": attack,
                    "num_features": int(x_train.shape[1]),
                    "train_samples": int(len(y_train)),
                    "test_samples": int(len(y_test)),
                    "train_positive_rate": float(y_train.mean()),
                    "test_positive_rate": float(y_test.mean()),
                    "majority_auroc": majority["auroc"],
                    "majority_auprc": majority["auprc"],
                    "majority_balanced_accuracy": majority["balanced_accuracy"],
                    "metrics": metrics,
                }
                rows.append(row)
                print(
                    f"{name} {ATTRS[attr_idx]:<20} {mode:<18} {attack:<6} "
                    f"AUROC={metrics['auroc']['value']:.4f} "
                    f"AUPRC={metrics['auprc']['value']:.4f} "
                    f"BalAcc={metrics['balanced_accuracy']['value']:.4f}"
                )
    return rows


def macro_rows(rows):
    grouped = {}
    for r in rows:
        key = (r["target_model"], r["checkpoint_kind"], r["hidden_attr_set"], r["mode"], r["attack"])
        grouped.setdefault(key, []).append(r)
    out = []
    for (model, ckpt_kind, attr_set, mode, attack), group in grouped.items():
        macro = {
            "target_model": model,
            "checkpoint_kind": ckpt_kind,
            "checkpoint": group[0]["checkpoint"],
            "hidden_attr_index": "macro",
            "hidden_attr_name": "macro",
            "hidden_attr_set": attr_set,
            "mode": mode,
            "attack": attack,
            "num_features": group[0]["num_features"],
            "train_samples": group[0]["train_samples"],
            "test_samples": group[0]["test_samples"],
        }
        metrics = {}
        for metric in ["auroc", "auprc", "accuracy", "balanced_accuracy", "f1"]:
            vals = [g["metrics"][metric]["value"] for g in group]
            metrics[metric] = {
                "value": float(np.nanmean(vals)),
                "se": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
            }
        macro["metrics"] = metrics
        out.append(macro)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-kind", choices=["best_val_macro_f1", "best_val_loss", "last", "canonical"], default="best_val_macro_f1")
    ap.add_argument("--sensitive-attrs", default="Eyeglasses", help="Comma-separated attr names or 0-based indices.")
    ap.add_argument("--modes", default="prior_only,prob_only,feature_only,feature_plus_prior")
    ap.add_argument("--attacks", default="logreg,rf")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--min-samples-leaf", type=int, default=5)
    ap.add_argument("--logreg-max-iter", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    seed_all(args.seed)
    sensitive_idx = parse_attrs(args.sensitive_attrs)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_records = load_json(STYLEGAN3_META / "official_val_records_40.json")
    test_records = load_json(META_DIR / "nonmember_records.json")
    train_loader = DataLoader(
        Records(train_records, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Records(test_records, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    for name in MODELS:
        rows.extend(run_model(name, train_loader, test_loader, sensitive_idx, args, device))
    rows_with_macro = rows + macro_rows(rows)

    suffix = "_".join(ATTRS[i] for i in sensitive_idx)
    tag = f"{args.checkpoint_kind}_{suffix}"
    save_json(OUT_DIR / f"b1_8_swin_flexible_aia_{tag}.json", rows_with_macro)
    write_csv(OUT_DIR / f"b1_8_swin_flexible_aia_{tag}.csv", [flatten(r) for r in rows_with_macro])
    print("=" * 70)
    print(f"Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
