#!/usr/bin/env python3
"""
b_8: Feature-based attribute inference attack for Swin-T.

Goal:
  Test whether the downstream model representation leaks the hidden sensitive
  attribute Male beyond the non-sensitive attribute correlations.

Protocol:
  attack train = official CelebA val records from a_1 metadata
  attack test  = nonmember_records.json saved by b_2
  target attr  = Male by default

Attack inputs:
  prior_only         : true non-sensitive 39 labels
  prob_only          : target model non-sensitive 39 output probabilities
  feature_only       : target model penultimate feature vector
  feature_plus_prior : penultimate feature vector + true non-sensitive labels

The main privacy signal is whether feature_plus_prior beats prior_only.
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
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
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
ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/swim_transformer")
CKPT_DIR = ROOT / "checkpoints"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "attacks" / "attribute_inference"
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
        image = Image.open(IMG_DIR / fname).convert("RGB")
        return self.tf(image), torch.tensor(y, dtype=torch.float32)


def build_model(is_dp):
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, 40)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head.")
    if is_dp:
        if ModuleValidator is None:
            raise ImportError("opacus is required to load the C_dp_40 checkpoint architecture.")
        model = ModuleValidator.fix(model)
    return model


def load_model(name, device):
    model = build_model(name == "C_dp_40")
    state = torch.load(CKPT_DIR / f"swin_t_{name}.pth", map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def swin_features_and_logits(model, x):
    z = model.features(x)
    z = model.norm(z)
    z = model.permute(z)
    z = model.avgpool(z)
    feat = model.flatten(z)
    logits = model.head(feat)
    return feat, logits


def extract(model, loader, device):
    labels, probs, feats = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="extract", leave=False):
            x = x.to(device, non_blocking=True)
            feat, logits = swin_features_and_logits(model, x)
            labels.append(y.numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            feats.append(feat.cpu().numpy())
    return np.vstack(labels).astype(int), np.vstack(probs).astype(np.float32), np.vstack(feats).astype(np.float32)


def features_for_mode(labels, probs, feats, target_idx, mode):
    prior = np.delete(labels, target_idx, axis=1).astype(np.float32)
    prob = np.delete(probs, target_idx, axis=1).astype(np.float32)
    if mode == "prior_only":
        return prior
    if mode == "prob_only":
        return prob
    if mode == "feature_only":
        return feats
    if mode == "feature_plus_prior":
        return np.concatenate([feats, prior], axis=1)
    raise ValueError(mode)


def metric_dict(y, score):
    pred = (score >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def bootstrap(y, score, n_boot, seed):
    base = metric_dict(y, score)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in base}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        row = metric_dict(y[idx], score[idx])
        for k, v in row.items():
            samples[k].append(v)
    out = {}
    for k, v in base.items():
        arr = np.asarray(samples[k], dtype=float)
        out[k] = {"value": v, "se": float(arr.std(ddof=1)), "ci95_low": float(np.percentile(arr, 2.5)), "ci95_high": float(np.percentile(arr, 97.5))}
    return out


def train_attacker(kind, x_train, y_train, args):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=args.logreg_max_iter, class_weight="balanced", solver="liblinear", random_state=args.seed),
        ).fit(x_train, y_train)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=args.n_jobs,
        ).fit(x_train, y_train)
    raise ValueError(kind)


def flat(row):
    out = {k: v for k, v in row.items() if k != "metrics"}
    for k, s in row["metrics"].items():
        out[k] = s["value"]
        out[f"{k}_se"] = s["se"]
        out[f"{k}_ci95_low"] = s["ci95_low"]
        out[f"{k}_ci95_high"] = s["ci95_high"]
    return out


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_one(name, train_loader, test_loader, target_idx, args, device):
    print("=" * 70)
    print(f"Feature-based AIA: {name}")
    model = load_model(name, device)
    y_train_all, p_train, f_train = extract(model, train_loader, device)
    y_test_all, p_test, f_test = extract(model, test_loader, device)
    y_train, y_test = y_train_all[:, target_idx], y_test_all[:, target_idx]
    majority_score = np.full(len(y_test), float(y_train.mean()))

    rows = []
    for mode in ["prior_only", "prob_only", "feature_only", "feature_plus_prior"]:
        x_train = features_for_mode(y_train_all, p_train, f_train, target_idx, mode)
        x_test = features_for_mode(y_test_all, p_test, f_test, target_idx, mode)
        for attack in ["logreg", "rf"]:
            clf = train_attacker(attack, x_train, y_train, args)
            score = clf.predict_proba(x_test)[:, 1]
            metrics = bootstrap(y_test, score, args.bootstrap, args.seed)
            rows.append(
                {
                    "target_model": name,
                    "target_attribute": ATTRS[target_idx],
                    "target_attribute_index": target_idx,
                    "mode": mode,
                    "attack": attack,
                    "num_features": int(x_train.shape[1]),
                    "train_samples": int(len(y_train)),
                    "test_samples": int(len(y_test)),
                    "majority_train_positive_rate": float(y_train.mean()),
                    "majority_test_positive_rate": float(y_test.mean()),
                    "majority_metrics": metric_dict(y_test, majority_score),
                    "metrics": metrics,
                }
            )
            print(f"{name} {mode:<18} {attack:<6} AUROC={metrics['auroc']['value']:.4f} AUPRC={metrics['auprc']['value']:.4f} BalAcc={metrics['balanced_accuracy']['value']:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-attribute", default="Male")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--min-samples-leaf", type=int, default=5)
    ap.add_argument("--logreg-max-iter", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    seed_everything(args.seed)
    target_idx = ATTRS.index(args.target_attribute)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_records = load_json(STYLEGAN3_META / "official_val_records_40.json")
    test_records = load_json(META_DIR / "nonmember_records.json")
    train_loader = DataLoader(Records(train_records, args.image_size), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(Records(test_records, args.image_size), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    rows = []
    for name in MODELS:
        rows.extend(run_one(name, train_loader, test_loader, target_idx, args, device))

    save_json(OUT_DIR / "b_8_swin_feature_aia_results.json", rows)
    write_csv(OUT_DIR / "b_8_swin_feature_aia_summary.csv", [flat(r) for r in rows])
    print("=" * 70)
    print(f"Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
