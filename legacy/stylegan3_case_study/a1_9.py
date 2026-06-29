#!/usr/bin/env python3
"""
a1_9: GMI-inspired StyleGAN3-prior model inversion for ResNet18 a1.

This adapts "The Secret Revealer" / GMI to the current CelebA-40 benchmark:
  - Use the trained conditional StyleGAN3 generator as the natural-image prior.
  - Freeze the downstream target model.
  - Optimize StyleGAN3 latent z so target_model(G(z, c)) matches a 40-dim
    attribute vector c.

It is not an identity-class reconstruction attack. It audits whether the
downstream model exposes reconstructable attribute/face information under a
strong white-box generator prior.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, utils as vutils
from torchvision.models.resnet import BasicBlock
from tqdm import tqdm

try:
    from opacus.validators import ModuleValidator
except ImportError:
    ModuleValidator = None


STYLEGAN3_MAIN = Path("/u/jliu80/priv-bench/stylegan3-main")
STYLEGAN_CKPT_ROOT = Path("/work/hdd/bcga/priv-bench/checkpoints")
STYLEGAN_DATA = Path("/work/hdd/bcga/priv-bench/datasets/stylegan3")
STYLEGAN_META = STYLEGAN_DATA / "metadata"

ROOT = Path("/work/hdd/bcga/priv-bench/downstream_models/resnet18_a1")
CKPT_DIR = ROOT / "checkpoints"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "attacks" / "gmi_stylegan3_inversion"

MEMBER_IMG_DIR = STYLEGAN_DATA / "paired_real_train_10k"
OFFICIAL_IMG_DIR = Path("/u/jliu80/priv-bench/data/celeba/img_align_celeba")
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
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def latest_stylegan_pkl():
    hits = glob.glob(str(STYLEGAN_CKPT_ROOT / "**" / "network-snapshot-*.pkl"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"No StyleGAN3 network-snapshot-*.pkl under {STYLEGAN_CKPT_ROOT}")
    return Path(sorted(hits, key=os.path.getmtime)[-1])


def load_stylegan(network_pkl, device):
    if not STYLEGAN3_MAIN.exists():
        raise FileNotFoundError(STYLEGAN3_MAIN)
    sys.path.insert(0, str(STYLEGAN3_MAIN))
    import dnnlib
    import legacy

    pkl = Path(network_pkl) if network_pkl else latest_stylegan_pkl()
    print(f"Using StyleGAN3 checkpoint: {pkl}")
    with dnnlib.util.open_url(str(pkl)) as f:
        nets = legacy.load_network_pkl(f)
    G = nets["G_ema"].to(device).eval()
    D = nets.get("D", None)
    if D is not None:
        D = D.to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)
    if D is not None:
        for p in D.parameters():
            p.requires_grad_(False)
    if G.c_dim != FEATURES:
        raise RuntimeError(f"Expected conditional StyleGAN3 c_dim={FEATURES}, got {G.c_dim}")
    return G, D, str(pkl)


def build_target(is_dp):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, FEATURES)
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    if is_dp:
        if ModuleValidator is None:
            raise ImportError("opacus is required to load C_dp_40.")
        model = ModuleValidator.fix(model)
        BasicBlock.forward = safe_basicblock_forward
    return model


def ckpt_path(name, kind):
    if kind == "canonical":
        return CKPT_DIR / f"resnet18_{name}.pth"
    return CKPT_DIR / f"resnet18_{name}_{kind}.pth"


def load_target(name, kind, device):
    model = build_target(name == "C_dp_40")
    path = ckpt_path(name, kind)
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device)
    state = {k.replace("_module.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval(), str(path)


def downstream_preprocess(img01, image_size):
    x = F.interpolate(img01, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def tv_loss(img):
    return (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean() + (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()


def attr_loss(logits, labels, mode, margin):
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(logits, labels)
    pos = labels * F.relu(margin - logits)
    neg = (1 - labels) * F.relu(margin + logits)
    return (pos + neg).mean()


def select_targets(args, source):
    if source == "member":
        records = load_json(META_DIR / "member_records.json")
    elif source == "nonmember":
        records = load_json(META_DIR / "nonmember_records.json")
    elif source == "official_val":
        records = load_json(STYLEGAN_META / "official_val_records_40.json")
    else:
        raise ValueError(source)
    items = list(records.items())
    random.Random(args.seed + {"member": 0, "nonmember": 17, "official_val": 31}[source]).shuffle(items)
    return items[: args.num_targets]


def select_all_targets(args):
    sources = [args.target_source] if args.target_source else args.target_sources
    return {source: select_targets(args, source) for source in sources}


def run_inversion(model_name, target_source, target_model, G, D, targets, args, device):
    labels_np = np.asarray([y for _, y in targets], dtype=np.float32)
    labels = torch.tensor(labels_np, dtype=torch.float32, device=device)
    best = {
        "bce": torch.full((len(targets),), float("inf"), device=device),
        "imgs": None,
        "z": None,
        "probs": None,
    }
    baseline = None

    for restart in range(args.restarts):
        z = torch.randn(len(targets), G.z_dim, device=device).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=args.lr) if args.optimizer == "adam" else None
        velocity = torch.zeros_like(z)
        for step in range(1, args.steps + 1):
            synth = G(z, labels, truncation_psi=args.truncation_psi, noise_mode=args.noise_mode)
            img01 = (synth + 1) / 2
            logits = target_model(downstream_preprocess(img01, args.image_size))
            loss_attr = attr_loss(logits, labels, args.loss, args.margin)
            if D is not None and args.lambda_d_prior > 0:
                d_out = D(synth, labels)
                if isinstance(d_out, (tuple, list)):
                    d_out = d_out[0]
                loss_d_prior = -d_out.float().mean()
            else:
                loss_d_prior = synth.new_tensor(0.0)
            loss_z = z.pow(2).mean()
            loss_tv = tv_loss(img01)
            loss = loss_attr + args.lambda_d_prior * loss_d_prior + args.lambda_z * loss_z + args.lambda_tv * loss_tv

            if args.optimizer == "adam":
                opt.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    opt.step()
                    if args.z_clip > 0:
                        z.clamp_(-args.z_clip, args.z_clip)
            else:
                if z.grad is not None:
                    z.grad.zero_()
                loss.backward()
                with torch.no_grad():
                    velocity_prev = velocity.clone()
                    velocity.mul_(args.momentum).add_(z.grad, alpha=args.lr)
                    z_next = z - velocity - args.momentum * (velocity - velocity_prev)
                    if args.z_clip > 0:
                        z_next.clamp_(-args.z_clip, args.z_clip)
                z = z_next.detach().requires_grad_(True)

            if step == 1 and restart == 0:
                with torch.no_grad():
                    baseline = ((synth + 1) / 2).clamp(0, 1).detach().cpu()
            if step % args.log_every == 0 or step == args.steps:
                print(
                    f"{model_name}/{target_source} restart={restart + 1}/{args.restarts} step={step}/{args.steps} "
                    f"loss={loss.item():.4f} attr={loss_attr.item():.4f} d_prior={loss_d_prior.item():.4f} "
                    f"z={loss_z.item():.4f} tv={loss_tv.item():.4f}"
                )

        with torch.no_grad():
            synth = G(z, labels, truncation_psi=args.truncation_psi, noise_mode=args.noise_mode)
            img01 = ((synth + 1) / 2).clamp(0, 1)
            logits = target_model(downstream_preprocess(img01, args.image_size))
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1)
            probs = torch.sigmoid(logits)
            update = bce < best["bce"]
            if best["imgs"] is None:
                best["imgs"] = img01.detach().clone()
                best["z"] = z.detach().clone()
                best["probs"] = probs.detach().clone()
                best["bce"] = bce.detach().clone()
            else:
                best["imgs"][update] = img01[update].detach()
                best["z"][update] = z[update].detach()
                best["probs"][update] = probs[update].detach()
                best["bce"][update] = bce[update].detach()

    return labels, best, baseline


class ImageRecords(Dataset):
    def __init__(self, records, img_dir, image_size, limit=0):
        items = list(records.items()) if isinstance(records, dict) else list(records)
        if limit and len(items) > limit:
            items = items[:limit]
        self.items = items
        self.img_dir = Path(img_dir)
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, _ = self.items[idx]
        return self.tf(Image.open(self.img_dir / fname).convert("RGB")), fname


def evaluator_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model.to(device).eval()


def extract_eval_features(model, dataset, batch_size, workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())
    feats, names = [], []
    with torch.no_grad():
        for x, fname in tqdm(loader, desc="NN features", leave=False):
            f = F.normalize(model(x.to(device, non_blocking=True)), dim=1)
            feats.append(f.cpu())
            names.extend(list(fname))
    return torch.cat(feats, 0), names


def nearest_neighbor_report(imgs, args, device):
    if args.skip_nn:
        return {}
    eval_model = evaluator_model(device)
    inv_ds = torch.utils.data.TensorDataset(downstream_preprocess(imgs.to(device), 224).cpu())
    inv_loader = DataLoader(inv_ds, batch_size=args.eval_batch_size)
    inv_feats = []
    with torch.no_grad():
        for (x,) in inv_loader:
            inv_feats.append(F.normalize(eval_model(x.to(device)), dim=1).cpu())
    inv_feats = torch.cat(inv_feats, 0)

    groups = {
        "member": (load_json(META_DIR / "member_records.json"), MEMBER_IMG_DIR, 0),
        "nonmember": (load_json(META_DIR / "nonmember_records.json"), OFFICIAL_IMG_DIR, 0),
        "official_val": (load_json(STYLEGAN_META / "official_val_records_40.json"), OFFICIAL_IMG_DIR, args.nn_public_size),
    }
    report = {}
    for group, (records, img_dir, limit) in groups.items():
        ds = ImageRecords(records, img_dir, 224, limit=limit)
        feats, names = extract_eval_features(eval_model, ds, args.eval_batch_size, args.num_workers, device)
        dist = torch.cdist(inv_feats, feats).numpy()
        idx = dist.argmin(axis=1)
        report[group] = {
            "mean_nn_l2": float(dist[np.arange(len(idx)), idx].mean()),
            "nearest": [{"name": names[int(i)], "dist": float(dist[j, i])} for j, i in enumerate(idx)],
        }
    return report


def nn_summary(nn):
    if not nn:
        return {}
    member = nn.get("member", {}).get("mean_nn_l2")
    nonmember = nn.get("nonmember", {}).get("mean_nn_l2")
    official_val = nn.get("official_val", {}).get("mean_nn_l2")
    out = {
        "mean_nn_l2_to_member_pool": member,
        "mean_nn_l2_to_nonmember_pool": nonmember,
        "mean_nn_l2_to_official_val_pool": official_val,
    }
    if member is not None and nonmember is not None:
        out["member_pool_minus_nonmember_pool_l2"] = float(member - nonmember)
        out["closer_to_member_pool"] = bool(member < nonmember)
    return out


def eval_generated(target_model, imgs, labels, args, device):
    with torch.no_grad():
        logits = target_model(downstream_preprocess(imgs.to(device), args.image_size))
        bce = F.binary_cross_entropy_with_logits(logits, labels.to(device), reduction="none").mean(1).cpu()
        probs = torch.sigmoid(logits).cpu()
    labels_cpu = labels.cpu()
    pred = (probs >= 0.5).float()
    attr_match = (pred == labels_cpu).float().mean(1)
    conf = (labels_cpu * probs + (1 - labels_cpu) * (1 - probs)).mean(1)
    return bce, probs, attr_match, conf


def save_images(imgs, baseline, targets, model_name, target_source, args):
    out = OUT_DIR / model_name / target_source
    out.mkdir(parents=True, exist_ok=True)
    for i, (src, _) in enumerate(targets):
        Image.fromarray((imgs[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).save(out / f"inversion_{i:03d}_{Path(src).stem}.png")
        if baseline is not None:
            Image.fromarray((baseline[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).save(out / f"baseline_{i:03d}_{Path(src).stem}.png")
    vutils.save_image(imgs, out / "grid_inversion.png", nrow=min(8, len(imgs)))
    if baseline is not None:
        vutils.save_image(baseline, out / "grid_generator_baseline.png", nrow=min(8, len(baseline)))


def source_summary(rows, nn):
    mean_bce = float(np.mean([r["target_bce"] for r in rows]))
    mean_base_bce = float(np.mean([r["baseline_bce"] for r in rows]))
    mean_match = float(np.mean([r["attribute_match"] for r in rows]))
    mean_base_match = float(np.mean([r["baseline_attribute_match"] for r in rows]))
    return {
        "mean_bce": mean_bce,
        "mean_baseline_bce": mean_base_bce,
        "mean_bce_gain_vs_baseline": float(mean_base_bce - mean_bce),
        "mean_attribute_match": mean_match,
        "mean_baseline_attribute_match": mean_base_match,
        "mean_attribute_match_gain_vs_baseline": float(mean_match - mean_base_match),
        "nearest_neighbor_summary": nn_summary(nn),
    }


def compare_sources(per_source):
    if "member" not in per_source or "nonmember" not in per_source:
        return {}
    m, n = per_source["member"], per_source["nonmember"]
    out = {
        "member_mean_bce_minus_nonmember_mean_bce": float(m["summary"]["mean_bce"] - n["summary"]["mean_bce"]),
        "member_baseline_bce_minus_nonmember_baseline_bce": float(m["summary"]["mean_baseline_bce"] - n["summary"]["mean_baseline_bce"]),
        "member_bce_gain_minus_nonmember_bce_gain": float(m["summary"]["mean_bce_gain_vs_baseline"] - n["summary"]["mean_bce_gain_vs_baseline"]),
        "member_attr_match_minus_nonmember_attr_match": float(m["summary"]["mean_attribute_match"] - n["summary"]["mean_attribute_match"]),
        "privacy_signal_note": "Negative member_mean_bce_minus_nonmember_mean_bce means member targets invert better/lower BCE.",
    }
    m_nn = m["summary"].get("nearest_neighbor_summary", {})
    n_nn = n["summary"].get("nearest_neighbor_summary", {})
    for key in ["member_pool_minus_nonmember_pool_l2", "mean_nn_l2_to_member_pool", "mean_nn_l2_to_nonmember_pool"]:
        if key in m_nn and key in n_nn:
            out[f"member_source_{key}_minus_nonmember_source_{key}"] = float(m_nn[key] - n_nn[key])
    return out


def run_one(model_name, G, targets_by_source, args, device):
    print("=" * 70)
    print(f"GMI-style inversion: {model_name}")
    target_model, ckpt = load_target(model_name, args.checkpoint_kind, device)
    per_source = {}
    flat_rows = []
    for target_source, targets in targets_by_source.items():
        print("-" * 70)
        print(f"{model_name}: target_source={target_source}")
        labels, best, baseline = run_inversion(model_name, target_source, target_model, G, args.D, targets, args, device)
        imgs = best["imgs"].detach().cpu()
        bce, probs, attr_match, conf = eval_generated(target_model, imgs, labels, args, device)
        if baseline is not None:
            baseline_bce, _, baseline_attr_match, baseline_conf = eval_generated(target_model, baseline, labels, args, device)
        else:
            baseline_bce = torch.full_like(bce, float("nan"))
            baseline_attr_match = torch.full_like(attr_match, float("nan"))
            baseline_conf = torch.full_like(conf, float("nan"))
        rows = []
        for i, (src, _) in enumerate(targets):
            row = {
                "model": model_name,
                "target_source": target_source,
                "source": src,
                "target_bce": float(bce[i]),
                "baseline_bce": float(baseline_bce[i]),
                "bce_gain_vs_baseline": float(baseline_bce[i] - bce[i]),
                "attribute_match": float(attr_match[i]),
                "baseline_attribute_match": float(baseline_attr_match[i]),
                "attribute_match_gain_vs_baseline": float(attr_match[i] - baseline_attr_match[i]),
                "mean_target_confidence": float(conf[i]),
                "baseline_mean_target_confidence": float(baseline_conf[i]),
            }
            rows.append(row)
            flat_rows.append(row)
        save_images(imgs, baseline, targets, model_name, target_source, args)
        nn = nearest_neighbor_report(imgs, args, device)
        per_source[target_source] = {
            "target_source": target_source,
            "num_targets": len(targets),
            "summary": source_summary(rows, nn),
            "nearest_neighbor": nn,
            "rows": rows,
        }
        save_json(OUT_DIR / model_name / target_source / "result_gmi_stylegan3_inversion.json", per_source[target_source])
        with open(OUT_DIR / model_name / target_source / "per_target_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        s = per_source[target_source]["summary"]
        print(
            f"{model_name}/{target_source}: mean_bce={s['mean_bce']:.4f} "
            f"baseline_bce={s['mean_baseline_bce']:.4f} gain={s['mean_bce_gain_vs_baseline']:.4f} "
            f"attr_match={s['mean_attribute_match']:.4f}"
        )

    result = {
        "model": model_name,
        "checkpoint": ckpt,
        "checkpoint_kind": args.checkpoint_kind,
        "target_sources": list(targets_by_source.keys()),
        "per_source": per_source,
        "member_vs_nonmember": compare_sources(per_source),
        "rows": flat_rows,
    }
    save_json(OUT_DIR / model_name / "result_gmi_stylegan3_inversion.json", result)
    with open(OUT_DIR / model_name / "per_target_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_rows)
    cmp = result["member_vs_nonmember"]
    if cmp:
        print(
            f"{model_name}: member_bce_minus_nonmember_bce="
            f"{cmp['member_mean_bce_minus_nonmember_mean_bce']:.4f} "
            f"member_gain_minus_nonmember_gain={cmp['member_bce_gain_minus_nonmember_bce_gain']:.4f}"
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--checkpoint-kind", choices=["best_val_macro_f1", "best_val_loss", "last", "canonical"], default="best_val_macro_f1")
    p.add_argument("--network-pkl", default="")
    p.add_argument("--target-source", choices=["", "member", "nonmember", "official_val"], default="", help="Optional legacy single source. Empty means use --target-sources.")
    p.add_argument("--target-sources", nargs="+", choices=["member", "nonmember", "official_val"], default=["member", "nonmember"])
    p.add_argument("--num-targets", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--optimizer", choices=["adam", "momentum"], default="adam")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--loss", choices=["bce", "margin"], default="bce")
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--lambda-z", type=float, default=1e-4)
    p.add_argument("--lambda-tv", type=float, default=1e-5)
    p.add_argument("--lambda-d-prior", type=float, default=0.0, help="Optional GMI-style discriminator prior weight: -D(G(z), c).")
    p.add_argument("--z-clip", type=float, default=2.5)
    p.add_argument("--truncation-psi", type=float, default=0.7)
    p.add_argument("--noise-mode", choices=["const", "random", "none"], default="const")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--nn-public-size", type=int, default=2000)
    p.add_argument("--skip-nn", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--device", default="")
    args = p.parse_args()

    seed_all(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    G, D, g_ckpt = load_stylegan(args.network_pkl, device)
    args.D = D
    targets_by_source = select_all_targets(args)
    save_json(OUT_DIR / "selected_targets.json", {src: {k: v for k, v in targets} for src, targets in targets_by_source.items()})
    config = {k: v for k, v in vars(args).items() if k != "D"}
    save_json(OUT_DIR / "config_gmi_stylegan3_inversion.json", {**config, "stylegan3_checkpoint": g_ckpt, "stylegan3_discriminator_loaded": D is not None})
    results = [run_one(m, G, targets_by_source, args, device) for m in args.models]
    save_json(OUT_DIR / "summary_gmi_stylegan3_inversion.json", results)


if __name__ == "__main__":
    main()
