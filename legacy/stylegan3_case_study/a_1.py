#!/usr/bin/env python3
"""
Short official-CelebA StyleGAN3 pipeline for Delta.

Paths are matched to the user's Delta layout:
  data root:
    /u/jliu80/priv-bench/data
  StyleGAN3 checkpoints:
    /work/hdd/bcga/priv-bench/checkpoints
  all other outputs:
    /work/hdd/bcga/priv-bench/datasets/stylegan3

Modes:
  --mode prepare   pack official train only into a StyleGAN3 zip
  --mode train     run StyleGAN3 training on that zip
  --mode sample    use a trained checkpoint to create 10k paired real/synth data
  --mode all       prepare + train + sample
"""

import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


DATA_DIR = Path("/u/jliu80/priv-bench/data")
IMG_DIR = DATA_DIR / "celeba" / "img_align_celeba"
ATTR_TXT = DATA_DIR / "celeba" / "list_attr_celeba.txt"
PARTITION_CSV = DATA_DIR / "list_eval_partition.csv"
STYLEGAN3_DIR = Path("/u/jliu80/priv-bench/stylegan3-main")

CKPT_DIR = Path("/work/hdd/bcga/priv-bench/checkpoints")
OUT_DIR = Path("/work/hdd/bcga/priv-bench/datasets/stylegan3")

EXPECTED = {0: 162770, 1: 19867, 2: 19962}


def read_attrs():
    with open(ATTR_TXT) as f:
        n = int(f.readline().strip())
        names = f.readline().split()
        attrs = {}
        for line in f:
            p = line.split()
            attrs[p[0]] = [1 if int(v) == 1 else 0 for v in p[1:41]]
    print(f"Loaded attributes: {len(attrs)} images, declared={n}, dims={len(names)}")
    return names, attrs


def read_partition():
    part = {}
    with open(PARTITION_CSV, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].lower() in {"image_id", "filename"}:
                continue
            part[row[0].strip()] = int(row[1])
    return part


def build_splits(attrs, part):
    splits = {0: [], 1: [], 2: []}
    for fname, split_id in part.items():
        if split_id not in splits:
            continue
        img_path = IMG_DIR / fname
        if fname not in attrs:
            raise RuntimeError(f"Missing attributes for {fname}")
        if not img_path.exists():
            raise RuntimeError(f"Missing image file: {img_path}")
        splits[split_id].append((fname, attrs[fname]))

    for k in splits:
        splits[k].sort()
        print(f"official split {k}: {len(splits[k])} expected={EXPECTED[k]}")
        if len(splits[k]) != EXPECTED[k]:
            raise RuntimeError(f"Official partition count mismatch for split {k}")
    return splits


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_records(path, records):
    save_json(path, {fname: labels for fname, labels in records})


def pack_zip(train_records, image_size, force):
    zip_path = OUT_DIR / f"celeba_official_train_cond_{image_size}.zip"
    if zip_path.exists() and not force:
        print(f"Reusing existing zip: {zip_path}")
        return zip_path

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_for_stylegan = []
    print(f"Packing official train only: {len(train_records)} images -> {zip_path}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for fname, labels in tqdm(train_records, desc="zip"):
            with Image.open(IMG_DIR / fname).convert("RGB") as im:
                im = im.resize((image_size, image_size), Image.LANCZOS)
                buf = BytesIO()
                out_name = f"{Path(fname).stem}.png"
                im.save(buf, format="PNG")
                zf.writestr(out_name, buf.getvalue())
            labels_for_stylegan.append([out_name, labels])
        zf.writestr("dataset.json", json.dumps({"labels": labels_for_stylegan}))

    return zip_path


def prepare(args):
    names, attrs = read_attrs()
    splits = build_splits(attrs, read_partition())
    meta = OUT_DIR / "metadata"
    save_json(meta / "celeba_attr_names_40.json", names)
    save_records(meta / "official_train_records_40.json", splits[0])
    save_records(meta / "official_val_records_40.json", splits[1])
    save_records(meta / "official_test_records_40.json", splits[2])
    zip_path = pack_zip(splits[0], args.image_size, args.force_repack)
    return splits, zip_path


def train(args, zip_path):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(STYLEGAN3_DIR / "train.py"),
        f"--outdir={CKPT_DIR}",
        "--cfg=stylegan3-t",
        f"--data={zip_path}",
        f"--gpus={args.gpus}",
        f"--batch={args.batch}",
        f"--gamma={args.gamma}",
        "--mirror=1",
        "--cond=1",
        f"--kimg={args.kimg}",
        "--snap=10",
        f"--metrics={args.metrics}",
    ]
    if args.resume:
        resume_pkl = Path(args.network_pkl) if args.network_pkl else latest_pkl()
        cmd.append(f"--resume={resume_pkl}")
        print(f"Resuming from: {resume_pkl}")
    print("Running:", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)


def latest_pkl():
    hits = glob.glob(str(CKPT_DIR / "**" / "network-snapshot-*.pkl"), recursive=True)
    if not hits:
        raise RuntimeError(f"No network-snapshot-*.pkl found under {CKPT_DIR}")
    hits = sorted(hits, key=lambda p: os.path.getmtime(p))
    return Path(hits[-1])


def sample(args, train_records):
    import torch

    sys.path.insert(0, str(STYLEGAN3_DIR))
    import dnnlib
    import legacy

    rng = random.Random(args.seed)
    picked = train_records[:]
    rng.shuffle(picked)
    picked = picked[: args.sample_size]

    real_dir = OUT_DIR / "paired_real_train_10k"
    synth_dir = OUT_DIR / "paired_synthetic_train_10k"
    meta = OUT_DIR / "metadata"
    real_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    pkl = Path(args.network_pkl) if args.network_pkl else latest_pkl()
    print(f"Using checkpoint: {pkl}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(str(pkl)) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    if G.c_dim != 40:
        raise RuntimeError(f"Expected c_dim=40, got {G.c_dim}")

    real_labels, synth_labels, pairs = {}, {}, []
    for i, (src_name, labels) in enumerate(tqdm(picked, desc="paired sample")):
        real_name = f"real_train_{i:05d}.png"
        synth_name = f"synth_train_cond_{i:05d}.png"

        with Image.open(IMG_DIR / src_name).convert("RGB") as im:
            im = im.resize((args.image_size, args.image_size), Image.LANCZOS)
            im.save(real_dir / real_name)

        z = torch.from_numpy(np.random.RandomState(args.seed + i).randn(1, G.z_dim)).float().to(device)
        c = torch.tensor([labels], dtype=torch.float32, device=device)
        with torch.no_grad():
            img = G(z, c, truncation_psi=args.truncation_psi, noise_mode="const")
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img, "RGB").save(synth_dir / synth_name)

        real_labels[real_name] = labels
        synth_labels[synth_name] = labels
        pairs.append({
            "source_official_train": src_name,
            "real_image": real_name,
            "synthetic_image": synth_name,
            "attributes_40dim": labels,
        })

    save_records(meta / "paired_real_train_10k_source_records_40.json", picked)
    save_json(meta / "paired_real_train_10k_labels_40.json", real_labels)
    save_json(meta / "paired_synthetic_train_10k_labels_40.json", synth_labels)
    save_json(meta / "paired_real_synthetic_train_10k_pairs.json", pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prepare", "train", "sample", "all"], default="all")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--sample-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260613)
    p.add_argument("--force-repack", action="store_true")
    p.add_argument("--network-pkl", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--kimg", type=int, default=20000)
    p.add_argument("--metrics", default="none")
    p.add_argument("--truncation-psi", type=float, default=0.7)
    args = p.parse_args()

    for path in [IMG_DIR, ATTR_TXT, PARTITION_CSV, STYLEGAN3_DIR]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    splits, zip_path = prepare(args)
    if args.mode in {"train", "all"}:
        train(args, zip_path)
    if args.mode in {"sample", "all"}:
        sample(args, splits[0])

    print("Done.")
    print(f"checkpoints: {CKPT_DIR}")
    print(f"datasets:    {OUT_DIR}")


if __name__ == "__main__":
    main()
