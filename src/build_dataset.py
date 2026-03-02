"""
Rebuilds data/processed/ from all raw sources and creates data/eval_by_source/
for per-source false positive evaluation.

Usage (from project root):
    python src/build_dataset.py
    python src/build_dataset.py --seed 42 --val_frac 0.1 --test_frac 0.1

WARNING: clears data/processed/ before rebuilding.
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp"}


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    )


def split_list(items: list, val_frac: float, test_frac: float, seed: int):
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    return items[n_test + n_val:], items[n_test:n_test + n_val], items[:n_test]


def copy_to(files: list[Path], dest: Path, prefix: str):
    dest.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        shutil.copy2(src, dest / f"{prefix}_{i:06d}{src.suffix.lower()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir",   type=Path, default=Path("data/raw"))
    ap.add_argument("--out_dir",   type=Path, default=Path("data/processed"))
    ap.add_argument("--eval_dir",  type=Path, default=Path("data/eval_by_source"))
    ap.add_argument("--val_frac",  type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed",      type=int,   default=42)
    args = ap.parse_args()

    # ── Gather raw images ────────────────────────────────────────────────────
    no_damage_sources = {
        "clean_cars":       args.raw_dir / "clean_cars",
        "clean_cars_extra": args.raw_dir / "clean_cars_extra",
        "cars_dataset_2":   args.raw_dir / "external_datasets" / "cars_dataset_2",
        "harder_negatives": args.raw_dir / "harder_negatives",
    }
    damage_source = args.raw_dir / "cardd_damage"

    print("=== Raw image counts ===")
    all_no_damage: list[Path] = []
    for name, folder in no_damage_sources.items():
        imgs = collect_images(folder)
        print(f"  no_damage / {name}: {len(imgs)}")
        all_no_damage.extend(imgs)

    all_damage = collect_images(damage_source)
    print(f"  damage    / cardd_damage: {len(all_damage)}")

    # ── Balance classes ──────────────────────────────────────────────────────
    n = min(len(all_no_damage), len(all_damage))
    print(f"\nBalancing at {n} images per class")
    rng = random.Random(args.seed)
    rng.shuffle(all_no_damage)
    rng.shuffle(all_damage)
    all_no_damage = all_no_damage[:n]
    all_damage    = all_damage[:n]

    # ── Split ────────────────────────────────────────────────────────────────
    nd_train, nd_val, nd_test = split_list(all_no_damage, args.val_frac, args.test_frac, args.seed)
    d_train,  d_val,  d_test  = split_list(all_damage,    args.val_frac, args.test_frac, args.seed)

    print(f"\nSplit — no_damage: train={len(nd_train)} val={len(nd_val)} test={len(nd_test)}")
    print(f"Split — damage:    train={len(d_train)}  val={len(d_val)}  test={len(d_test)}")

    # ── Rebuild processed/ ───────────────────────────────────────────────────
    if args.out_dir.exists():
        print(f"\nClearing {args.out_dir} ...")
        shutil.rmtree(args.out_dir)

    print("Copying to processed/ ...")
    for split_name, nd_imgs, d_imgs in [
        ("train", nd_train, d_train),
        ("val",   nd_val,   d_val),
        ("test",  nd_test,  d_test),
    ]:
        copy_to(nd_imgs, args.out_dir / split_name / "no_damage", "no_damage")
        copy_to(d_imgs,  args.out_dir / split_name / "damage",    "damage")

    print(f"Done → {args.out_dir}")

    # ── Build eval_by_source/ ────────────────────────────────────────────────
    if args.eval_dir.exists():
        print(f"\nClearing {args.eval_dir} ...")
        shutil.rmtree(args.eval_dir)

    print("Building eval_by_source/ ...")
    for name, folder in no_damage_sources.items():
        imgs = collect_images(folder)
        dest = args.eval_dir / name / "no_damage"
        dest.mkdir(parents=True, exist_ok=True)
        for src in imgs:
            shutil.copy2(src, dest / src.name)
        print(f"  {name}: {len(imgs)} images")

    print(f"Done → {args.eval_dir}")
    print("\nTo evaluate false positive rate per source:")
    for name in no_damage_sources:
        print(f"  python src/eval_clean_only.py --data_dir {args.eval_dir}/{name}")


if __name__ == "__main__":
    main()
