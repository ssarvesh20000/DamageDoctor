"""
Rebuilds data/processed/ from all raw sources and creates data/eval_by_source/
for per-source false positive evaluation.

eval_by_source/<source>/ holds only that source's images that landed in the
classifier's TEST split, so per-source FP rates are measured on held-out
images. Pass --eval_all_images for the old behavior (every image of the
source, including train/val images — optimistic FP rates).

Note: these images are held out from the binary classifier only. All 165
harder_negatives (and 35 clean_cars) were v7 DETECTOR training backgrounds.

Usage (from project root):
    python src/build_dataset.py
    python src/build_dataset.py --seed 42 --val_frac 0.1 --test_frac 0.1
    python src/build_dataset.py --eval_all_images

WARNING: clears data/processed/ and data/eval_by_source/ before rebuilding.
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
    ap.add_argument("--eval_all_images", action="store_true",
                    help="Old behavior: copy EVERY image of each source into "
                         "eval_by_source/, including train/val images (biased)")
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
    source_of: dict[Path, str] = {}  # image path -> source name (read-only lookup)
    for name, folder in no_damage_sources.items():
        imgs = collect_images(folder)
        print(f"  no_damage / {name}: {len(imgs)}")
        all_no_damage.extend(imgs)
        for img in imgs:
            source_of[img] = name

    all_damage = collect_images(damage_source)
    print(f"  damage    / cardd_damage: {len(all_damage)}")

    # ── Balance classes ──────────────────────────────────────────────────────
    # DO NOT change the order of operations from here through the split
    # (list order, seed, shuffles, truncation). The current classifier's test
    # numbers depend on this exact split. Source tracking above is a separate
    # dict lookup and does not touch the lists being shuffled.
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

    if args.eval_all_images:
        print("\nWARNING: --eval_all_images copies every image of each source, "
              "including the ones in the classifier's train/val splits.")
        print("WARNING: per-source FP rates from this eval set are optimistic "
              "(the model has seen most of these images).")
        print("Building eval_by_source/ (ALL images) ...")
        eval_imgs = {name: collect_images(folder)
                     for name, folder in no_damage_sources.items()}
    else:
        print("\nBuilding eval_by_source/ (test-split images only) ...")
        eval_imgs = {name: [] for name in no_damage_sources}
        for img in nd_test:
            eval_imgs[source_of[img]].append(img)

    for name, imgs in eval_imgs.items():
        if not imgs:
            # ImageFolder fails on an empty class dir, so skip the source
            print(f"  {name}: 0 images (skipped)")
            continue
        dest = args.eval_dir / name / "no_damage"
        dest.mkdir(parents=True, exist_ok=True)
        for src in sorted(imgs):
            shutil.copy2(src, dest / src.name)
        print(f"  {name}: {len(imgs)} images")

    print(f"Done → {args.eval_dir}")
    print("\nTo evaluate false positive rate per source:")
    for name, imgs in eval_imgs.items():
        if imgs:
            print(f"  python src/eval_clean_only.py --data_dir {args.eval_dir}/{name}")


if __name__ == "__main__":
    main()
