"""
Rebuilds data/processed/ from all raw sources and creates data/eval_by_source/
for per-source false positive evaluation.

eval_by_source/<source>/ holds only that source's images that landed in the
classifier's TEST split, so per-source FP rates are measured on held-out
images. Pass --eval_all_images for the old behavior (every image of the
source, including train/val images — optimistic FP rates).

Note: these images are held out from the binary classifier only. All 165
harder_negatives (and 35 clean_cars) were v7 DETECTOR training backgrounds.

25kUndmged (--undmged_dir, ~17.8k clean cars) is added on top of the balanced
portion as extra no_damage images. It keeps ITS OWN train/val/test split (no
reshuffle) and its original filenames, copied flat from every subfolder of
<undmged_dir>/<split>/. eval_by_source/undmged_25k/ gets its test split.
Pass --no_undmged to build only the balanced portion.

Usage (from project root):
    python src/build_dataset.py
    python src/build_dataset.py --seed 42 --val_frac 0.1 --test_frac 0.1
    python src/build_dataset.py --eval_all_images

Refuses to touch an existing out_dir / eval_dir unless --force is passed.
With --force it DELETES them first. data/processed/ is the only copy of the
current classifier dataset layout, so build into a new folder instead:
    python src/build_dataset.py --out_dir runs/rebuild/processed \
        --eval_dir runs/rebuild/eval_by_source
"""

import argparse
import random
import shutil
import sys
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


def collect_undmged(undmged_dir: Path) -> dict:
    """25kUndmged's own split: {split: [paths]}, sorted by filename."""
    out = {}
    for split in ("train", "val", "test"):
        imgs = collect_images(undmged_dir / split)
        names = [p.name for p in imgs]
        if len(set(names)) != len(names):
            sys.exit(f"ERROR: duplicate filenames inside {undmged_dir / split}; "
                     "flat copy would overwrite files")
        out[split] = sorted(imgs, key=lambda p: p.name)
    return out


def refuse_existing(path: Path, force: bool):
    if not path.exists():
        return
    if not force:
        print("!" * 72)
        print(f"!! REFUSING to delete existing {path}")
        print("!! It may hold data that this script cannot rebuild.")
        print("!! Pass --force to delete it, or choose a new --out_dir/--eval_dir.")
        print("!" * 72)
        sys.exit(1)
    print("!" * 72)
    print(f"!! --force: DELETING {path}")
    print("!" * 72)
    shutil.rmtree(path)


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
    ap.add_argument("--undmged_dir", type=Path, default=Path("25kUndmged/data"))
    ap.add_argument("--no_undmged", action="store_true",
                    help="Skip 25kUndmged (balanced portion only)")
    ap.add_argument("--force", action="store_true",
                    help="Delete existing out_dir / eval_dir before rebuilding")
    args = ap.parse_args()

    # Check both before touching anything
    if not args.force:
        refuse_existing(args.out_dir, False)
        refuse_existing(args.eval_dir, False)

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

    undmged = {} if args.no_undmged else collect_undmged(args.undmged_dir)
    for split, imgs in undmged.items():
        print(f"Extra  — no_damage / undmged_25k {split}: {len(imgs)}")

    # ── Rebuild processed/ ───────────────────────────────────────────────────
    refuse_existing(args.out_dir, args.force)

    print("Copying to processed/ ...")
    for split_name, nd_imgs, d_imgs in [
        ("train", nd_train, d_train),
        ("val",   nd_val,   d_val),
        ("test",  nd_test,  d_test),
    ]:
        copy_to(nd_imgs, args.out_dir / split_name / "no_damage", "no_damage")
        copy_to(d_imgs,  args.out_dir / split_name / "damage",    "damage")
        dest = args.out_dir / split_name / "no_damage"
        for src in undmged.get(split_name, []):
            shutil.copy2(src, dest / src.name)

    print(f"Done → {args.out_dir}")

    # ── Build eval_by_source/ ────────────────────────────────────────────────
    refuse_existing(args.eval_dir, args.force)

    if args.eval_all_images:
        print("\nWARNING: --eval_all_images copies every image of each source, "
              "including the ones in the classifier's train/val splits.")
        print("WARNING: per-source FP rates from this eval set are optimistic "
              "(the model has seen most of these images).")
        print("Building eval_by_source/ (ALL images) ...")
        eval_imgs = {name: collect_images(folder)
                     for name, folder in no_damage_sources.items()}
        if undmged:
            # 16 filenames repeat across 25kUndmged splits; flat copy keeps one
            eval_imgs["undmged_25k"] = [p for s in ("train", "val", "test")
                                        for p in undmged[s]]
    else:
        print("\nBuilding eval_by_source/ (test-split images only) ...")
        eval_imgs = {name: [] for name in no_damage_sources}
        for img in nd_test:
            eval_imgs[source_of[img]].append(img)
        if undmged:
            eval_imgs["undmged_25k"] = list(undmged["test"])

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
