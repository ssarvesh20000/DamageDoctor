"""
Write a CSV manifest (split,class,filename,source,md5) of a classifier
dataset folder such as data/processed/. Read-only on all inputs.

`source` is found by matching md5s against the raw sources
(data/raw/* and 25kUndmged/data/<split>/). Files whose md5 matches no
source get source "UNKNOWN"; files matching several get "a|b".

Usage (from project root):
    python src/make_manifest.py --data_dir data/processed \
        --out data/manifests/classifier_split_current.csv
    python src/make_manifest.py --data_dir runs/x/processed --out x.csv --no_sources
"""

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp"}

RAW_SOURCES = {
    "cardd_damage":     Path("data/raw/cardd_damage"),
    "clean_cars":       Path("data/raw/clean_cars"),
    "clean_cars_extra": Path("data/raw/clean_cars_extra"),
    "cars_dataset_2":   Path("data/raw/external_datasets/cars_dataset_2"),
    "harder_negatives": Path("data/raw/harder_negatives"),
    "undmged_25k_train": Path("25kUndmged/data/train"),
    "undmged_25k_val":   Path("25kUndmged/data/val"),
    "undmged_25k_test":  Path("25kUndmged/data/test"),
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def images_in(folder: Path):
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no_sources", action="store_true",
                    help="Skip source lookup (source column left empty)")
    args = ap.parse_args()

    md5_sources = defaultdict(set)
    if not args.no_sources:
        for name, folder in RAW_SOURCES.items():
            imgs = images_in(folder)
            print(f"hashing {name}: {len(imgs)}")
            for p in imgs:
                md5_sources[md5_of(p)].add(name)

    rows = []
    for split_dir in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for p in images_in(cls_dir):
                m = md5_of(p)
                src = "|".join(sorted(md5_sources.get(m, ()))) or "UNKNOWN"
                if args.no_sources:
                    src = ""
                rows.append((split_dir.name, cls_dir.name,
                             str(p.relative_to(cls_dir)), src, m))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "filename", "source", "md5"])
        w.writerows(rows)

    counts = defaultdict(int)
    for split, cls, _, src, _ in rows:
        counts[(split, cls, src)] += 1
    print(f"\n{len(rows)} files → {args.out}")
    for k in sorted(counts):
        print(f"  {k[0]:5s} {k[1]:9s} {k[2] or '-':30s} {counts[k]}")
    unknown = [r for r in rows if r[3] == "UNKNOWN"]
    if unknown:
        print(f"\n{len(unknown)} files with UNKNOWN source:")
        for r in unknown[:50]:
            print("  ", "/".join(r[:3]))


if __name__ == "__main__":
    main()
