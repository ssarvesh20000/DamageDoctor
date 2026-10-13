"""
Kaggle training script for DamageDoctor detector — CarDD-only v4.

Goal: beat the v1 baseline (mAP50 0.734 / mAP50-95 0.581) by combining two
upgrades that have never both been trained to completion:

  1. Bigger model: YOLOv8m (26M params) vs. v1's YOLOv8s (11M).
     Weak classes (crack 0.425, scratch 0.548, dent 0.584 on v1) need more
     capacity to discriminate subtle damage textures, not more data.

  2. Full v2-style recipe: 100 epochs, cos_lr, freeze=10, moderate aug
     (mosaic=1.0, mixup=0.1, copy_paste=0.1, scale=0.9). Designed locally
     but never finished — the 16 GB M5 ran out of runway at epoch 29.

P100 has tons of headroom (v3 used only 5.93/16 GB at YOLOv8s+batch=32), so
YOLOv8m at batch=32 is well within budget and finishes in ~2 hours.

The Kaggle torch→2.5.1 downgrade for P100 sm_60 compatibility is identical
to kernels/train_detect_combined/train.py — see that file's docstring for
the full explanation.

Produces best.pt at /kaggle/working/best.pt for download via
`kaggle kernels output`.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
STAGED_DATASET = WORK_DIR / "dataset"


def _pip_install():
    """Install pinned torch + ultralytics (see combined kernel for rationale)."""
    print("Downgrading torch to 2.5.1+cu121 (sm_60 compatible for P100)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall",
        "torch==2.5.1", "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    print("Installing ultralytics...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "ultralytics",
    ])


def _show_tree(root: Path, prefix: str = "", max_depth: int = 3, depth: int = 0):
    if not root.exists():
        print(f"{prefix}(does not exist: {root})")
        return
    if depth >= max_depth:
        return
    try:
        items = sorted(root.iterdir())
    except PermissionError:
        print(f"{prefix}(permission denied)")
        return
    for item in items[:20]:
        tag = "DIR " if item.is_dir() else "FILE"
        size = ""
        if item.is_file():
            size = f" ({item.stat().st_size / 1e6:.2f} MB)"
        print(f"{prefix}[{tag}] {item.name}{size}")
        if item.is_dir():
            _show_tree(item, prefix + "    ", max_depth, depth + 1)
    if len(items) > 20:
        print(f"{prefix}... ({len(items) - 20} more entries)")


def _locate_dataset_root() -> Path:
    """Find the cardd_yolo dataset inside /kaggle/input/ via rglob."""
    print("Full tree of /kaggle/input/ (depth 3):")
    _show_tree(KAGGLE_INPUT, max_depth=3)

    if not KAGGLE_INPUT.exists() or not any(KAGGLE_INPUT.iterdir()):
        raise SystemExit(
            "\nERROR: /kaggle/input/ is empty. No datasets attached to this kernel.\n"
            "Check kernel-metadata.json dataset_sources."
        )

    for yaml_file in KAGGLE_INPUT.rglob("data.yaml"):
        parent = yaml_file.parent
        print(f"\nFound data.yaml at: {yaml_file}")
        return parent

    for train_images in KAGGLE_INPUT.rglob("images"):
        if train_images.parent.name == "train":
            parent = train_images.parent.parent
            print(f"\nFound train/images at: {train_images}")
            return parent

    zips = list(KAGGLE_INPUT.rglob("*.zip"))
    if zips:
        parent = zips[0].parent
        print(f"\nFound {len(zips)} zip files in: {parent}")
        return parent

    raise SystemExit(
        "\nERROR: Could not find data.yaml, train/images/, or *.zip anywhere in /kaggle/input/."
    )


def _stage_dataset(source: Path) -> Path:
    """Stage dataset into /kaggle/working/dataset/ (handles already-unpacked or zipped)."""
    print(f"\nSource dataset contents ({source}):")
    _show_tree(source, max_depth=2)

    if (source / "train" / "images").exists():
        print(f"\nDataset already unpacked at {source} — using in place.")
        return source

    zips = sorted(source.glob("*.zip"))
    if not zips:
        raise SystemExit(f"\nERROR: {source} has neither train/images/ nor *.zip files.")

    STAGED_DATASET.mkdir(parents=True, exist_ok=True)
    for zf in zips:
        print(f"Extracting {zf.name} -> {STAGED_DATASET}/")
        with zipfile.ZipFile(zf) as archive:
            archive.extractall(STAGED_DATASET)

    if not (STAGED_DATASET / "train" / "images").exists():
        raise SystemExit(
            f"\nERROR: After extraction, {STAGED_DATASET / 'train' / 'images'} still does not exist."
        )
    return STAGED_DATASET


def _write_runtime_yaml(dataset_root: Path) -> Path:
    """Write a fresh data.yaml pointing at the resolved dataset root."""
    import yaml

    data_config = {
        "path": str(dataset_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 6,
        "names": {
            0: "dent",
            1: "scratch",
            2: "crack",
            3: "glass shatter",
            4: "lamp broken",
            5: "tire flat",
        },
    }
    data_yaml = WORK_DIR / "data.yaml"
    data_yaml.write_text(yaml.dump(data_config, sort_keys=False))
    return data_yaml


def _verify_dataset(dataset_root: Path):
    for split in ("train", "val", "test"):
        img_dir = dataset_root / split / "images"
        if not img_dir.exists():
            raise SystemExit(f"Missing split directory: {img_dir}")
        n = sum(1 for _ in img_dir.iterdir())
        print(f"  {split}: {n} images")


def _print_gpu():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible. Verify enable_gpu=true in kernel-metadata.json "
            "and that your Kaggle account is phone-verified."
        )
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}")
    print(f"VRAM: {gpu.total_memory / 1e9:.1f} GB")
    print(f"torch: {torch.__version__}")


def main():
    print("=== Install torch 2.5.1 + ultralytics ===")
    _pip_install()

    print("\n=== Locate dataset ===")
    source_root = _locate_dataset_root()

    print("\n=== Stage dataset ===")
    dataset_root = _stage_dataset(source_root)

    print("\n=== Verify dataset ===")
    _verify_dataset(dataset_root)

    print("\n=== GPU info ===")
    _print_gpu()

    print("\n=== Write runtime data.yaml ===")
    data_yaml = _write_runtime_yaml(dataset_root)
    print(data_yaml.read_text())

    from ultralytics import YOLO

    print("=== Train YOLOv8m on CarDD-only (v4 recipe) ===")
    # v4 config:
    #   Model:  YOLOv8m (26M params) — 2.3x v1's YOLOv8s (11M), more capacity
    #           for crack/scratch/dent texture features.
    #   Epochs: 100 — full v2-style recipe. v2 locally crashed at epoch 29
    #           with mAP50 still climbing.
    #   Batch:  32 — P100 has tons of headroom (v3 YOLOv8s/batch=32 = 5.93 GB).
    #           YOLOv8m is ~2.3x bigger but still easily fits.
    #   cos_lr: cosine annealing — helped the binary classifier.
    #   freeze=10: backbone frozen for first 10 epochs for stable head init.
    #   Aug:    scale=0.9, mosaic=1.0, mixup=0.1, copy_paste=0.1 — moderate,
    #           matching the designed-but-never-finished v2 recipe.
    #   patience=25: don't give up early; v2 was still improving at epoch 29.
    model = YOLO("yolov8m.pt")
    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=32,
        patience=25,
        workers=4,
        cache=False,
        cos_lr=True,
        freeze=10,
        scale=0.9,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        device=0,
        project=str(WORK_DIR / "runs"),
        name="cardd_v4",
        exist_ok=True,
    )

    print("\n=== Evaluate on CarDD test split ===")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        project=str(WORK_DIR / "runs"),
        name="cardd_v4_test",
        exist_ok=True,
    )
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\n=== Copy best.pt to /kaggle/working root ===")
    best_src = WORK_DIR / "runs" / "cardd_v4" / "weights" / "best.pt"
    if best_src.exists():
        best_dst = WORK_DIR / "best.pt"
        shutil.copy2(best_src, best_dst)
        size_mb = best_dst.stat().st_size / 1e6
        print(f"Saved {best_dst} ({size_mb:.1f} MB)")
    else:
        print(f"WARNING: {best_src} not found — training may have failed early")


if __name__ == "__main__":
    main()
