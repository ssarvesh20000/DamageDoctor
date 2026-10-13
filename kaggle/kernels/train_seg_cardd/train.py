"""
Kaggle training script for DamageDoctor detector — v5 seg experiment.

Goal: beat v1 baseline (mAP50 0.734) by addressing the two dominant error
types identified by TIDE analysis:
  - Localization (6.34 dAP lost) → imgsz 640→1280
  - Background FPs (4.43 dAP lost) → conservative aug + copy_paste

Key changes from v1:
  - Task:  detect → segment (needed for copy_paste to use real masks)
  - Model: yolov8s → yolo11s-seg (C3k2 + C2PSA attention, ANE-friendly)
  - imgsz: 640 → 1280 (retains texture detail for crack/scratch)
  - Aug:   conservative — mosaic=0.3, mixup=0, copy_paste=0.3
  - Optimizer: AdamW with cosine LR

Trains for 80 epochs (diagnostic run — fits in one 9hr session at ~7 min/epoch).
If the trend is good, extend to 150 with resume.

P100 16GB VRAM budget at 1280: YOLO11s-seg + batch=8-10 + AMP + cache=ram.
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
    """Install pinned torch + ultralytics for P100 sm_60 compatibility."""
    print("Downgrading torch to 2.5.1+cu121 (sm_60 compatible for P100)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall",
        "torch==2.5.1", "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    # Need ultralytics >= 8.3.207 for YOLO11 non-80-class CoreML fix
    print("Installing ultralytics >= 8.3.207...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "ultralytics>=8.3.207",
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
    """Find the cardd_yolo_seg dataset inside /kaggle/input/."""
    print("Full tree of /kaggle/input/ (depth 3):")
    _show_tree(KAGGLE_INPUT, max_depth=3)

    if not KAGGLE_INPUT.exists() or not any(KAGGLE_INPUT.iterdir()):
        raise SystemExit(
            "\nERROR: /kaggle/input/ is empty. No datasets attached.\n"
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
        "\nERROR: Could not find data.yaml, train/images/, or *.zip in /kaggle/input/."
    )


def _stage_dataset(source: Path) -> Path:
    """Stage dataset into /kaggle/working/dataset/."""
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
            f"\nERROR: After extraction, {STAGED_DATASET / 'train' / 'images'} does not exist."
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
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            raise SystemExit(f"Missing split directory: {img_dir}")
        n_imgs = sum(1 for _ in img_dir.iterdir())
        n_lbls = sum(1 for _ in lbl_dir.iterdir()) if lbl_dir.exists() else 0
        # Verify seg format: labels should have many numbers per line (not just 5)
        sample_lbl = next(lbl_dir.glob("*.txt"), None)
        fmt = "unknown"
        if sample_lbl:
            first_line = sample_lbl.read_text().strip().split("\n")[0]
            n_fields = len(first_line.split())
            fmt = "seg" if n_fields > 5 else "detect"
        print(f"  {split}: {n_imgs} images, {n_lbls} labels, format={fmt}")


def _verify_seg_format(dataset_root: Path):
    """Verify labels are in segmentation format (polygon), not detect (bbox)."""
    sample = next((dataset_root / "train" / "labels").glob("*.txt"))
    first_line = sample.read_text().strip().split("\n")[0]
    n_fields = len(first_line.split())
    if n_fields <= 5:
        raise SystemExit(
            f"\nERROR: Labels appear to be in detect format ({n_fields} fields per line).\n"
            f"Seg format requires polygon coordinates (many fields per line).\n"
            f"Sample: {sample}\n"
            f"Line: {first_line[:100]}"
        )
    print(f"  Label format verified: seg ({n_fields} fields in sample line)")


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
    _verify_seg_format(dataset_root)

    print("\n=== GPU info ===")
    _print_gpu()

    print("\n=== Write runtime data.yaml ===")
    data_yaml = _write_runtime_yaml(dataset_root)
    print(data_yaml.read_text())

    from ultralytics import YOLO

    print("=== Train YOLO11s-seg on CarDD (v5: 1280 + copy_paste + conservative aug) ===")
    # v5 config — driven by TIDE error analysis:
    #
    # TIDE findings on v1:
    #   Loc=6.34 dAP (dominant) → higher resolution captures texture detail
    #   Bkg=4.43 dAP           → conservative aug reduces hallucinated FPs
    #   Cls=1.22 dAP (low)     → class confusion is NOT the problem
    #   Miss=1.49 dAP          → copy_paste augments rare classes (crack)
    #
    # Model:   YOLO11s-seg — C3k2+C2PSA attention, CoreML/ANE-friendly
    # imgsz:   1280 — retains scratch/crack texture at native resolution
    # batch:   10 — P100 16GB budget with AMP + cache=ram
    # Aug:     conservative to preserve low-contrast damage features
    # Epochs:  80 (diagnostic) — if promising, extend to 150 with resume
    model = YOLO("yolo11s-seg.pt")
    model.train(
        data=str(data_yaml),
        task="segment",
        epochs=80,
        imgsz=1280,
        batch=10,
        patience=25,
        workers=4,
        cache="ram",
        amp=True,
        rect=True,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,
        # Conservative augmentation — preserve low-contrast texture
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        scale=0.2,
        translate=0.1,
        degrees=5.0,
        fliplr=0.5,
        flipud=0.0,
        perspective=0.0,
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.3,
        close_mosaic=20,
        # Training
        device=0,
        project=str(WORK_DIR / "runs"),
        name="cardd_v5_seg",
        exist_ok=True,
    )

    print("\n=== Evaluate on CarDD test split ===")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=1280,
        project=str(WORK_DIR / "runs"),
        name="cardd_v5_seg_test",
        exist_ok=True,
    )
    print(f"Box mAP50:    {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    if hasattr(metrics, "seg"):
        print(f"Seg mAP50:    {metrics.seg.map50:.4f}")
        print(f"Seg mAP50-95: {metrics.seg.map:.4f}")

    # Per-class box AP50
    names = model.names
    if hasattr(metrics.box, "ap50"):
        print("\nPer-class Box AP50:")
        for i, ap50 in enumerate(metrics.box.ap50):
            print(f"  {names[i]:20s}  {ap50:.4f}")

    print("\n=== Copy best.pt to /kaggle/working root ===")
    best_src = WORK_DIR / "runs" / "cardd_v5_seg" / "weights" / "best.pt"
    if best_src.exists():
        best_dst = WORK_DIR / "best.pt"
        shutil.copy2(best_src, best_dst)
        size_mb = best_dst.stat().st_size / 1e6
        print(f"Saved {best_dst} ({size_mb:.1f} MB)")
    else:
        print(f"WARNING: {best_src} not found — training may have failed early")

    # Also copy last.pt for potential resume
    last_src = WORK_DIR / "runs" / "cardd_v5_seg" / "weights" / "last.pt"
    if last_src.exists():
        last_dst = WORK_DIR / "last.pt"
        shutil.copy2(last_src, last_dst)
        print(f"Saved {last_dst} (for resume if needed)")


if __name__ == "__main__":
    main()
