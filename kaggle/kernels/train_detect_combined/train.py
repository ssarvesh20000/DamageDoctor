"""
Kaggle training script for DamageDoctor detector (combined CarDD + VeHIDE).

Runs on Kaggle free P100 GPU. Produces best.pt at /kaggle/working/best.pt
for download via `kaggle kernels output`.

Mirrors src/train_detect.py --combined but uses aggressive GPU-friendly
settings that OOM on local 16GB M5 MPS but fit comfortably in a 16GB P100:
  batch=32, mosaic=1.0, mixup=0.1, copy_paste=0.1, workers=4, epochs=40

The whole run takes roughly 45-75 minutes depending on queue time.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
# Where we'll stage a unified dataset regardless of how Kaggle presents the input
STAGED_DATASET = WORK_DIR / "dataset"


def _pip_install():
    """Install pinned torch + ultralytics into the Kaggle kernel.

    Kaggle's default image (as of 2026-04) ships torch 2.10+cu128, which
    was compiled WITHOUT sm_60 kernels. But Kaggle's free tier hands out
    P100 GPUs (Pascal, sm_60, 2016). So the pre-installed torch literally
    cannot run on the GPU we get. Fix: downgrade to torch 2.5.1, the last
    release that still ships sm_60 kernels for Pascal.

    We install from the PyTorch CUDA 12.1 wheel index (compatible with
    Kaggle's CUDA runtime) and use --force-reinstall to overwrite the
    pre-installed 2.10 build.
    """
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
    """Print a recursive directory listing for diagnostics."""
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
    for i, item in enumerate(items[:20]):  # cap to 20 entries per level
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
    """Find the combined_yolo dataset inside /kaggle/input/.

    Kaggle normally mounts attached datasets at /kaggle/input/<slug>/, but
    the exact layout can vary. We dump the full tree for diagnostics, then
    use rglob to find any data.yaml or train/images/ anywhere in the input
    and treat its parent directory as the dataset root.
    """
    print("Full tree of /kaggle/input/ (depth 3):")
    _show_tree(KAGGLE_INPUT, max_depth=3)

    if not KAGGLE_INPUT.exists() or not any(KAGGLE_INPUT.iterdir()):
        raise SystemExit(
            "\nERROR: /kaggle/input/ is empty. No datasets attached to this kernel.\n"
            "Check kernel-metadata.json dataset_sources."
        )

    # Recursive search: find data.yaml anywhere under /kaggle/input/
    for yaml_file in KAGGLE_INPUT.rglob("data.yaml"):
        parent = yaml_file.parent
        print(f"\nFound data.yaml at: {yaml_file}")
        return parent

    # Fallback: find train/images anywhere under /kaggle/input/
    for train_images in KAGGLE_INPUT.rglob("images"):
        if train_images.parent.name == "train":
            parent = train_images.parent.parent
            print(f"\nFound train/images at: {train_images}")
            print(f"Using dataset root: {parent}")
            return parent

    # Last resort: find any *.zip files (maybe zip-upload wasn't extracted)
    zips = list(KAGGLE_INPUT.rglob("*.zip"))
    if zips:
        parent = zips[0].parent
        print(f"\nFound {len(zips)} zip files in: {parent}")
        return parent

    raise SystemExit(
        "\nERROR: Could not find data.yaml, train/images/, or *.zip anywhere "
        "in /kaggle/input/. See tree above."
    )


def _stage_dataset(source: Path) -> Path:
    """Stage the dataset into /kaggle/working/dataset/ with the expected tree.

    Handles three possible layouts Kaggle might give us:
      (a) Already-unpacked tree with train/, val/, test/ subdirs
      (b) Zip files (train.zip, val.zip, test.zip) at the root from -r zip upload
      (c) Mixed — e.g. data.yaml loose + zipped splits

    We always end up with:
      /kaggle/working/dataset/
        ├── train/{images,labels}/
        ├── val/{images,labels}/
        └── test/{images,labels}/
    """
    print(f"\nSource dataset contents ({source}):")
    _show_tree(source, max_depth=2)

    # Case A: already unpacked
    if (source / "train" / "images").exists():
        print(f"\nDataset already unpacked at {source} — using in place.")
        return source

    # Case B/C: there are zip files we need to extract
    zips = sorted(source.glob("*.zip"))
    if not zips:
        raise SystemExit(
            f"\nERROR: {source} has neither train/images/ nor *.zip files.\n"
            "Cannot locate training data."
        )

    STAGED_DATASET.mkdir(parents=True, exist_ok=True)
    for zf in zips:
        print(f"Extracting {zf.name} -> {STAGED_DATASET}/")
        with zipfile.ZipFile(zf) as archive:
            archive.extractall(STAGED_DATASET)

    print(f"\nStaged dataset contents ({STAGED_DATASET}):")
    _show_tree(STAGED_DATASET, max_depth=2)

    if not (STAGED_DATASET / "train" / "images").exists():
        raise SystemExit(
            f"\nERROR: After extraction, {STAGED_DATASET / 'train' / 'images'} "
            "still does not exist."
        )
    return STAGED_DATASET


def _write_runtime_yaml(dataset_root: Path) -> Path:
    """Write a fresh data.yaml pointing at the resolved dataset root.

    The local data/combined_yolo/data.yaml has an absolute macOS path
    (/Users/sarvesh/...), which is meaningless inside the Kaggle container.
    """
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
    print("=== Install ultralytics ===")
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

    from ultralytics import YOLO  # imported after pip install

    print("=== Train ===")
    # Aggressive P100-friendly config:
    #   16 GB VRAM easily holds batch=32 + mosaic 4-tile + copy_paste.
    #   workers=4 — P100 kernels have 4 vCPU; saturates dataloader.
    #   cos_lr — cosine schedule helped the binary classifier.
    #   freeze=10 — freeze backbone first 10 epochs for stable head init.
    #   scale=0.9, mosaic=1.0, mixup=0.1, copy_paste=0.1 — strong aug to
    #     counter the weaker classes (crack, dent, scratch) and combat
    #     dataset-bias artifacts from VeHIDE.
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=640,
        batch=32,
        patience=15,
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
        name="combined_v3",
        exist_ok=True,
    )

    print("\n=== Evaluate on CarDD test split ===")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        project=str(WORK_DIR / "runs"),
        name="combined_v3_test",
        exist_ok=True,
    )
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\n=== Copy best.pt to /kaggle/working root ===")
    best_src = WORK_DIR / "runs" / "combined_v3" / "weights" / "best.pt"
    if best_src.exists():
        best_dst = WORK_DIR / "best.pt"
        shutil.copy2(best_src, best_dst)
        size_mb = best_dst.stat().st_size / 1e6
        print(f"Saved {best_dst} ({size_mb:.1f} MB)")
    else:
        print(f"WARNING: {best_src} not found — training may have failed early")


if __name__ == "__main__":
    main()
