"""
Kaggle training script for DamageDoctor detector — v9 (PREPARED, NOT RUN).

Copied from train_detect_v7/train.py. The training recipe is IDENTICAL to v7
(= v1 recipe). The only addition is the V9 DATA HOOK (_apply_v9_data_change),
which changes the TRAIN split only, from an optional second Kaggle dataset
("v9 overlay"). With no overlay attached the hook is a no-op and this script
trains on exactly the same data as the attached cardd-yolo dataset.

Overlay dataset layout (upload as its own Kaggle dataset, add its slug to
kernel-metadata.json "dataset_sources"):
  V9_OVERLAY              marker file (required, may be empty)
  train_images/           images to ADD to train (or replace, same filename)
  train_labels/           YOLO .txt labels to ADD/REPLACE in train
                          (e.g. relabeled crack/scratch boxes, same stem)
  remove_train.txt        optional: one train image stem per line to REMOVE
                          (image + label), e.g. to swap the background set
Never touches val/ or test/, so metrics stay comparable with v1/v7.

BASE DATASET VERSION: kernel-metadata.json attaches the LATEST version of
ssarvesh20000/damagedoctor-cardd-yolo. v7 trained on dataset v2 (200 bg);
v8 uploaded v3 (280 bg). If v9 should be "v7 data + one change", make sure
the attached version has v7's 200 backgrounds (or remove the 80 extra ones
via remove_train.txt) — check the "train: N images, M labels, K background"
line printed by _verify_dataset (v7: 200 background).
Do not put data.yaml or a train/images/ dir in the overlay (the dataset
locator below would pick it up instead of CarDD).

Previous v7 notes:

v1 recipe (from args.yaml):
  - YOLOv8s, detect, imgsz=640, batch=16, 50 epochs, patience=10
  - SGD (optimizer=auto → SGD), lr0=0.01, lrf=0.01
  - Default augmentation (mosaic=1.0, hsv_s=0.7, hsv_v=0.4, scale=0.5)
  - close_mosaic=10, amp=True, seed=0

Expected: background images reduce Bkg FPs (4.43 dAP in TIDE)
without hurting strong classes. If this beats v1, it's the new baseline.
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
    """Install pinned torch for P100 sm_60 compatibility."""
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
    """Find the cardd_yolo dataset inside /kaggle/input/."""
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


# ==========================================================================
# ===== V9 DATA HOOK — the ONLY intended difference from train_detect_v7 =====
# ==========================================================================
V9_STAGED_DATASET = WORK_DIR / "dataset_v9"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_v9_overlay():
    """Return the overlay dataset root (dir containing V9_OVERLAY), or None."""
    for marker in KAGGLE_INPUT.rglob("V9_OVERLAY"):
        return marker.parent
    return None


def _apply_v9_data_change(dataset_root: Path) -> Path:
    """Apply the v9 data-side change to the TRAIN split only.

    No overlay attached -> returns dataset_root unchanged (same data as v7's
    dataset version). Otherwise copies the dataset to a writable dir (if it is
    still under read-only /kaggle/input), then removes / adds / replaces train
    images and labels from the overlay. Prints exactly what changed.
    """
    overlay = _find_v9_overlay()
    if overlay is None:
        print("No V9_OVERLAY dataset attached — training on the unmodified dataset.")
        return dataset_root
    print(f"V9 overlay found at {overlay}")

    if KAGGLE_INPUT in dataset_root.parents:
        print(f"Copying {dataset_root} -> {V9_STAGED_DATASET} (input is read-only)...")
        if V9_STAGED_DATASET.exists():
            shutil.rmtree(V9_STAGED_DATASET)
        shutil.copytree(dataset_root, V9_STAGED_DATASET)
        dataset_root = V9_STAGED_DATASET

    train_imgs = dataset_root / "train" / "images"
    train_lbls = dataset_root / "train" / "labels"
    train_lbls.mkdir(parents=True, exist_ok=True)

    removed = 0
    remove_list = overlay / "remove_train.txt"
    if remove_list.exists():
        stems = {ln.strip() for ln in remove_list.read_text().splitlines() if ln.strip()}
        for img in list(train_imgs.iterdir()):
            if img.stem in stems:
                img.unlink()
                removed += 1
        for stem in stems:
            lbl = train_lbls / f"{stem}.txt"
            if lbl.exists():
                lbl.unlink()

    added_imgs = replaced_imgs = added_lbls = replaced_lbls = 0
    if (overlay / "train_images").is_dir():
        for src in sorted((overlay / "train_images").iterdir()):
            if src.suffix.lower() not in IMG_EXTS:
                continue
            dst = train_imgs / src.name
            replaced_imgs += dst.exists()
            added_imgs += not dst.exists()
            shutil.copy2(src, dst)
    if (overlay / "train_labels").is_dir():
        for src in sorted((overlay / "train_labels").glob("*.txt")):
            dst = train_lbls / src.name
            replaced_lbls += dst.exists()
            added_lbls += not dst.exists()
            shutil.copy2(src, dst)

    print(f"V9 data change (train only): removed {removed} images, "
          f"added {added_imgs} / replaced {replaced_imgs} images, "
          f"added {added_lbls} / replaced {replaced_lbls} labels")
    return dataset_root
# ======================== END V9 DATA HOOK ========================


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
        bg = n_imgs - n_lbls
        print(f"  {split}: {n_imgs} images, {n_lbls} labels, {bg} background")


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

    print("\n=== V9 data hook ===")
    dataset_root = _apply_v9_data_change(dataset_root)

    print("\n=== Verify dataset ===")
    _verify_dataset(dataset_root)

    print("\n=== GPU info ===")
    _print_gpu()

    print("\n=== Write runtime data.yaml ===")
    data_yaml = _write_runtime_yaml(dataset_root)
    print(data_yaml.read_text())

    from ultralytics import YOLO

    print("=== Train YOLOv8s (v9: exact v1/v7 recipe, data from V9 hook) ===")
    # Exact v1 recipe from args.yaml — DO NOT CHANGE. Only the data differs.
    # v1 used optimizer='auto' which resolves to SGD for detect.
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(data_yaml),
        task="detect",
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        workers=4,
        cache=False,
        amp=True,
        cos_lr=False,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        weight_decay=0.0005,
        momentum=0.937,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        nbs=64,
        # Default augmentation — exactly matching v1
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.4,
        close_mosaic=10,
        # Training
        device=0,
        seed=0,
        deterministic=True,
        project=str(WORK_DIR / "runs"),
        name="cardd_v9_detect",
        exist_ok=True,
    )

    print("\n=== Evaluate on CarDD test split ===")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        project=str(WORK_DIR / "runs"),
        name="cardd_v9_detect_test",
        exist_ok=True,
    )
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # Per-class AP50
    names = model.names
    if hasattr(metrics.box, "ap50"):
        print("\nPer-class AP50:")
        for i, ap50 in enumerate(metrics.box.ap50):
            print(f"  {names[i]:20s}  {ap50:.4f}")

    print("\n=== Copy best.pt to /kaggle/working root ===")
    best_src = WORK_DIR / "runs" / "cardd_v9_detect" / "weights" / "best.pt"
    if best_src.exists():
        best_dst = WORK_DIR / "best.pt"
        shutil.copy2(best_src, best_dst)
        size_mb = best_dst.stat().st_size / 1e6
        print(f"Saved {best_dst} ({size_mb:.1f} MB)")
    else:
        print(f"WARNING: {best_src} not found — training may have failed early")

    last_src = WORK_DIR / "runs" / "cardd_v9_detect" / "weights" / "last.pt"
    if last_src.exists():
        last_dst = WORK_DIR / "last.pt"
        shutil.copy2(last_src, last_dst)
        print(f"Saved {last_dst} (for resume if needed)")


if __name__ == "__main__":
    main()
