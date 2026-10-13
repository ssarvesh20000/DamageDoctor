"""
Convert CarDD COCO annotations to YOLO segmentation format.

COCO format: polygon segmentation in absolute pixel coordinates, 1-indexed classes.
YOLO seg format: cls x1 y1 x2 y2 ... xn yn (normalized 0-1), 0-indexed classes.

Creates data/cardd_yolo_seg/ with the same image structure as data/cardd_yolo/
but symlinked images and polygon labels.

Usage:
    python src/convert_coco_to_yolo_seg.py
"""
import json
import os
from pathlib import Path

COCO_ROOT = Path("CarDD_release/CarDD_COCO")
DETECT_ROOT = Path("data/cardd_yolo")
SEG_ROOT = Path("data/cardd_yolo_seg")

SPLITS = {
    "train": "instances_train2017.json",
    "val": "instances_val2017.json",
    "test": "instances_test2017.json",
}


def convert_split(split: str, ann_file: str):
    ann_path = COCO_ROOT / "annotations" / ann_file
    with open(ann_path) as f:
        coco = json.load(f)

    # Build image lookup: id -> {file_name, width, height}
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    from collections import defaultdict
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # Create output directories
    img_dir = SEG_ROOT / split / "images"
    lbl_dir = SEG_ROOT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Source image directory (from existing detect dataset)
    src_img_dir = DETECT_ROOT / split / "images"

    n_images = 0
    n_anns = 0
    n_skipped = 0

    for img_id, img_info in images.items():
        fname = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]
        stem = Path(fname).stem

        # Symlink image from detect dataset
        src_img = src_img_dir / fname
        dst_img = img_dir / fname
        if src_img.exists() and not dst_img.exists():
            # Use absolute path for symlink target
            os.symlink(src_img.resolve(), dst_img)

        # Write seg-format labels
        anns = anns_by_img.get(img_id, [])
        lines = []
        for ann in anns:
            # COCO category_id is 1-indexed, YOLO is 0-indexed
            cls_id = ann["category_id"] - 1
            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, list) or len(seg) == 0:
                n_skipped += 1
                continue

            # Take the first polygon (most annotations have exactly one)
            poly = seg[0]
            if len(poly) < 6:  # need at least 3 points
                n_skipped += 1
                continue

            # Normalize polygon coordinates
            norm_pts = []
            for i in range(0, len(poly), 2):
                nx = poly[i] / w
                ny = poly[i + 1] / h
                # Clamp to [0, 1]
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                norm_pts.extend([f"{nx:.6f}", f"{ny:.6f}"])

            lines.append(f"{cls_id} " + " ".join(norm_pts))
            n_anns += 1

        lbl_path = lbl_dir / f"{stem}.txt"
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_images += 1

    print(f"  {split}: {n_images} images, {n_anns} annotations, {n_skipped} skipped")


def write_data_yaml():
    yaml_content = f"""path: {SEG_ROOT.resolve()}
train: train/images
val: val/images
test: test/images

nc: 6
names:
  0: dent
  1: scratch
  2: crack
  3: glass shatter
  4: lamp broken
  5: tire flat
"""
    (SEG_ROOT / "data.yaml").write_text(yaml_content)


def main():
    print("Converting CarDD COCO → YOLO segmentation format")
    print(f"  Source: {COCO_ROOT}")
    print(f"  Output: {SEG_ROOT}")
    print()

    for split, ann_file in SPLITS.items():
        convert_split(split, ann_file)

    write_data_yaml()

    # Verify a sample label
    sample = next((SEG_ROOT / "train" / "labels").glob("*.txt"))
    print(f"\nSample label ({sample.name}):")
    for line in sample.read_text().strip().split("\n")[:3]:
        parts = line.split()
        print(f"  cls={parts[0]}, {(len(parts)-1)//2} polygon points")

    print(f"\nDone. Dataset at {SEG_ROOT}/")


if __name__ == "__main__":
    main()
