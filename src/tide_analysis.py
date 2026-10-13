"""
Per-class error analysis using TIDE on the CarDD val set.

Diagnoses whether each weak class (crack, scratch, dent) is failing from:
  - Cls  (classification confusion — right box, wrong class)
  - Loc  (localization error — right class, bad box)
  - Both (wrong class AND bad box)
  - Dupe (duplicate detection)
  - Bkg  (false positive on background)
  - Miss (missed ground truth — low recall)

Usage:
  python src/tide_analysis.py
  python src/tide_analysis.py --split test
  python src/tide_analysis.py --ckpt path/to/weights.pt
"""
import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tidecv import TIDE, datasets
from ultralytics import YOLO

CLASS_NAMES = {0: "dent", 1: "scratch", 2: "crack",
               3: "glass shatter", 4: "lamp broken", 5: "tire flat"}


def yolo_to_coco_gt(img_dir: Path, label_dir: Path) -> dict:
    """Convert YOLO-format ground truth to COCO JSON."""
    images, annotations = [], []
    ann_id = 1

    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_id = int(img_path.stem) if img_path.stem.isdigit() else hash(img_path.stem) % (2**31)
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        for line in label_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            box_w = bw * w
            box_h = bh * h
            # TIDE requires segmentation — use bbox polygon as placeholder
            seg = [[x1, y1, x1 + box_w, y1, x1 + box_w, y1 + box_h, x1, y1 + box_h]]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [x1, y1, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0,
                "segmentation": seg,
            })
            ann_id += 1

    categories = [{"id": k, "name": v} for k, v in CLASS_NAMES.items()]
    return {"images": images, "annotations": annotations, "categories": categories}


def run_predictions(model: YOLO, img_dir: Path, imgsz: int) -> list:
    """Run model on all images, return COCO-format predictions."""
    preds = []
    img_paths = sorted(p for p in img_dir.glob("*.*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    # Process one image at a time to avoid MPS OOM
    for img_path in img_paths:
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            device="mps",
            conf=0.001,  # low threshold — TIDE needs the full score distribution
            iou=0.6,
            verbose=False,
        )
        for r in results:
            img_id = int(img_path.stem) if img_path.stem.isdigit() else hash(img_path.stem) % (2**31)
            boxes = r.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                preds.append({
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": conf,
                })

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("runs/kaggle_cardd_v7_detect/best.pt"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not args.ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {args.ckpt.resolve()}")

    data_root = Path("data/cardd_yolo")
    img_dir = data_root / args.split / "images"
    label_dir = data_root / args.split / "labels"

    print(f"Model:  {args.ckpt}")
    print(f"Split:  {args.split} ({len(list(img_dir.glob('*.*')))} images)")
    print(f"imgsz:  {args.imgsz}")

    # Step 1: Build COCO ground truth
    print("\nConverting ground truth to COCO format ...")
    gt_coco = yolo_to_coco_gt(img_dir, label_dir)
    print(f"  {len(gt_coco['images'])} images, {len(gt_coco['annotations'])} annotations")

    # Count per-class GT instances
    cls_counts = {}
    for ann in gt_coco["annotations"]:
        name = CLASS_NAMES[ann["category_id"]]
        cls_counts[name] = cls_counts.get(name, 0) + 1
    print("\n  GT instance counts:")
    for name in CLASS_NAMES.values():
        print(f"    {name:20s}  {cls_counts.get(name, 0)}")

    # Step 2: Run predictions
    print("\nRunning inference (conf=0.001 for full TIDE analysis) ...")
    model = YOLO(str(args.ckpt))
    pred_list = run_predictions(model, img_dir, args.imgsz)
    print(f"  {len(pred_list)} raw predictions")

    # Step 3: Write temp files and run TIDE
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = Path(tmpdir) / "gt.json"
        pred_path = Path(tmpdir) / "pred.json"

        with open(gt_path, "w") as f:
            json.dump(gt_coco, f)
        with open(pred_path, "w") as f:
            json.dump(pred_list, f)

        gt_data = datasets.COCO(str(gt_path))
        pred_data = datasets.COCOResult(str(pred_path))

        tide = TIDE()
        tide.evaluate(gt_data, pred_data, mode=TIDE.BOX)

        print("\n" + "=" * 70)
        print("TIDE ERROR SUMMARY")
        print("=" * 70)
        tide.summarize()

        # --- Per-class error breakdown from TIDE internals ---
        print("\n" + "=" * 70)
        print("PER-CLASS ERROR BREAKDOWN")
        print("=" * 70)

        run = list(tide.runs.values())[0]

        # Collect per-class error counts by type
        # Error objects have different attributes depending on type:
        #   Cls/Loc: pred['class'] and gt['class']
        #   Both/Bkg/Dupe: pred['class'] only
        #   Miss: gt['class'] only
        from collections import defaultdict
        cls_errors = defaultdict(lambda: defaultdict(int))
        for err in run.errors:
            if hasattr(err, 'gt') and err.gt is not None:
                cls_id = err.gt['class']
            elif hasattr(err, 'pred') and err.pred is not None:
                cls_id = err.pred['class']
            else:
                continue
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
            cls_errors[cls_name][err.short_name] += 1

        # Also count false negatives per class
        fn_counts = {}
        for cls_id, fn_list in run.false_negatives.items():
            fn_counts[CLASS_NAMES.get(cls_id, str(cls_id))] = len(fn_list)

        # Per-class AP
        print("\n  Per-class AP50:")
        for cls_id, name in CLASS_NAMES.items():
            if cls_id in run.ap_data.objs:
                ap = run.ap_data.objs[cls_id].get_ap()
                print(f"    {name:20s}  {ap:.4f}")

        error_types = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
        print(f"\n  {'Class':20s}  {'Cls':>5s}  {'Loc':>5s}  {'Both':>5s}  {'Dupe':>5s}  {'Bkg':>5s}  {'Miss':>5s}  {'FN':>5s}")
        print("  " + "-" * 75)
        for name in CLASS_NAMES.values():
            errs = cls_errors.get(name, {})
            fn = fn_counts.get(name, 0)
            row = f"  {name:20s}"
            for et in error_types:
                row += f"  {errs.get(et, 0):5d}"
            row += f"  {fn:5d}"
            print(row)

        # Confusion matrix: which classes get confused with which
        print("\n" + "=" * 70)
        print("CLASSIFICATION CONFUSION MATRIX")
        print("(rows = true class, cols = predicted class)")
        print("=" * 70)
        confusion = defaultdict(lambda: defaultdict(int))
        for err in run.errors:
            if err.short_name == "Cls":
                true_name = CLASS_NAMES.get(err.gt['class'], str(err.gt['class']))
                pred_name = CLASS_NAMES.get(err.pred['class'], str(err.pred['class']))
                confusion[true_name][pred_name] += 1

        all_names = list(CLASS_NAMES.values())
        label = "True \\ Pred"
        header = f"  {label:20s}" + "".join(f"  {n[:8]:>8s}" for n in all_names)
        print(header)
        print("  " + "-" * (20 + 10 * len(all_names)))
        for true_name in all_names:
            row = f"  {true_name:20s}"
            for pred_name in all_names:
                val = confusion[true_name][pred_name]
                row += f"  {val:8d}" if val > 0 else f"  {'·':>8s}"
                # row += f"  {val:8d}"
            print(row)


if __name__ == "__main__":
    main()
