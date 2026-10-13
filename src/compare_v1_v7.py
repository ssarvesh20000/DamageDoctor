"""
Compare v1 vs v7 detector: FP rate on clean cars + recall on CarDD test.

Runs both PyTorch and CoreML models, checks for prediction drift.

Outputs:
  1. Summary table: FP counts at 0.25/0.5/0.7, mean FP conf, test recall @0.5
  2. Per-class FP breakdown on harder_negatives
  3. Side-by-side PNGs for disagreement cases (conf >= 0.5)
  4. CoreML vs PyTorch drift report

Usage:
  python src/compare_v1_v7.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --- Config ---
CLASS_NAMES = {0: "dent", 1: "scratch", 2: "crack",
               3: "glass shatter", 4: "lamp broken", 5: "tire flat"}

MODELS = {
    "v1_pt": Path("runs/detect/runs/detect/cardd_v1/weights/best.pt"),
    "v7_pt": Path("runs/kaggle_cardd_v7_detect/best.pt"),
    "v1_coreml": Path("models/DamageDetector_v1.mlpackage"),
    "v7_coreml": Path("models/DamageDetector.mlpackage"),
}

FP_DIR = Path("data/raw/harder_negatives")
TEST_IMG_DIR = Path("data/cardd_yolo/test/images")
TEST_LBL_DIR = Path("data/cardd_yolo/test/labels")
OUT_DIR = Path("runs/compare_v1_v7")

CONF_LOW = 0.05  # inference threshold — post-filter later
IMGSZ = 640
THRESHOLDS = [0.25, 0.5, 0.7]


def run_inference(model: YOLO, img_dir: Path, device: str = "mps") -> list:
    """Run model on all images at low conf. Returns list of dicts."""
    rows = []
    img_paths = sorted(
        p for p in img_dir.glob("*.*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    for img_path in img_paths:
        results = model.predict(
            source=str(img_path), imgsz=IMGSZ, device=device,
            conf=CONF_LOW, iou=0.6, verbose=False,
        )
        for r in results:
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                rows.append({
                    "image": img_path.name,
                    "class_id": int(r.boxes.cls[i].cpu()),
                    "class_name": CLASS_NAMES[int(r.boxes.cls[i].cpu())],
                    "conf": float(r.boxes.conf[i].cpu()),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                })
    return rows


def load_gt_boxes(label_dir: Path, img_dir: Path) -> dict:
    """Load YOLO ground truth. Returns {image_name: [(cls_id, x1, y1, x2, y2), ...]}."""
    gt = {}
    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl_path = label_dir / (img_path.stem + ".txt")
        boxes = []
        if lbl_path.exists():
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for line in lbl_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append((cls_id, x1, y1, x2, y2))
        gt[img_path.name] = boxes
    return gt


def iou(box_a, box_b):
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_recall(preds: list, gt: dict, conf_thresh: float, iou_thresh: float = 0.5):
    """Compute per-class recall at a given confidence threshold."""
    # Count GT instances per class
    gt_counts = defaultdict(int)
    for boxes in gt.values():
        for cls_id, *_ in boxes:
            gt_counts[cls_id] += 1

    # Match predictions to GT (greedy, highest conf first)
    matched = defaultdict(int)
    preds_sorted = sorted(
        [p for p in preds if p["conf"] >= conf_thresh],
        key=lambda p: -p["conf"]
    )

    # Track which GT boxes have been matched per image
    gt_matched = defaultdict(set)

    for p in preds_sorted:
        img = p["image"]
        if img not in gt:
            continue
        p_box = (p["x1"], p["y1"], p["x2"], p["y2"])
        best_iou, best_idx = 0, -1
        for idx, (cls_id, *gt_box) in enumerate(gt[img]):
            if cls_id != p["class_id"]:
                continue
            if idx in gt_matched[img]:
                continue
            v = iou(p_box, tuple(gt_box))
            if v > best_iou:
                best_iou, best_idx = v, idx
        if best_iou >= iou_thresh and best_idx >= 0:
            matched[p["class_id"]] += 1
            gt_matched[img].add(best_idx)

    # Per-class recall
    recall = {}
    for cls_id in sorted(gt_counts.keys()):
        total = gt_counts[cls_id]
        found = matched[cls_id]
        recall[cls_id] = found / total if total > 0 else 0.0
    return recall, gt_counts


def fp_summary(preds: list, thresholds: list):
    """FP summary stats at various thresholds."""
    results = {}
    for t in thresholds:
        above = [p for p in preds if p["conf"] >= t]
        results[t] = len(above)

    confs = [p["conf"] for p in preds if p["conf"] >= CONF_LOW]
    results["mean_conf"] = np.mean(confs) if confs else 0.0
    results["total_raw"] = len(preds)
    return results


def fp_per_class(preds: list, threshold: float):
    """Per-class FP counts at given threshold."""
    counts = defaultdict(int)
    for p in preds:
        if p["conf"] >= threshold:
            counts[p["class_name"]] += 1
    return dict(counts)


def render_disagreements(v1_preds: list, v7_preds: list,
                         img_dir: Path, out_dir: Path, conf_thresh: float = 0.5):
    """Render side-by-side images where models disagree at conf >= threshold."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by image
    v1_by_img = defaultdict(list)
    v7_by_img = defaultdict(list)
    for p in v1_preds:
        if p["conf"] >= conf_thresh:
            v1_by_img[p["image"]].append(p)
    for p in v7_preds:
        if p["conf"] >= conf_thresh:
            v7_by_img[p["image"]].append(p)

    all_imgs = set(v1_by_img.keys()) | set(v7_by_img.keys())
    rendered = 0

    for img_name in sorted(all_imgs):
        v1_dets = v1_by_img.get(img_name, [])
        v7_dets = v7_by_img.get(img_name, [])

        # Skip if both models agree (same number of detections)
        if len(v1_dets) == len(v7_dets) == 0:
            continue
        if len(v1_dets) == len(v7_dets):
            # Same count — check if same classes
            v1_cls = sorted(d["class_name"] for d in v1_dets)
            v7_cls = sorted(d["class_name"] for d in v7_dets)
            if v1_cls == v7_cls:
                continue

        # Load image
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Draw v1 (left) and v7 (right)
        left = img.copy()
        right = img.copy()

        for det in v1_dets:
            pt1 = (int(det["x1"]), int(det["y1"]))
            pt2 = (int(det["x2"]), int(det["y2"]))
            cv2.rectangle(left, pt1, pt2, (0, 0, 255), 2)
            label = f"{det['class_name']} {det['conf']:.2f}"
            cv2.putText(left, label, (pt1[0], pt1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for det in v7_dets:
            pt1 = (int(det["x1"]), int(det["y1"]))
            pt2 = (int(det["x2"]), int(det["y2"]))
            cv2.rectangle(right, pt1, pt2, (0, 255, 0), 2)
            label = f"{det['class_name']} {det['conf']:.2f}"
            cv2.putText(right, label, (pt1[0], pt1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Add model labels
        cv2.putText(left, "v1", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(right, "v7", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        combined = np.hstack([left, right])
        stem = Path(img_name).stem
        cv2.imwrite(str(out_dir / f"{stem}_compare.jpg"), combined)
        rendered += 1

    return rendered


def coreml_drift_check(pt_preds: list, coreml_preds: list, label: str):
    """Compare PyTorch vs CoreML predictions for drift."""
    # Match by image — compare detection counts and max conf diff
    pt_by_img = defaultdict(list)
    cm_by_img = defaultdict(list)
    for p in pt_preds:
        pt_by_img[p["image"]].append(p)
    for p in coreml_preds:
        cm_by_img[p["image"]].append(p)

    all_imgs = set(pt_by_img.keys()) | set(cm_by_img.keys())
    count_diffs = []
    conf_diffs = []

    for img in all_imgs:
        pt_dets = sorted(pt_by_img.get(img, []), key=lambda d: -d["conf"])
        cm_dets = sorted(cm_by_img.get(img, []), key=lambda d: -d["conf"])

        # Compare detection count at conf >= 0.25
        pt_n = sum(1 for d in pt_dets if d["conf"] >= 0.25)
        cm_n = sum(1 for d in cm_dets if d["conf"] >= 0.25)
        count_diffs.append(abs(pt_n - cm_n))

        # Compare top conf per image
        pt_top = pt_dets[0]["conf"] if pt_dets else 0
        cm_top = cm_dets[0]["conf"] if cm_dets else 0
        conf_diffs.append(abs(pt_top - cm_top))

    print(f"\n  CoreML drift ({label}):")
    print(f"    Images compared: {len(all_imgs)}")
    print(f"    Detection count diff (mean): {np.mean(count_diffs):.2f}")
    print(f"    Detection count diff (max):  {max(count_diffs)}")
    print(f"    Top-conf diff (mean): {np.mean(conf_diffs):.4f}")
    print(f"    Top-conf diff (max):  {max(conf_diffs):.4f}")
    exact = sum(1 for d in count_diffs if d == 0)
    print(f"    Exact count match: {exact}/{len(all_imgs)} images")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load models ---
    print("Loading models...")
    v1_pt = YOLO(str(MODELS["v1_pt"]))
    v7_pt = YOLO(str(MODELS["v7_pt"]))

    # --- Part 1: FP analysis on harder_negatives ---
    print(f"\n{'='*70}")
    print("PART 1: FALSE POSITIVES ON HARDER_NEGATIVES (clean cars)")
    print(f"{'='*70}")

    n_imgs = len(list(FP_DIR.glob("*.jpeg"))) + len(list(FP_DIR.glob("*.jpg")))
    print(f"  {n_imgs} images (all clean — every detection is a false positive)\n")

    print("  Running v1 inference...")
    v1_fp = run_inference(v1_pt, FP_DIR)
    print(f"    {len(v1_fp)} raw detections (conf >= {CONF_LOW})")

    print("  Running v7 inference...")
    v7_fp = run_inference(v7_pt, FP_DIR)
    print(f"    {len(v7_fp)} raw detections (conf >= {CONF_LOW})")

    # Summary table
    v1_stats = fp_summary(v1_fp, THRESHOLDS)
    v7_stats = fp_summary(v7_fp, THRESHOLDS)

    print(f"\n  FP Summary:")
    print(f"  {'':15s}  {'v1':>8s}  {'v7':>8s}  {'delta':>8s}")
    print(f"  {'-'*45}")
    print(f"  {'raw (conf>.05)':15s}  {v1_stats['total_raw']:8d}  {v7_stats['total_raw']:8d}  {v7_stats['total_raw'] - v1_stats['total_raw']:+8d}")
    for t in THRESHOLDS:
        delta = v7_stats[t] - v1_stats[t]
        print(f"  {'FP @'+str(t):15s}  {v1_stats[t]:8d}  {v7_stats[t]:8d}  {delta:+8d}")
    print(f"  {'mean FP conf':15s}  {v1_stats['mean_conf']:8.3f}  {v7_stats['mean_conf']:8.3f}  {v7_stats['mean_conf'] - v1_stats['mean_conf']:+8.3f}")

    # Per-class FP at 0.25
    print(f"\n  Per-class FPs (conf >= 0.25):")
    v1_cls = fp_per_class(v1_fp, 0.25)
    v7_cls = fp_per_class(v7_fp, 0.25)
    all_classes = sorted(set(v1_cls.keys()) | set(v7_cls.keys()))
    print(f"  {'class':20s}  {'v1':>6s}  {'v7':>6s}  {'delta':>6s}")
    print(f"  {'-'*40}")
    for c in CLASS_NAMES.values():
        v1_n = v1_cls.get(c, 0)
        v7_n = v7_cls.get(c, 0)
        if v1_n == 0 and v7_n == 0:
            continue
        print(f"  {c:20s}  {v1_n:6d}  {v7_n:6d}  {v7_n - v1_n:+6d}")

    # Per-class FP at 0.5
    print(f"\n  Per-class FPs (conf >= 0.5):")
    v1_cls5 = fp_per_class(v1_fp, 0.5)
    v7_cls5 = fp_per_class(v7_fp, 0.5)
    print(f"  {'class':20s}  {'v1':>6s}  {'v7':>6s}  {'delta':>6s}")
    print(f"  {'-'*40}")
    for c in CLASS_NAMES.values():
        v1_n = v1_cls5.get(c, 0)
        v7_n = v7_cls5.get(c, 0)
        if v1_n == 0 and v7_n == 0:
            continue
        print(f"  {c:20s}  {v1_n:6d}  {v7_n:6d}  {v7_n - v1_n:+6d}")

    # --- Part 2: Recall on CarDD test ---
    print(f"\n{'='*70}")
    print("PART 2: RECALL ON CARDD TEST SET (labeled damage)")
    print(f"{'='*70}")

    gt = load_gt_boxes(TEST_LBL_DIR, TEST_IMG_DIR)
    n_test = len(gt)
    print(f"  {n_test} images, {sum(len(v) for v in gt.values())} GT boxes\n")

    print("  Running v1 inference on test set...")
    v1_test = run_inference(v1_pt, TEST_IMG_DIR)
    print(f"    {len(v1_test)} raw detections")

    print("  Running v7 inference on test set...")
    v7_test = run_inference(v7_pt, TEST_IMG_DIR)
    print(f"    {len(v7_test)} raw detections")

    # Recall at 0.5
    v1_recall, gt_counts = compute_recall(v1_test, gt, conf_thresh=0.5)
    v7_recall, _ = compute_recall(v7_test, gt, conf_thresh=0.5)

    v1_total_matched = sum(int(v1_recall[c] * gt_counts[c]) for c in gt_counts)
    v7_total_matched = sum(int(v7_recall[c] * gt_counts[c]) for c in gt_counts)
    v1_overall = v1_total_matched / sum(gt_counts.values())
    v7_overall = v7_total_matched / sum(gt_counts.values())

    print(f"\n  Recall @ conf >= 0.5, IoU >= 0.5:")
    print(f"  {'class':20s}  {'GT':>5s}  {'v1':>8s}  {'v7':>8s}  {'delta':>8s}")
    print(f"  {'-'*50}")
    for cls_id in sorted(gt_counts.keys()):
        name = CLASS_NAMES[cls_id]
        n = gt_counts[cls_id]
        v1_r = v1_recall.get(cls_id, 0)
        v7_r = v7_recall.get(cls_id, 0)
        print(f"  {name:20s}  {n:5d}  {v1_r:8.3f}  {v7_r:8.3f}  {v7_r - v1_r:+8.3f}")
    print(f"  {'-'*50}")
    print(f"  {'OVERALL':20s}  {sum(gt_counts.values()):5d}  {v1_overall:8.3f}  {v7_overall:8.3f}  {v7_overall - v1_overall:+8.3f}")

    # --- Part 3: Disagreement side-by-sides ---
    print(f"\n{'='*70}")
    print("PART 3: DISAGREEMENT VISUALIZATIONS")
    print(f"{'='*70}")

    vis_dir = OUT_DIR / "disagreements"
    n_rendered = render_disagreements(v1_fp, v7_fp, FP_DIR, vis_dir, conf_thresh=0.5)
    print(f"  Rendered {n_rendered} disagreement images to {vis_dir}/")

    # --- Part 4: CoreML drift check ---
    print(f"\n{'='*70}")
    print("PART 4: COREML vs PYTORCH DRIFT CHECK")
    print(f"{'='*70}")

    v1_coreml_path = MODELS["v1_coreml"]
    v7_coreml_path = MODELS["v7_coreml"]

    if v1_coreml_path.exists() and v7_coreml_path.exists():
        # Run CoreML on a subset (harder_negatives is small enough)
        # CoreML via ultralytics must run on CPU to avoid MPS float64 error
        print("\n  Running v1 CoreML inference on harder_negatives...")
        v1_cm = YOLO(str(v1_coreml_path), task="detect")
        v1_cm_fp = run_inference(v1_cm, FP_DIR, device="cpu")
        coreml_drift_check(v1_fp, v1_cm_fp, "v1")

        print("\n  Running v7 CoreML inference on harder_negatives...")
        v7_cm = YOLO(str(v7_coreml_path), task="detect")
        v7_cm_fp = run_inference(v7_cm, FP_DIR, device="cpu")
        coreml_drift_check(v7_fp, v7_cm_fp, "v7")
    else:
        missing = []
        if not v1_coreml_path.exists():
            missing.append(str(v1_coreml_path))
        if not v7_coreml_path.exists():
            missing.append(str(v7_coreml_path))
        print(f"  Skipped — missing CoreML models: {', '.join(missing)}")

    # --- Save raw detections CSV ---
    csv_path = OUT_DIR / "detections.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "dataset", "image", "class_id", "class_name",
            "conf", "x1", "y1", "x2", "y2",
        ])
        writer.writeheader()
        for row in v1_fp:
            writer.writerow({"model": "v1", "dataset": "harder_negatives", **row})
        for row in v7_fp:
            writer.writerow({"model": "v7", "dataset": "harder_negatives", **row})
        for row in v1_test:
            writer.writerow({"model": "v1", "dataset": "cardd_test", **row})
        for row in v7_test:
            writer.writerow({"model": "v7", "dataset": "cardd_test", **row})
    print(f"\n  Raw detections saved to {csv_path}")

    # --- Decision framework ---
    print(f"\n{'='*70}")
    print("DECISION FRAMEWORK")
    print(f"{'='*70}")
    fewer_fps = v7_stats[0.5] < v1_stats[0.5]
    holds_recall = v7_overall >= v1_overall - 0.02  # 2% tolerance
    if fewer_fps and holds_recall:
        print("  --> v7 has fewer FPs AND holds recall. Ship v7.")
    elif fewer_fps and not holds_recall:
        print("  --> v7 has fewer FPs but LOSES recall. Check per-class.")
    elif not fewer_fps and holds_recall:
        print("  --> v7 has similar/more FPs but holds recall. Ship whichever is simpler.")
    else:
        print("  --> v7 has more FPs AND loses recall. Debug eval pipeline.")


if __name__ == "__main__":
    main()
