"""
Compare the iOS app's detector output against the Python (ultralytics) pipeline.

Input is the CSV written by DD-ios DD-iOSTests/DetectorBenchmarkTests.swift
(columns: filename,state,detection_count,top_label,top_confidence,all_labels;
all_labels is "label:pct|label:pct" with integer percents). The same images
are run through ultralytics and compared per image and overall:
  - damage / no_damage agreement (+ 2x2 confusion)
  - top-label agreement (images where both sides found damage)
  - mean |top confidence difference| and detection-count difference
  - image-level damage recall vs ground truth, if a labels/ dir sits next to --images
  - the images that disagree most

Matching the app's threshold:
  The app has no threshold of its own. DamageDetectionViewModel keeps every
  VNRecognizedObjectObservation Vision returns, and an image is "damage" if
  there is at least one. Filtering happens in the NMS stage baked into
  DamageDetector.mlpackage by `ultralytics export format=coreml nms=True`:
    confidenceThreshold = 0.25, iouThreshold = 0.7, per-class NMS
  (read from the model spec; Vision does not override them). These match
  ultralytics' predict defaults, so --conf 0.25 --iou 0.7 is the default here.

Known preprocessing differences (expect some disagreement from these alone):
  - .pt weights in ultralytics letterbox to a stride-32 rectangle; the CoreML
    model always takes a 640x640 square. Pass the .mlpackage as --ckpt (macOS
    only) to compare the exact exported model and isolate Vision preprocessing.
  - iOS builds before the letterbox fix stretch the photo to 640x640
    (imageCropAndScaleOption = .scaleFill), which ultralytics never does.
  - iOS all_labels percents are truncated to integers; top_confidence is exact.

Usage:
  python src/compare_ios_benchmark.py --csv cardd_ios_benchmark.csv
  python src/compare_ios_benchmark.py --csv cardd_ios_benchmark.csv --out runs/ios_compare.csv
  python src/compare_ios_benchmark.py --csv cardd_ios_benchmark.csv --ckpt models/DamageDetector.mlpackage
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from paths import DETECTOR_CKPT

APP_CONF = 0.25  # CoreML NMS confidenceThreshold in DamageDetector.mlpackage
APP_IOU = 0.7    # CoreML NMS iouThreshold in DamageDetector.mlpackage
VALID_IOS_STATES = {"damage", "no_damage"}


def pick_device(ckpt: Path) -> str:
    """mps -> cuda -> cpu. CoreML models run on CPU (ultralytics MPS float64 error)."""
    if ckpt.suffix == ".mlpackage":
        return "cpu"
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_all_labels(s: str) -> list:
    """'dent:87|scratch:41' -> [('dent', 0.87), ('scratch', 0.41)]"""
    out = []
    for part in filter(None, (s or "").split("|")):
        label, _, pct = part.rpartition(":")
        out.append((label, int(pct) / 100.0))
    return out


def load_ios_csv(path: Path) -> dict:
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["filename"]] = {
                "state": r["state"],
                "count": int(r["detection_count"] or 0),
                "top_label": r["top_label"],
                "top_conf": float(r["top_confidence"]) if r["top_confidence"] else 0.0,
                "labels": parse_all_labels(r["all_labels"]),
            }
    return rows


def run_python(model: YOLO, img_path: Path, args) -> dict:
    r = model.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf,
                      iou=args.iou, device=args.device, verbose=False)[0]
    confs = r.boxes.conf.cpu().numpy().tolist()
    labels = [model.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
    dets = sorted(zip(labels, confs), key=lambda d: -d[1])
    return {
        "state": "damage" if dets else "no_damage",
        "count": len(dets),
        "top_label": dets[0][0] if dets else "",
        "top_conf": dets[0][1] if dets else 0.0,
        "labels": dets,
    }


def has_gt_boxes(label_dir: Path, img_path: Path):
    """True/False if a labels dir exists, else None."""
    if not label_dir.is_dir():
        return None
    lbl = label_dir / (img_path.stem + ".txt")
    return lbl.exists() and bool(lbl.read_text().strip())


def fmt_labels(dets: list) -> str:
    return "|".join(f"{l}:{c:.2f}" for l, c in dets)


def main():
    ap = argparse.ArgumentParser(description="Compare iOS benchmark CSV vs Python detector")
    ap.add_argument("--csv", type=Path, required=True,
                    help="cardd_ios_benchmark.csv from DetectorBenchmarkTests")
    ap.add_argument("--ckpt", type=Path, default=DETECTOR_CKPT,
                    help="Detector .pt (default: v7) or .mlpackage (macOS only)")
    ap.add_argument("--images", type=Path, default=Path("data/cardd_yolo/test/images"))
    ap.add_argument("--conf", type=float, default=APP_CONF,
                    help="Confidence threshold (default matches the app: 0.25)")
    ap.add_argument("--iou", type=float, default=APP_IOU,
                    help="NMS IoU threshold (default matches the app: 0.7)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="Default: mps -> cuda -> cpu")
    ap.add_argument("--top", type=int, default=15, help="How many worst disagreements to list")
    ap.add_argument("--out", type=Path, default=None, help="Optional per-image CSV")
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"iOS CSV not found: {args.csv}")
    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    if args.device is None:
        args.device = pick_device(args.ckpt)

    ios = load_ios_csv(args.csv)
    states = Counter(r["state"] for r in ios.values())
    print(f"iOS CSV:  {args.csv} ({len(ios)} rows: "
          + ", ".join(f"{k}={v}" for k, v in sorted(states.items())) + ")")
    print(f"Model:    {args.ckpt}  (device={args.device}, conf={args.conf}, iou={args.iou}, imgsz={args.imgsz})")
    print(f"Images:   {args.images}")

    label_dir = args.images.parent / "labels"
    model = YOLO(str(args.ckpt), task="detect")

    rows, missing, skipped = [], [], []
    for i, name in enumerate(sorted(ios)):
        io_r = ios[name]
        if io_r["state"] not in VALID_IOS_STATES:
            skipped.append(name)
            continue
        img_path = args.images / name
        if not img_path.is_file():
            missing.append(name)
            continue
        py_r = run_python(model, img_path, args)

        state_agree = io_r["state"] == py_r["state"]
        both_damage = io_r["state"] == py_r["state"] == "damage"
        label_agree = (io_r["top_label"] == py_r["top_label"]) if both_damage else None
        conf_diff = io_r["top_conf"] - py_r["top_conf"]  # no_damage counts as 0
        # Disagreement score: state flip dominates, then label flip, then confidence gap
        score = (0 if state_agree else 2) + (1 if label_agree is False else 0) + abs(conf_diff)
        rows.append({
            "filename": name,
            "gt_damage": has_gt_boxes(label_dir, img_path),
            "ios_state": io_r["state"], "py_state": py_r["state"],
            "ios_count": io_r["count"], "py_count": py_r["count"],
            "ios_top_label": io_r["top_label"], "py_top_label": py_r["top_label"],
            "ios_top_conf": round(io_r["top_conf"], 4), "py_top_conf": round(py_r["top_conf"], 4),
            "conf_diff": round(conf_diff, 4),
            "state_agree": state_agree, "label_agree": label_agree,
            "score": round(score, 4),
            "ios_labels": fmt_labels(io_r["labels"]),
            "py_labels": fmt_labels(py_r["labels"]),
        })
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(ios)}")

    if missing:
        print(f"\n  {len(missing)} iOS rows have no matching image in {args.images} "
              f"(e.g. {', '.join(missing[:3])}) — skipped")
    if skipped:
        print(f"  {len(skipped)} iOS rows not damage/no_damage (load_failed/stuck/idle) — skipped")
    if not rows:
        raise SystemExit("No images to compare.")

    n = len(rows)
    agree = sum(r["state_agree"] for r in rows)
    both = [r for r in rows if r["label_agree"] is not None]
    lab_agree = sum(r["label_agree"] for r in both)
    conf_abs = np.array([abs(r["conf_diff"]) for r in rows])
    conf_both = np.array([r["conf_diff"] for r in both]) if both else np.array([0.0])
    count_abs = np.array([abs(r["ios_count"] - r["py_count"]) for r in rows])

    def cell(i_state, p_state):
        return sum(1 for r in rows if r["ios_state"] == i_state and r["py_state"] == p_state)

    print(f"\n{'=' * 70}")
    print(f"OVERALL ({n} images)")
    print(f"{'=' * 70}")
    print(f"  Damage/no-damage agreement: {agree}/{n} = {agree / n:.1%}")
    print(f"\n  {'':16s}  {'py damage':>10s}  {'py no_damage':>12s}")
    print(f"  {'ios damage':16s}  {cell('damage', 'damage'):10d}  {cell('damage', 'no_damage'):12d}")
    print(f"  {'ios no_damage':16s}  {cell('no_damage', 'damage'):10d}  {cell('no_damage', 'no_damage'):12d}")
    if both:
        print(f"\n  Top-label agreement (both found damage): {lab_agree}/{len(both)} = {lab_agree / len(both):.1%}")
        print(f"  Top-conf diff, both damage (ios - py): mean {conf_both.mean():+.4f}, "
              f"mean |diff| {np.abs(conf_both).mean():.4f}, max |diff| {np.abs(conf_both).max():.4f}")
    print(f"  Top-conf |diff|, all images (no_damage = 0): mean {conf_abs.mean():.4f}")
    print(f"  Detection count |diff|: mean {count_abs.mean():.2f}, max {count_abs.max()}, "
          f"exact match {int((count_abs == 0).sum())}/{n}")

    gt_rows = [r for r in rows if r["gt_damage"] is not None]
    if gt_rows:
        gt_pos = [r for r in gt_rows if r["gt_damage"]]
        gt_neg = [r for r in gt_rows if not r["gt_damage"]]
        print(f"\n  Ground truth ({label_dir}): {len(gt_pos)} damaged, {len(gt_neg)} without labels")
        for side in ("ios", "py"):
            if gt_pos:
                rec = sum(r[f"{side}_state"] == "damage" for r in gt_pos) / len(gt_pos)
                print(f"    {side:3s} image-level damage recall: {rec:.1%}")
            if gt_neg:
                fpr = sum(r[f"{side}_state"] == "damage" for r in gt_neg) / len(gt_neg)
                print(f"    {side:3s} image-level FP rate:       {fpr:.1%}")

    # Per-class top-label confusion where both sides found damage
    if both:
        pairs = Counter((r["ios_top_label"], r["py_top_label"]) for r in both if not r["label_agree"])
        if pairs:
            print("\n  Top-label mismatches (ios -> py):")
            for (a, b), c in pairs.most_common(10):
                print(f"    {a:15s} -> {b:15s} {c}")

    worst = sorted(rows, key=lambda r: -r["score"])[:args.top]
    worst = [r for r in worst if r["score"] > 0]
    if worst:
        print(f"\n{'=' * 70}")
        print(f"TOP {len(worst)} DISAGREEMENTS")
        print(f"{'=' * 70}")
        for r in worst:
            print(f"  {r['filename']:14s} ios={r['ios_state']:9s} {r['ios_top_label'] or '-':13s} {r['ios_top_conf']:.2f}"
                  f" | py={r['py_state']:9s} {r['py_top_label'] or '-':13s} {r['py_top_conf']:.2f}"
                  f" | n={r['ios_count']}/{r['py_count']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: -r["score"]))
        print(f"\nPer-image results written to {args.out}")


if __name__ == "__main__":
    main()
