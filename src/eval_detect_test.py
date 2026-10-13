"""
Evaluate YOLOv8 detector on the CarDD test set.
Prints per-class and overall metrics, and saves prediction visualizations.

Usage:
  python src/eval_detect_test.py
  python src/eval_detect_test.py --ckpt runs/detect/runs/detect/cardd_v2/weights/best.pt
  python src/eval_detect_test.py --save-imgs 20   # save 20 images with drawn boxes
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path("runs/detect/runs/detect/cardd_v1/weights/best.pt"),
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("data/cardd_yolo/data.yaml"),
    )
    ap.add_argument(
        "--save-imgs",
        type=int,
        default=0,
        help="Number of test images to save with predicted boxes (0 = skip)",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not args.ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {args.ckpt.resolve()}")

    model = YOLO(str(args.ckpt))

    # --- Full test-set metrics ---
    print(f"\nEvaluating {args.ckpt} on test split ...\n")
    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        batch=16,
        device="mps",
        verbose=True,
    )

    print("\n=== Test Set Results ===")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")

    names = model.names
    if hasattr(metrics.box, "ap50"):
        print("\n  Per-class AP50:")
        for i, ap50 in enumerate(metrics.box.ap50):
            print(f"    {names[i]:20s}  {ap50:.4f}")

    # --- Save sample predictions ---
    if args.save_imgs > 0:
        test_img_dir = Path("data/cardd_yolo/test/images")
        out_dir = Path("runs/detect/test_predictions")
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = sorted(test_img_dir.glob("*.*"))[:args.save_imgs]
        print(f"\nSaving {len(imgs)} prediction images to {out_dir} ...")

        results = model.predict(
            source=[str(p) for p in imgs],
            imgsz=args.imgsz,
            device="mps",
            save=False,
            verbose=False,
        )
        for r in results:
            name = Path(r.path).name
            r.save(str(out_dir / name))

        print(f"Done. Open {out_dir}/ to inspect.")


if __name__ == "__main__":
    main()
