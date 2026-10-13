"""
Run damage detection on one or more images.
Prints detected damage types + confidence, and saves annotated images.

Usage:
  python src/detect.py photo.jpg
  python src/detect.py photo1.jpg photo2.jpg --conf 0.3
  python src/detect.py images_folder/ --out results/
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from paths import DETECTOR_CKPT


def main():
    ap = argparse.ArgumentParser(description="Detect car damage in images")
    ap.add_argument("source", nargs="+", help="Image path(s) or directory")
    ap.add_argument("--ckpt", type=Path, default=DETECTOR_CKPT,
                    help="Detector weights (default: v7)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--out", type=Path, default=Path("runs/detect/inference"),
                    help="Output directory for annotated images")
    args = ap.parse_args()

    if not args.ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    model = YOLO(str(args.ckpt))

    # ultralytics accepts a directory only as a single source, not inside a list
    sources = [str(s) for s in args.source]
    results = model.predict(
        source=sources[0] if len(sources) == 1 else sources,
        conf=args.conf,
        imgsz=640,
        device="mps",
        save=True,
        project=str(args.out.parent),
        name=args.out.name,
        exist_ok=True,
    )

    for r in results:
        name = Path(r.path).name
        boxes = r.boxes
        if len(boxes) == 0:
            print(f"{name}: no damage detected")
            continue

        print(f"\n{name}:")
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {label:20s}  conf={conf:.2f}  box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

    print(f"\nAnnotated images saved to {args.out}/")


if __name__ == "__main__":
    main()
