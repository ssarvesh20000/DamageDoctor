"""
Diagnose model accuracy on specific images.
Runs both the classifier and detector through the Python pipeline
so you can compare against CoreML/iOS results.

Usage:
  python src/diagnose.py photo1.jpg photo2.jpg photo3.jpg
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image


def load_classifier(ckpt_path):
    m = models.mobilenet_v3_small(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = torch.nn.Linear(in_f, 1)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    img_size = ck.get("img_size", 224)
    class_to_idx = ck["class_to_idx"]
    return m, img_size, class_to_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="Image paths to test")
    ap.add_argument("--classifier-ckpt", type=Path,
                    default=Path("models/binary_mobilenetv3_best.pt"))
    ap.add_argument("--detector-ckpt", type=Path,
                    default=Path("runs/detect/runs/detect/cardd_v1/weights/best.pt"))
    args = ap.parse_args()

    # --- Classifier ---
    print("=" * 60)
    print("BINARY CLASSIFIER")
    print("=" * 60)

    clf, img_size, class_to_idx = load_classifier(args.classifier_ckpt)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"  class_to_idx: {class_to_idx}")
    print(f"  img_size: {img_size}")

    tfm = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for img_path in args.images:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        x = tfm(img).unsqueeze(0)

        with torch.no_grad():
            logit = clf(x).item()

        prob = 1 / (1 + np.exp(-logit))
        pred_idx = int(prob >= 0.5)
        pred_label = idx_to_class[pred_idx]

        print(f"\n  {Path(img_path).name} ({w}x{h}):")
        print(f"    logit  = {logit:.4f}")
        print(f"    sigmoid = {prob:.4f}")
        print(f"    prediction: {pred_label} (idx={pred_idx})")
        print(f"    interpretation: sigmoid >= 0.5 → class 1 ({idx_to_class[1]})")
        print(f"                    sigmoid <  0.5 → class 0 ({idx_to_class[0]})")

    # --- Detector ---
    print("\n" + "=" * 60)
    print("YOLO DETECTOR")
    print("=" * 60)

    if not args.detector_ckpt.is_file():
        print(f"  Detector checkpoint not found: {args.detector_ckpt}")
        return

    from ultralytics import YOLO
    model = YOLO(str(args.detector_ckpt))
    print(f"  Classes: {model.names}")

    for img_path in args.images:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        results = model.predict(
            source=img_path,
            imgsz=640,
            device="mps",
            verbose=False,
        )

        r = results[0]
        print(f"\n  {Path(img_path).name} ({w}x{h}):")
        if len(r.boxes) == 0:
            print("    No detections")
        else:
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"    {label:20s}  conf={conf:.3f}  box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")


if __name__ == "__main__":
    main()
