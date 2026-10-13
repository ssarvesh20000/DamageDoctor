"""
Export both DamageDoctor models to CoreML for iOS deployment.

Usage:
  python src/export_coreml.py                   # export both models
  python src/export_coreml.py --classifier      # export binary classifier only
  python src/export_coreml.py --detector        # export YOLOv8 detector only

Outputs:
  models/DamageClassifier.mlpackage    — binary classifier (damage vs no_damage)
  models/DamageDetector.mlpackage      — YOLOv8s object detector (6 damage types)
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class NormalizedClassifier(nn.Module):
    """Wraps the classifier with built-in ImageNet normalization.
    Expects input in [0, 1] range (CoreML default for image inputs)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)


# --- Binary classifier export ---

def export_classifier(ckpt_path: Path, out_path: Path):
    print(f"\n=== Exporting binary classifier ===")
    print(f"  Checkpoint: {ckpt_path}")

    import coremltools as ct

    # Rebuild model architecture
    m = models.mobilenet_v3_small(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, 1)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()

    # Wrap with normalization baked in
    wrapped = NormalizedClassifier(m)
    wrapped.eval()

    img_size = ck.get("img_size", 224)

    # Trace with dummy input in [0, 1] range (what CoreML will provide)
    dummy = torch.rand(1, 3, img_size, img_size)
    traced = torch.jit.trace(wrapped, dummy)

    # Convert to CoreML — scale=1/255 converts [0,255] → [0,1],
    # then the baked-in NormalizedClassifier handles ImageNet mean/std
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, img_size, img_size),
                scale=1.0 / 255.0,
                bias=[0.0, 0.0, 0.0],
                color_layout="RGB",
            )
        ],
        minimum_deployment_target=ct.target.iOS15,
    )

    # Add metadata
    mlmodel.author = "DamageDoctor"
    mlmodel.short_description = "Binary classifier: damage vs no_damage"
    mlmodel.version = "2.1"

    mlmodel.user_defined_metadata["class_to_idx"] = '{"damage": 0, "no_damage": 1}'
    mlmodel.user_defined_metadata["threshold"] = "0.5"
    mlmodel.user_defined_metadata["output_note"] = (
        "Single logit output. Apply sigmoid: < 0.5 means DAMAGE (class 0), >= 0.5 means NO DAMAGE (class 1)."
    )

    mlmodel.save(str(out_path))
    print(f"  Saved to: {out_path}")
    print(f"  Input: {img_size}x{img_size} RGB image")
    print(f"  Output: single logit → sigmoid < 0.5 = damage, >= 0.5 = no_damage")


# --- YOLOv8 detector export ---

def export_detector(ckpt_path: Path, out_dir: Path):
    print(f"\n=== Exporting YOLOv8 detector ===")
    print(f"  Checkpoint: {ckpt_path}")

    from ultralytics import YOLO

    model = YOLO(str(ckpt_path))
    model.export(
        format="coreml",
        imgsz=640,
        nms=True,
    )

    # Ultralytics saves the .mlpackage next to the .pt file
    exported = ckpt_path.with_suffix(".mlpackage")
    dest = out_dir / "DamageDetector.mlpackage"

    if exported.exists():
        # Move to models/ directory
        if dest.exists():
            import shutil
            shutil.rmtree(dest)
        exported.rename(dest)
        print(f"  Saved to: {dest}")
    else:
        print(f"  Exported file location: check {ckpt_path.parent}/")

    print(f"  Input: 640x640 RGB image")
    print(f"  Output: bounding boxes with class labels")
    print(f"  Classes: dent, scratch, crack, glass shatter, lamp broken, tire flat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", action="store_true", help="Export classifier only")
    ap.add_argument("--detector", action="store_true", help="Export detector only")
    ap.add_argument("--classifier-ckpt", type=Path,
                    default=Path("models/binary_mobilenetv3_best.pt"))
    ap.add_argument("--detector-ckpt", type=Path,
                    default=Path("runs/kaggle_cardd_v7_detect/best.pt"))
    args = ap.parse_args()

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    export_both = not args.classifier and not args.detector

    if args.classifier or export_both:
        export_classifier(args.classifier_ckpt, out_dir / "DamageClassifier.mlpackage")

    if args.detector or export_both:
        export_detector(args.detector_ckpt, out_dir)

    print("\n✓ Export complete. Add .mlpackage files to your Xcode project.")


if __name__ == "__main__":
    main()
