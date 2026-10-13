"""
Single source of truth for model paths and detector class names.

Scripts are run from the project root as `python src/<script>.py`, so src/ is
on sys.path and `from paths import ...` works without making src/ a package.
All paths are relative to the project root.
"""
from pathlib import Path

# --- Binary classifier (MobileNetV3-Small, damage=0 / no_damage=1) ---
CLASSIFIER_CKPT = Path("models/binary_mobilenetv3_best.pt")
CLASSIFIER_COREML = Path("models/DamageClassifier.mlpackage")

# --- Damage detector (YOLOv8s) ---
# v7 = production (v1 recipe + 200 background images)
DETECTOR_CKPT = Path("runs/kaggle_cardd_v7_detect/best.pt")
DETECTOR_COREML = Path("models/DamageDetector.mlpackage")

# v1 = previous baseline, kept for comparisons
DETECTOR_V1_CKPT = Path("runs/detect/runs/detect/cardd_v1/weights/best.pt")
DETECTOR_V1_COREML = Path("models/DamageDetector_v1.mlpackage")

# Detector class ids → names (must match data/cardd_yolo/data.yaml)
CLASS_NAMES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass shatter",
    4: "lamp broken",
    5: "tire flat",
}
