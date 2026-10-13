# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DamageDoctor has two models:
1. **Binary classifier** — detects whether a car photo has visible damage. MobileNetV3-Small, single logit, BCEWithLogitsLoss. Classes: `damage` (0), `no_damage` (1).
2. **Damage detector** — localizes and classifies damage types. YOLOv8s, 6 classes: dent, scratch, crack, glass shatter, lamp broken, tire flat.

Pipeline: binary classifier gates the detector (run boxes only when damage detected).

## Environment Setup

Uses Python 3.9 with a `.venv` in the project root.

```bash
source .venv/bin/activate
```

Key dependencies (no requirements.txt — installed in .venv): `torch 2.8`, `torchvision 0.23`, `ultralytics`, `numpy`, `tqdm`, `scikit-learn`, `flask`, `Pillow`.

All scripts must be run from the **project root**, not from `src/`.

## Commands

### Binary Classifier

```bash
python src/build_dataset.py           # Rebuild data/processed/ from raw sources
python src/train_binary.py            # Train (--epochs, --batch, --lr, --img, --out)
python src/eval_test.py               # Confusion matrix + classification report
python src/eval_clean_only.py --data_dir data/eval_by_source/harder_negatives  # FP rate
python src/eval_no_damage_all.py      # FP rate across all splits
```

### Damage Detector

```bash
python src/eval_detect_test.py                   # Evaluate detector on test set
python src/detect.py                             # Run inference on images
```

### Utilities

```bash
python src/demo.py                    # Flask web demo (both models)
python src/diagnose.py img1.jpg       # CLI inference (both models)
python src/export_coreml.py           # Export binary classifier to CoreML
```

## Data Layout

```
data/
  cardd_yolo/                       # CarDD in YOLO format (2816/810/374 train/val/test)
    {train,val,test}/{images,labels}/
    data.yaml
  archive (3)/                      # VeHIDE raw download (kept as reference, not used)
  raw/                              # Binary classifier raw images
    cardd_damage/                   # 4000 damaged
    clean_cars/                     # 1253 clean
    clean_cars_extra/               # 199 clean
    external_datasets/cars_dataset_2/  # 951 clean
    harder_negatives/               # Phone photos (glare, stickers edge cases)
  processed/                        # Binary classifier splits (rebuilt by build_dataset.py)
  eval_by_source/                   # Per-source FP eval dirs
```

## Current Model Status

### Binary Classifier (v2) — DONE
- Test accuracy: 99.61%, damage recall: 100%
- Remaining weakness: extreme specular glare (1.2% FP on harder_negatives)
- Checkpoint: `models/binary_mobilenetv3_best.pt`

### Detector (v7) — CarDD + 200 Background Images — PRODUCTION
- mAP50: 0.749, mAP50-95: 0.589
- Strong: glass shatter (0.988), tire flat (0.975), lamp broken (0.906)
- Weak: crack (0.459), scratch (0.567), dent (0.599)
- Weights: `runs/kaggle_cardd_v7_detect/best.pt`
- Recipe: exact v1 config (YOLOv8s, SGD, 50ep, batch=16, default aug) + 200 background images (7% of train set)
- Background images: 165 harder_negatives + 35 clean_cars, no label files
- Kaggle dataset version: `ssarvesh20000/damagedoctor-cardd-yolo` v2
- TIDE (v1): Loc=6.34 dAP (dominant), Bkg=4.43, Cls=1.22 — weak classes are label-limited, not model-limited

### Detector (v1) — Previous Baseline (Superseded by v7)
- mAP50: 0.734, mAP50-95: 0.581
- Weights: `runs/detect/runs/detect/cardd_v1/weights/best.pt`
- Training config: `runs/detect/runs/detect/cardd_v1/args.yaml`

### Experiment History
Seven experiments tested; only v7 (data-side: background images) beat v1:
- v1: YOLOv8s, 50ep, default aug → mAP50 0.734 (original baseline)
- v3: CarDD+VeHIDE combined → 0.677 (regressed)
- v4: YOLOv8m, heavy aug → 0.704 (regressed)
- v5: YOLO11s-seg, 1280, copy_paste → 0.681 (regressed, but crack +0.080)
- v6: YOLO11s, 640, conservative aug → 0.696 (regressed)
- **v7: YOLOv8s, exact v1 recipe + 200 bg images → 0.749 (NEW BASELINE)**
- v8: Same as v7 + 280 bg images → 0.726 (10% bg ratio too high)

Lesson: five hyperparameter experiments all failed. First data-side experiment succeeded immediately. Remaining errors are addressable through data, not training config.

### Kaggle Training Workflow
- `kaggle/` directory has a working Makefile-driven workflow for Kaggle's free P100 GPU
- Reusable for future training if needed
- Key gotcha: must downgrade torch to 2.5.1+cu121 for P100 sm_60 compatibility

## Immediate Next Steps

1. **Export v7 detector to CoreML** — first detector CoreML export for iOS pipeline
2. **Run TIDE on v7** — confirm Bkg error dropped, establish new error profile for next intervention
3. **Binary classifier** — collect more extreme glare photos, retrain, export to CoreML
4. **End-to-end pipeline** — binary classifier gates detector for iOS app
5. **iOS app integration**
