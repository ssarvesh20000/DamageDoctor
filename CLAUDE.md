# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DamageDoctor detects car damage from phone photos. Two repos:
- **DamageDoctor** (this repo): Python ML — training, evaluation, CoreML export.
- **DD-ios** (github.com/ssarvesh20000/DD-ios, active branch `working`): SwiftUI app that runs the detector on device via CoreML + Vision.

Two models:
1. **Binary classifier** — detects whether a car photo has visible damage. MobileNetV3-Small, single logit, BCEWithLogitsLoss. Classes: `damage` (0), `no_damage` (1), so sigmoid < 0.5 means DAMAGE.
2. **Damage detector** — localizes and classifies damage types. YOLOv8s, 6 classes: 0 dent, 1 scratch, 2 crack, 3 glass shatter, 4 lamp broken, 5 tire flat.

Pipeline today:
- **iOS app: detector only.** The classifier gate was removed from the app (DD-ios commit 588977d, "faulty"). An image is "damage" if the detector returns at least one box above 0.25.
- **Python `demo.py` / `diagnose.py`** still run classifier → detector (demo only runs boxes when the classifier says damage).
- **Open question:** bring the classifier gate back in the app or not. Decide after the iOS letterbox/orientation fix (see Known Problems P3), since the app's preprocessing differed from training for both models.

## Environment Setup

Uses Python 3.9 with a `.venv` in the project root.

```bash
source .venv/bin/activate
```

Key dependencies (no requirements.txt — installed in .venv): `torch 2.8`, `torchvision 0.23`, `ultralytics`, `numpy`, `tqdm`, `scikit-learn`, `flask`, `Pillow`, `coremltools`, `opencv-python`, `tidecv`.

All scripts must be run from the **project root**, not from `src/`. Code must stay Python 3.9 compatible (no `match`, no runtime `X | Y` unions).

Scripts default to `device="mps"`; ultralytics falls back to CPU when MPS is unavailable. New code that picks a device should use mps → cuda → cpu.

## Model Paths

All model paths and the detector class names live in `src/paths.py` (`CLASSIFIER_CKPT`, `DETECTOR_CKPT` = v7, `DETECTOR_V1_CKPT`, `CLASSIFIER_COREML`, `DETECTOR_COREML`, `DETECTOR_V1_COREML`, `CLASS_NAMES`). Scripts import from it (`from paths import ...` works because `python src/x.py` puts `src/` on `sys.path`; `src/` is not a package). Change paths there, not in individual scripts.

| What | Path |
|---|---|
| Classifier checkpoint | `models/binary_mobilenetv3_best.pt` |
| Classifier CoreML | `models/DamageClassifier.mlpackage` |
| Detector v7 (production) | `runs/kaggle_cardd_v7_detect/best.pt` |
| Detector v7 CoreML (done; shipped in DD-ios) | `models/DamageDetector.mlpackage` |
| Detector v1 (old baseline) | `runs/detect/runs/detect/cardd_v1/weights/best.pt` |
| Detector v1 CoreML | `models/DamageDetector_v1.mlpackage` |

Weights, `data/` and `runs/` are gitignored.

## Commands

### Binary Classifier

```bash
python src/build_dataset.py           # Rebuild data/processed/ + data/eval_by_source/ (test-split images only)
python src/build_dataset.py --eval_all_images   # Old eval_by_source (includes train/val images — biased)
python src/train_binary.py            # Train (--epochs, --batch, --lr, --img, --out)
python src/eval_test.py               # Confusion matrix + classification report
python src/eval_clean_only.py --data_dir data/eval_by_source/harder_negatives  # FP rate (held-out)
python src/eval_no_damage_all.py      # FP rate across all splits
```

### Damage Detector

```bash
python src/eval_detect_test.py                   # Evaluate detector (v7) on test set
python src/detect.py photo.jpg                   # Run inference on images or a folder
python src/tide_analysis.py --split test         # TIDE error breakdown
python src/compare_v1_v7.py --fp_dir data/eval_by_source/cars_dataset_2/no_damage   # v1 vs v7 FPs/recall/CoreML drift
python src/compare_ios_benchmark.py --csv cardd_ios_benchmark.csv   # iOS app vs Python on the same images
```

### Utilities

```bash
python src/demo.py                    # Flask web demo (classifier → detector), http://localhost:8080
python src/diagnose.py img1.jpg       # CLI inference (both models)
python src/export_coreml.py           # Export classifier + detector to CoreML (--classifier / --detector)
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
    harder_negatives/               # 165 phone photos (glare, stickers edge cases)
  processed/                        # Binary classifier splits (rebuilt by build_dataset.py)
  eval_by_source/                   # Per-source FP eval dirs: only each source's classifier TEST-split images
```

Which images each model has trained on (matters for any FP evaluation):
- Classifier: everything in `processed/train` (≈80% of each no_damage source).
- Detector v7: CarDD train + 200 background images = **all 165 `harder_negatives` + 35 `clean_cars`** (which 35 is not recorded in this repo). `clean_cars_extra` and `cars_dataset_2` were never detector training data.

## Current Model Status

### Binary Classifier (v2) — trained, CoreML exported, NOT used by the app
- Test accuracy: 99.61%, damage recall: 100%
- Per-source FP rates (clean_cars 0.1%, clean_cars_extra 1.0%, cars_dataset_2 0.9%, harder_negatives 1.2%) were measured on the old `eval_by_source/`, which included training images → **optimistic; re-measure** with the new held-out eval sets.
- Remaining weakness: extreme specular glare.
- Checkpoint: `models/binary_mobilenetv3_best.pt`; CoreML: `models/DamageClassifier.mlpackage`
- Preprocessing it was trained with: `Resize(int(224*1.15))` + `CenterCrop(224)` + ImageNet normalization (baked into the CoreML export).

### Detector (v7) — CarDD + 200 Background Images — PRODUCTION
- **mAP50: 0.754 test / 0.749 val**
- Per-class AP50 (test): dent 0.642, scratch 0.627, crack 0.463, glass shatter 0.991, lamp broken 0.868, tire flat 0.932
- Weak classes: crack, scratch, dent
- Weights: `runs/kaggle_cardd_v7_detect/best.pt`; CoreML: `models/DamageDetector.mlpackage` (done, in the DD-ios app)
- CoreML export (`ultralytics export format=coreml nms=True`) has NMS built in: conf 0.25, IoU 0.7, per-class. Class scores are padded 6 → 80 before NMS (labels "6"–"79" are placeholders).
- Recipe: exact v1 config (YOLOv8s, SGD, 50ep, batch=16, default aug) + 200 background images (7% of train set), no label files
- Kaggle dataset version: `ssarvesh20000/damagedoctor-cardd-yolo` v2
- TIDE (v1): Loc=6.34 dAP (dominant), Bkg=4.43, Cls=1.22 — weak classes are label-limited, not model-limited. TIDE not yet run on v7.

### Detector (v1) — Previous Baseline (Superseded by v7)
- mAP50: 0.734
- Weights: `runs/detect/runs/detect/cardd_v1/weights/best.pt`
- Training config: `runs/detect/runs/detect/cardd_v1/args.yaml`

### Experiment History (mAP50)
Only v7 (data-side: background images) beat v1:
- v1: YOLOv8s, 50ep, default aug → 0.734 (original baseline)
- v3: CarDD+VeHIDE combined → 0.677 (regressed)
- v4: YOLOv8m, heavy aug → 0.704 (regressed)
- v5: YOLO11s-seg, 1280, copy_paste → 0.681 (regressed, but crack +0.080)
- v6: YOLO11s, 640, conservative aug → 0.696 (regressed)
- **v7: YOLOv8s, exact v1 recipe + 200 bg images → 0.749 val / 0.754 test (PRODUCTION)**
- v8: Same as v7 + 280 bg images → 0.726 (10% bg ratio too high)

Lesson: five hyperparameter experiments all failed. The first data-side experiment succeeded immediately. Do not change training recipes or hyperparameters; remaining errors are addressable through data.

### Kaggle Training Workflow
- `kaggle/` directory has a working Makefile-driven workflow for Kaggle's free P100 GPU
- Key gotcha: must downgrade torch to 2.5.1+cu121 for P100 sm_60 compatibility

## iOS App (DD-ios) — current state
- Runs only the v7 detector (`DD-iOS/DD-iOS/ViewModels/DamageDetectionViewModel.swift`).
- `DD-iOSTests/DetectorBenchmarkTests.swift` runs the app pipeline over the 374 CarDD test images and writes `cardd_ios_benchmark.csv`; compare it with Python using `src/compare_ios_benchmark.py`.
- `ios/DamageDoctorUITests/` in THIS repo is stale (tests a classifier-gate design that doesn't exist). Replacement UI tests live in DD-ios `DD-iOSUITests/`; delete the old folder once reviewed.

## Known Problems (still open)
- **P3 — iOS preprocessing.** The app stretched photos to 640×640 (`.scaleFill`) instead of letterboxing like ultralytics, and ignored `UIImage.imageOrientation` (portrait camera photos reached the model rotated 90°). Fix: DD-ios letterbox PR — needs an Xcode build, the unit tests, and a before/after benchmark run through `compare_ios_benchmark.py`.
- **P2 — biased FP numbers.** Both are fixed in code (`build_dataset.py` held-out eval sets, `compare_v1_v7.py --fp_dir` + warning) but the FP rates in this file and README have not been re-measured yet. The held-out `harder_negatives` set is only ~16 images, so it can't give a precise FP rate.
- **No held-out set of clean phone photos** exists for the detector: every `harder_negatives` image was a v7 training background.
- **P4 — stale UI tests** in `ios/DamageDoctorUITests/` (see above).
- DD-ios: the camera button crashes in the simulator (no `isSourceTypeAvailable` check); `AnalysisLogView` shows scripted fake log lines ("Found dent · 86%") during analysis.

## Immediate Next Steps

1. **Decide classifier gate yes/no** after the iOS letterbox fix: run the benchmark before/after, then measure the classifier on device with the training preprocessing (Resize + CenterCrop) before calling it faulty.
2. **Rebuild eval sets without training images and re-measure FP rates**: `python src/build_dataset.py`, then `eval_clean_only.py` per source and `compare_v1_v7.py --fp_dir <clean dir v7 never trained on>`.
3. **Run TIDE on v7** (`python src/tide_analysis.py --split test`) — confirm the Bkg error dropped and get the new error profile.
4. **Build a held-out set of clean phone photos** (never used for training either model) for honest FP measurement.
5. **Data-side work on the weak classes** (crack / scratch / dent): relabeling, more examples — not hyperparameters.
