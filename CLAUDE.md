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
python src/build_dataset.py --out_dir runs/rebuild/processed --eval_dir runs/rebuild/eval   # NEVER default dirs, see below
python src/build_dataset.py --eval_all_images   # Old eval_by_source (includes train/val images — biased)
python src/train_binary.py            # Train (--epochs, --batch, --lr, --img, --out)
python src/eval_test.py               # Confusion matrix + classification report
python src/eval_clean_only.py --data_dir data/eval_by_source/harder_negatives  # FP rate (held-out)
python src/eval_no_damage_all.py      # FP rate across all splits
```

**DANGER: `build_dataset.py` (on main) `shutil.rmtree`s `--out_dir` (default `data/processed`) and `--eval_dir` (default `data/eval_by_source`).** `data/processed` also holds 17,825 25kUndmged images that main's `build_dataset.py` does not know about, so running it with the defaults destroys the current classifier dataset. Branch `overnight2/build-dataset-25k` adds the 25kUndmged source and refuses to delete existing dirs without `--force`; a rebuild into a fresh dir reproduced `data/processed` exactly (22,961/22,961 files, same split/class/filename/md5).

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
  processed/                        # Classifier splits: build_dataset.py balanced part (damage_*/no_damage_*: 2054/257/257 per class)
                                    #   + 25kUndmged copied flat with original names, its own split (10,695/1,742/5,388)
                                    #   totals: train 2054 damage / 12749 no_damage; val 257 / 1999; test 257 / 5645
  eval_by_source/                   # Per-source FP eval dirs (currently the OLD all-images version: includes train images)
  manifests/                        # Text manifests (classifier_split_current.csv: split,class,filename,source,md5)
25kUndmged/data/{train,val,test}/<~300 subfolders>/*.jpg   # 17,825 clean-car images, mostly THUMBNAILS
                                    #   (median max side 100 px; 96% < 200 px). CarDD is ~1000 px.
```

Which images each model has trained on (matters for any FP evaluation):
- Classifier: everything in `processed/train` (≈80% of each raw no_damage source + all 10,695 25kUndmged train images + 2054 CarDD images). 192 of the 374 CarDD **detector test** images are in the classifier's train split.
- Detector v7: CarDD train + 200 background images = **all 165 `harder_negatives` + 35 `clean_cars`** (the 35 were recovered from file birth times in `data/cardd_yolo/train/images`: list in `data/manifests/detector_v7_backgrounds.txt` on branch `overnight2/v9-experiments`; 3 of them are in the classifier's held-out clean_cars test images). `clean_cars_extra`, `cars_dataset_2` and 25kUndmged were never detector training data.
- Local `data/cardd_yolo/train` has 280 backgrounds (= Kaggle dataset v3, v8's data), not v7's 200.
- 25kUndmged has 17 byte-identical duplicates across its own splits (15 train↔test, 2 train↔val).

## Current Model Status

### Binary Classifier (v2) — trained, CoreML exported, NOT used by the app
Measured 2026-10-13 (overnight2) on the current `data/processed/test` (n=5902):
- Accuracy 5896/5902 = 99.90%, damage recall 256/257 = 99.61%, no_damage FP 5/5645 = 0.09%.
- Held-out FP by source: 25kUndmged 2/5388, clean_cars 0/124, clean_cars_extra 0/16, cars_dataset_2 2/100, harder_negatives 1/17. All 165 harder_negatives (128 were training images): 4/165 = 2.4%.
- On the 514-image balanced part only: 510/514 = 99.22%. The old "99.61% / recall 100%" does not reproduce with this checkpoint.
- **Resolution shortcut:** damaged CarDD test images downscaled to max side 100 px are called clean (recall 0/374 after JPEG save, 30/374 in memory; 200 px 268/374; ≥300 px 369–373/374). Its near-zero FP on 25kUndmged thumbnails is mostly "small image ⇒ no damage". The older `binary_mobilenetv3_v2.pt` (not trained on 25k) flags 944/5388 of those thumbnails as damage.
- Remaining weakness: extreme specular glare.
- Checkpoint: `models/binary_mobilenetv3_best.pt`; CoreML: `models/DamageClassifier.mlpackage`
- Preprocessing it was trained with: `Resize(int(224*1.15))` + `CenterCrop(224)` + ImageNet normalization (baked into the CoreML export).

### Detector (v7) — CarDD + 200 Background Images — PRODUCTION
- **mAP50: 0.754 test / 0.749 val**; mAP50-95 0.589 (val)
- Per-class AP50 (test): dent 0.642, scratch 0.627, crack 0.463, glass shatter 0.991, lamp broken 0.868, tire flat 0.932
- Per-class AP50 (val): dent 0.599, scratch 0.567, crack 0.459, glass shatter 0.988, lamp broken 0.906, tire flat 0.975
- Weak classes: crack, scratch, dent
- Weights: `runs/kaggle_cardd_v7_detect/best.pt`; CoreML: `models/DamageDetector.mlpackage` (done, in the DD-ios app)
- CoreML export (`ultralytics export format=coreml nms=True`) has NMS built in: conf 0.25, IoU 0.7, per-class. Class scores are padded 6 → 80 before NMS (labels "6"–"79" are placeholders).
- Recipe: exact v1 config (YOLOv8s, SGD, 50ep, batch=16, default aug) + 200 background images (7% of train set), no label files
- Kaggle dataset version: `ssarvesh20000/damagedoctor-cardd-yolo` v2
- TIDE dAP (overnight2, `src/tide_analysis.py`):

  | | AP50 | Cls | Loc | Both | Dupe | Bkg | Miss |
  |---|---|---|---|---|---|---|---|
  | v1 val | 72.72 | 1.22 | 6.34 | 0.78 | 0.22 | 4.43 | 1.49 |
  | v7 val | 73.60 | 0.86 | 5.80 | 0.70 | 0.16 | 4.93 | 1.77 |
  | v1 test | 74.61 | 1.53 | 4.87 | 0.64 | 0.20 | 5.25 | 2.10 |
  | v7 test | 74.85 | 1.54 | 5.10 | 0.60 | 0.18 | 4.29 | 3.34 |

  The old "v1 baseline Loc 6.34 / Bkg 4.43 / Cls 1.22" was v1 on **val**. Loc dominates everywhere; Bkg on CarDD went down on test but up on val. Scratch errors are Loc-dominated (not Cls/Miss).
- Image-level FP (any box ≥ 0.25) on never-seen clean images, v7 vs v1: 25kUndmged test 652/5388 (12.1%) vs 2483/5388 (46.1%); cars_dataset_2 18/100 vs 50/100; held-out clean_cars 8/121 vs 39/121. v1's 25k FPs are mostly whole-frame "glass shatter" on blurry thumbnails.

### Detector (v1) — Previous Baseline (Superseded by v7)
- mAP50: 0.734, mAP50-95: 0.581
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
- Gotcha (2026-10): Kaggle's image is now Python 3.13 and torchvision 0.20.1 has no cp313 wheel, so the pip downgrade fails. The v9 E1/E2 kernels (`overnight2/v9-experiments`) create a Python 3.12 venv with `uv`, install the same pin + ultralytics 8.4.40 (v7's version) there, and re-exec.
- Gotcha: `kernel-metadata.json` attaches the LATEST version of `damagedoctor-cardd-yolo`. v7 trained on v2 (200 bg), but v8 uploaded v3 (280 bg), so re-pushing an old kernel does not reproduce its data. Check the `train: N images, M labels, K background` line in the kernel log.

## iOS App (DD-ios) — current state
- Runs only the v7 detector (`DD-iOS/DD-iOS/ViewModels/DamageDetectionViewModel.swift`).
- `DD-iOSTests/DetectorBenchmarkTests.swift` runs the app pipeline over the 374 CarDD test images and writes `cardd_ios_benchmark.csv`; compare it with Python using `src/compare_ios_benchmark.py`.
- `ios/DamageDoctorUITests/` in THIS repo is stale (tests a classifier-gate design that doesn't exist). Replacement UI tests live in DD-ios `DD-iOSUITests/`; delete the old folder once reviewed.

## Known Problems (still open)
- **P3 — iOS preprocessing.** The app stretched photos to 640×640 (`.scaleFill`) instead of letterboxing like ultralytics, and ignored `UIImage.imageOrientation`. Measured on the simulator (CarDD test n=374, vs Python v7): damage/no-damage agreement stretch 365/374 → letterbox 373/374, exact box-count match 178 → 271, top-label agreement 82.7% → 85.6%; recall 368/374 on both. Letterbox PR not merged yet.
- **App confidence display.** The app shows `observation.labels.first.confidence`, which is ~0.99 for almost every box (looks normalized across the 80 padded classes); it is not the detector score.
- **P2 — biased FP numbers.** Re-measured on held-out sets (above). The held-out `harder_negatives` set is only 17 images, so it can't give a precise FP rate.
- **No held-out set of clean phone photos** exists for the detector: every `harder_negatives` image was a v7 training background.
- **P4 — stale UI tests** in `ios/DamageDoctorUITests/` (see above).
- DD-ios: the camera button crashes in the simulator (no `isSourceTypeAvailable` check); `AnalysisLogView` shows scripted fake log lines ("Found dent · 86%") during analysis.

## Immediate Next Steps

1. **Decide classifier gate yes/no.** Python numbers (v7, CarDD test n=374 vs 237 non-25k held-out clean photos): detector ≥0.25 alone recall 369/374, FP 30/237; classifier gate + detector recall 368/374, FP 0/237. See `runs/overnight2/eval/PHASE3_SUMMARY.txt`. Then measure the classifier on device with the training preprocessing (Resize + CenterCrop).
2. **Merge or reject the letterbox PR** (numbers above).
3. **Build a held-out set of clean phone photos** (never used for training either model) for honest FP measurement — the only big held-out clean set (25kUndmged) is thumbnails.
4. **Fix the classifier's resolution shortcut** before relying on it for small images (e.g., train with random downscaling of damaged images, or drop 25k thumbnails).
5. **Data-side work on the weak classes** (crack / scratch / dent): relabeling, more examples — not hyperparameters.
