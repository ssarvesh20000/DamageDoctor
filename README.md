# DamageDoctor

Detects visible car damage in phone photos. This repo holds the Python ML side (training, evaluation, CoreML export); the iOS app lives in [DD-ios](https://github.com/ssarvesh20000/DD-ios) (branch `working`).

Two models:

| Model | Task | Architecture | Status |
|---|---|---|---|
| Damage detector (v7) | Localize + classify 6 damage types: dent, scratch, crack, glass shatter, lamp broken, tire flat | YOLOv8s | **Production** — exported to CoreML, runs in the iOS app |
| Binary classifier (v2) | Damage vs no damage | MobileNetV3-Small | Trained + exported to CoreML, **not used by the app** |

The iOS app currently runs the **detector only**: a photo is "damaged" if the detector finds at least one box with confidence ≥ 0.25. The classifier gate was removed from the app because it looked faulty; whether to bring it back is an open question (see Next Steps).

## Detector (v7)

- Training data: CarDD (YOLO format, 2816/810/374 train/val/test) + 200 background images (no labels): all 165 `harder_negatives` + 35 `clean_cars`.
- Recipe: YOLOv8s, SGD, 50 epochs, batch 16, default augmentation (same as v1). Trained on Kaggle (`kaggle/`).
- Weights: `runs/kaggle_cardd_v7_detect/best.pt` → CoreML `models/DamageDetector.mlpackage` (NMS built in: conf 0.25, IoU 0.7).

| Metric | Value |
|---|---|
| mAP50 (test) | 0.754 |
| mAP50 (val) | 0.749 |

| Class (test AP50) | dent | scratch | crack | glass shatter | lamp broken | tire flat |
|---|---|---|---|---|---|---|
| v7 | 0.642 | 0.627 | 0.463 | 0.991 | 0.868 | 0.932 |

Experiments (mAP50): v1 baseline 0.734 · v3 +VeHIDE 0.677 · v4 YOLOv8m heavy aug 0.704 · v5 YOLO11s-seg 1280 0.681 · v6 YOLO11s 0.696 · **v7 +200 background images 0.749 val / 0.754 test** · v8 +280 background images 0.726. Hyperparameter changes never helped; the only win came from data.

## Binary classifier (v2)

- Dataset: 2568 images per class, 80/10/10 split (seed 42)
  - no_damage: `clean_cars` (1253) + `clean_cars_extra` (199) + `cars_dataset_2` (951) + `harder_negatives` (165)
  - damage: `cardd_damage` (2568 of 4000 available)
- Model: `mobilenet_v3_small` (ImageNet pretrained), BCEWithLogitsLoss, AdamW 3e-4, cosine LR, 25 epochs, RandomGlare augmentation
- Preprocessing: `Resize(int(224*1.15))` + `CenterCrop(224)` + ImageNet normalization
- Checkpoint: `models/binary_mobilenetv3_best.pt` → CoreML `models/DamageClassifier.mlpackage`
- Main remaining weakness: extreme specular glare (sun on the windshield or hood). More phone photos of clean cars with heavy glare are the planned fix.

Measured 2026-10-13 on the current `data/processed/test` (n=5902; 5388 of the 5645 clean images are 25kUndmged thumbnails):

| Metric | Value |
|---|---|
| Test accuracy | 5896/5902 = 99.90% (balanced 514-image part: 510/514 = 99.22%) |
| Damage recall | 256/257 = 99.61% |
| Held-out FP by source | 25kUndmged 2/5388 · clean_cars 0/124 · clean_cars_extra 0/16 · cars_dataset_2 2/100 · harder_negatives 1/17 |

Caveat: the classifier treats small images as clean (damaged CarDD images downscaled to 100 px: recall ≤ 30/374), so its near-zero FP on the 25kUndmged thumbnails is not evidence of real-world performance.

## Setup

Python 3.9, `.venv` in the repo root (torch 2.8, torchvision 0.23, ultralytics, scikit-learn, flask, Pillow, coremltools, opencv-python, tidecv). Run every script from the repo root:

```bash
source .venv/bin/activate
python src/<script>.py --help
```

Model paths are defined once in `src/paths.py`.

## Data layout (gitignored)

```
data/
  cardd_yolo/{train,val,test}/{images,labels}/   # detector data + data.yaml
  raw/
    cardd_damage/             # 4000 damaged car images
    clean_cars/               # 1253 clean
    clean_cars_extra/         # 199 clean
    external_datasets/cars_dataset_2/  # 951 clean
    harder_negatives/         # 165 phone photos (glare, stickers, ...) — also v7 detector backgrounds
  processed/                  # classifier train/val/test = build_dataset.py balanced part
                              #   + 17,825 25kUndmged images (own 60/10/30 split, original names)
  eval_by_source/<source>/no_damage/   # per-source FP eval images
25kUndmged/data/{train,val,test}/      # clean cars, mostly ~100 px thumbnails
```

**Never run `build_dataset.py` with its default `--out_dir`/`--eval_dir` on main:** it deletes `data/processed`, including the 25kUndmged images it does not know how to rebuild. Branch `overnight2/build-dataset-25k` fixes this (adds the 25k source, refuses to delete without `--force`).

## Common commands

```bash
# Classifier
python src/build_dataset.py --out_dir <new dir> --eval_dir <new dir>   # never the defaults (see above)
python src/train_binary.py                       # --epochs, --lr, --batch
python src/eval_test.py
python src/eval_clean_only.py --data_dir data/eval_by_source/cars_dataset_2

# Detector
python src/eval_detect_test.py                   # test-set mAP (v7)
python src/detect.py photo.jpg                   # or a folder
python src/tide_analysis.py --split test
python src/compare_v1_v7.py --fp_dir data/eval_by_source/cars_dataset_2/no_damage
python src/compare_ios_benchmark.py --csv cardd_ios_benchmark.csv   # iOS app vs Python

# Both
python src/demo.py                               # web demo, http://localhost:8080
python src/diagnose.py photo.jpg
python src/export_coreml.py                      # --classifier / --detector
```

## Known problems

- **iOS preprocessing differs from training.** The app stretches photos to 640×640 instead of letterboxing, and ignores photo orientation. The DD-ios letterbox fix (not merged) was benchmarked on the simulator: agreement with Python on CarDD test went from 365/374 to 373/374 images, recall unchanged at 368/374.
- **App confidence display** shows ~99% for nearly every box (`labels.first.confidence`), not the detector score.
- **Detector false positives on clean cars** (any box ≥ 0.25, never-seen images): v7 12.1% of 5388 25kUndmged test thumbnails, 18/100 cars_dataset_2, 8/121 clean_cars. Every `harder_negatives` image was a v7 training background.
- **No held-out set of clean phone photos.** The held-out `harder_negatives` slice is only 17 images.
- `ios/DamageDoctorUITests/` is stale: it tests a classifier-gate UI that doesn't exist. Replacement UI tests are in DD-ios.

## Next steps

1. Decide on the classifier gate (yes/no). In Python, gating v7 with the classifier removed all 30 false alarms on 237 held-out clean photos and lost 1 of 374 damaged images.
2. Decide on the DD-ios letterbox PR.
3. Build a held-out set of clean phone photos.
5. Data-side work on the weak classes (crack / scratch / dent).
