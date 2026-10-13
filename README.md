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

| Metric | Value |
|---|---|
| Test accuracy | 99.61% |
| Damage recall | 100% |
| Per-source FP rates* | clean_cars 0.1% · clean_cars_extra 1.0% · cars_dataset_2 0.9% · harder_negatives 1.2% |

\* Measured on the old `eval_by_source/`, which included the classifier's own training images, so these are **optimistic**. `build_dataset.py` now builds held-out eval sets; re-measure before relying on these.

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
  processed/                  # classifier train/val/test — rebuilt by build_dataset.py
  eval_by_source/<source>/no_damage/   # each source's classifier TEST-split images only
```

## Common commands

```bash
# Classifier
python src/build_dataset.py                      # rebuild processed/ + held-out eval_by_source/
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

- **iOS preprocessing differs from training.** The app stretched photos to 640×640 instead of letterboxing, and ignored photo orientation. A fix is in review in DD-ios; it still needs an Xcode build and a before/after benchmark.
- **False-positive numbers above are not trustworthy yet.** Classifier FP rates were measured partly on training images. The detector's FP comparison on `harder_negatives` uses v7's own training backgrounds (`compare_v1_v7.py` now warns about this).
- **No held-out set of clean phone photos.** The held-out `harder_negatives` slice is only ~16 images.
- `ios/DamageDoctorUITests/` is stale: it tests a classifier-gate UI that doesn't exist. Replacement UI tests are in DD-ios.

## Next steps

1. Decide on the classifier gate (yes/no) after the iOS letterbox fix.
2. Rebuild the eval sets without training images and re-measure false-positive rates.
3. Run TIDE on v7.
4. Build a held-out set of clean phone photos.
5. Data-side work on the weak classes (crack / scratch / dent).
