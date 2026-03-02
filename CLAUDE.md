# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DamageDoctor is a binary image classifier that detects visible car damage using a fine-tuned `mobilenet_v3_small` model (ImageNet pretrained). The model outputs a single logit and uses `BCEWithLogitsLoss` with a sigmoid threshold of 0.5. Classes are `damage` (index 0) and `no_damage` (index 1).

## Environment Setup

Uses Python 3.9 with a `.venv` in the project root.

```bash
source .venv/bin/activate
```

Key dependencies (no requirements.txt — installed in .venv): `torch 2.8`, `torchvision 0.23`, `numpy`, `tqdm`, `scikit-learn`.

All scripts must be run from the **project root**, not from `src/`.

## Commands

**Rebuild dataset** (run whenever new raw images are added):
```bash
python src/build_dataset.py
# Clears data/processed/ and rebuilds from all raw sources
# Also rebuilds data/eval_by_source/ for per-source FP evaluation
```

**Train:**
```bash
python src/train_binary.py
# Optional args: --data_dir, --epochs (default 25), --batch, --lr, --img, --out
```

**Evaluate on test set** (confusion matrix + classification report):
```bash
python src/eval_test.py
# Optional: --ckpt, --data_dir
```

**Evaluate false positive rate on a specific source:**
```bash
python src/eval_clean_only.py --data_dir data/eval_by_source/harder_negatives
python src/eval_clean_only.py --data_dir data/eval_by_source/clean_cars
# Optional: --ckpt, --threshold
```

**Evaluate false positive rate across all splits:**
```bash
python src/eval_no_damage_all.py
# Optional: --ckpt, --data_dir, --threshold
```

## Architecture

**Model** (`src/train_binary.py`): `mobilenet_v3_small` with the final classifier layer replaced by `nn.Linear(in_features, 1)`. Trained with AdamW lr=3e-4, weight_decay=1e-4, CosineAnnealingLR over 25 epochs (decays to 1e-6).

**Augmentation**: Train pipeline includes `RandomGlare(p=0.7, max_spots=5)` — simulates specular highlights and windshield glare using soft blurred ellipses composited onto the image. Applied only during training, not eval.

**Device selection**: Auto-detects MPS (Apple Silicon) → CUDA → CPU.

**Checkpoint format** (`models/binary_mobilenetv3_best.pt`):
```python
{"model": state_dict, "class_to_idx": {"damage": 0, "no_damage": 1}, "img_size": 224}
```

## Data Layout

```
data/
  raw/                          # source images (gitignored)
    cardd_damage/               # 4000 damaged car images
    clean_cars/                 # 1253 clean
    clean_cars_extra/           # 199 clean
    external_datasets/
      cars_dataset_2/           # 951 clean
    harder_negatives/           # real phone photos targeting edge cases (glare, stickers)
  processed/                    # rebuilt by build_dataset.py — do not edit manually
    train/val/test/
      damage/
      no_damage/
  eval_by_source/               # flat per-source folders for targeted FP eval
    clean_cars/no_damage/
    clean_cars_extra/no_damage/
    cars_dataset_2/no_damage/
    harder_negatives/no_damage/
```

To add a new no_damage source: add the folder to `data/raw/`, register it in `no_damage_sources` dict in `src/build_dataset.py`, then run `build_dataset.py` and retrain.

## Current Status & Immediate Next Step

**v2 results** (25 epochs, cosine LR, RandomGlare augmentation):
- Test accuracy: 99.61%, damage recall: 100%
- harder_negatives FP rate: 1.2% (2 images with extreme glare still failing)

**Remaining weakness:** Heavy specular glare on windshields/hoods. The model has learned to handle moderate glare but the most extreme cases (direct sunlight dead-on) still trigger false positives.

**Next action — collect more glare photos:**
- Direct sunlight hitting the windshield dead-on
- Bright overhead lights reflecting off the hood or roof
- Sun glare on side windows and mirrors
- Any angle where glare covers a large portion of the car

Drop photos into `data/raw/harder_negatives/`, then:
```bash
python src/build_dataset.py
python src/train_binary.py
python src/eval_clean_only.py --data_dir data/eval_by_source/harder_negatives
```

**After glare is solved:**
1. Export to CoreML for iOS
2. Test on real-world phone images in-app
3. Move to damage localization (object detection)
