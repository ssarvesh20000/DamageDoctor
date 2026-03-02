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

**Train:**
```bash
python src/train_binary.py
# Optional args: --data_dir, --epochs, --batch, --lr, --img, --out
```

**Evaluate on test set** (confusion matrix + classification report):
```bash
python src/eval_test.py
# Optional: --ckpt, --data_dir
```

**Evaluate false positive rate on clean images only** (requires explicit `--data_dir`):
```bash
python src/eval_clean_only.py --data_dir data/processed_external/some_dir
# Optional: --ckpt, --threshold
```

**Evaluate false positive rate across all splits** (train/val/test no_damage):
```bash
python src/eval_no_damage_all.py
# Optional: --ckpt, --data_dir, --threshold
```

## Architecture

**Model** (`src/train_binary.py`): `mobilenet_v3_small` with the final classifier layer replaced by `nn.Linear(in_features, 1)`. Trained with AdamW, lr=3e-4, weight_decay=1e-4.

**Device selection**: Auto-detects MPS (Apple Silicon) → CUDA → CPU.

**Data pipeline**: `torchvision.datasets.ImageFolder` on `data/processed/{train,val,test}/`. Train uses RandomResizedCrop + ColorJitter + RandomRotation augmentation; eval uses Resize + CenterCrop with scale factor 1.15.

**Checkpoint format** (`models/binary_mobilenetv3_best.pt`):
```python
{"model": state_dict, "class_to_idx": {"damage": 0, "no_damage": 1}, "img_size": 224}
```

## Data Layout

```
data/
  processed/          # train/val/test split (80/10/10), 1253 images per class
    train/val/test/
      damage/
      no_damage/
  processed_external/ # additional external datasets for targeted eval
  raw/
    external_datasets/
    cardd_damage/
    clean_cars/
    clean_cars_extra/
```

## Current Status & Next Steps

v1 achieves ~99% test accuracy on the balanced dataset. Known gaps:
1. Add more undamaged data to reduce dataset bias (more real-world variety)
2. Test on real-world phone images
3. Add harder negatives (shadows, reflections, dirt)
4. Export to CoreML for iOS
5. Move to damage localization (object detection)
