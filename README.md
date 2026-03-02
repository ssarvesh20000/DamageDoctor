# DamageDoctor

Binary image classifier that detects visible car damage.

## Current Status (v2)

- Dataset: 2568 images per class (damage / no_damage), 80/10/10 split
  - no_damage sources: `clean_cars` (1253) + `clean_cars_extra` (199) + `cars_dataset_2` (951) + `harder_negatives` (165)
  - damage source: `cardd_damage` (2568 of 4000 available)
- Model: `mobilenet_v3_small` (ImageNet pretrained), fine-tuned with BCEWithLogitsLoss
- Training: AdamW lr=3e-4, CosineAnnealingLR over 25 epochs, RandomGlare augmentation (p=0.7)
- Best model: `models/binary_mobilenetv3_best.pt`

### Latest eval results

| Metric | Value |
|---|---|
| Test accuracy | 99.61% |
| Damage recall | 100% (0 missed damage) |
| clean_cars FP rate | 0.1% |
| clean_cars_extra FP rate | 1.0% |
| cars_dataset_2 FP rate | 0.9% |
| harder_negatives FP rate | 1.2% |

## Data Layout

```
data/
  raw/                        # source images (gitignored)
    cardd_damage/             # 4000 damaged car images
    clean_cars/               # 1253 clean car images
    clean_cars_extra/         # 199 clean car images
    external_datasets/
      cars_dataset_2/         # 951 clean car images
    harder_negatives/         # 165 real phone photos (glare, stickers, etc.)
  processed/                  # train/val/test split — rebuilt by build_dataset.py
  eval_by_source/             # per-source eval folders (one per no_damage source)
```

## Scripts

**Rebuild dataset** (run after adding new raw images):
```bash
python src/build_dataset.py
```

**Train:**
```bash
python src/train_binary.py
# Key args: --epochs (default 25), --lr (default 3e-4), --batch (default 32)
```

**Evaluate on test set:**
```bash
python src/eval_test.py
```

**Evaluate false positive rate per source:**
```bash
python src/eval_clean_only.py --data_dir data/eval_by_source/harder_negatives
python src/eval_clean_only.py --data_dir data/eval_by_source/clean_cars
python src/eval_clean_only.py --data_dir data/eval_by_source/cars_dataset_2
```

**Evaluate across all no_damage splits:**
```bash
python src/eval_no_damage_all.py
```

**Find false positives and copy them to a folder:**
```bash
# See src/ scripts or run inline — check false_positives/ folder
```

## Immediate Next Step

The model's main remaining weakness is **heavy glare** (bright sunlight on windshields/hoods). The `harder_negatives` source still has a 1.2% FP rate — the 2 remaining failures are extreme glare cases.

**Action required:** Collect more phone photos of undamaged cars with heavy glare:
- Direct sunlight hitting the windshield dead-on
- Bright overhead lights reflecting off the hood or roof
- Sun glare on side windows and mirrors
- Any angle where the glare covers a large portion of the car

Drop them into `data/raw/harder_negatives/`, then run:
```bash
python src/build_dataset.py
python src/train_binary.py
```

## Roadmap

1. **Collect more heavy-glare photos** ← current focus
2. Export to CoreML for iOS
3. Test on real-world phone images in-app
4. Move to damage localization (object detection)
