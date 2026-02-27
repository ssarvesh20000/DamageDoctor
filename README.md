# DamageDoctor

Binary image classifier that detects visible car damage.

## Current Status (v1)

- Built balanced dataset:
  - 1253 damage
  - 1253 no_damage
- Split into train / val / test (80/10/10)
- Trained `mobilenet_v3_small` (ImageNet pretrained)
- Achieved ~99% test accuracy
- Best model saved to:
  models/binary_mobilenetv3_best.pt

## Data Structure

data/processed/
  train/
    damage/
    no_damage/
  val/
    damage/
    no_damage/
  test/
    damage/
    no_damage/

## Train

source .venv/bin/activate  
python src/train_binary.py  

## Evaluate

python src/eval_test.py  

## Next Steps

1. Add additional undamaged datasets (reduce dataset bias)
2. Test on real-world phone images
3. Add harder negatives (shadows, reflections, dirt)
4. Export to CoreML for iOS
5. Move to damage localization (object detection)

Baseline complete. Improving generalization next.