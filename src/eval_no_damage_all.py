"""
Run the binary model on every no_damage image (train + val + test) and report
false positive rate per split and overall. Answers: have we tested everything in no_damage?
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from paths import CLASSIFIER_CKPT

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_model():
    m = models.mobilenet_v3_small(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = torch.nn.Linear(in_f, 1)
    return m

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CLASSIFIER_CKPT)
    ap.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    dev = get_device()
    ck = torch.load(args.ckpt, map_location="cpu")
    class_to_idx = ck["class_to_idx"]
    no_damage_idx = class_to_idx["no_damage"]  # 1 in typical mapping
    img_size = ck.get("img_size", 224)

    tfm = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = build_model()
    model.load_state_dict(ck["model"])
    model.to(dev).eval()

    totals = []
    for split in ("train", "val", "test"):
        split_dir = args.data_dir / split
        if not split_dir.exists():
            print(f"Skip {split}: {split_dir} not found")
            continue
        ds = datasets.ImageFolder(split_dir, transform=tfm)
        no_damage_indices = [i for i in range(len(ds)) if ds.targets[i] == no_damage_idx]
        if not no_damage_indices:
            print(f"{split}: no no_damage images")
            continue
        subset = Subset(ds, no_damage_indices)
        dl = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)

        preds = []
        for x, _ in dl:
            x = x.to(dev)
            logits = model(x).cpu().numpy().reshape(-1)
            probs = 1 / (1 + np.exp(-logits))
            preds.extend((probs >= args.threshold).astype(int).tolist())
        preds = np.array(preds)
        n = len(preds)
        correct = (preds == 1).sum()
        false_pos = (preds == 0).sum()
        print(f"{split:5}  no_damage n={n:4}  correct(no_damage)={correct:4}  false_pos(damage)={false_pos:4}  FP%={100*false_pos/n:.2f}")
        totals.append((n, correct, false_pos))

    if totals:
        n_all = sum(t[0] for t in totals)
        fp_all = sum(t[2] for t in totals)
        print(f"{'total':5}  no_damage n={n_all:4}  false_pos(damage)={fp_all:4}  FP%={100*fp_all/n_all:.2f}")
    print("Done. All no_damage images (train/val/test) have been run through the model.")

if __name__ == "__main__":
    main()
