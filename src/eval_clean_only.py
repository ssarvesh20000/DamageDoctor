import argparse
from pathlib import Path
import torch
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

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
    ap.add_argument("--data_dir", type=Path, required=True)  # points to .../test/no_damage parent
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    img = ck.get("img_size", 224)

    tfm = transforms.Compose([
        transforms.Resize(int(img * 1.15)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    ds = ImageFolder(args.data_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    dev = get_device()
    model = build_model()
    model.load_state_dict(ck["model"])
    model.to(dev).eval()

    preds = []
    for x, _ in dl:
        x = x.to(dev)
        logits = model(x).cpu().numpy().reshape(-1)
        probs = 1 / (1 + np.exp(-logits))
        pred = (probs >= args.threshold).astype(int)  # 1=no_damage, 0=damage based on your training mapping
        preds.extend(pred.tolist())

    # In your trained mapping: {'damage':0, 'no_damage':1}
    # False positive here = predicted DAMAGE (0) on clean images
    preds = np.array(preds)
    false_damage = (preds == 0).mean()
    true_clean = (preds == 1).mean()

    print("Samples:", len(preds))
    print(f"Predicted NO_DAMAGE: {true_clean:.3f}")
    print(f"Predicted DAMAGE (false positive rate): {false_damage:.3f}")

if __name__ == "__main__":
    main()