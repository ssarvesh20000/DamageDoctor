import argparse
from pathlib import Path
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

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
    args = ap.parse_args()

    dev = get_device()
    ck = torch.load(args.ckpt, map_location="cpu")
    class_to_idx = ck["class_to_idx"]
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    print("class_to_idx:", class_to_idx)

    img = ck.get("img_size", 224)
    tfm = transforms.Compose([
        transforms.Resize(int(img * 1.15)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    ds = datasets.ImageFolder(args.data_dir / "test", transform=tfm)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model()
    model.load_state_dict(ck["model"])
    model.to(dev).eval()

    y_true, y_pred = [], []
    for x, y in dl:
        x = x.to(dev)
        logits = model(x).cpu().numpy().reshape(-1)
        probs = 1 / (1 + np.exp(-logits))
        pred = (probs >= 0.5).astype(int)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nReport:\n", classification_report(y_true, y_pred, target_names=[idx_to_class[0], idx_to_class[1]]))

if __name__ == "__main__":
    main()