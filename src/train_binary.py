import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter

from paths import CLASSIFIER_CKPT


class RandomGlare:
    """Simulates specular highlights and windshield glare on car surfaces."""

    def __init__(self, p=0.7, max_spots=5):
        self.p = p
        self.max_spots = max_spots

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        img = img.convert("RGBA")
        w, h = img.size

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for _ in range(random.randint(1, self.max_spots)):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            rx = random.randint(w // 8, w // 2)
            ry = random.randint(h // 8, h // 2)
            brightness = random.randint(230, 255)
            alpha = random.randint(140, 245)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                         fill=(brightness, brightness, brightness, alpha))

        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.randint(8, 25)))
        return Image.alpha_composite(img, overlay).convert("RGB")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_model():
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, 1)  # single logit for BCEWithLogitsLoss
    return m

@torch.no_grad()
def eval_loop(model, loader, dev, criterion):
    model.eval()
    losses, correct, total = [], 0, 0
    for x, y in loader:
        x = x.to(dev)
        y = y.float().unsqueeze(1).to(dev)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
        total += y.numel()

    return float(np.mean(losses)), (correct / total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--out", type=Path, default=CLASSIFIER_CKPT)
    args = ap.parse_args()

    dev = get_device()
    print("device:", dev)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(args.img, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.15, 0.15, 0.10, 0.02),
        transforms.RandomRotation(10),
        RandomGlare(p=0.7, max_spots=5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(int(args.img * 1.15)),
        transforms.CenterCrop(args.img),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(args.data_dir / "train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(args.data_dir / "val",   transform=eval_tfms)

    print("class_to_idx:", train_ds.class_to_idx)

    targets = torch.tensor(train_ds.targets)
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"class counts: {dict(enumerate(class_counts.tolist()))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_model().to(dev)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for e in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        correct, total = 0, 0

        for x, y in tqdm(train_loader, desc=f"epoch {e}/{args.epochs}"):
            x = x.to(dev)
            y = y.float().unsqueeze(1).to(dev)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            tr_losses.append(loss.item())
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()

        tr_loss = float(np.mean(tr_losses))
        tr_acc  = correct / total
        val_loss, val_acc = eval_loop(model, val_loader, dev, criterion)

        scheduler.step()
        print(f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "img_size": args.img,
            }, args.out)
            print("saved:", args.out)

    print("best val acc:", best_val_acc)

if __name__ == "__main__":
    main()