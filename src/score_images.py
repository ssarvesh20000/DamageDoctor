"""
Score image sets with the classifier or a detector and save per-image /
per-box CSVs, so threshold analyses (eval_app_level.py) never re-run models.

Image sets come from a CSV with columns set,path (see --sets). Outputs:
  classifier: <out>  columns set,path,p_damage   (p_damage = 1 - sigmoid;
              DAMAGE iff sigmoid < 0.5 iff p_damage > 0.5)
  detector:   <out>  columns set,path,n_boxes,max_conf,max_cls
              <out>.boxes.csv  columns set,path,cls,conf,x1,y1,x2,y2
              (all boxes with conf >= --min_conf, NMS IoU 0.7 per class like
              the app's CoreML export)

Usage (from project root):
    python src/score_images.py classifier --sets sets.csv --out clf.csv
    python src/score_images.py detector --ckpt runs/kaggle_cardd_v7_detect/best.pt \
        --sets sets.csv --out v7.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from paths import CLASSIFIER_CKPT, DETECTOR_CKPT


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def read_sets(path: Path):
    with open(path) as f:
        return [(r["set"], r["path"]) for r in csv.DictReader(f)]


@torch.no_grad()
def score_classifier(rows, ckpt: Path, out: Path, batch: int):
    from torchvision import models, transforms

    ck = torch.load(ckpt, map_location="cpu")
    assert ck["class_to_idx"] == {"damage": 0, "no_damage": 1}, ck["class_to_idx"]
    img = ck.get("img_size", 224)
    # Same eval transform as train_binary.py / eval_test.py
    tfm = transforms.Compose([
        transforms.Resize(int(img * 1.15)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, 1)
    m.load_state_dict(ck["model"])
    dev = get_device()
    m.to(dev).eval()

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "path", "p_damage"])
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            x = torch.stack([tfm(Image.open(p).convert("RGB")) for _, p in chunk])
            logits = m(x.to(dev)).cpu().numpy().reshape(-1)
            sig = 1 / (1 + np.exp(-logits))
            for (s, p), v in zip(chunk, sig):
                w.writerow([s, p, f"{1 - v:.6f}"])
            if (i // batch) % 20 == 0:
                print(f"  {i + len(chunk)}/{len(rows)}")


def score_detector(rows, ckpt: Path, out: Path, min_conf: float, imgsz: int):
    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    dev = get_device()
    box_out = out.with_suffix(".boxes.csv")
    with open(out, "w", newline="") as f, open(box_out, "w", newline="") as fb:
        w, wb = csv.writer(f), csv.writer(fb)
        w.writerow(["set", "path", "n_boxes", "max_conf", "max_cls"])
        wb.writerow(["set", "path", "cls", "conf", "x1", "y1", "x2", "y2"])
        # one image at a time (MPS memory)
        for i, (s, p) in enumerate(rows):
            r = model.predict(source=p, imgsz=imgsz, conf=min_conf, iou=0.7,
                              device=dev, verbose=False)[0]
            b = r.boxes
            confs = b.conf.cpu().numpy() if len(b) else np.zeros(0)
            clss = b.cls.cpu().numpy().astype(int) if len(b) else np.zeros(0, int)
            xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
            for c, k, xy in zip(confs, clss, xyxy):
                wb.writerow([s, p, k, f"{c:.5f}"] + [f"{v:.1f}" for v in xy])
            if len(confs):
                j = int(confs.argmax())
                w.writerow([s, p, len(confs), f"{confs[j]:.5f}", clss[j]])
            else:
                w.writerow([s, p, 0, "0", -1])
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["classifier", "detector"])
    ap.add_argument("--sets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="Only these set names")
    ap.add_argument("--min_conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    rows = read_sets(args.sets)
    if args.only:
        rows = [r for r in rows if r[0] in args.only]
    print(f"{len(rows)} images")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.model == "classifier":
        score_classifier(rows, args.ckpt or CLASSIFIER_CKPT, args.out, args.batch)
    else:
        score_detector(rows, args.ckpt or DETECTOR_CKPT, args.out,
                       args.min_conf, args.imgsz)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
