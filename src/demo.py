"""
Interactive web demo for testing both DamageDoctor models.
Drop in a photo → see classifier result + detector boxes.

Usage:
  python src/demo.py
  python src/demo.py --detector-ckpt runs/detect/runs/detect/cardd_v1/weights/best.pt
  # Opens browser at http://localhost:8080
"""
import argparse
import io
import base64
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw
from flask import Flask, request, render_template_string

from paths import CLASSIFIER_CKPT, DETECTOR_CKPT

# --- Models (loaded once in main() via load_models) ---

_clf = None
_detector = None
IMG_SIZE = 224
CLF_TRANSFORM = None


def load_models(classifier_ckpt: Path, detector_ckpt: Path):
    global _clf, _detector, IMG_SIZE, CLF_TRANSFORM

    _clf = models.mobilenet_v3_small(weights=None)
    in_f = _clf.classifier[-1].in_features
    _clf.classifier[-1] = nn.Linear(in_f, 1)
    ck = torch.load(classifier_ckpt, map_location="cpu", weights_only=False)
    _clf.load_state_dict(ck["model"])
    _clf.eval()

    IMG_SIZE = ck.get("img_size", 224)

    CLF_TRANSFORM = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    _detector = YOLO(str(detector_ckpt))


COLORS = {
    "dent": "#FF4444",
    "scratch": "#FF8800",
    "crack": "#FFCC00",
    "glass shatter": "#4488FF",
    "lamp broken": "#AA44FF",
    "tire flat": "#44FF44",
}

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>DamageDoctor Tester</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #111; color: #eee; }
  h1 { margin-bottom: 5px; }
  .subtitle { color: #888; margin-bottom: 30px; }
  .drop-zone { border: 2px dashed #555; border-radius: 12px; padding: 60px; text-align: center; cursor: pointer; transition: border-color 0.2s; margin-bottom: 30px; }
  .drop-zone:hover, .drop-zone.dragover { border-color: #4488FF; }
  .drop-zone input { display: none; }
  .result-img { max-width: 100%; border-radius: 8px; margin: 20px 0; }
  .card { background: #1a1a1a; border-radius: 8px; padding: 20px; margin: 15px 0; }
  .damage { border-left: 4px solid #FF4444; }
  .no-damage { border-left: 4px solid #44FF44; }
  table { width: 100%; border-collapse: collapse; }
  td, th { padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }
  .conf { font-family: monospace; }
  .color-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
  .loading { display: none; color: #888; margin: 20px 0; }
</style>
</head>
<body>
<h1>DamageDoctor</h1>
<p class="subtitle">Drop a car photo to test the classifier + detector</p>

<div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
  <p>Click or drag & drop an image here</p>
  <input type="file" id="fileInput" accept="image/*">
</div>
<div class="loading" id="loading">Analyzing...</div>
<div id="results"></div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) upload(fileInput.files[0]); });

function upload(file) {
  const fd = new FormData();
  fd.append('image', file);
  loading.style.display = 'block';
  results.innerHTML = '';
  fetch('/analyze', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      loading.style.display = 'none';
      let html = '';
      const cls = data.is_damage ? 'damage' : 'no-damage';
      html += `<div class="card ${cls}">`;
      html += `<strong>Classifier:</strong> ${data.is_damage ? 'DAMAGE' : 'NO DAMAGE'}`;
      html += ` (${(data.damage_confidence * 100).toFixed(1)}% confidence, logit=${data.logit})`;
      html += `<br>Image: ${data.width}x${data.height}`;
      html += `</div>`;
      if (data.annotated_image) {
        html += `<img class="result-img" src="data:image/jpeg;base64,${data.annotated_image}">`;
      }
      if (data.detections && data.detections.length > 0) {
        html += `<div class="card"><strong>Detections: ${data.detections.length}</strong>`;
        html += `<table><tr><th>Type</th><th>Confidence</th><th>Box</th></tr>`;
        data.detections.forEach(d => {
          html += `<tr><td><span class="color-dot" style="background:${d.color}"></span>${d.label}</td>`;
          html += `<td class="conf">${(d.conf * 100).toFixed(1)}%</td>`;
          html += `<td class="conf">${d.box}</td></tr>`;
        });
        html += `</table></div>`;
      } else if (data.is_damage) {
        html += `<div class="card">No specific damage regions detected.</div>`;
      }
      results.innerHTML = html;
    });
}
</script>
</body>
</html>
"""

app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if not file:
        return {"error": "No image"}, 400

    img = Image.open(file.stream).convert("RGB")
    w, h = img.size

    # Classifier
    x = CLF_TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        logit = _clf(x).item()
    prob = 1 / (1 + np.exp(-logit))
    is_damage = bool(prob < 0.5)
    damage_confidence = float(1 - prob)

    result = {
        "is_damage": is_damage,
        "damage_confidence": round(damage_confidence, 4),
        "logit": round(logit, 3),
        "width": w,
        "height": h,
    }

    if not is_damage:
        return result

    # Detector
    det_results = _detector.predict(source=img, imgsz=640, device="mps", verbose=False, conf=0.25)
    r = det_results[0]

    draw = ImageDraw.Draw(img)
    detections = []
    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = _detector.names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        color = COLORS.get(label, "#FFFFFF")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {conf:.0%}"
        bbox = draw.textbbox((x1, y1 - 14), text)
        draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=color)
        draw.text((x1, y1 - 14), text, fill="black")

        detections.append({
            "label": label,
            "conf": round(conf, 3),
            "box": f"[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]",
            "color": color,
        })

    # Encode annotated image
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    result["annotated_image"] = base64.b64encode(buf.getvalue()).decode()
    result["detections"] = detections

    return result


def main():
    ap = argparse.ArgumentParser(description="DamageDoctor web demo")
    ap.add_argument("--classifier-ckpt", type=Path, default=CLASSIFIER_CKPT)
    ap.add_argument("--detector-ckpt", type=Path, default=DETECTOR_CKPT,
                    help="Detector weights (default: v7)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    print(f"Classifier: {args.classifier_ckpt}")
    print(f"Detector:   {args.detector_ckpt}")
    load_models(args.classifier_ckpt, args.detector_ckpt)

    # Warmup both models so first request isn't slow
    print("Warming up models...")
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        _clf(dummy)
    _detector.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print("Ready.")

    print(f"\n  Open http://localhost:{args.port} in your browser\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
