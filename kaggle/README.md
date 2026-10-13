# Kaggle training workflow

Run DamageDoctor detector training on Kaggle's free P100 GPU (16 GB VRAM, 30 h/week).
Side-steps the 16 GB unified-memory constraint on the local M5 MacBook Pro,
trains ~10x faster, and doesn't require closing other apps.

All commands below run from the `kaggle/` directory (`cd kaggle`).

## One-time setup

### 1. Install the Kaggle CLI in the project venv

```bash
../.venv/bin/pip install kaggle
```

### 2. Get your API token

1. Go to <https://www.kaggle.com/ssarvesh20000/account>
2. Scroll to **API** and click **Create New API Token** — downloads `kaggle.json`
3. Place it on disk (local to your Mac only, never share):
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Verify:
   ```bash
   make check-cli
   ```
   Should print `OK: kaggle CLI authenticated`.

### 3. Phone-verify your Kaggle account

- <https://www.kaggle.com/settings> → Phone verification
- Required once to enable **GPU** and **internet** in kernels
- Free, no credit card, ~2 minutes

## Typical workflow

```bash
cd kaggle
make upload-dataset   # one-time: ~3.5 GB upload (resume-safe)
make push-kernel      # uploads train.py and triggers a P100 training run
make status           # poll every few minutes until status = "complete"
make pull-output      # downloads best.pt + logs to runs/kaggle_combined_v3/
```

Expected timings:

| Step | Time |
|---|---|
| `upload-dataset` | 5–50 min (one-time, depends on upload speed) |
| `push-kernel` | < 10 s |
| training (runs on Kaggle) | 45–75 min |
| `pull-output` | 1–2 min (downloads ~50 MB) |

After `pull-output`, `best.pt` lands at `runs/kaggle_combined_v3/best.pt` and
can be used directly by `src/detect.py`, `src/eval_detect_test.py`, and
`src/export_coreml.py`.

## When to re-upload the dataset

`upload-dataset` is a one-time thing. Run `update-dataset` only if you
regenerate `data/combined_yolo/` locally — e.g. after tweaking
`--max-per-class` in `src/convert_vehide_to_yolo.py` or adding new source data.
It creates a new dataset version; old versions remain accessible.

## Cost

**Free.** No credit card required. Kaggle's free tier:

- P100 GPU (16 GB VRAM) — 30 hours per week
- 9 hours maximum per kernel run
- 100 GB total private dataset storage
- Free internet in kernels (needed for `pip install ultralytics` + pretrained weight download)

A full detector training run is under 1 hour, so you could run ~30 trainings
per week and stay under quota.

## Files

- `Makefile` — all workflow commands
- `datasets/combined_yolo/dataset-metadata.json` — metadata describing the uploaded dataset
- `kernels/train_detect_combined/kernel-metadata.json` — Kaggle kernel config (GPU on, internet on, dataset attached)
- `kernels/train_detect_combined/train.py` — the training script that runs inside the Kaggle kernel

## Troubleshooting

**`check-cli` fails with auth error**
Double-check `~/.kaggle/kaggle.json` exists and has `chmod 600`. Regenerate the token from the Kaggle account page if needed.

**`push-kernel` fails with "GPU not enabled"**
Phone-verify your account at <https://www.kaggle.com/settings> first.

**Training OOMs inside the Kaggle kernel**
Shouldn't happen with the current config (batch=32 on a 16 GB P100), but if it does, lower `batch` in `kernels/train_detect_combined/train.py` to 24 or 16 and re-push.

**`pull-output` returns nothing**
Run `make status` — the kernel might still be running, queued, or failed. Kaggle's queue can take a few minutes during peak hours.

**Want to view live logs**
Open <https://www.kaggle.com/code/ssarvesh20000/damagedoctor-train-detector-combined> in a browser. The web UI streams logs while the kernel runs.
