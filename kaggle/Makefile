# DamageDoctor — Kaggle training workflow
#
# Run `make help` for the list of commands.
# All paths are relative to the kaggle/ directory.

KAGGLE      := ../.venv/bin/kaggle
KAGGLE_USER := ssarvesh20000

# --- v8 detect + 280 bg images (current) ---
V8_KERNEL_SLUG  := $(KAGGLE_USER)/damagedoctor-train-detect-v8
V8_KERNEL_DIR   := kernels/train_detect_v8
V8_OUTPUT_DIR   := ../runs/kaggle_cardd_v8_detect

# --- v7 detect + 200 bg images (done) ---
V7_KERNEL_SLUG  := $(KAGGLE_USER)/damagedoctor-train-detect-v7
V7_KERNEL_DIR   := kernels/train_detect_v7
V7_OUTPUT_DIR   := ../runs/kaggle_cardd_v7_detect

# --- v6 detect ablation (done) ---
V6_KERNEL_SLUG  := $(KAGGLE_USER)/damagedoctor-train-detect-v6
V6_KERNEL_DIR   := kernels/train_detect_v6
V6_OUTPUT_DIR   := ../runs/kaggle_cardd_v6_detect

# --- v5 seg experiment (done) ---
SEG_DATASET_SLUG := $(KAGGLE_USER)/damagedoctor-cardd-yolo-seg
SEG_KERNEL_SLUG  := $(KAGGLE_USER)/damagedoctor-train-seg-cardd-v5
SEG_DATA_DIR     := ../data/cardd_yolo_seg
SEG_DATASET_META := datasets/cardd_yolo_seg/dataset-metadata.json
SEG_KERNEL_DIR   := kernels/train_seg_cardd
SEG_OUTPUT_DIR   := ../runs/kaggle_cardd_v5_seg

.PHONY: help check-cli \
        push-v8 v8-status pull-v8 \
        push-v7 v7-status pull-v7 \
        push-v6 v6-status pull-v6 \
        upload-seg-dataset push-seg-kernel seg-status pull-seg-output \
        clean-tmp

help:
	@echo "DamageDoctor — Kaggle training workflow"
	@echo ""
	@echo "One-time setup:"
	@echo "  ../.venv/bin/pip install kaggle"
	@echo "  # Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)"
	@echo "  # Phone-verify at https://www.kaggle.com/settings to enable GPU + internet"
	@echo ""
	@echo "v8 detect + 280 bg images (v1 recipe, 10% background):"
	@echo "  make push-v8        Push training script and trigger GPU run"
	@echo "  make v8-status      Poll current training run status"
	@echo "  make pull-v8        Download best.pt + logs"
	@echo ""
	@echo "Previous experiments (done):"
	@echo "  v7: push-v7 / v7-status / pull-v7   (200 bg, new baseline)"
	@echo "  v6: push-v6 / v6-status / pull-v6"
	@echo "  v5: push-seg-kernel / seg-status / pull-seg-output"
	@echo ""
	@echo "Utilities:"
	@echo "  make check-cli      Verify kaggle CLI is installed and authenticated"

check-cli:
	@test -f ~/.kaggle/kaggle.json || { \
		echo "ERROR: ~/.kaggle/kaggle.json not found."; \
		echo "Get token at https://www.kaggle.com/$(KAGGLE_USER)/account -> Create New API Token"; \
		exit 1; \
	}
	@$(KAGGLE) kernels list --mine --page-size 1 >/dev/null && echo "OK: kaggle CLI authenticated"

# --- v8 detect + 280 bg images ---
# Uses cardd-yolo dataset v3 (with 280 bg images)
push-v8: check-cli
	$(KAGGLE) kernels push -p $(V8_KERNEL_DIR)
	@echo ""
	@echo "Kernel running at: https://www.kaggle.com/code/$(V8_KERNEL_SLUG)"

v8-status: check-cli
	$(KAGGLE) kernels status $(V8_KERNEL_SLUG)

pull-v8: check-cli
	@mkdir -p $(V8_OUTPUT_DIR)
	$(KAGGLE) kernels output $(V8_KERNEL_SLUG) -p $(V8_OUTPUT_DIR)
	@echo ""
	@echo "Downloaded to $(V8_OUTPUT_DIR)/"
	@ls -la $(V8_OUTPUT_DIR) || true

# --- v7 detect + 200 bg images ---
push-v7: check-cli
	$(KAGGLE) kernels push -p $(V7_KERNEL_DIR)
	@echo ""
	@echo "Kernel running at: https://www.kaggle.com/code/$(V7_KERNEL_SLUG)"

v7-status: check-cli
	$(KAGGLE) kernels status $(V7_KERNEL_SLUG)

pull-v7: check-cli
	@mkdir -p $(V7_OUTPUT_DIR)
	$(KAGGLE) kernels output $(V7_KERNEL_SLUG) -p $(V7_OUTPUT_DIR)
	@echo ""
	@echo "Downloaded to $(V7_OUTPUT_DIR)/"
	@ls -la $(V7_OUTPUT_DIR) || true

# --- v6 detect ablation ---
push-v6: check-cli
	$(KAGGLE) kernels push -p $(V6_KERNEL_DIR)
	@echo ""
	@echo "Kernel running at: https://www.kaggle.com/code/$(V6_KERNEL_SLUG)"

v6-status: check-cli
	$(KAGGLE) kernels status $(V6_KERNEL_SLUG)

pull-v6: check-cli
	@mkdir -p $(V6_OUTPUT_DIR)
	$(KAGGLE) kernels output $(V6_KERNEL_SLUG) -p $(V6_OUTPUT_DIR)
	@echo ""
	@echo "Downloaded to $(V6_OUTPUT_DIR)/"
	@ls -la $(V6_OUTPUT_DIR) || true

# --- v5 seg dataset ---
upload-seg-dataset: check-cli
	@test -d $(SEG_DATA_DIR) || { echo "ERROR: $(SEG_DATA_DIR) does not exist. Run src/convert_coco_to_yolo_seg.py first."; exit 1; }
	cp $(SEG_DATASET_META) $(SEG_DATA_DIR)/dataset-metadata.json
	$(KAGGLE) datasets create -p $(SEG_DATA_DIR) -r zip || { rm -f $(SEG_DATA_DIR)/dataset-metadata.json; exit 1; }
	rm -f $(SEG_DATA_DIR)/dataset-metadata.json
	@echo ""
	@echo "Dataset live at: https://www.kaggle.com/datasets/$(SEG_DATASET_SLUG)"

push-seg-kernel: check-cli
	$(KAGGLE) kernels push -p $(SEG_KERNEL_DIR)
	@echo ""
	@echo "Kernel running at: https://www.kaggle.com/code/$(SEG_KERNEL_SLUG)"

seg-status: check-cli
	$(KAGGLE) kernels status $(SEG_KERNEL_SLUG)

pull-seg-output: check-cli
	@mkdir -p $(SEG_OUTPUT_DIR)
	$(KAGGLE) kernels output $(SEG_KERNEL_SLUG) -p $(SEG_OUTPUT_DIR)
	@echo ""
	@echo "Downloaded to $(SEG_OUTPUT_DIR)/"
	@ls -la $(SEG_OUTPUT_DIR) || true

clean-tmp:
	rm -f $(SEG_DATA_DIR)/dataset-metadata.json
