# Downstream Classifier Verification & Merge Guide

**Branch Owner:** Chetan (`classifier/` pipeline)  
**Target Reviewer:** Dhruv Kapoor (CNN backbone & feature contract owner)  
**Purpose:** Step-by-step verification guide for testing the `classifier/` downstream pipeline on Dhruv's machine (with full dataset/feature access) before merging into `main`.

---

## 1. What This Branch Adds

All downstream classification code is isolated under `classifier/` and does not modify any existing CNN training scripts (`shared_train.py`, `train_models.py`, `runs/`, `outputs/`):

```text
classifier/
├── README.md
├── requirements.txt
├── configs/default.json
├── src/
│   ├── data_loader.py       # Contract + exploratory loader, group-leakage & NaN/Inf checks
│   ├── train_xgboost.py     # Tunes on val (Macro-F1), evaluates once on test
│   ├── train_adaboost.py    # Tunes on val (Macro-F1), evaluates once on test
│   ├── train_lightgbm.py    # Tunes on val (Macro-F1), evaluates once on test
│   ├── evaluate.py          # Accuracy, Macro-F1, per-class F1, confusion matrix, latency, size
│   └── sensor_join.py       # sample_id join + duplicate-key & schema validation
└── tests/
    ├── test_data_loader.py
    └── test_sensor_join.py
```

---

## 2. Environment Setup & Unit Tests (Run First)

From the repository root (`shared-cnn-backbone-food-and-water/`):

```bash
cd classifier
python3 -m pip install -r requirements.txt
python3 -m pytest tests/
```

*(macOS users: if `xgboost` or `lightgbm` raises a missing `libomp.dylib` error, run `brew install libomp` first.)*

**Expected Evidence:**
- All unit tests in `tests/test_data_loader.py` and `tests/test_sensor_join.py` pass (`6 passed`).
- Confirms enforcement of: row-count alignment, `NaN`/`Inf` rejection, `group_id` cross-split leakage detection, `exploratory=False` strict contract enforcement, and duplicate `sample_id` rejection in `sensor_join.py`.

---

## 3. Exploratory Baseline Verification (Already Passed on Chetan's Machine)

To verify against the current `features/` directory (`mobilenetv2_shared` and `customcnn_shared` with 3-column `index.csv`):

```bash
for backbone in mobilenetv2_shared customcnn_shared; do
  for domain in food water; do
    for algo in xgboost adaboost lightgbm; do
      echo "=== Running $algo | $backbone | $domain ==="
      python3 -m src.train_${algo} --project-root .. --backbone-id ${backbone} --domain ${domain} --exploratory || break 3
    done
  done
done
```

### Reference Exploratory Results (Seed 42, 128-d Embeddings)

> **Note:** These results are **exploratory only** because the current `index.csv` lacks `bb_train` / `head_train` separation and `group_id`.

| Backbone | Domain | Classifier | Exploratory Macro-F1 | Exploratory Accuracy |
|---|---|---|---|---|
| `mobilenetv2_shared` | `food` | XGBoost | `0.9965` | `0.9961` |
| `mobilenetv2_shared` | `food` | AdaBoost | `0.9929` | `0.9909` |
| `mobilenetv2_shared` | `food` | LightGBM | `0.9975` | `0.9974` |
| `mobilenetv2_shared` | `water` | XGBoost | `0.9774` | `0.9677` |
| `mobilenetv2_shared` | `water` | AdaBoost | `0.9774` | `0.9677` |
| `mobilenetv2_shared` | `water` | LightGBM | `0.9030` | `0.9570` |
| `customcnn_shared` | `food` | XGBoost | `0.9797` | `0.9721` |
| `customcnn_shared` | `food` | AdaBoost | `0.8416` | `0.8145` |
| `customcnn_shared` | `food` | LightGBM | `0.9864` | `0.9812` |
| `customcnn_shared` | `water` | XGBoost | `0.9615` | `0.9462` |
| `customcnn_shared` | `water` | AdaBoost | `0.9566` | `0.9462` |
| `customcnn_shared` | `water` | LightGBM | `0.9615` | `0.9462` |

---

## 4. Validated Contract Verification (Run Once Final Features Are Ready)

When Dhruv's finalized feature contract is in place under `features/<backbone_id>/<domain>/`, `data_loader.py` expects:

1. `emb_128.npy` (`float32`, shape `(N, 128)`) and optional `pooled.npy` (`float32`).
2. `index.csv` with exact columns: `sample_id, domain, split, label, produce, state, group_id, source_dataset`.
   - `split` values must be in `{"bb_train", "head_train", "val", "test"}`.
   - `bb_train` is automatically excluded from downstream training.
   - Any `group_id` appearing in more than one split raises a `ValueError`.
3. Optional `own_head_probs.npy` (row-aligned with `index.csv`) and `meta.json`.

Run **without** `--exploratory` to enforce strict contract mode across all approved backbones:

```bash
for backbone in mnv2_food mnv2_water mnv2_shared cnn_food cnn_water cnn_shared; do
  for domain in food water; do
    for algo in xgboost adaboost lightgbm; do
      echo "=== Validated Run: $algo | $backbone | $domain ==="
      python3 -m src.train_${algo} --project-root .. --backbone-id ${backbone} --domain ${domain} || break 3
    done
  done
done
```

*(To run the pooled-feature ablation, add `--feature-type pooled`.)*

---

## 5. Expected Output Evidence Per Run

Every completed run writes a self-contained folder under `classifier/runs/<algo>/<backbone>_<domain>_<feature_type>_seed42_<status>/` containing:

- `config.json` — Backbone ID, domain, feature dimension, best hyperparameters (selected on `val` only), seed, and label mapping.
- `val_metrics.json` — Validation Accuracy, Macro-F1, per-class F1, and confusion matrix.
- `test_metrics.json` — Final test Accuracy, Macro-F1, per-class F1, confusion matrix, `model_size_bytes` / `model_size_mb`, and `latency` (`total_inference_ms`, `ms_per_sample`).
- `confusion_matrix.csv` — Labeled confusion matrix for `test`.
- `test_predictions.npz` — Saved `y_true`, `y_pred`, and `y_prob` arrays.
- `model.joblib` — Serialized classifier artifact.

---

## 6. Merge Sign-Off Checklist

- [ ] `python3 -m pytest tests/` passes cleanly.
- [ ] Exploratory runs execute without `NaN`/`Inf` or path errors.
- [ ] `git status` shows changes scoped cleanly to `classifier/` (no accidental large binary or `.venv/` commits).
- [ ] Ready to merge feature branch into `main`.
