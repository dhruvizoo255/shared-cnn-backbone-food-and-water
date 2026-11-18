# Downstream Classification Pipeline (`classifier/`)

Modular downstream pipeline for **AI-Based Food Spoilage and Water-Condition Detection Using a Shared CNN Backbone**.

## Structure

- `src/data_loader.py` — Loads `emb_128.npy` / `pooled.npy` and `index.csv` with configurable `project_root` and `backbone_id`. Enforces contract validation (row alignment, finite values, `group_id` split isolation, and `bb_train` exclusion) and supports explicit `--exploratory` mode for the preliminary `path,label_id,label_name` index files.
- `src/train_xgboost.py` — Tunes XGBoost on `val` (Macro-F1) and evaluates once on `test`.
- `src/train_adaboost.py` — Tunes AdaBoost on `val` (Macro-F1) and evaluates once on `test`.
- `src/train_lightgbm.py` — Tunes LightGBM on `val` (Macro-F1) and evaluates once on `test`.
- `src/evaluate.py` — Computes Accuracy, Macro-F1, per-class F1, confusion matrices, serialized model size, and inference latency; supports CNN-only baseline evaluation via `own_head_probs.npy`.
- `src/sensor_join.py` — Validates and joins image indices with water/food sensor CSVs on `sample_id`, raising clear errors on duplicate keys.
- `tests/` — Pytest suite for contract validation, leakage prevention, and sensor joining.

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/

# Exploratory run on existing 128-d embeddings (until final manifest is delivered)
python -m src.train_xgboost --project-root .. --backbone-id mobilenetv2_shared --domain food --exploratory
python -m src.train_adaboost --project-root .. --backbone-id mobilenetv2_shared --domain food --exploratory
python -m src.train_lightgbm --project-root .. --backbone-id mobilenetv2_shared --domain food --exploratory

# Validated run once Dhruv supplies the final contract (omit --exploratory)
python -m src.train_xgboost --project-root .. --backbone-id mobilenetv2_shared --domain food
```
