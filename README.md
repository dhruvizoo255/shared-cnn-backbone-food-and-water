# Shared-Backbone CNN for Food Spoilage and Water Condition Classification

An ECE micro-project exploring whether a single CNN backbone can learn useful visual features from both food freshness images and visible water-condition images, compared with domain-specific models.

**Institution:** Bharati Vidyapeeth's College of Engineering, New Delhi  
**Academic Year:** 2026–27  
**Framework:** TensorFlow / Keras  
**Project Status:** In progress

## Overview

The broader project aims to develop an ESP32-CAM-based sensing platform with interchangeable food and water sensor modules.

This repository focuses on the CNN pipeline: training visual feature extractors, generating embeddings, comparing shared and domain-specific models, and preparing features for downstream classification.

The central research question is:

> Can a shared CNN backbone trained on food and water images perform comparably to separate domain-specific backbones?

## Current Features

- Shared MobileNetV2 training
- Shared custom CNN training
- Separate food/water model artifacts
- 128-dimensional image embeddings
- Pooled feature extraction
- Saved checkpoints and training histories
- Preliminary feature and training audits

## Repository Structure

```text
sharedcnn_project/
├── training.ipynb
├── shared_train.py
├── train_models.py
├── runs/
│   ├── mobilenetv2_shared/
│   ├── customcnn_shared/
│   └── customcnn_v2_shared/
├── outputs/
├── features/
├── training.log
├── project_audit.txt
├── README.md
└── docs/
```

The exact contents may differ between project versions. Consult the files included in the repository before assuming a particular experiment is complete.

## Experimental Plan

The intended comparison includes six configurations:

| Model ID | Architecture | Training Data |
|---|---|---|
| `mnv2_food` | MobileNetV2 | Food |
| `mnv2_water` | MobileNetV2 | Water |
| `mnv2_shared` | MobileNetV2 | Food and water |
| `cnn_food` | Custom CNN | Food |
| `cnn_water` | Custom CNN | Water |
| `cnn_shared` | Custom CNN | Food and water |

The complete six-model experiment has not yet been established as finished. Exploratory custom-CNN variants must be identified separately.

## Setup

Create a Python environment:

```bash
conda create -n sharedcnn python=3.12 -y
conda activate sharedcnn
python -m pip install --upgrade pip
pip install tensorflow numpy pandas scikit-learn matplotlib pillow
```

These are starter dependencies, not a verified, fully pinned environment. Check the imports required by each script and record the actual TensorFlow, Keras, Python and library versions used.

### Verify GPU availability

```python
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
```

Modern TensorFlow GPU execution on Windows generally requires a supported Linux or WSL2 environment. Verify GPU visibility before running lengthy training jobs.

## Existing Feature Artifacts

The current audit recorded the following feature-array shapes:

| Model | Domain | Embeddings | Pooled Features |
|---|---|---|---|
| MobileNetV2 shared | Food | `(10277, 128)` | `(10277, 1280)` |
| MobileNetV2 shared | Water | `(617, 128)` | `(617, 1280)` |
| Custom CNN shared | Food | `(10277, 128)` | `(10277, 128)` |
| Custom CNN shared | Water | `(617, 128)` | `(617, 128)` |

These are recorded artifact dimensions, not evidence of final model performance or a validated feature-to-label mapping.

## Important Data Limitations

The source datasets are not included in the supplied project archive. Some saved paths may reference the original machine.

Before final evaluation:

1. Obtain the approved source datasets and record their licences.
2. Audit labels, class counts and unreadable images.
3. Deduplicate images across datasets and domains.
4. Group related images, including frames from the same video.
5. Create and freeze a group-aware, stratified manifest.
6. Maintain separate backbone-training, classifier-training, validation and test splits.

The current feature indices contain only `path`, `label_id` and `label_name`. They do not yet satisfy the full contract requiring stable sample IDs, domain, split, group ID and source metadata.

**Do not use the current feature arrays for final research claims until their row-to-label mapping and split membership have been verified.**

## Downstream Classification

The downstream classifier pipeline is intended to evaluate:

- XGBoost
- AdaBoost
- LightGBM

The classifier owner can develop the training pipeline and evaluation code in parallel. Final comparisons must use validated embeddings, approved split assignments and consistent evaluation procedures.

Metrics should include accuracy, macro-F1, per-class F1 and confusion matrices. Hyperparameter tuning must use training and validation data, never the held-out test set.

## Sensor Fusion

The planned water sensor schema includes pH, TDS, turbidity and temperature. The food module may use MQ-135 gas readings, temperature and optional TCS3200 colour measurements.

Image and sensor records must share a stable `sample_id`.

No paired real image-and-sensor dataset is established by the current artifacts. Sensor-fusion results must not be claimed until real paired measurements have been collected and validated.

## Current Limitations

- The final deduplicated data splits are not yet verified.
- The complete six-backbone comparison is not yet established.
- Feature metadata and row mappings require validation.
- Cross-domain probes and domain-separability diagnostics remain to be completed.
- Final latency and model-size measurements remain to be completed.
- Sensor fusion and ESP32-CAM deployment have not been established by the current artifacts.

Water-image classification concerns visible water condition; it does not establish potability or the absence of dissolved or microbial contaminants.

## Collaboration

Keep dataset manifests, label mappings, split definitions and feature schemas consistent across contributors. Record changes to model configurations and clearly distinguish exploratory runs from final evaluations.

Every reported metric should be traceable to saved experiment outputs.

## License

The repository's code licence has not yet been specified. Add an appropriate licence after the team decides how the code may be reused. Dataset licences and attribution requirements remain separate and must be respected.

## Disclaimer

This repository represents an academic research prototype, not a certified food-safety or water-quality assessment system. Real-world performance, sensor fusion and edge-device deployment require separate validation.
