"""Evaluation metrics, timing, model size, and report generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mapping: Dict[int, str],
) -> Dict[str, Any]:
    labels = sorted(label_mapping.keys())
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = {label_mapping[lbl]: float(score) for lbl, score in zip(labels, per_class)}
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "labels": [label_mapping[lbl] for lbl in labels],
    }


def measure_inference_latency_ms(model: Any, X: np.ndarray) -> Dict[str, float]:
    t0 = time.perf_counter()
    _ = model.predict(X)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_sample_ms = elapsed_ms / max(len(X), 1)
    return {
        "total_inference_ms": float(elapsed_ms),
        "ms_per_sample": float(per_sample_ms),
        "num_samples": int(len(X)),
    }


def save_run_artifacts(
    run_dir: Path,
    model: Any,
    config: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)
    model_size_bytes = model_path.stat().st_size

    test_metrics_full = {
        **test_metrics,
        "model_size_bytes": int(model_size_bytes),
        "model_size_mb": round(model_size_bytes / (1024 * 1024), 4),
    }

    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics_full, indent=2), encoding="utf-8")

    cm_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=test_metrics["labels"],
        columns=test_metrics["labels"],
    )
    cm_df.to_csv(run_dir / "confusion_matrix.csv")

    np.savez(
        run_dir / "test_predictions.npz",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob if y_prob is not None else np.array([]),
    )
    return model_path


def evaluate_cnn_baseline(
    own_head_probs_test: np.ndarray,
    y_test: np.ndarray,
    label_mapping: Dict[int, str],
) -> Dict[str, Any]:
    """Experiment A: Evaluate saved CNN-only probabilities on the test split."""
    y_pred = np.argmax(own_head_probs_test, axis=1)
    return compute_classification_metrics(y_test, y_pred, label_mapping)
