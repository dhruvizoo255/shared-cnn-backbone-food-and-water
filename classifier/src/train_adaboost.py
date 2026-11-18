"""Train and evaluate ADABOOST on CNN-extracted features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_features
from src.evaluate import (
    compute_classification_metrics,
    measure_inference_latency_ms,
    save_run_artifacts,
)

DEFAULT_PARAM_GRID = [{'n_estimators': 100, 'learning_rate': 0.5, 'max_depth': 1}, {'n_estimators': 200, 'learning_rate': 0.5, 'max_depth': 2}]


def build_model(params: dict, num_classes: int, seed: int):
    p = dict(params)
    max_depth = p.pop("max_depth", 1)
    base = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    return AdaBoostClassifier(estimator=base, random_state=seed, **p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train adaboost downstream classifier")
    parser.add_argument("--project-root", type=str, default="..")
    parser.add_argument("--backbone-id", type=str, required=True)
    parser.add_argument("--domain", type=str, choices=["food", "water"], required=True)
    parser.add_argument("--feature-type", type=str, choices=["emb_128", "pooled"], default="emb_128")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs/adaboost")
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args()

    dataset = load_features(
        project_root=args.project_root,
        backbone_id=args.backbone_id,
        domain=args.domain,
        feature_type=args.feature_type,
        seed=args.seed,
        exploratory=args.exploratory,
    )
    num_classes = len(dataset.label_mapping)

    best_model = None
    best_params = None
    best_val_metrics = None
    best_score = -1.0

    for params in DEFAULT_PARAM_GRID:
        model = build_model(params, num_classes=num_classes, seed=args.seed)
        model.fit(dataset.head_train.X, dataset.head_train.y)
        val_pred = model.predict(dataset.val.X)
        val_metrics = compute_classification_metrics(
            dataset.val.y, val_pred, dataset.label_mapping
        )
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_model = model
            best_params = params
            best_val_metrics = val_metrics

    test_pred = best_model.predict(dataset.test.X)
    test_prob = best_model.predict_proba(dataset.test.X) if hasattr(best_model, "predict_proba") else None
    test_metrics = compute_classification_metrics(dataset.test.y, test_pred, dataset.label_mapping)
    test_metrics["latency"] = measure_inference_latency_ms(best_model, dataset.test.X)

    status_tag = "exploratory" if dataset.is_exploratory else "validated"
    run_dir = Path(args.output_dir) / f"{args.backbone_id}_{args.domain}_{args.feature_type}_seed{args.seed}_{status_tag}"
    config = {
        "algorithm": "adaboost",
        "backbone_id": args.backbone_id,
        "domain": args.domain,
        "feature_type": args.feature_type,
        "feature_dim": dataset.feature_dim,
        "seed": args.seed,
        "status": status_tag,
        "best_params": best_params,
        "label_mapping": dataset.label_mapping,
    }

    save_run_artifacts(
        run_dir=run_dir,
        model=best_model,
        config=config,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        y_test=dataset.test.y,
        y_pred=test_pred,
        y_prob=test_prob,
    )
    print(json.dumps({"run_dir": str(run_dir), "status": status_tag, "test_macro_f1": test_metrics["macro_f1"], "test_accuracy": test_metrics["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
