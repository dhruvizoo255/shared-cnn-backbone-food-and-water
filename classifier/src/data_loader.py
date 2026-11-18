"""Feature loading and contract validation for downstream classifiers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

REQUIRED_CONTRACT_COLUMNS = {
    "sample_id",
    "domain",
    "split",
    "label",
    "produce",
    "state",
    "group_id",
    "source_dataset",
}
EXPLORATORY_COLUMNS = {"path", "label_id", "label_name"}
VALID_SPLITS = {"bb_train", "head_train", "val", "test"}


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    index_df: pd.DataFrame


@dataclass
class LoadedDataset:
    backbone_id: str
    domain: str
    feature_type: str
    feature_dim: int
    is_exploratory: bool
    label_mapping: Dict[int, str]
    head_train: SplitData
    val: SplitData
    test: SplitData
    own_head_probs_test: Optional[np.ndarray] = None
    meta: Optional[dict] = None


def resolve_feature_dir(project_root: str | Path, backbone_id: str, domain: str) -> Path:
    root = Path(project_root).expanduser().resolve()
    feature_dir = root / "features" / backbone_id / domain
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    return feature_dir


def validate_feature_array(X: np.ndarray, index_df: pd.DataFrame) -> None:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {X.shape}")
    if X.shape[0] != len(index_df):
        raise ValueError(
            f"Feature row count ({X.shape[0]}) != index row count ({len(index_df)})"
        )
    if not np.isfinite(X).all():
        bad = int(np.size(X) - np.isfinite(X).sum())
        raise ValueError(f"Feature array contains {bad} non-finite (NaN/Inf) values")


def validate_contract_index(index_df: pd.DataFrame, expected_domain: str) -> None:
    missing = REQUIRED_CONTRACT_COLUMNS - set(index_df.columns)
    if missing:
        raise ValueError(
            f"index.csv is missing required contract columns: {sorted(missing)}"
        )

    if index_df["sample_id"].duplicated().any():
        dups = index_df.loc[index_df["sample_id"].duplicated(), "sample_id"].tolist()[:5]
        raise ValueError(f"Duplicate sample_id values found in index.csv: {dups}")

    domains = set(index_df["domain"].astype(str).str.lower().unique())
    if domains != {expected_domain.lower()}:
        raise ValueError(
            f"Domain mismatch in index.csv: expected {{{expected_domain}}}, found {domains}"
        )

    splits = set(index_df["split"].unique())
    unknown_splits = splits - VALID_SPLITS
    if unknown_splits:
        raise ValueError(f"Unknown split labels in index.csv: {unknown_splits}")

    for required_split in ("head_train", "val", "test"):
        if required_split not in splits:
            raise ValueError(f"Missing required split '{required_split}' in index.csv")

    # Enforce group_id isolation across splits
    group_splits = index_df.groupby("group_id")["split"].nunique()
    leaked_groups = group_splits[group_splits > 1]
    if not leaked_groups.empty:
        raise ValueError(
            f"Group leakage detected: {len(leaked_groups)} group_id(s) span multiple splits "
            f"(e.g. {leaked_groups.index.tolist()[:5]})"
        )


def _encode_labels(series: pd.Series) -> Tuple[np.ndarray, Dict[int, str]]:
    if pd.api.types.is_integer_dtype(series):
        y = series.to_numpy(dtype=np.int64)
        mapping = {int(v): str(v) for v in sorted(np.unique(y))}
        return y, mapping
    categories = sorted(series.astype(str).unique())
    str_to_id = {name: idx for idx, name in enumerate(categories)}
    y = series.astype(str).map(str_to_id).to_numpy(dtype=np.int64)
    id_to_str = {idx: name for name, idx in str_to_id.items()}
    return y, id_to_str


def load_features(
    project_root: str | Path,
    backbone_id: str,
    domain: str,
    feature_type: str = "emb_128",
    seed: int = 42,
    exploratory: bool = False,
) -> LoadedDataset:
    """Load and validate CNN features and splits for downstream training."""
    if feature_type not in {"emb_128", "pooled"}:
        raise ValueError("feature_type must be 'emb_128' or 'pooled'")

    feature_dir = resolve_feature_dir(project_root, backbone_id, domain)
    npy_path = feature_dir / f"{feature_type}.npy"
    index_path = feature_dir / "index.csv"
    meta_path = feature_dir / "meta.json"
    probs_path = feature_dir / "own_head_probs.npy"

    if not npy_path.exists():
        raise FileNotFoundError(f"Feature array not found: {npy_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_path}")

    X = np.load(npy_path)
    index_df = pd.read_csv(index_path)
    validate_feature_array(X, index_df)

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    has_contract = REQUIRED_CONTRACT_COLUMNS.issubset(set(index_df.columns))
    if has_contract:
        validate_contract_index(index_df, expected_domain=domain)
        y, label_mapping = _encode_labels(index_df["label"])

        def _subset(split_name: str) -> Tuple[SplitData, np.ndarray]:
            mask = (index_df["split"] == split_name).to_numpy()
            return SplitData(X=X[mask], y=y[mask], index_df=index_df.loc[mask].copy()), mask

        head_train, _ = _subset("head_train")
        val, _ = _subset("val")
        test, test_mask = _subset("test")

        own_head_probs_test = None
        if probs_path.exists():
            probs = np.load(probs_path)
            if probs.shape[0] != len(index_df):
                raise ValueError(
                    f"own_head_probs.npy rows ({probs.shape[0]}) != index rows ({len(index_df)})"
                )
            own_head_probs_test = probs[test_mask]

        return LoadedDataset(
            backbone_id=backbone_id,
            domain=domain,
            feature_type=feature_type,
            feature_dim=int(X.shape[1]),
            is_exploratory=False,
            label_mapping=label_mapping,
            head_train=head_train,
            val=val,
            test=test,
            own_head_probs_test=own_head_probs_test,
            meta=meta,
        )

    # Exploratory fallback for preliminary index.csv (path, label_id, label_name)
    if not exploratory:
        raise ValueError(
            "index.csv does not satisfy the final contract (missing sample_id/split/group_id). "
            "Pass exploratory=True (or --exploratory) to run preliminary experiments."
        )

    if not EXPLORATORY_COLUMNS.issubset(set(index_df.columns)):
        raise ValueError(
            f"Exploratory index.csv must contain {sorted(EXPLORATORY_COLUMNS)}, "
            f"got {list(index_df.columns)}"
        )

    logger.warning(
        "EXPLORATORY MODE: Using preliminary index.csv without canonical splits or group_id. "
        "Do not report these metrics as final."
    )

    y = index_df["label_id"].to_numpy(dtype=np.int64)
    label_mapping = (
        index_df[["label_id", "label_name"]]
        .drop_duplicates()
        .set_index("label_id")["label_name"]
        .to_dict()
    )
    label_mapping = {int(k): str(v) for k, v in label_mapping.items()}

    indices = np.arange(len(index_df))
    # 70% train, 15% val, 15% test stratified exploratory split
    idx_train, idx_hold = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=y
    )
    idx_val, idx_test = train_test_split(
        idx_hold, test_size=0.50, random_state=seed, stratify=y[idx_hold]
    )

    return LoadedDataset(
        backbone_id=backbone_id,
        domain=domain,
        feature_type=feature_type,
        feature_dim=int(X.shape[1]),
        is_exploratory=True,
        label_mapping=label_mapping,
        head_train=SplitData(X=X[idx_train], y=y[idx_train], index_df=index_df.iloc[idx_train].copy()),
        val=SplitData(X=X[idx_val], y=y[idx_val], index_df=index_df.iloc[idx_val].copy()),
        test=SplitData(X=X[idx_test], y=y[idx_test], index_df=index_df.iloc[idx_test].copy()),
        own_head_probs_test=None,
        meta=meta,
    )
