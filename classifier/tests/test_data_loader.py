import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_features


def _write_dataset(tmp_path, index_df, X):
    feat_dir = tmp_path / "features" / "mobilenetv2_shared" / "food"
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_dir / "emb_128.npy", X.astype(np.float32))
    index_df.to_csv(feat_dir / "index.csv", index=False)


def _make_contract_df(n=12):
    splits = ["bb_train"] * 4 + ["head_train"] * 4 + ["val"] * 2 + ["test"] * 2
    return pd.DataFrame(
        {
            "sample_id": [f"s_{i}" for i in range(n)],
            "domain": ["food"] * n,
            "split": splits,
            "label": [i % 2 for i in range(n)],
            "produce": ["apple"] * n,
            "state": ["fresh" if i % 2 == 0 else "spoiled" for i in range(n)],
            "group_id": [f"g_{i}" for i in range(n)],
            "source_dataset": ["ds1"] * n,
        }
    )


def test_contract_load_success(tmp_path):
    df = _make_contract_df(12)
    X = np.ones((12, 128), dtype=np.float32)
    _write_dataset(tmp_path, df, X)

    ds = load_features(tmp_path, "mobilenetv2_shared", "food", exploratory=False)
    assert not ds.is_exploratory
    assert ds.head_train.X.shape == (4, 128)
    assert ds.val.X.shape == (2, 128)
    assert ds.test.X.shape == (2, 128)


def test_group_leakage_raises(tmp_path):
    df = _make_contract_df(12)
    df.loc[4, "group_id"] = "g_0"  # leak across bb_train and head_train
    X = np.ones((12, 128), dtype=np.float32)
    _write_dataset(tmp_path, df, X)

    with pytest.raises(ValueError, match="Group leakage"):
        load_features(tmp_path, "mobilenetv2_shared", "food", exploratory=False)


def test_non_finite_features_raise(tmp_path):
    df = _make_contract_df(12)
    X = np.ones((12, 128), dtype=np.float32)
    X[0, 0] = np.nan
    _write_dataset(tmp_path, df, X)

    with pytest.raises(ValueError, match="non-finite"):
        load_features(tmp_path, "mobilenetv2_shared", "food", exploratory=False)


def test_exploratory_mode_fallback(tmp_path):
    n = 20
    df = pd.DataFrame(
        {
            "path": [f"/old/img_{i}.jpg" for i in range(n)],
            "label_id": [i % 2 for i in range(n)],
            "label_name": ["fresh" if i % 2 == 0 else "spoiled" for i in range(n)],
        }
    )
    X = np.ones((n, 128), dtype=np.float32)
    _write_dataset(tmp_path, df, X)

    with pytest.raises(ValueError, match="exploratory=True"):
        load_features(tmp_path, "mobilenetv2_shared", "food", exploratory=False)

    ds = load_features(tmp_path, "mobilenetv2_shared", "food", exploratory=True)
    assert ds.is_exploratory
    assert ds.head_train.X.shape[0] + ds.val.X.shape[0] + ds.test.X.shape[0] == n
