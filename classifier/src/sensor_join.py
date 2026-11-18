"""Sensor-fusion schema validation and sample_id join interface."""

from __future__ import annotations

from typing import List

import pandas as pd

WATER_REQUIRED_FIELDS: List[str] = ["sample_id", "ph", "tds_ppm", "turbidity_ntu", "temp_c"]
FOOD_REQUIRED_FIELDS: List[str] = ["sample_id", "mq135_raw", "temp_c"]
FOOD_OPTIONAL_FIELDS: List[str] = ["tcs3200_r", "tcs3200_g", "tcs3200_b"]


def validate_sensor_table(sensor_df: pd.DataFrame, domain: str) -> List[str]:
    domain_l = domain.lower()
    if domain_l == "water":
        required = WATER_REQUIRED_FIELDS
        optional: List[str] = []
    elif domain_l == "food":
        required = FOOD_REQUIRED_FIELDS
        optional = FOOD_OPTIONAL_FIELDS
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    missing = [col for col in required if col not in sensor_df.columns]
    if missing:
        raise ValueError(f"Sensor table for '{domain_l}' is missing columns: {missing}")

    if sensor_df["sample_id"].isna().any():
        raise ValueError("Sensor table contains null sample_id values")

    if sensor_df["sample_id"].duplicated().any():
        dups = sensor_df.loc[sensor_df["sample_id"].duplicated(), "sample_id"].tolist()[:5]
        raise ValueError(f"Duplicate sensor sample_id keys detected: {dups}")

    present_sensor_cols = [c for c in required + optional if c != "sample_id" and c in sensor_df.columns]
    for col in present_sensor_cols:
        if not pd.api.types.is_numeric_dtype(sensor_df[col]):
            raise ValueError(f"Sensor column '{col}' must be numeric")

    return present_sensor_cols


def join_image_and_sensor(
    index_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    domain: str,
    how: str = "inner",
) -> pd.DataFrame:
    """Join image index metadata with physical sensor readings on sample_id."""
    if "sample_id" not in index_df.columns:
        raise ValueError("index_df must contain 'sample_id' before sensor joining")
    if index_df["sample_id"].duplicated().any():
        dups = index_df.loc[index_df["sample_id"].duplicated(), "sample_id"].tolist()[:5]
        raise ValueError(f"Duplicate image sample_id keys detected: {dups}")

    sensor_cols = validate_sensor_table(sensor_df, domain=domain)
    trimmed_sensor = sensor_df[["sample_id"] + sensor_cols]
    merged = index_df.merge(trimmed_sensor, on="sample_id", how=how, validate="one_to_one")
    return merged
