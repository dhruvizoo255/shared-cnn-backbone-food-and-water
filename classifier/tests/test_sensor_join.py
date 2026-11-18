import pandas as pd
import pytest

from src.sensor_join import join_image_and_sensor


def test_water_sensor_join_success():
    img_df = pd.DataFrame({"sample_id": ["w1", "w2"], "label": [0, 1]})
    sen_df = pd.DataFrame(
        {
            "sample_id": ["w1", "w2"],
            "ph": [7.1, 5.4],
            "tds_ppm": [120.0, 480.0],
            "turbidity_ntu": [1.2, 18.5],
            "temp_c": [24.5, 26.0],
        }
    )
    joined = join_image_and_sensor(img_df, sen_df, domain="water")
    assert len(joined) == 2
    assert "ph" in joined.columns


def test_duplicate_sensor_key_raises():
    img_df = pd.DataFrame({"sample_id": ["f1", "f2"], "label": [0, 1]})
    sen_df = pd.DataFrame(
        {
            "sample_id": ["f1", "f1"],
            "mq135_raw": [310, 315],
            "temp_c": [25.0, 25.1],
        }
    )
    with pytest.raises(ValueError, match="Duplicate sensor sample_id"):
        join_image_and_sensor(img_df, sen_df, domain="food")
