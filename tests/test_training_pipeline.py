import numpy as np
import pandas as pd

from config import FEATURE_COLS, TARGET_COLS
from src.feature_engineering import build_feature_matrix
from src.train import create_time_series_split, train_and_compare


def make_sample_weather_df(rows: int = 220) -> pd.DataFrame:
    rng = np.random.RandomState(7)
    dates = pd.date_range("2014-01-01", periods=rows, freq="D")
    temp = 22 + np.sin(np.arange(rows) / 20) * 6 + np.arange(rows) / 250
    humidity = 60 + np.sin(np.arange(rows) / 18) * 10 + rng.normal(0, 3, rows)
    dewpoint = temp - 3 + rng.normal(0, 2, rows)
    pressure = 1010 + np.sin(np.arange(rows) / 35) * 4 + rng.normal(0, 1, rows)
    windspeed = np.abs(np.sin(np.arange(rows) / 15)) * 7 + rng.normal(0, 1.5, rows)
    cloudcover = np.clip(np.abs(np.sin(np.arange(rows) / 13)) * 100 + rng.normal(0, 8, rows), 0, 100)
    visibility = np.clip(10 + np.cos(np.arange(rows) / 16) * 2 + rng.normal(0, 0.8, rows), 0, None)
    precip = np.clip(np.abs(np.sin(np.arange(rows) / 10)) * 2 + rng.normal(0, 0.2, rows), 0, None)

    df = pd.DataFrame(
        {
            "datetime": dates,
            "temp": temp,
            "temp_min": temp - 2 + rng.normal(0, 1, rows),
            "temp_max": temp + 2 + rng.normal(0, 1, rows),
            "humidity": humidity,
            "dewpoint": dewpoint,
            "pressure": pressure,
            "windspeed": windspeed,
            "cloudcover": cloudcover,
            "visibility": visibility,
            "precip": precip,
        }
    )
    return df


def test_create_time_series_split_keeps_chronological_order():
    df = make_sample_weather_df(220)

    train_df, val_df, test_df = create_time_series_split(df, train_ratio=0.7, val_ratio=0.15)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert train_df["datetime"].max() < val_df["datetime"].min()
    assert val_df["datetime"].max() < test_df["datetime"].min()


def test_train_and_compare_includes_baseline_and_experiment_log(tmp_path):
    df = make_sample_weather_df(220)
    df_features = build_feature_matrix(df)

    best_model, results_df, _, _, _, _, experiment_log = train_and_compare(
        df_features,
        experiment_path=tmp_path / "experiment_log.csv",
    )

    assert best_model is not None
    assert "Baseline" in results_df["Model"].values
    assert "CV R²" in experiment_log.columns
    assert (experiment_log["Model"] == "Baseline").any()
    assert experiment_log.shape[0] >= 2
    assert set(FEATURE_COLS).issubset(set(df_features.columns))
    assert set(TARGET_COLS).issubset(set(df_features.columns))
