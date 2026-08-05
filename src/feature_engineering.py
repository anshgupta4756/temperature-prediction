"""
Feature Engineering Module
──────────────────────────
Transforms cleaned weather data into ML-ready feature matrices.

Key engineering choices:
  1. Cyclical Encoding — Months and day-of-year are encoded as
     sin/cos pairs so the model understands that December (12) is
     close to January (1), not far away.
  2. Lag Features — Previous day's temp and 3-day lag capture
     short-term weather persistence (autocorrelation).
  3. Rolling Averages — 7-day and 30-day windows capture weekly
     and monthly climate trends.
  4. Interaction Features — Humidity × Dewpoint and pressure
     change rate capture nonlinear atmospheric dynamics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))      # look 2 folders in
from config import FEATURE_COLS, SEASON_MAP       # from config.py


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical (sin/cos) encoding for temporal features.
    
    Why cyclical? A linear month number means the model sees month 12
    as very different from month 1, when in reality they're adjacent.
    Sin/cos encoding preserves this circular relationship.
    """
    df = df.copy()

    # Month cyclical encoding (period = 12)
    df["month"] = df["datetime"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Day-of-year cyclical encoding (period = 365)
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Season (for analysis, not model input)
    df["season"] = df["month"].map(SEASON_MAP)         # config.py 

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add autoregressive lag features.
    
    Weather is highly autocorrelated — today's temperature is strongly
    correlated with yesterday's. Lag features capture this.
    """
    df = df.copy()

    # Lag features
    df["temp_lag_1"] = df["temp"].shift(1)
    df["temp_lag_3"] = df["temp"].shift(3)

    # Rolling statistics
    df["temp_rolling_7"]  = df["temp"].rolling(window=7, min_periods=1).mean()
    df["temp_rolling_30"] = df["temp"].rolling(window=30, min_periods=1).mean()

    # Create a feature containing the previous day's temperature.
# shift(1) moves the temperature values down by one row, so each day gets yesterday's temperature.
# Example: Jan 5 temperature row will contain Jan 4 temperature as temp_lag_1
# Create a feature containing the temperature from 3 days ago.
# shift(3) moves the values down by three rows, allowing the model to see temperature from three days earlier.

# Rolling statistics calculate the average temperature over a moving window of previous days.
# They capture short-term and long-term temperature trends instead of a single previous value.


# Calculate the 7-day moving average temperature.
# window=7 means look at the last 7 temperature values.
# min_periods=1 allows calculation even when fewer than 7 days of data are available at the beginning.

# Calculate the 30-day moving average temperature.
# This captures longer-term temperature patterns and seasonal trends.
# window=30 means the average is calculated using the previous 30 temperature values.


    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-inspired interaction features.
    
    These features capture relationships between weather variables
    that may provide additional information to the ML model.
    """
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame


    # Create a combined moisture feature using humidity and dew point.
    # This captures the relationship between air moisture level and condensation tendency.
    # Dividing by 100 scales the value because humidity is represented as a percentage.
    df["humidity_dewpoint_interaction"] = df["humidity"] * df["dewpoint"] / 100


    # Calculate daily pressure change compared to the previous observation.
    # diff() computes: current pressure - previous day's pressure.
    # A falling or rising pressure trend can indicate changing weather conditions.
    df["pressure_change"] = df["pressure"].diff().fillna(0)


    return df  # Return the DataFrame with the newly added features


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    
    Applies all transformations and returns a DataFrame with only
    the columns needed for training.
    """
    df = add_cyclical_time_features(df)
    df = add_lag_features(df)
    df = add_interaction_features(df)

    # Drop rows with NaN from lag operations
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True) # because of lag

    return df


def get_feature_importance_names() -> list[str]:
    """Return human-readable feature names for display."""
    return [
        "Humidity (%)", "Dew Point (°F)", "Pressure (hPa)", "Wind Speed (km/h)",
        "Cloud Cover (%)", "Visibility (km)", "Precipitation (mm)",
        "Month (sin)", "Month (cos)", "Day of Year (sin)", "Day of Year (cos)",
        "Temp Lag 1-day", "Temp Lag 3-day", "Temp Rolling 7-day", "Temp Rolling 30-day",
        "Humidity×Dewpoint", "Pressure Change",
    ]
