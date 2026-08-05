"""
Data Preprocessing Module
─────────────────────────
Handles loading, cleaning, deduplication, and type-casting of the raw
Visual Crossing Weather CSV.

Design Decision:
  • Duplicate dates exist (two readings per day in many cases).
    We aggregate them by date to get daily statistics instead of
    randomly dropping one — this preserves information.
  • Missing values are forward-filled then back-filled (common for
    time-series weather data where brief sensor gaps occur).
  • Temperatures remain in Fahrenheit during training for consistency
    with the source data; conversion happens only at the UI layer.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COLUMN_RENAME


def load_raw_data(path: str | Path) -> pd.DataFrame:      # helper func 1
    """Load the raw CSV and apply column standardisation."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns=COLUMN_RENAME)
    return df

# I expect a pandas DataFrame and I return a pandas DataFrame
# meaning the input and output types are both pandas DataFrames
def clean_data(df: pd.DataFrame) -> pd.DataFrame:     # helper func 2

    """
    Clean and aggregate the raw weather data.
    
    The dataset contains multiple observations per day (typically two:
    morning and afternoon readings). We aggregate to daily level to
    create a clean time-series:
      - temp, dewpoint, humidity, etc. → daily mean
      - temp_min → daily minimum
      - temp_max → daily maximum
    """
    df = df.copy() # create a copy to avoid modifying the original DataFrame

    # Parse dates
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", dayfirst=True)
    # Converts the datetime column from string/text format into pandas datetime
    # format so Python can understand and manipulate dates.
    df["date"] = df["datetime"].dt.date
    # Extracts only the date part (year-month-day) from the datetime
    # column and stores it in a new date column for grouping daily weather records.

    # Define "NUMERIC" columns we care about
    numeric_cols = [
        "temp", "temp_min", "temp_max", "humidity", "dewpoint",
        "windspeed", "pressure", "cloudcover", "visibility", "precip",
    ]
    # Only use columns that exist
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Coerce to numeric (handles stray strings)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Go through every numerical weather column and make sure all values are actually 
    # numbers. it DOES not throw error, converts them to NaN

    # data is multiple for one day, average most weather features,
    #  but preserve daily minimum and maximum temperatures
    # also this is a loop (as you can see "for" in there)
    agg_funcs = {col: "mean" for col in numeric_cols}
    if "temp_min" in agg_funcs:
        agg_funcs["temp_min"] = "min"
    if "temp_max" in agg_funcs:
        agg_funcs["temp_max"] = "max"

    # Group by date and aggregate
    # This groups all rows with the same date and applies different
    # aggregation functions to each column
    daily = df.groupby("date").agg(agg_funcs).reset_index()
    daily["datetime"] = pd.to_datetime(daily["date"])
    daily = daily.drop(columns=["date"])
    # Removes the old date column because datetime

    # Sort chronologically cause of groupby and reset_index, the order of rows may not be chronological anymore
    daily = daily.sort_values("datetime").reset_index(drop=True)

    # Fill missing values (forward-fill + back-fill for edge cases)
    daily[numeric_cols] = daily[numeric_cols].ffill().bfill()
    # forward-fill: if a value is missing, fill it with the last known value
    # back-fill: if a value is still missing after forward-fill, fill it with the

    return daily


def get_data_summary(df: pd.DataFrame) -> dict:
    """Generate a summary dict for the dataset overview."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "date_range": f"{df['datetime'].min().date()} → {df['datetime'].max().date()}",
        "missing_pct": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        "features": list(df.columns),
    }


def load_and_clean_data(path: str | Path) -> pd.DataFrame:
    """One-shot convenience function: load → clean → return."""
    raw = load_raw_data(path)
    return clean_data(raw)
