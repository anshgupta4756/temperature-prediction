"""
Central configuration for the Temperature Prediction System.
All paths, constants, and hyperparameters are managed here.
"""
from pathlib import Path

# ──────────────────────────── Paths ────────────────────────────
BASE_DIR   = Path(__file__).parent       # naming C:\Users\Ansh\Temperature_Prediction 
DATA_DIR   = BASE_DIR / "data"           # Go inside this folder and add this next folder/file name."
MODEL_DIR  = BASE_DIR / "models"         # C:\Users\Ansh\Temperature_Prediction\models
REPORT_DIR = BASE_DIR / "reports"        # C:\Users\Ansh\Temperature_Prediction\reports
                                         # / means that access this folder inside the BASE_DIR folder.
DATA_PATH  = DATA_DIR / "Bangalore Weather Data (Visual Crossing Weather).csv"
EXPERIMENT_LOG_PATH = REPORT_DIR / "experiment_log.csv"

# ──────────────────────────── Features ─────────────────────────
# Raw column mappings from the Visual Crossing CSV
COLUMN_RENAME = {
    "date time":           "datetime",
    "temperature":         "temp",
    "minimum temperature": "temp_min",
    "maximum temperature": "temp_max",
    "relative humidity":   "humidity",
    "wind speed":          "windspeed",
    "wind gust":           "windgust",
    "wind direction":      "winddir",
    "sea level pressure":  "pressure",
    "dew point":           "dewpoint",
    "cloud cover":         "cloudcover",
    "visibility":          "visibility",
    "precipitation":       "precip",
    "precipitation cover": "precip_cover",
    "heat index":          "heatindex",
    "conditions":          "conditions",
}

# Features used for model training (order matters)
FEATURE_COLS = [
    "humidity", "dewpoint", "pressure", "windspeed",
    "cloudcover", "visibility", "precip",
    "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
    "temp_lag_1", "temp_lag_3", "temp_rolling_7", "temp_rolling_30",
    "humidity_dewpoint_interaction", "pressure_change",
]

# Multi-output targets: Average, Low, and High
TARGET_COLS = ["temp", "temp_min", "temp_max"]

# Indian climate seasons: maps month → season index (1-4)
SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon", 12: "Post-Monsoon",
}

# ──────────────────────────── Model Hyperparameters ────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

