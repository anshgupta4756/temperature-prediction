"""
Model Training Module
─────────────────────
Trains, compares, and persists multiple ML models for temperature
prediction.

Architecture:
  • Multi-Output Regression — Each model simultaneously predicts
    temp (mean), temp_min, and temp_max using scikit-learn's
    MultiOutputRegressor wrapper.
  • Model Zoo — We train Random Forest, XGBoost, and LightGBM,
    then select the best performer on the test set.
  • Pipeline — Each model is wrapped in a Pipeline with
    StandardScaler for numerical stability.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURE_COLS, TARGET_COLS, MODEL_DIR, REPORT_DIR, EXPERIMENT_LOG_PATH,
    RANDOM_STATE,
    RF_PARAMS, XGB_PARAMS, LGBM_PARAMS,
)


def create_time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a chronological dataset into train/validation/test slices."""
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be between 0 and 1")

    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    val_end = train_end + int(total_rows * val_ratio)

    if train_end <= 0 or val_end <= train_end or val_end >= total_rows:
        raise ValueError("The requested split ratios produce an invalid time-series split")

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def _build_pipelines() -> dict:
    """Create model pipelines for comparison."""
    pipelines = {}

    # Baseline: simple persistence-like mean predictor
    pipelines["Baseline"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MultiOutputRegressor(DummyRegressor(strategy="mean"))),
    ])

    # 1. Random Forest
    pipelines["Random Forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MultiOutputRegressor(
            RandomForestRegressor(**RF_PARAMS)
        )),
    ])

    # 2. Gradient Boosting (scikit-learn, always available)
    pipelines["Gradient Boosting"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, random_state=RANDOM_STATE,
            )
        )),
    ])

    # 3. XGBoost (optional)
    try:
        from xgboost import XGBRegressor
        pipelines["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MultiOutputRegressor(
                XGBRegressor(**XGB_PARAMS)
            )),
        ])
    except ImportError:
        pass

    # 4. LightGBM (optional)
    try:
        from lightgbm import LGBMRegressor
        pipelines["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MultiOutputRegressor(
                LGBMRegressor(**LGBM_PARAMS)
            )),
        ])
    except ImportError:
        pass

    return pipelines


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² for multi-output predictions."""
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²":   float(r2_score(y_true, y_pred)),
    }


def train_and_compare(
    df: pd.DataFrame,
    experiment_path: str | Path | None = None,
) -> tuple:
    """
    Train all models, compare them, and return results.

    Returns:
        (best_model, results_df, X_test, y_test, y_pred_best, trained_models, experiment_log)
    """
    train_df, val_df, test_df = create_time_series_split(df)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COLS].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[TARGET_COLS].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COLS].values

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.vstack([y_train, y_val])

    pipelines = _build_pipelines()
    results = []
    trained_models = {}

    for name, pipeline in pipelines.items():
        print(f"  Training {name}...")
        if name != "Baseline":
            tscv = TimeSeriesSplit(n_splits=min(3, max(2, len(train_df) // 30)))
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=tscv,
                scoring="r2",
                n_jobs=None,
            )
            cv_r2 = float(np.mean(cv_scores))
        else:
            cv_r2 = np.nan

        pipeline.fit(X_train_full, y_train_full)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        metrics["CV R²"] = cv_r2
        results.append({"Model": name, **metrics})
        trained_models[name] = (pipeline, y_pred)

    results_df = pd.DataFrame(results).sort_values("R²", ascending=False)

    # Select best model
    best_name = results_df.iloc[0]["Model"]
    best_model, y_pred_best = trained_models[best_name]

    # Save best model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)

    # Also save metadata
    meta = {
        "best_model": best_name,
        "features": FEATURE_COLS,
        "targets": TARGET_COLS,
        "metrics": results_df.to_dict("records"),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    joblib.dump(meta, MODEL_DIR / "model_metadata.pkl")

    experiment_path = Path(experiment_path) if experiment_path else EXPERIMENT_LOG_PATH
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_log = results_df.copy()
    experiment_log.to_csv(experiment_path, index=False)

    print(f"\n  ✅ Best model: {best_name} (R² = {results_df.iloc[0]['R²']:.4f})")
    print(f"  💾 Saved to {model_path}")

    return best_model, results_df, X_test, y_test, y_pred_best, trained_models, experiment_log


def load_model():
    """Load the persisted best model and metadata."""
    model_path = MODEL_DIR / "best_model.pkl"
    meta_path  = MODEL_DIR / "model_metadata.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run the training pipeline first."
        )

    model = joblib.load(model_path)
    meta  = joblib.load(meta_path) if meta_path.exists() else {}
    return model, meta
