"""
Training Pipeline
─────────────────
End-to-end script that loads data, engineers features, trains all
models, compares performance, and saves the best model.

Usage:
    python pipeline.py
"""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_PATH, REPORT_DIR
from src.preprocessing import load_and_clean_data, get_data_summary
from src.feature_engineering import build_feature_matrix
from src.train import train_and_compare


def main():
    print("=" * 60)
    print("  🌡️  Temperature Prediction System — Training Pipeline")
    print("=" * 60)

    # ─── Step 1: Load & Clean ───
    print("\n📂 Step 1: Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    summary = get_data_summary(df)
    print(f"   Rows: {summary['rows']}")
    print(f"   Date Range: {summary['date_range']}")
    print(f"   Missing Data: {summary['missing_pct']:.2f}%")

    # ─── Step 2: Feature Engineering ───
    print("\n🔧 Step 2: Engineering features...")
    df_features = build_feature_matrix(df)
    print(f"   Feature matrix shape: {df_features.shape}")

    # ─── Step 3: Train & Compare ───
    print("\n🤖 Step 3: Training and comparing models...\n")
    best_model, results_df, X_test, y_test, y_pred, _, experiment_log = train_and_compare(df_features)

    # ─── Step 4: Report ───
    print("\n📊 Model Comparison Results:")
    print("-" * 55)
    for _, row in results_df.iterrows():
        print(f"  {row['Model']:22s}  MAE={row['MAE']:.2f}  RMSE={row['RMSE']:.2f}  R²={row['R²']:.4f}")
    print("-" * 55)

    # Save results CSV
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    experiment_log.to_csv(REPORT_DIR / "experiment_log.csv", index=False)

    print("\n✅ Pipeline complete! Run the dashboard:")
    print("   streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
