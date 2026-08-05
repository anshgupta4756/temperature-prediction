# 🌡️ Temperature Prediction System

> **Multi-Model ML Pipeline for Bangalore Weather Forecasting**  
> An end-to-end machine learning project that predicts daily temperatures using 10+ years of historical weather data, comparing multiple models and serving predictions via an interactive Streamlit dashboard.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Comparison** | Random Forest, Gradient Boosting, XGBoost, LightGBM trained and compared automatically |
| **Advanced Feature Engineering** | Cyclical temporal encoding, lag features, rolling statistics, physics-inspired interactions |
| **Multi-Output Prediction** | Simultaneously predicts daily average, minimum, and maximum temperatures |
| **Interactive Dashboard** | 4-page Streamlit app with dark theme, interactive Plotly visualizations |
| **Production Code Quality** | Modular architecture, type hints, docstrings, configuration management |

## 🏗️ Architecture

```
temp-pred/
├── config.py                  # Central configuration & hyperparameters
├── pipeline.py                # Training pipeline entry point
├── requirements.txt           # Python dependencies
├── data/
│   └── weather.csv            # Bangalore weather dataset (6,600+ records)
├── models/
│   ├── best_model.pkl         # Trained best model (auto-generated)
│   └── model_metadata.pkl     # Training metadata & metrics
├── src/
│   ├── preprocessing.py       # Data loading, cleaning, aggregation
│   ├── feature_engineering.py # Feature transforms & cyclical encoding
│   ├── train.py               # Multi-model training & evaluation
│   └── visualizations.py      # Plotly chart library (20+ chart types)
├── app/
│   └── dashboard.py           # Streamlit interactive dashboard
└── reports/
    └── model_comparison.csv   # Model performance comparison
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python pipeline.py
```

### 3. Launch Dashboard
```bash
streamlit run app/dashboard.py
```

## 📊 Dataset

- **Source**: Visual Crossing Weather API
- **Location**: Bangalore (Bengaluru), Karnataka, India
- **Period**: 2014–2023 (10+ years)
- **Records**: ~6,600 daily observations
- **Features**: Temperature, Humidity, Dew Point, Pressure, Wind, Cloud Cover, Visibility, Precipitation

## 🧠 ML Approach

### Feature Engineering
1. **Cyclical Encoding**: Month and day-of-year encoded as sin/cos pairs to preserve circular relationships (Dec→Jan continuity)
2. **Autoregressive Lags**: 1-day, 3-day temperature lags capture weather persistence
3. **Rolling Statistics**: 7-day and 30-day moving averages for trend signals
4. **Interaction Terms**: Humidity×Dewpoint (moisture saturation) and pressure derivatives (front movements)

### Models Compared
| Model | Type | Key Strength |
|-------|------|-------------|
| Random Forest | Ensemble (Bagging) | Robust, handles non-linearity |
| Gradient Boosting | Ensemble (Boosting) | Strong sequential learning |
| XGBoost | Gradient Boosting | Regularization, speed |
| LightGBM | Gradient Boosting | Histogram-based, very fast |

### Multi-Output Strategy
All models use `MultiOutputRegressor` to simultaneously predict:
- **Daily Average Temperature**
- **Daily Minimum Temperature**  
- **Daily Maximum Temperature**

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — ML pipelines, preprocessing
- **XGBoost / LightGBM** — Gradient boosting models
- **Pandas / NumPy** — Data processing
- **Plotly** — Interactive visualizations
- **Streamlit** — Web dashboard

## 🔍 What was added for hiring-readiness

- Chronological train/validation/test splits for realistic time-series evaluation
- A simple baseline comparator to show whether the advanced models are actually improving performance
- An experiment log saved to reports/experiment_log.csv for reproducibility and discussion
- A few regression tests under tests/ to keep the pipeline stable
- A clearer story around business value, such as energy planning, agriculture, and climate-aware decision support

## 📄 License

MIT License — Feel free to use this project in your portfolio.
