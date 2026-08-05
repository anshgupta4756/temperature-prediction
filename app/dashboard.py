"""
🌡️ Temperature Prediction System — Interactive Dashboard
Built with Streamlit • Multi-model ML • Bangalore Weather Data
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, date
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_PATH, MODEL_DIR, FEATURE_COLS, TARGET_COLS,
    SEASON_MAP,
)
from src.utils import fahrenheit_to_celsius
from src.preprocessing import load_and_clean_data, get_data_summary
from src.feature_engineering import (
    build_feature_matrix, get_feature_importance_names,
)
from src.train import train_and_compare, load_model, evaluate_model
from src.visualizations import (
    plot_temperature_timeseries, plot_seasonal_boxplot,
    plot_monthly_heatmap, plot_correlation_matrix,
    plot_humidity_vs_temp, plot_precipitation_monthly,
    plot_model_comparison, plot_actual_vs_predicted,
    plot_residuals, plot_feature_importance, COLORS,
)

# ──────────────── Page Config ────────────────
st.set_page_config(
    page_title="Bangalore Temp Forecast",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────── Custom CSS ─────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(145deg, #060b18 0%, #0d1425 50%, #060b18 100%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1425 0%, #060b18 100%);
    border-right: 1px solid rgba(99,102,241,0.15);
}
section[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; font-size: 0.95rem; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 1.25rem;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 60% 40%, rgba(99,102,241,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-city {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 0.5rem;
}
.hero h1 {
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    line-height: 1.15;
}
.hero p {
    color: #64748b;
    font-size: 1rem;
    margin: 0;
}

/* ── Weather Card (big result) ── */
.weather-card {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(16px);
    border-radius: 1.25rem;
    padding: 2rem;
    border: 1px solid rgba(99,102,241,0.2);
    text-align: center;
    transition: transform 0.25s, border-color 0.25s, box-shadow 0.25s;
}
.weather-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99,102,241,0.45);
    box-shadow: 0 12px 40px rgba(99,102,241,0.12);
}
.weather-card .label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.weather-card .temp-c {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.weather-card .temp-f {
    font-size: 1rem;
    color: #475569;
    font-weight: 500;
}

.card-low  .label { color: #60a5fa; }
.card-low  .temp-c { color: #60a5fa; }
.card-avg  .label { color: #fbbf24; }
.card-avg  .temp-c { color: #fbbf24; }
.card-high .label { color: #f87171; }
.card-high .temp-c { color: #f87171; }

/* ── Advisory banner ── */
.advisory {
    border-radius: 0.85rem;
    padding: 1rem 1.5rem;
    margin-top: 1.5rem;
    border-left: 4px solid;
    font-size: 0.95rem;
}
.adv-hot  { background: rgba(239,68,68,0.08);  border-color: #ef4444; color: #fca5a5; }
.adv-cold { background: rgba(59,130,246,0.08); border-color: #3b82f6; color: #93c5fd; }
.adv-nice { background: rgba(16,185,129,0.08); border-color: #10b981; color: #6ee7b7; }

/* ── Context chips ── */
.chip {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 2rem;
    padding: 0.3rem 0.9rem;
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
}

/* ── Metric cards (overview) ── */
.metric-card {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 0.85rem;
    padding: 1.1rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(99,102,241,0.4); }
.metric-card h4 { color: #64748b; font-size: 0.78rem; margin: 0 0 0.3rem 0; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card h2 { font-size: 1.8rem; margin: 0; font-weight: 700; color: #e2e8f0; }
.metric-card small { color: #475569; font-size: 0.75rem; }

/* ── Section header ── */
.section-hdr {
    border-left: 3px solid #6366f1;
    padding-left: 0.9rem;
    margin: 1.8rem 0 1rem 0;
}
.section-hdr h3 { color: #e2e8f0; margin: 0; font-weight: 600; font-size: 1.05rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.6rem !important;
    padding: 0.65rem 2.2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

/* ── Sliders & inputs ── */
.stSlider label, .stNumberInput label, .stDateInput label, .stSelectbox label {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* ── Divider ── */
hr { border-color: rgba(99,102,241,0.12) !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────── Data Loading (cached) ──────
@st.cache_data
def get_data():
    df = load_and_clean_data(DATA_PATH)
    df_feat = build_feature_matrix(df)
    summary = get_data_summary(df)
    return df, df_feat, summary


@st.cache_resource
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None, None


# ──────────────── Sidebar Navigation ─────────
with st.sidebar:
    st.markdown("## 🌡️ Bangalore Forecast")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮 Forecast", "📊 Explore Data", "🤖 Model Lab"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style="color:#374151; font-size:0.75rem; text-align:center; line-height:1.8;">
        10 years of Bangalore weather<br>
        4 ML models compared<br>
        Visual Crossing dataset
    </div>
    """, unsafe_allow_html=True)

# Load data
df_clean, df_feat, data_summary = get_data()
model, meta = get_model()

experiment_log_path = PROJECT_ROOT / "reports" / "experiment_log.csv"
if experiment_log_path.exists():
    experiment_log = pd.read_csv(experiment_log_path)
    st.subheader("📈 Experiment Log")
    st.dataframe(experiment_log, use_container_width=True)


# ════════════════════════════════════════════════
#  PAGE 1: FORECAST (default)
# ════════════════════════════════════════════════
if page == "🔮 Forecast":

    st.markdown("""
    <div class="hero">
        <div class="hero-city">📍 Bangalore, Karnataka</div>
        <h1>Temperature Forecast</h1>
        <p>Pick a date — we'll predict the expected high, low, and average temperature</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model status check ──
    if model is None:
        st.error("⚠️ No trained model found. Go to **🤖 Model Lab** and click **Train All Models** first.")
        st.stop()

    best_name = meta.get("best_model", "Unknown")

    # ── Input: date + optional toggles ──
    col_left, col_mid, col_right = st.columns([1.4, 0.2, 1.4])

    with col_left:
        st.markdown('<div class="section-hdr"><h3>📅 Select a Date</h3></div>', unsafe_allow_html=True)
        pred_date = st.date_input(
            "Prediction date",
            value=datetime.now().date(),
            label_visibility="collapsed",
        )

        # Season info
        month = pred_date.month
        season = SEASON_MAP.get(month, "Unknown")
        day_of_year = pred_date.timetuple().tm_yday

        # Pull historical averages for this month (smart defaults)
        df_feat["month"] = df_feat["datetime"].dt.month
        hist_month = df_feat[df_feat["month"] == month]
        
        defaults = {
            "humidity":   float(hist_month["humidity"].mean()) if len(hist_month) else 65.0,
            "dewpoint":   float(hist_month["dewpoint"].mean()) if len(hist_month) else 62.0,
            "pressure":   float(hist_month["pressure"].mean()) if len(hist_month) else 1014.0,
            "windspeed":  float(hist_month["windspeed"].mean()) if len(hist_month) else 12.0,
            "cloudcover": float(hist_month["cloudcover"].mean()) if len(hist_month) else 50.0,
            "visibility": float(hist_month["visibility"].mean()) if len(hist_month) else 5.5,
            "precip":     float(hist_month["precip"].mean()) if len(hist_month) else 0.0,
        }

        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); 
                    border-radius:0.75rem; padding:1rem 1.2rem; margin-top:1rem;">
            <div style="color:#818cf8; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">
                📍 Context for {pred_date.strftime("%B %d, %Y")}
            </div>
            <div style="color:#e2e8f0; font-size:0.9rem;">
                🌏 Season: <b style="color:#a5b4fc;">{season}</b><br>
                🌡️ Historical avg for {pred_date.strftime("%B")}: 
                <b style="color:#fbbf24;">{fahrenheit_to_celsius(float(hist_month["temp"].mean())):.1f}°C</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        predict_btn = st.button("🌡️ Generate Forecast", type="primary")

    with col_right:
        st.markdown('<div class="section-hdr"><h3>⚙️ Adjust Conditions <span style="color:#475569;font-size:0.78rem;font-weight:400;">(optional)</span></h3></div>', unsafe_allow_html=True)
        st.caption("Pre-filled with historical averages for the selected month. Adjust if you know the forecast conditions.")

        humidity   = st.number_input("💧 Humidity (%)", 10.0, 100.0, round(defaults["humidity"], 1), step=0.5)
        cloudcover = st.number_input("☁️ Cloud Cover (%)", 0.0, 100.0, round(defaults["cloudcover"], 1), step=1.0)
        precip     = st.number_input("🌧️ Precipitation (mm)", 0.0, 50.0, round(min(defaults["precip"], 50.0), 1), step=0.5)
        windspeed  = st.number_input("💨 Wind Speed (km/h)", 0.0, 50.0, round(defaults["windspeed"], 1), step=0.5)

    # ── Run prediction ──
    if predict_btn:
        # Cyclical time features
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        doy_sin   = np.sin(2 * np.pi * day_of_year / 365)
        doy_cos   = np.cos(2 * np.pi * day_of_year / 365)

        # Lag features from recent historical data
        recent = df_feat.tail(30)
        temp_lag_1 = float(recent["temp"].iloc[-1])
        temp_lag_3 = float(recent["temp"].iloc[-3]) if len(recent) >= 3 else temp_lag_1
        temp_r7    = float(recent["temp"].tail(7).mean())
        temp_r30   = float(recent["temp"].mean())
        hdi        = humidity * defaults["dewpoint"] / 100
        pc         = 0.0

        features = np.array([[
            humidity, defaults["dewpoint"], defaults["pressure"], windspeed,
            cloudcover, defaults["visibility"], precip,
            month_sin, month_cos, doy_sin, doy_cos,
            temp_lag_1, temp_lag_3, temp_r7, temp_r30,
            hdi, pc,
        ]])

        preds = model.predict(features)[0]
        avg_f, min_f, max_f = preds
        avg_c = fahrenheit_to_celsius(avg_f)
        min_c = fahrenheit_to_celsius(min_f)
        max_c = fahrenheit_to_celsius(max_f)

        st.markdown("---")
        st.markdown(f"#### 🌤️ Forecast for **{pred_date.strftime('%B %d, %Y')}** — {season} Season")

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(f"""
            <div class="weather-card card-low">
                <div class="label">🌙 Expected Low</div>
                <div class="temp-c">{min_c:.1f}°C</div>
                <div class="temp-f">{min_f:.1f}°F</div>
            </div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
            <div class="weather-card card-avg">
                <div class="label">☀️ Daily Average</div>
                <div class="temp-c">{avg_c:.1f}°C</div>
                <div class="temp-f">{avg_f:.1f}°F</div>
            </div>""", unsafe_allow_html=True)
        with rc3:
            st.markdown(f"""
            <div class="weather-card card-high">
                <div class="label">🔥 Expected High</div>
                <div class="temp-c">{max_c:.1f}°C</div>
                <div class="temp-f">{max_f:.1f}°F</div>
            </div>""", unsafe_allow_html=True)

        # Advisory
        if max_c > 35:
            advisory_cls, advisory_txt = "adv-hot", f"🔥 <b>Heat Alert</b> — High of {max_c:.1f}°C expected. Stay hydrated and avoid peak sun hours."
        elif min_c < 15:
            advisory_cls, advisory_txt = "adv-cold", f"🧥 <b>Cool Night</b> — Low of {min_c:.1f}°C expected. Carry a jacket in the evening."
        else:
            advisory_cls, advisory_txt = "adv-nice", f"✨ <b>Comfortable Day</b> — Pleasant {season.lower()} conditions for Bangalore. Great day to be outdoors."

        st.markdown(f'<div class="advisory {advisory_cls}">{advisory_txt}</div>', unsafe_allow_html=True)

        # Subtle context footer
        if meta.get("metrics"):
            m = meta["metrics"][0]
            st.markdown(f"""
            <div style="margin-top:1.2rem; display:flex; gap:0.5rem; flex-wrap:wrap;">
                <span class="chip">Model: {best_name}</span>
                <span class="chip">R² {m.get('R²', 0):.4f}</span>
                <span class="chip">MAE ±{m.get('MAE', 0):.1f}°F</span>
                <span class="chip">Temp range: {max_c - min_c:.1f}°C spread</span>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  PAGE 2: DATA EXPLORER
# ════════════════════════════════════════════════
elif page == "📊 Explore Data":

    st.markdown('<div class="section-hdr"><h3>📊 Bangalore Weather — 10 Years of Data</h3></div>', unsafe_allow_html=True)

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📅 Records", f"{data_summary['rows']:,}", "Daily observations"),
        ("📆 Span", "10 Years", data_summary['date_range']),
        ("🔢 Features", f"{len(FEATURE_COLS)}", "Engineered for ML"),
        ("📍 Location", "Bangalore", "12.97°N, 77.59°E"),
    ]
    for col, (title, val, sub) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""<div class="metric-card">
                <h4>{title}</h4><h2>{val}</h2><small>{sub}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌡️ Temperature", "🌍 Seasons", "🗓️ Heatmap", "🔗 Correlations", "🌧️ Rainfall"
    ])

    with tab1:
        st.plotly_chart(plot_temperature_timeseries(df_clean), use_container_width=True)
        st.caption("Daily temperature range band with 30-day moving average trend.")
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_seasonal_boxplot(df_clean), use_container_width=True)
        with col2:
            st.plotly_chart(plot_humidity_vs_temp(df_clean), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_monthly_heatmap(df_clean), use_container_width=True)
        st.caption("Year × Month heatmap revealing annual climate trends.")
    with tab4:
        st.plotly_chart(plot_correlation_matrix(df_clean), use_container_width=True)
        st.info("💡 **Key Insight**: Dew point is the strongest predictor of temperature. Humidity has a counterintuitive negative relationship.")
    with tab5:
        st.plotly_chart(plot_precipitation_monthly(df_clean), use_container_width=True)
        st.caption("Bangalore's monsoon (Jun–Sep) dominates annual rainfall.")

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df_clean.head(50), use_container_width=True)
        st.caption(f"Showing first 50 of {len(df_clean):,} daily records.")


# ════════════════════════════════════════════════
#  PAGE 3: MODEL LAB
# ════════════════════════════════════════════════
elif page == "🤖 Model Lab":

    st.markdown('<div class="section-hdr"><h3>🤖 Model Training & Comparison</h3></div>', unsafe_allow_html=True)
    st.caption("Trains Random Forest, Gradient Boosting, XGBoost, and LightGBM — picks the best automatically.")

    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training 4 models... this takes ~2 minutes."):
            best_model, results_df, X_test, y_test, y_pred, trained = train_and_compare(df_feat)

        st.success(f"✅ Best model: **{results_df.iloc[0]['Model']}** — R² = {results_df.iloc[0]['R²']:.4f}")

        st.plotly_chart(plot_model_comparison(results_df), use_container_width=True)

        st.markdown("#### 📊 All Model Metrics")
        st.dataframe(
            results_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.4f}"}),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_actual_vs_predicted(y_test[:, 0], y_pred[:, 0], "Avg Temperature"), use_container_width=True)
        with col2:
            st.plotly_chart(plot_residuals(y_test[:, 0], y_pred[:, 0]), use_container_width=True)

        st.markdown("#### 🔬 Feature Importance")
        st.plotly_chart(plot_feature_importance(best_model, get_feature_importance_names()), use_container_width=True)

        st.cache_resource.clear()

    else:
        if model is not None and meta:
            st.info(f"📦 Active model: **{meta.get('best_model', 'Unknown')}** — click above to retrain.")
            results_df = pd.DataFrame(meta["metrics"])
            st.plotly_chart(plot_model_comparison(results_df), use_container_width=True)
            st.dataframe(
                results_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.4f}"}),
                use_container_width=True,
            )
            st.markdown("#### 🔬 Feature Importance")
            st.plotly_chart(plot_feature_importance(model, get_feature_importance_names()), use_container_width=True)
        else:
            st.warning("⚠️ No trained model found. Click **Train All Models** above to get started.")
