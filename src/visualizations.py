"""
Visualization Module
────────────────────
Generates publication-quality charts for the Streamlit dashboard
and analysis notebooks. All functions return Plotly figures for
interactive rendering.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SEASON_MAP
from src.utils import fahrenheit_to_celsius


# ────────────────── Color Palette ──────────────────
COLORS = {
    "primary":    "#6366f1",  # Indigo
    "secondary":  "#8b5cf6",  # Violet
    "accent":     "#06b6d4",  # Cyan
    "warm":       "#f59e0b",  # Amber
    "hot":        "#ef4444",  # Red
    "cold":       "#3b82f6",  # Blue
    "success":    "#10b981",  # Emerald
    "bg_dark":    "#0f172a",
    "bg_card":    "#1e293b",
    "text":       "#e2e8f0",
    "muted":      "#94a3b8",
}

TEMPLATE = "plotly_dark"


def style_figure(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent dark theme styling to all figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS["text"])),
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor=COLORS["muted"],
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)", zeroline=False)
    return fig


# ──────────────── EDA Visualizations ───────────────

def plot_temperature_timeseries(df: pd.DataFrame) -> go.Figure:
    """Interactive time-series of daily temperatures with range band."""
    fig = go.Figure()

    # Temperature range band (min to max)
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["temp_max"],
        mode="lines", line=dict(width=0),
        name="Daily Max", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["temp_min"],
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(99,102,241,0.15)",
        name="Daily Range",
    ))

    # Average temperature line
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["temp"],
        mode="lines",
        line=dict(color=COLORS["primary"], width=1.5),
        name="Daily Average",
    ))

    # 30-day rolling average
    rolling = df["temp"].rolling(30).mean()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=rolling,
        mode="lines",
        line=dict(color=COLORS["warm"], width=2.5, dash="dot"),
        name="30-Day Trend",
    ))

    return style_figure(fig, "📈 Temperature Time Series (°F)")


def plot_seasonal_boxplot(df: pd.DataFrame) -> go.Figure:
    """Box plots of temperature distribution by Indian climate season."""
    plot_df = df.copy()
    plot_df["season"] = plot_df["datetime"].dt.month.map(SEASON_MAP)

    season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
    season_colors = {
        "Winter": COLORS["cold"],
        "Summer": COLORS["hot"],
        "Monsoon": COLORS["accent"],
        "Post-Monsoon": COLORS["warm"],
    }

    fig = go.Figure()
    for season in season_order:
        sdf = plot_df[plot_df["season"] == season]
        fig.add_trace(go.Box(
            y=sdf["temp"],
            name=season,
            marker_color=season_colors[season],
            boxmean=True,
        ))

    return style_figure(fig, "🌍 Temperature Distribution by Season (°F)")


def plot_monthly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of average temperature by month and year."""
    plot_df = df.copy()
    plot_df["year"]  = plot_df["datetime"].dt.year
    plot_df["month"] = plot_df["datetime"].dt.month

    pivot = plot_df.pivot_table(
        values="temp", index="year", columns="month", aggfunc="mean"
    )

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=month_names,
        y=pivot.index.astype(str),
        colorscale="RdYlBu_r",
        colorbar=dict(title="Temp (°F)"),
    ))

    return style_figure(fig, "🗓️ Monthly Temperature Heatmap (°F)")


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of key weather variables."""
    num_cols = ["temp", "temp_min", "temp_max", "humidity", "dewpoint",
                "pressure", "windspeed", "cloudcover", "visibility", "precip"]
    num_cols = [c for c in num_cols if c in df.columns]

    corr = df[num_cols].corr()

    # Rename for display
    labels = [c.replace("_", " ").title() for c in num_cols]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="Correlation"),
    ))

    return style_figure(fig, "🔗 Feature Correlation Matrix")


def plot_humidity_vs_temp(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of humidity vs temperature colored by season."""
    plot_df = df.copy()
    plot_df["season"] = plot_df["datetime"].dt.month.map(SEASON_MAP)

    fig = px.scatter(
        plot_df, x="humidity", y="temp",
        color="season",
        color_discrete_map={
            "Winter": COLORS["cold"],
            "Summer": COLORS["hot"],
            "Monsoon": COLORS["accent"],
            "Post-Monsoon": COLORS["warm"],
        },
        opacity=0.6,
        template=TEMPLATE,
    )

    return style_figure(fig, "💧 Humidity vs Temperature by Season")


def plot_precipitation_monthly(df: pd.DataFrame) -> go.Figure:
    """Monthly precipitation pattern — critical for Bangalore's monsoon."""
    plot_df = df.copy()
    plot_df["month"] = plot_df["datetime"].dt.month

    monthly_precip = plot_df.groupby("month")["precip"].agg(["mean", "std"]).reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=month_names,
        y=monthly_precip["mean"],
        marker_color=[
            COLORS["cold"] if m in [1,2] else
            COLORS["hot"] if m in [3,4,5] else
            COLORS["accent"] if m in [6,7,8,9] else
            COLORS["warm"]
            for m in range(1, 13)
        ],
        name="Avg Precipitation",
    ))

    return style_figure(fig, "🌧️ Monthly Precipitation Pattern (mm)")


# ──────────────── Model Visualizations ─────────────

def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing model performance metrics."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["MAE (Lower is Better)", "RMSE (Lower is Better)", "R² Score (Higher is Better)"],
    )

    colors = [COLORS["primary"], COLORS["accent"], COLORS["warm"], COLORS["success"]]

    for i, metric in enumerate(["MAE", "RMSE", "R²"]):
        for j, row in results_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["Model"]],
                y=[row[metric]],
                name=row["Model"] if i == 0 else None,
                marker_color=colors[j % len(colors)],
                showlegend=(i == 0),
            ), row=1, col=i+1)

    fig.update_layout(barmode="group")
    return style_figure(fig, "🏆 Model Performance Comparison")


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, target_name: str = "Temperature") -> go.Figure:
    """Scatter plot of actual vs predicted values with perfect-prediction line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(
            color=COLORS["primary"],
            size=5,
            opacity=0.5,
        ),
        name="Predictions",
    ))

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color=COLORS["hot"], width=2, dash="dash"),
        name="Perfect Prediction",
    ))

    fig.update_xaxes(title_text="Actual (°F)")
    fig.update_yaxes(title_text="Predicted (°F)")

    return style_figure(fig, f"🎯 Actual vs Predicted — {target_name}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Residual distribution plot."""
    residuals = y_true - y_pred

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Residual Scatter", "Residual Distribution"])

    # Scatter
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(color=COLORS["accent"], size=4, opacity=0.4),
        name="Residuals",
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=COLORS["hot"], dash="dash"), row=1, col=1)

    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=40,
        marker_color=COLORS["primary"],
        opacity=0.7,
        name="Distribution",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Predicted (°F)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (°F)", row=1, col=1)
    fig.update_xaxes(title_text="Residual (°F)", row=1, col=2)

    return style_figure(fig, "📊 Residual Analysis")


def plot_feature_importance(model, feature_names: list[str]) -> go.Figure:
    """Extract and plot feature importances from the best model."""
    try:
        # Get the underlying estimator from Pipeline → MultiOutputRegressor
        multi_model = model.named_steps["model"]
        importances_list = [est.feature_importances_ for est in multi_model.estimators_]
        importances = np.mean(importances_list, axis=0)
    except (AttributeError, KeyError):
        # Fallback: uniform importance
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Sort by importance
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]

    fig = go.Figure(go.Bar(
        x=sorted_vals,
        y=sorted_names,
        orientation="h",
        marker=dict(
            color=sorted_vals,
            colorscale="Viridis",
        ),
    ))

    fig.update_xaxes(title_text="Importance Score")
    return style_figure(fig, "🔬 Feature Importance (Averaged Across Targets)")
