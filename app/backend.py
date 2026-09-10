"""
backend.py — Logique métier : chargement des données, KPIs, modélisation Prophet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

DATASET_PATH = Path(__file__).parent.parent / "train.csv"


# ---------------------------------------------------------------------------
# Chargement & validation
# ---------------------------------------------------------------------------

def prepare_forecast_data(source: str | Path | Any) -> pd.DataFrame:
    """Charge et valide un CSV de ventes pour Prophet.

    Colonnes acceptées :
    - ``date`` + ``sales``
    - ``date`` + ``store`` + ``item`` + ``sales``  (agrégation automatique par jour)

    Returns:
        DataFrame avec colonnes ``ds`` (datetime) et ``y`` (float).
    """
    if source is None or source == "":
        source = DATASET_PATH

    if hasattr(source, "name") and not isinstance(source, (str, Path)):
        csv_path = Path(source.name)
    else:
        csv_path = Path(source)

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Le fichier CSV importé est vide.")

    required_columns = {"date", "sales"}
    if not required_columns.issubset(df.columns):
        if {"date", "store", "item", "sales"}.issubset(df.columns):
            df = df[["date", "sales"]].copy()
        else:
            raise ValueError(
                "Format CSV invalide. Colonnes attendues : date + sales, "
                "ou date + store + item + sales."
            )

    df = df[["date", "sales"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    if df["date"].isna().any() or df["sales"].isna().any():
        raise ValueError(
            "Des valeurs invalides ont été détectées dans les colonnes date ou sales."
        )

    daily_sales = df.groupby("date", as_index=False)["sales"].sum()
    daily_sales.columns = ["ds", "y"]
    daily_sales = daily_sales.sort_values("ds").reset_index(drop=True)
    return daily_sales


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict:
    """Calcule les indicateurs clés à partir de l'historique de ventes."""
    total_sales = df["y"].sum()
    avg_daily = df["y"].mean()
    max_day = df.loc[df["y"].idxmax()]
    nb_days = len(df)
    date_min = df["ds"].min().strftime("%d/%m/%Y")
    date_max = df["ds"].max().strftime("%d/%m/%Y")

    recent = df.tail(30)["y"].mean()
    previous = df.iloc[-60:-30]["y"].mean() if len(df) >= 60 else avg_daily
    trend_pct = ((recent - previous) / previous * 100) if previous else 0.0

    return {
        "total_sales": total_sales,
        "avg_daily": avg_daily,
        "nb_days": nb_days,
        "date_min": date_min,
        "date_max": date_max,
        "max_day_date": max_day["ds"].strftime("%d/%m/%Y"),
        "max_day_sales": max_day["y"],
        "trend_pct": trend_pct,
    }


# ---------------------------------------------------------------------------
# Modélisation Prophet
# ---------------------------------------------------------------------------

def run_prophet(df: pd.DataFrame, forecast_days: int) -> tuple[Prophet, pd.DataFrame]:
    """Entraîne un modèle Prophet et retourne (model, forecast)."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=int(forecast_days))
    forecast = model.predict(future)
    return model, forecast


def evaluate_model(
    forecast: pd.DataFrame,
    df: pd.DataFrame,
    eval_days: int = 90,
) -> dict:
    """Évalue la précision du modèle sur les derniers *eval_days* jours.

    Returns:
        dict avec clés ``mae`` et ``mape`` (ou ``None`` si données insuffisantes).
    """
    test_df = df.tail(eval_days).copy()
    merged = test_df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    if merged.empty:
        return {"mae": None, "mape": None}

    mae = (merged["y"] - merged["yhat"]).abs().mean()
    mape = (
        (merged["y"] - merged["yhat"]).abs()
        / merged["y"].replace(0, np.nan)
    ).mean() * 100
    return {"mae": mae, "mape": mape}


def build_future_table(forecast: pd.DataFrame, df: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
    """Construit le tableau des prévisions pour les jours *futurs uniquement*."""
    future_only = forecast[forecast["ds"] > df["ds"].max()].copy()
    future_only = future_only[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(forecast_days).copy()
    future_only["ds"] = future_only["ds"].dt.strftime("%d/%m/%Y")
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        future_only[col] = future_only[col].round(0).astype(int)
    future_only.columns = ["Date", "Prévision (unités)", "Borne basse", "Borne haute"]
    return future_only
