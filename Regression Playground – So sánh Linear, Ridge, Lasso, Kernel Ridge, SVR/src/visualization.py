"""Ve bieu do cho cac thi nghiem regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from src.models import get_estimator, predict_model


def _save_figure(figure: plt.Figure, save_path: str | Path | None) -> None:
    """Luu figure neu co duong dan dich."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")


def plot_predictions_1d(
    model: RegressorMixin,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Ve diem du lieu va duong du doan cho dataset mot chieu."""
    if x_train.shape[1] != 1:
        raise ValueError("plot_predictions_1d chi ho tro dataset co mot feature.")

    x_all = pd.concat([x_train, x_test])
    x_grid = np.linspace(x_all.iloc[:, 0].min(), x_all.iloc[:, 0].max(), 400)
    x_grid_frame = pd.DataFrame({x_train.columns[0]: x_grid})
    y_grid = predict_model(model, x_grid_frame)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(x_train.iloc[:, 0], y_train, alpha=0.55, label="Train data")
    axis.scatter(x_test.iloc[:, 0], y_test, alpha=0.75, label="Test data")
    axis.plot(x_grid, y_grid, color="crimson", linewidth=2, label="Prediction")
    axis.set(title=title, xlabel=x_train.columns[0], ylabel="target")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Ve bar chart so sanh model theo mot metric."""
    if metric not in results_df.columns:
        raise ValueError(f"Metric khong ton tai: {metric}")

    ascending = metric != "r2"
    ordered = results_df.sort_values(metric, ascending=ascending)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(ordered["model_name"], ordered[metric], color="steelblue")
    axis.set(title=f"So sanh model theo {metric.upper()}", ylabel=metric.upper())
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_coefficients(
    model: RegressorMixin,
    feature_names: list[str],
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Ve he so de quan sat tac dong cua regularization va Lasso."""
    estimator = get_estimator(model)
    if not hasattr(estimator, "coef_"):
        raise ValueError("Model nay khong co coef_.")

    coefficients = np.ravel(estimator.coef_)
    figure, axis = plt.subplots(figsize=(10, 5))
    colors = ["crimson" if value == 0 else "steelblue" for value in coefficients]
    axis.bar(feature_names, coefficients, color=colors)
    axis.set(title=title, ylabel="Coefficient")
    axis.tick_params(axis="x", rotation=45)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure

