"""Danh gia va so sanh cac mo hinh regression."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models import predict_model, train_model


def evaluate_regression(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Tinh cac metric regression pho bien."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compare_models(
    models: dict[str, RegressorMixin],
    x_train: Any,
    x_test: Any,
    y_train: Any,
    y_test: Any,
) -> pd.DataFrame:
    """Train tung model va tra ve bang metric kem thoi gian xu ly."""
    rows: list[dict[str, float | str]] = []
    for model_name, model in models.items():
        fit_start = perf_counter()
        train_model(model, x_train, y_train)
        fit_time = perf_counter() - fit_start

        predict_start = perf_counter()
        predictions = predict_model(model, x_test)
        predict_time = perf_counter() - predict_start

        rows.append(
            {
                "model_name": model_name,
                **evaluate_regression(y_test, predictions),
                "fit_time": fit_time,
                "predict_time": predict_time,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

