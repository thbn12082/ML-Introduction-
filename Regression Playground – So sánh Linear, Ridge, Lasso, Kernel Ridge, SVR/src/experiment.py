"""Chay cac thi nghiem minh hoa chuong Linear Models for Regression."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import (
    load_real_regression_dataset,
    make_linear_dataset,
    make_nonlinear_dataset,
    make_sparse_dataset,
)
from src.evaluation import compare_models
from src.models import create_model, get_models, train_model
from src.visualization import (
    plot_coefficients,
    plot_model_comparison,
    plot_predictions_1d,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"


def _prepare_outputs() -> None:
    """Tao folder output neu chua ton tai."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _split(x: pd.DataFrame, y: pd.Series):
    """Chia train/test thong nhat cho cac thi nghiem."""
    return train_test_split(x, y, test_size=0.25, random_state=42)


def _save_results(results: pd.DataFrame, report_name: str) -> None:
    """Luu report CSV va in bang ket qua ra terminal."""
    results.to_csv(REPORTS_DIR / report_name, index=False)
    print(results.round(4).to_string(index=False))


def run_linear_experiment() -> pd.DataFrame:
    """So sanh cac linear model tren dataset tuyen tinh."""
    _prepare_outputs()
    x, y = make_linear_dataset()
    x_train, x_test, y_train, y_test = _split(x, y)
    models = {
        "Linear Regression": create_model("Linear Regression"),
        "Ridge alpha=1.0": create_model("Ridge", alpha=1.0),
        "Lasso alpha=0.1": create_model("Lasso", alpha=0.1),
    }
    results = compare_models(models, x_train, x_test, y_train, y_test)
    print("\n=== Linear experiment ===")
    _save_results(results, "linear_experiment.csv")
    figure = plot_predictions_1d(
        models["Linear Regression"],
        x_train,
        y_train,
        x_test,
        y_test,
        "Linear Regression tren du lieu tuyen tinh",
        FIGURES_DIR / "linear_experiment.png",
    )
    plt.close(figure)
    return results


def run_nonlinear_experiment() -> pd.DataFrame:
    """So sanh linear model va kernel model tren dataset cong."""
    _prepare_outputs()
    x, y = make_nonlinear_dataset()
    x_train, x_test, y_train, y_test = _split(x, y)
    models = {
        "Linear Regression": create_model("Linear Regression"),
        "Ridge alpha=1.0": create_model("Ridge", alpha=1.0),
        "Kernel Ridge RBF": create_model("Kernel Ridge", alpha=0.1, gamma=5.0),
        "SVR RBF": create_model("SVR", C=10.0, epsilon=0.1, gamma=5.0),
    }
    results = compare_models(models, x_train, x_test, y_train, y_test)
    print("\n=== Nonlinear experiment ===")
    _save_results(results, "nonlinear_experiment.csv")
    figure = plot_model_comparison(
        results, "rmse", FIGURES_DIR / "nonlinear_experiment.png"
    )
    plt.close(figure)
    prediction_figure = plot_predictions_1d(
        models["SVR RBF"],
        x_train,
        y_train,
        x_test,
        y_test,
        "SVR RBF tren du lieu phi tuyen",
        FIGURES_DIR / "nonlinear_svr_predictions.png",
    )
    plt.close(prediction_figure)
    return results


def run_sparse_experiment() -> pd.DataFrame:
    """Minh hoa Lasso dat nhieu he so khong quan trong ve 0."""
    _prepare_outputs()
    x, y = make_sparse_dataset()
    x_train, x_test, y_train, y_test = _split(x, y)
    models = {
        "Linear Regression": create_model("Linear Regression"),
        "Ridge alpha=10.0": create_model("Ridge", alpha=10.0),
        "Lasso alpha=1.0": create_model("Lasso", alpha=1.0),
    }
    results = compare_models(models, x_train, x_test, y_train, y_test)
    print("\n=== Sparse experiment ===")
    _save_results(results, "sparse_experiment.csv")
    for model_name, model in models.items():
        file_name = model_name.lower().replace(" ", "_").replace("=", "_")
        figure = plot_coefficients(
            model,
            list(x.columns),
            f"He so: {model_name}",
            FIGURES_DIR / f"sparse_{file_name}.png",
        )
        plt.close(figure)
    return results


def run_real_dataset_experiment() -> pd.DataFrame:
    """So sanh toan bo model tren dataset diabetes."""
    _prepare_outputs()
    x, y, _ = load_real_regression_dataset()
    x_train, x_test, y_train, y_test = _split(x, y)
    results = compare_models(get_models(), x_train, x_test, y_train, y_test)
    print("\n=== Diabetes real dataset experiment ===")
    _save_results(results, "diabetes_experiment.csv")
    figure = plot_model_comparison(
        results, "rmse", FIGURES_DIR / "diabetes_experiment.png"
    )
    plt.close(figure)
    return results


def main() -> None:
    """Chay tat ca thi nghiem."""
    run_linear_experiment()
    run_nonlinear_experiment()
    run_sparse_experiment()
    run_real_dataset_experiment()


if __name__ == "__main__":
    main()
