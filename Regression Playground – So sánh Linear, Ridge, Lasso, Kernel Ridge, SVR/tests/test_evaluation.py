"""Tests cho metric va bang so sanh model."""

from sklearn.model_selection import train_test_split

from src.data import make_linear_dataset
from src.evaluation import compare_models, evaluate_regression
from src.models import create_model


def test_evaluate_regression_has_expected_keys() -> None:
    metrics = evaluate_regression([1, 2, 3], [1, 2, 4])
    assert set(metrics) == {"mse", "rmse", "mae", "r2"}


def test_compare_models_returns_non_empty_dataframe() -> None:
    x, y = make_linear_dataset(n_samples=80)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    results = compare_models(
        {"Linear Regression": create_model("Linear Regression")},
        x_train,
        x_test,
        y_train,
        y_test,
    )
    assert not results.empty
    assert set(results.columns) == {
        "model_name",
        "mse",
        "rmse",
        "mae",
        "r2",
        "fit_time",
        "predict_time",
    }

