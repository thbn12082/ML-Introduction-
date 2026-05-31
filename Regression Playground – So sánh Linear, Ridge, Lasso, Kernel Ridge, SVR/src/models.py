"""Khoi tao, train va predict cac mo hinh regression."""

from __future__ import annotations

from typing import Any

from sklearn.base import RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def _scaled(model: RegressorMixin) -> Pipeline:
    """Chuan hoa feature truoc khi train model nhay voi scale."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def create_model(model_name: str, **kwargs: Any) -> RegressorMixin:
    """Tao mot model theo ten de dung chung cho app va experiment."""
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Ridge":
        return _scaled(Ridge(alpha=float(kwargs.get("alpha", 1.0))))
    if model_name == "Lasso":
        return _scaled(
            Lasso(
                alpha=float(kwargs.get("alpha", 0.1)),
                max_iter=20_000,
                random_state=kwargs.get("random_state", 42),
            )
        )
    if model_name == "Kernel Ridge":
        return _scaled(
            KernelRidge(
                alpha=float(kwargs.get("alpha", 1.0)),
                kernel=str(kwargs.get("kernel", "rbf")),
                gamma=kwargs.get("gamma", None),
            )
        )
    if model_name == "SVR":
        return _scaled(
            SVR(
                kernel=str(kwargs.get("kernel", "rbf")),
                C=float(kwargs.get("C", 10.0)),
                epsilon=float(kwargs.get("epsilon", 0.1)),
                gamma=kwargs.get("gamma", "scale"),
            )
        )
    raise ValueError(f"Model khong duoc ho tro: {model_name}")


def get_models(random_state: int = 42) -> dict[str, RegressorMixin]:
    """Tra ve danh sach model voi nhieu muc regularization de so sanh."""
    return {
        "Linear Regression": create_model("Linear Regression"),
        "Ridge alpha=0.1": create_model("Ridge", alpha=0.1),
        "Ridge alpha=1.0": create_model("Ridge", alpha=1.0),
        "Ridge alpha=10.0": create_model("Ridge", alpha=10.0),
        "Lasso alpha=0.001": create_model(
            "Lasso", alpha=0.001, random_state=random_state
        ),
        "Lasso alpha=0.01": create_model(
            "Lasso", alpha=0.01, random_state=random_state
        ),
        "Lasso alpha=0.1": create_model(
            "Lasso", alpha=0.1, random_state=random_state
        ),
        "Kernel Ridge RBF": create_model("Kernel Ridge", alpha=1.0, kernel="rbf"),
        "SVR RBF": create_model("SVR", kernel="rbf", C=10.0, epsilon=0.1),
    }


def train_model(
    model: RegressorMixin, x_train: Any, y_train: Any
) -> RegressorMixin:
    """Train model va tra lai model da fit."""
    return model.fit(x_train, y_train)


def predict_model(model: RegressorMixin, x_test: Any) -> Any:
    """Du doan target tu model da train."""
    return model.predict(x_test)


def get_estimator(model: RegressorMixin) -> RegressorMixin:
    """Lay estimator cuoi cung neu model nam trong pipeline."""
    if isinstance(model, Pipeline):
        return model.named_steps["model"]
    return model

