"""Tao dataset synthetic va nap dataset regression co san."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, make_regression


def make_linear_dataset(
    n_samples: int = 300, noise: float = 10, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Tao dataset 1 chieu voi quan he tuyen tinh ro rang."""
    rng = np.random.default_rng(random_state)
    x = rng.uniform(-10, 10, size=n_samples)
    y = 4.5 * x + 12 + rng.normal(0, noise, size=n_samples)
    return pd.DataFrame({"x": x}), pd.Series(y, name="target")


def make_nonlinear_dataset(
    n_samples: int = 300, noise: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Tao dataset 1 chieu dang sin de minh hoa quan he phi tuyen."""
    rng = np.random.default_rng(random_state)
    x = rng.uniform(-4 * np.pi, 4 * np.pi, size=n_samples)
    y = np.sin(x) + rng.normal(0, noise, size=n_samples)
    return pd.DataFrame({"x": x}), pd.Series(y, name="target")


def make_sparse_dataset(
    n_samples: int = 300,
    n_features: int = 20,
    n_informative: int = 5,
    noise: float = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Tao dataset nhieu bien, nhung chi mot so bien thuc su quan trong."""
    x, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )
    feature_names = [f"feature_{index + 1}" for index in range(n_features)]
    return pd.DataFrame(x, columns=feature_names), pd.Series(y, name="target")


def load_real_regression_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Nap dataset diabetes built-in cua scikit-learn."""
    dataset = load_diabetes(as_frame=True)
    x = dataset.data.copy()
    y = dataset.target.copy().rename("target")
    return x, y, list(x.columns)

