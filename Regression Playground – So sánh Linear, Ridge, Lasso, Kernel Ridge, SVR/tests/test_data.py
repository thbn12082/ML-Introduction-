"""Tests cho module data."""

from src.data import (
    load_real_regression_dataset,
    make_linear_dataset,
    make_nonlinear_dataset,
    make_sparse_dataset,
)


def test_linear_dataset_shape() -> None:
    x, y = make_linear_dataset(n_samples=50)
    assert x.shape == (50, 1)
    assert y.shape == (50,)


def test_nonlinear_dataset_shape() -> None:
    x, y = make_nonlinear_dataset(n_samples=60)
    assert x.shape == (60, 1)
    assert y.shape == (60,)


def test_sparse_dataset_shape() -> None:
    x, y = make_sparse_dataset(n_samples=70, n_features=12, n_informative=4)
    assert x.shape == (70, 12)
    assert y.shape == (70,)


def test_diabetes_dataset_has_feature_names() -> None:
    x, y, feature_names = load_real_regression_dataset()
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == len(feature_names)
    assert x.shape[1] > 1

