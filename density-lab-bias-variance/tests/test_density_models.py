"""Tests for density model helpers."""

import numpy as np

from density_lab.data_generator import generate_gaussian_data, generate_mixture_data
from density_lab.density_models import (
    fit_gaussian_mle,
    fit_gmm,
    fit_kde,
    gaussian_pdf,
    gmm_pdf,
    integrated_absolute_error,
    kde_pdf,
    mean_squared_error_density,
)


def test_gaussian_mle_returns_reasonable_mean_and_std() -> None:
    """MLE estimates should be close to the known Gaussian parameters."""

    dataset = generate_gaussian_data(n=5000, random_state=42)
    params = fit_gaussian_mle(dataset["samples"])
    assert abs(params["mean"] - 170.0) < 0.3
    assert abs(params["std"] - 7.0) < 0.3


def test_kde_pdf_returns_non_negative_values() -> None:
    """A density curve cannot be negative."""

    dataset = generate_gaussian_data(n=200, random_state=42)
    kde = fit_kde(dataset["samples"], bandwidth=1.0)
    pdf = kde_pdf(kde, dataset["grid"])
    assert np.all(pdf >= 0)


def test_gmm_pdf_returns_non_negative_values() -> None:
    """GMM density values should be non-negative."""

    dataset = generate_mixture_data(n=200, random_state=42)
    gmm = fit_gmm(dataset["samples"], n_components=2, random_state=42)
    pdf = gmm_pdf(gmm, dataset["grid"])
    assert np.all(pdf >= 0)


def test_mse_function_returns_non_negative_value() -> None:
    """MSE is always zero or positive."""

    dataset = generate_gaussian_data(n=200, random_state=42)
    params = fit_gaussian_mle(dataset["samples"])
    estimated_pdf = gaussian_pdf(dataset["grid"], params["mean"], params["std"])
    mse = mean_squared_error_density(dataset["true_pdf"], estimated_pdf)
    assert mse >= 0


def test_integrated_absolute_error_returns_non_negative_value() -> None:
    """IAE is an area and should be zero or positive."""

    dataset = generate_gaussian_data(n=200, random_state=42)
    params = fit_gaussian_mle(dataset["samples"])
    estimated_pdf = gaussian_pdf(dataset["grid"], params["mean"], params["std"])
    iae = integrated_absolute_error(dataset["true_pdf"], estimated_pdf, dataset["grid"])
    assert iae >= 0
