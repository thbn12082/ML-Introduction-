"""Density models used throughout the educational lab."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from density_lab.utils import ensure_1d, ensure_2d_column


def fit_gaussian_mle(x: np.ndarray | list[float]) -> dict[str, float]:
    """Fit a one-dimensional Gaussian using maximum likelihood estimates.

    For a Gaussian distribution, MLE estimates are:

    - mean: average of the data
    - variance: average squared deviation from the mean
    - standard deviation: square root of the variance
    """

    values = ensure_1d(x)
    mean = float(np.mean(values))
    variance = float(np.mean((values - mean) ** 2))
    std = float(np.sqrt(variance))
    return {"mean": mean, "variance": variance, "std": std}


def gaussian_pdf(grid: np.ndarray | list[float], mean: float, std: float) -> np.ndarray:
    """Evaluate the Gaussian probability density function on a grid."""

    if std <= 0:
        raise ValueError("std must be positive.")
    return norm.pdf(ensure_1d(grid), loc=mean, scale=std)


def fit_kde(x: np.ndarray | list[float], bandwidth: float) -> KernelDensity:
    """Fit Gaussian-kernel KDE with the selected bandwidth.

    KDE can be understood as placing one smooth bump around every data point.
    The bandwidth controls how wide each bump is.
    """

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    return kde.fit(ensure_2d_column(x))


def kde_pdf(kde_model: KernelDensity, grid: np.ndarray | list[float]) -> np.ndarray:
    """Evaluate a fitted KDE model and return ordinary density values."""

    log_density = kde_model.score_samples(ensure_2d_column(grid))
    return np.exp(log_density)


def fit_gmm(
    x: np.ndarray | list[float],
    n_components: int,
    random_state: int = 42,
) -> GaussianMixture:
    """Fit a one-dimensional Gaussian Mixture Model."""

    if n_components < 1:
        raise ValueError("n_components must be at least 1.")
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    return gmm.fit(ensure_2d_column(x))


def gmm_pdf(gmm_model: GaussianMixture, grid: np.ndarray | list[float]) -> np.ndarray:
    """Evaluate a fitted Gaussian Mixture Model as a density curve."""

    log_density = gmm_model.score_samples(ensure_2d_column(grid))
    return np.exp(log_density)


def mean_squared_error_density(true_pdf: np.ndarray, estimated_pdf: np.ndarray) -> float:
    """Return the mean squared difference between two density curves."""

    true_pdf = np.asarray(true_pdf, dtype=float)
    estimated_pdf = np.asarray(estimated_pdf, dtype=float)
    if true_pdf.shape != estimated_pdf.shape:
        raise ValueError("true_pdf and estimated_pdf must have the same shape.")
    return float(np.mean((true_pdf - estimated_pdf) ** 2))


def integrated_absolute_error(
    true_pdf: np.ndarray,
    estimated_pdf: np.ndarray,
    grid: np.ndarray,
) -> float:
    """Approximate the area between two density curves using the trapezoid rule."""

    true_pdf = np.asarray(true_pdf, dtype=float)
    estimated_pdf = np.asarray(estimated_pdf, dtype=float)
    grid_1d = ensure_1d(grid)
    if true_pdf.shape != estimated_pdf.shape:
        raise ValueError("true_pdf and estimated_pdf must have the same shape.")
    return float(np.trapezoid(np.abs(true_pdf - estimated_pdf), x=grid_1d))
