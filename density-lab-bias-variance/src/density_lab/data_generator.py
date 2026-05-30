"""Synthetic datasets for density estimation experiments.

Each generator returns a dictionary with:

- ``name``: short dataset name
- ``samples``: NumPy array with shape ``(n, 1)``
- ``grid``: x-values used for plotting density curves
- ``true_pdf``: true density values on the grid
- ``description``: beginner-friendly explanation of the data
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma, norm

from density_lab.utils import Dataset


def _make_grid(low: float, high: float, points: int = 500) -> np.ndarray:
    """Create a plotting grid with shape ``(points, 1)``."""

    return np.linspace(low, high, points).reshape(-1, 1)


def generate_gaussian_data(n: int = 300, random_state: int = 42) -> Dataset:
    """Generate Dataset A: one Gaussian distribution for human-height-like data."""

    rng = np.random.default_rng(random_state)
    mean = 170.0
    std = 7.0
    samples = rng.normal(loc=mean, scale=std, size=n).reshape(-1, 1)
    grid = _make_grid(mean - 4 * std, mean + 4 * std)
    true_pdf = norm.pdf(grid.ravel(), loc=mean, scale=std)

    return {
        "name": "Gaussian",
        "samples": samples,
        "grid": grid,
        "true_pdf": true_pdf,
        "description": "Dataset A: chiều cao người, giả sử đến từ một phân phối Gaussian.",
    }


def generate_mixture_data(n: int = 300, random_state: int = 42) -> Dataset:
    """Generate Dataset B: a mixture of two Gaussian subgroups."""

    rng = np.random.default_rng(random_state)
    weights = np.array([0.6, 0.4])
    means = np.array([160.0, 175.0])
    stds = np.array([5.0, 6.0])

    component_ids = rng.choice([0, 1], size=n, p=weights)
    samples = rng.normal(loc=means[component_ids], scale=stds[component_ids]).reshape(-1, 1)

    grid = _make_grid(135.0, 200.0)
    grid_1d = grid.ravel()
    true_pdf = (
        weights[0] * norm.pdf(grid_1d, loc=means[0], scale=stds[0])
        + weights[1] * norm.pdf(grid_1d, loc=means[1], scale=stds[1])
    )

    return {
        "name": "Mixture",
        "samples": samples,
        "grid": grid,
        "true_pdf": true_pdf,
        "description": "Dataset B: hai nhóm con được trộn lại thành một bộ dữ liệu.",
    }


def generate_gamma_data(n: int = 300, random_state: int = 42) -> Dataset:
    """Generate Dataset C: a right-skewed Gamma distribution."""

    rng = np.random.default_rng(random_state)
    shape = 2.0
    scale = 2.0
    samples = rng.gamma(shape=shape, scale=scale, size=n).reshape(-1, 1)
    grid = _make_grid(0.0, 20.0)
    true_pdf = gamma.pdf(grid.ravel(), a=shape, scale=scale)

    return {
        "name": "Gamma",
        "samples": samples,
        "grid": grid,
        "true_pdf": true_pdf,
        "description": "Dataset C: dữ liệu lệch phải, giống thời gian chờ hoặc thu nhập.",
    }


def generate_all_datasets(n: int = 300, random_state: int = 42) -> dict[str, Dataset]:
    """Generate all three datasets with deterministic random seeds."""

    return {
        "gaussian": generate_gaussian_data(n=n, random_state=random_state),
        "mixture": generate_mixture_data(n=n, random_state=random_state),
        "gamma": generate_gamma_data(n=n, random_state=random_state),
    }
