"""Matplotlib plotting functions for the density lab."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from density_lab.data_generator import generate_gaussian_data
from density_lab.density_models import (
    fit_gaussian_mle,
    fit_gmm,
    fit_kde,
    gaussian_pdf,
    gmm_pdf,
    kde_pdf,
)
from density_lab.utils import Dataset, ensure_1d, make_directory


def _prepare_save_path(save_path: str | Path) -> Path:
    """Create the parent folder for a plot and return a Path object."""

    path = Path(save_path)
    make_directory(path.parent)
    return path


def _plot_histogram(ax: plt.Axes, samples: np.ndarray, label: str = "Histogram") -> None:
    """Draw a normalized histogram that can be compared with density curves."""

    ax.hist(
        ensure_1d(samples),
        bins=30,
        density=True,
        alpha=0.35,
        edgecolor="black",
        linewidth=0.5,
        label=label,
    )


def _plot_true_pdf(ax: plt.Axes, dataset: Dataset) -> None:
    """Draw the known true density if the dataset provides one."""

    true_pdf = dataset.get("true_pdf")
    if true_pdf is not None:
        ax.plot(ensure_1d(dataset["grid"]), true_pdf, linewidth=2.5, label="True PDF")


def plot_histogram_with_true_density(dataset: Dataset, save_path: str | Path) -> Path:
    """Plot samples as a histogram and overlay the true density curve."""

    save_path = _prepare_save_path(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_histogram(ax, dataset["samples"])
    _plot_true_pdf(ax, dataset)
    ax.set_title(f"{dataset['name']}: histogram and true density")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_gaussian_mle_fit(dataset: Dataset, save_path: str | Path) -> Path:
    """Plot a Gaussian MLE estimate against the data and true density."""

    save_path = _prepare_save_path(save_path)
    grid = dataset["grid"]
    params = fit_gaussian_mle(dataset["samples"])
    estimated_pdf = gaussian_pdf(grid, params["mean"], params["std"])

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_histogram(ax, dataset["samples"])
    _plot_true_pdf(ax, dataset)
    ax.plot(ensure_1d(grid), estimated_pdf, linewidth=2.2, label="Gaussian MLE")
    ax.set_title(f"{dataset['name']}: Gaussian MLE fit, good only when Gaussian assumption fits")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_kde_bandwidth_comparison(
    dataset: Dataset,
    bandwidths: list[float],
    save_path: str | Path,
) -> Path:
    """Compare KDE curves for multiple bandwidth values."""

    save_path = _prepare_save_path(save_path)
    grid = dataset["grid"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_histogram(ax, dataset["samples"])
    _plot_true_pdf(ax, dataset)
    for bandwidth in bandwidths:
        kde = fit_kde(dataset["samples"], bandwidth=bandwidth)
        ax.plot(ensure_1d(grid), kde_pdf(kde, grid), linewidth=1.8, label=f"KDE bw={bandwidth}")

    ax.set_title(f"{dataset['name']}: bandwidth controls KDE smoothness")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_bias_variance_experiment(
    distribution_name: str,
    bandwidths: list[float],
    save_path: str | Path,
) -> Path:
    """Visualize bias-variance behavior by fitting KDE on many training sets."""

    if distribution_name != "gaussian":
        raise ValueError("This educational experiment currently supports distribution_name='gaussian'.")

    save_path = _prepare_save_path(save_path)
    n_train_sets = 20
    n_samples = 80
    base_dataset = generate_gaussian_data(n=n_samples, random_state=0)
    grid = base_dataset["grid"]

    fig, axes = plt.subplots(1, len(bandwidths), figsize=(5.2 * len(bandwidths), 4.5), sharey=True)
    if len(bandwidths) == 1:
        axes = [axes]

    for ax, bandwidth in zip(axes, bandwidths):
        for seed in range(n_train_sets):
            dataset = generate_gaussian_data(n=n_samples, random_state=seed)
            kde = fit_kde(dataset["samples"], bandwidth=bandwidth)
            ax.plot(ensure_1d(grid), kde_pdf(kde, grid), color="tab:blue", alpha=0.18, linewidth=1.0)

        ax.plot(ensure_1d(grid), base_dataset["true_pdf"], color="black", linewidth=2.3, label="True PDF")
        ax.set_title(f"Bandwidth = {bandwidth}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Bias-variance in KDE: small bandwidth varies more, large bandwidth oversmooths")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_gmm_components_comparison(
    dataset: Dataset,
    components: list[int],
    save_path: str | Path,
) -> Path:
    """Compare GMM density estimates with different component counts."""

    save_path = _prepare_save_path(save_path)
    grid = dataset["grid"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_histogram(ax, dataset["samples"])
    _plot_true_pdf(ax, dataset)
    for n_components in components:
        gmm = fit_gmm(dataset["samples"], n_components=n_components, random_state=42)
        ax.plot(ensure_1d(grid), gmm_pdf(gmm, grid), linewidth=2.0, label=f"GMM components={n_components}")

    ax.set_title("GMM on mixture data: 1 too simple, 2 good, 5 more flexible")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_curse_of_dimensionality(save_path: str | Path) -> Path:
    """Plot how nearest-neighbor distances behave as dimension increases.

    The saved figure contains two panels:

    1. Average nearest-neighbor distance increases with dimension.
    2. Nearest/farthest distance contrast becomes less helpful.
    """

    save_path = _prepare_save_path(save_path)
    dimensions = [1, 2, 5, 10, 50, 100]
    n = 1000
    rng = np.random.default_rng(42)
    mean_nearest_distances: list[float] = []
    contrast_ratios: list[float] = []

    for dim in dimensions:
        x = rng.normal(size=(n, dim))
        nearest_model = NearestNeighbors(n_neighbors=2)
        nearest_model.fit(x)
        distances, _ = nearest_model.kneighbors(x)
        nearest_distances = distances[:, 1]

        farthest_distances = []
        for row in x:
            all_distances = np.linalg.norm(x - row, axis=1)
            farthest_distances.append(np.max(all_distances))

        mean_nearest_distances.append(float(np.mean(nearest_distances)))
        contrast_ratios.append(float(np.mean(nearest_distances) / np.mean(farthest_distances)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(dimensions, mean_nearest_distances, marker="o")
    axes[0].set_title("Higher dimensions make nearby points farther away")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Average nearest-neighbor distance")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dimensions, contrast_ratios, marker="o", color="tab:orange")
    axes[1].set_title("Distance contrast becomes less useful")
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Average nearest / average farthest distance")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
