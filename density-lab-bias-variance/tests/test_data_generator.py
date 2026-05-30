"""Tests for deterministic synthetic data generators."""

import numpy as np

from density_lab.data_generator import (
    generate_all_datasets,
    generate_gamma_data,
    generate_gaussian_data,
    generate_mixture_data,
)


def test_generated_data_has_correct_shape() -> None:
    """Every generator should return samples with shape (n, 1)."""

    for generator in [generate_gaussian_data, generate_mixture_data, generate_gamma_data]:
        dataset = generator(n=123, random_state=42)
        assert dataset["samples"].shape == (123, 1)


def test_generators_are_deterministic_with_fixed_random_state() -> None:
    """The same random_state should produce the same samples."""

    first = generate_mixture_data(n=50, random_state=7)
    second = generate_mixture_data(n=50, random_state=7)
    np.testing.assert_allclose(first["samples"], second["samples"])


def test_true_pdf_and_grid_have_same_length() -> None:
    """The true density should align with the plotting grid."""

    datasets = generate_all_datasets(n=100, random_state=42)
    for dataset in datasets.values():
        assert dataset["grid"].shape[0] == dataset["true_pdf"].shape[0]
