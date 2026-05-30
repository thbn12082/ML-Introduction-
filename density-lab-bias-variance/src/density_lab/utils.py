"""Small utility helpers shared across the density lab project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def ensure_2d_column(x: np.ndarray | list[float]) -> np.ndarray:
    """Return data as a NumPy array with shape ``(n_samples, 1)``.

    Many scikit-learn estimators expect a two-dimensional array. Beginners often
    have a one-dimensional vector, so this helper makes the conversion explicit.
    """

    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2 and array.shape[1] == 1:
        return array
    raise ValueError("Expected x to have shape (n,) or (n, 1).")


def ensure_1d(x: np.ndarray | list[float]) -> np.ndarray:
    """Return data as a one-dimensional NumPy array."""

    return ensure_2d_column(x).ravel()


def make_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def relative_markdown_link(target: str | Path, base_dir: str | Path) -> str:
    """Return a POSIX-style relative path suitable for Markdown links."""

    target_path = Path(target)
    base_path = Path(base_dir)
    try:
        rel_path = target_path.relative_to(base_path)
    except ValueError:
        rel_path = Path("..") / target_path
    return rel_path.as_posix()


Dataset = dict[str, Any]
