"""Streamlit app for interactively exploring density estimation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from density_lab.data_generator import (
    generate_gamma_data,
    generate_gaussian_data,
    generate_mixture_data,
)
from density_lab.density_models import (
    fit_gaussian_mle,
    fit_gmm,
    fit_kde,
    gaussian_pdf,
    gmm_pdf,
    kde_pdf,
)
from density_lab.utils import ensure_1d


def _load_dataset(dataset_name: str) -> dict:
    """Map a sidebar label to one synthetic dataset."""

    if dataset_name == "Gaussian":
        return generate_gaussian_data(n=300, random_state=42)
    if dataset_name == "Mixture":
        return generate_mixture_data(n=300, random_state=42)
    return generate_gamma_data(n=300, random_state=42)


def _explain(dataset_name: str, model_name: str, bandwidth: float | None, n_components: int | None) -> str:
    """Return a beginner-friendly Vietnamese explanation for the current view."""

    if model_name == "Gaussian MLE":
        if dataset_name == "Gaussian":
            return "Gaussian MLE đang fit tốt vì dữ liệu thật cũng có dạng Gaussian."
        return "Gaussian MLE chỉ tạo được một đỉnh đối xứng, nên dễ bị bias cao khi dữ liệu lệch hoặc có nhiều cụm."

    if model_name == "KDE":
        assert bandwidth is not None
        if bandwidth < 0.5:
            return "Đường cong đang bám sát từng điểm dữ liệu, dễ bị variance cao."
        if bandwidth > 2.0:
            return "Đường cong quá mượt, dễ bị bias cao vì nhiều chi tiết bị làm phẳng."
        return "Bandwidth ở mức vừa phải thường cho cân bằng tốt hơn giữa bias và variance."

    assert n_components is not None
    if dataset_name == "Mixture" and n_components == 1:
        return "Mô hình quá đơn giản vì dữ liệu có hai cụm nhưng GMM chỉ được dùng một component."
    if dataset_name == "Mixture" and n_components == 2:
        return "Hai components phù hợp với hai cụm chính trong dữ liệu mixture."
    if n_components > 4:
        return "Mô hình rất linh hoạt. Hãy cẩn thận vì quá nhiều components có thể fit cả nhiễu."
    return "GMM biểu diễn density bằng cách cộng nhiều Gaussian nhỏ lại với nhau."


def main() -> None:
    """Render the Streamlit educational visualizer."""

    st.set_page_config(page_title="Density Lab", layout="wide")
    st.title("Density Lab: Bias, Variance và Density Estimation")

    dataset_name = st.sidebar.selectbox("Dataset", ["Gaussian", "Mixture", "Gamma"])
    model_name = st.sidebar.selectbox("Model", ["Gaussian MLE", "KDE", "GMM"])

    bandwidth: float | None = None
    n_components: int | None = None
    if model_name == "KDE":
        bandwidth = st.sidebar.slider("Bandwidth", min_value=0.05, max_value=5.0, value=1.0, step=0.05)
    if model_name == "GMM":
        n_components = st.sidebar.slider("n_components", min_value=1, max_value=8, value=2, step=1)

    dataset = _load_dataset(dataset_name)
    samples = dataset["samples"]
    grid = dataset["grid"]

    if model_name == "Gaussian MLE":
        params = fit_gaussian_mle(samples)
        estimated_pdf = gaussian_pdf(grid, params["mean"], params["std"])
        estimate_label = f"Gaussian MLE (mean={params['mean']:.2f}, std={params['std']:.2f})"
    elif model_name == "KDE":
        kde = fit_kde(samples, bandwidth=bandwidth or 1.0)
        estimated_pdf = kde_pdf(kde, grid)
        estimate_label = f"KDE (bandwidth={bandwidth:.2f})"
    else:
        gmm = fit_gmm(samples, n_components=n_components or 2, random_state=42)
        estimated_pdf = gmm_pdf(gmm, grid)
        estimate_label = f"GMM ({n_components} components)"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ensure_1d(samples), bins=30, density=True, alpha=0.35, edgecolor="black", label="Histogram")
    ax.plot(ensure_1d(grid), dataset["true_pdf"], linewidth=2.5, label="True density")
    ax.plot(ensure_1d(grid), estimated_pdf, linewidth=2.2, label=estimate_label)
    ax.set_title(f"{dataset_name} dataset with {model_name}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Bạn đang thấy gì?")
    st.write(_explain(dataset_name, model_name, bandwidth, n_components))
    st.write(dataset["description"])


if __name__ == "__main__":
    main()
