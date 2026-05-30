"""Experiment pipeline for generating figures, metrics, and the report."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

import numpy as np

from density_lab.data_generator import generate_all_datasets
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
from density_lab.plots import (
    plot_bias_variance_experiment,
    plot_curse_of_dimensionality,
    plot_gaussian_mle_fit,
    plot_gmm_components_comparison,
    plot_histogram_with_true_density,
    plot_kde_bandwidth_comparison,
)
from density_lab.utils import Dataset, make_directory


def _evaluate_density(true_pdf: np.ndarray, estimated_pdf: np.ndarray, grid: np.ndarray) -> dict[str, float]:
    """Compute simple error metrics for one estimated density curve."""

    return {
        "mse": mean_squared_error_density(true_pdf, estimated_pdf),
        "iae": integrated_absolute_error(true_pdf, estimated_pdf, grid),
    }


def _compute_metrics(datasets: dict[str, Dataset]) -> dict[str, dict[str, dict[str, float]]]:
    """Compute comparable metrics for Gaussian MLE, KDE, and GMM."""

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for key, dataset in datasets.items():
        grid = dataset["grid"]
        true_pdf = dataset["true_pdf"]

        gaussian_params = fit_gaussian_mle(dataset["samples"])
        gaussian_estimate = gaussian_pdf(grid, gaussian_params["mean"], gaussian_params["std"])

        kde = fit_kde(dataset["samples"], bandwidth=1.0)
        kde_estimate = kde_pdf(kde, grid)

        # Two components are especially meaningful for the mixture dataset, but
        # this still gives a simple comparison for the other datasets.
        gmm = fit_gmm(dataset["samples"], n_components=2, random_state=42)
        gmm_estimate = gmm_pdf(gmm, grid)

        metrics[key] = {
            "gaussian_mle": _evaluate_density(true_pdf, gaussian_estimate, grid),
            "kde_bandwidth_1": _evaluate_density(true_pdf, kde_estimate, grid),
            "gmm_2_components": _evaluate_density(true_pdf, gmm_estimate, grid),
        }
    return metrics


def _metric_line(metrics: dict[str, dict[str, float]], model_key: str) -> str:
    """Format one model's metrics for the Markdown report."""

    values = metrics[model_key]
    return f"MSE = {values['mse']:.6f}, IAE = {values['iae']:.4f}"


def _write_report(
    report_path: str | Path,
    figure_paths: dict[str, str],
    metrics: dict[str, dict[str, dict[str, float]]],
) -> Path:
    """Write a beginner-friendly Vietnamese Markdown report."""

    report_path = Path(report_path)
    make_directory(report_path.parent)

    content = f"""# Density Lab: Bias, Variance và Density Estimation

## 1. Project này chứng minh điều gì?

Ta thường chỉ có dữ liệu mẫu, nhưng không biết đường cong phân phối thật đã sinh ra dữ liệu đó. Density estimation là bài toán ước lượng đường cong ẩn này từ dữ liệu quan sát được.

Trong project này, ta thử ba cách ước lượng: Gaussian MLE, KDE và GMM. Sau đó ta quan sát bandwidth, bias, variance và curse of dimensionality.

## 2. Dataset A: Gaussian

![Dataset A]({figure_paths['gaussian_mle']})

Gaussian MLE: {_metric_line(metrics['gaussian'], 'gaussian_mle')}.

KDE bandwidth = 1.0: {_metric_line(metrics['gaussian'], 'kde_bandwidth_1')}.

Vì dữ liệu thật là Gaussian, Gaussian MLE hoạt động tốt. Đây là ví dụ khi giả định mô hình đúng thì mô hình đơn giản vẫn có thể rất hiệu quả.

## 3. Dataset B: Mixture of Gaussians

![Dataset B]({figure_paths['mixture_gmm']})

Gaussian MLE: {_metric_line(metrics['mixture'], 'gaussian_mle')}.

GMM 2 components: {_metric_line(metrics['mixture'], 'gmm_2_components')}.

Gaussian đơn bị bias cao vì dữ liệu có hai cụm. GMM 2 components hợp lý hơn vì nó cho phép hai đỉnh mật độ. GMM nhiều components linh hoạt hơn, nhưng nếu dùng quá nhiều components thì mô hình có thể trở nên phức tạp không cần thiết.

## 4. Dataset C: Gamma / Skewed

![Dataset C]({figure_paths['gamma_mle']})

Gaussian MLE: {_metric_line(metrics['gamma'], 'gaussian_mle')}.

KDE bandwidth = 1.0: {_metric_line(metrics['gamma'], 'kde_bandwidth_1')}.

Dữ liệu Gamma bị lệch phải nên Gaussian đơn không mô tả tốt phần đuôi và độ lệch. KDE có thể linh hoạt hơn nếu chọn bandwidth phù hợp.

## 5. KDE và bandwidth/sigma

![KDE bandwidth]({figure_paths['gaussian_kde']})

Bandwidth nhỏ làm mỗi điểm dữ liệu tạo ra một bump hẹp. Đường cong có thể răng cưa, bám sát dữ liệu huấn luyện, bias thấp nhưng variance cao.

Bandwidth lớn làm các bump rất rộng. Đường cong mượt và ổn định hơn, variance thấp hơn, nhưng dễ bị bias cao vì bỏ qua chi tiết thật của phân phối.

## 6. Bias - Variance Tradeoff

![Bias variance]({figure_paths['bias_variance']})

Mô hình quá đơn giản giống dùng thước thẳng để vẽ một đường cong: dễ sai có hệ thống, tức bias cao.

Mô hình quá phức tạp giống học thuộc từng điểm dữ liệu: thay đổi mạnh khi đổi mẫu huấn luyện, tức variance cao.

Mục tiêu thực tế là chọn mức linh hoạt vừa đủ.

## 7. Curse of Dimensionality

![Curse of dimensionality]({figure_paths['curse']})

Khi số chiều tăng, không gian phình to rất nhanh. Với cùng số điểm dữ liệu, các điểm trở nên thưa hơn. Khoảng cách giữa các điểm cũng kém ý nghĩa hơn, nên các phương pháp dựa vào lân cận như KDE khó dùng trong dữ liệu nhiều chiều.

## 8. Kết luận

Gaussian MLE tốt khi giả định Gaussian đúng. KDE linh hoạt nhưng nhạy với bandwidth. GMM hữu ích cho dữ liệu nhiều cụm. Với high-dimensional data, ta thường cần giảm chiều, chọn đặc trưng tốt hơn hoặc dùng domain knowledge thay vì cố ước lượng density trực tiếp trong không gian quá lớn.
"""

    report_path.write_text(content, encoding="utf-8")
    return report_path


def run_all_experiments(
    output_dir: str | Path = "figures",
    report_path: str | Path = "reports/density_lab_summary.md",
) -> dict[str, object]:
    """Generate all figures, metrics, and the Markdown report."""

    output_dir = make_directory(output_dir)
    report_path = Path(report_path)
    make_directory(report_path.parent)

    datasets = generate_all_datasets(n=300, random_state=42)
    bandwidths = [0.1, 0.3, 0.7, 1.5, 3.0]

    figures: dict[str, Path] = {}
    for key, dataset in datasets.items():
        figures[f"{key}_hist"] = plot_histogram_with_true_density(
            dataset, output_dir / f"{key}_histogram_true_density.png"
        )
        figures[f"{key}_mle"] = plot_gaussian_mle_fit(dataset, output_dir / f"{key}_gaussian_mle.png")
        figures[f"{key}_kde"] = plot_kde_bandwidth_comparison(
            dataset, bandwidths, output_dir / f"{key}_kde_bandwidths.png"
        )

    figures["bias_variance"] = plot_bias_variance_experiment(
        "gaussian", [0.3, 1.0, 3.0], output_dir / "bias_variance_kde.png"
    )
    figures["mixture_gmm"] = plot_gmm_components_comparison(
        datasets["mixture"], [1, 2, 5], output_dir / "mixture_gmm_components.png"
    )
    figures["curse"] = plot_curse_of_dimensionality(output_dir / "curse_of_dimensionality.png")

    metrics = _compute_metrics(datasets)

    report_dir = report_path.parent
    markdown_figures = {
        key: Path("..", value).as_posix() if not Path(value).is_absolute() else value.as_posix()
        for key, value in figures.items()
    }
    written_report = _write_report(report_path, markdown_figures, metrics)

    return {
        "figures": {key: str(value) for key, value in figures.items()},
        "metrics": metrics,
        "report_path": str(written_report),
    }


def main() -> None:
    """Run the experiment pipeline from the command line."""

    results = run_all_experiments()
    pprint(results)


if __name__ == "__main__":
    main()
