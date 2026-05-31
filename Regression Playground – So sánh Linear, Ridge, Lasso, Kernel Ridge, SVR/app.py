"""Streamlit dashboard cho regression playground."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.data import (
    load_real_regression_dataset,
    make_linear_dataset,
    make_nonlinear_dataset,
    make_sparse_dataset,
)
from src.evaluation import evaluate_regression
from src.models import create_model, get_estimator, predict_model, train_model
from src.visualization import plot_coefficients, plot_predictions_1d

DATASET_DESCRIPTIONS = {
    "Linear synthetic": "Quan hệ thật gần tuyến tính. Linear Regression thường là lựa chọn tốt.",
    "Nonlinear synthetic": "Quan hệ dạng sin. Dataset này cho thấy giới hạn của đường thẳng.",
    "Sparse synthetic": "Có nhiều feature nhưng chỉ vài feature quan trọng. Dùng để quan sát feature selection.",
    "Diabetes real dataset": "Dataset thật built-in của scikit-learn với 10 feature liên quan bệnh tiểu đường.",
}

MODEL_DESCRIPTIONS = {
    "Linear Regression": "Tìm đường hoặc siêu phẳng giảm tổng bình phương sai số.",
    "Ridge": "Thêm hình phạt L2 để hạn chế hệ số quá lớn và giảm overfitting.",
    "Lasso": "Thêm hình phạt L1, có thể đưa một số hệ số về 0 để chọn feature.",
    "Kernel Ridge": "Kết hợp Ridge với kernel để học quan hệ phi tuyến.",
    "SVR": "Tập trung vào sai số nằm ngoài epsilon margin và có thể dùng kernel phi tuyến.",
}


def load_selected_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Nap dataset theo lua chon cua nguoi dung."""
    if name == "Linear synthetic":
        return make_linear_dataset()
    if name == "Nonlinear synthetic":
        return make_nonlinear_dataset()
    if name == "Sparse synthetic":
        return make_sparse_dataset()
    x, y, _ = load_real_regression_dataset()
    return x, y


def select_model(model_name: str):
    """Hien thi hyperparameter phu hop va tao model."""
    if model_name in {"Ridge", "Lasso", "Kernel Ridge"}:
        default_alpha = 0.1 if model_name == "Lasso" else 1.0
        alpha = st.sidebar.number_input(
            "alpha", min_value=0.0001, value=default_alpha, format="%.4f"
        )
    else:
        alpha = 1.0

    if model_name in {"Kernel Ridge", "SVR"}:
        kernel = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly"])
    else:
        kernel = "rbf"

    if model_name == "SVR":
        c_value = st.sidebar.number_input("C", min_value=0.01, value=10.0)
        epsilon = st.sidebar.number_input("epsilon", min_value=0.0, value=0.1)
        gamma = st.sidebar.selectbox("gamma", ["scale", "auto"])
        return create_model(
            model_name, kernel=kernel, C=c_value, epsilon=epsilon, gamma=gamma
        )

    if model_name == "Kernel Ridge":
        gamma = st.sidebar.number_input("gamma", min_value=0.0001, value=5.0)
        return create_model(model_name, alpha=alpha, kernel=kernel, gamma=gamma)
    return create_model(model_name, alpha=alpha)


def show_automatic_notes(
    model_name: str,
    dataset_name: str,
    model,
    train_r2: float,
    test_r2: float,
    x: pd.DataFrame,
) -> None:
    """Sinh nhan xet de nguoi moi hoc biet cach doc ket qua."""
    st.subheader("Nhận xét tự động")
    notes: list[str] = []
    if train_r2 - test_r2 > 0.2:
        notes.append(
            "Train R2 cao hơn test R2 đáng kể: model có dấu hiệu overfitting."
        )
    else:
        notes.append("Chênh lệch train/test R2 chưa cho thấy overfitting rõ rệt.")

    estimator = get_estimator(model)
    if model_name == "Lasso" and hasattr(estimator, "coef_"):
        zero_count = int(np.isclose(estimator.coef_, 0).sum())
        notes.append(
            f"Lasso đưa {zero_count}/{x.shape[1]} hệ số về 0. "
            "Đây là cách Lasso thực hiện feature selection."
        )

    if dataset_name == "Nonlinear synthetic":
        if model_name in {"Kernel Ridge", "SVR"}:
            notes.append(
                "Model kernel phù hợp hơn đường thẳng vì dữ liệu có quan hệ phi tuyến."
            )
        else:
            notes.append(
                "Nếu R2 thấp, nguyên nhân chính là model tuyến tính khó mô tả đường cong."
            )
    for note in notes:
        st.write(f"- {note}")


def main() -> None:
    """Render Streamlit app."""
    st.set_page_config(page_title="Regression Playground Chapter 7", layout="wide")
    st.title("Regression Playground - Chapter 7")
    st.write(
        "Mini-lab so sánh Linear Regression, Ridge, Lasso, Kernel Ridge và SVR."
    )

    st.sidebar.header("Cấu hình thí nghiệm")
    dataset_name = st.sidebar.selectbox("Dataset", list(DATASET_DESCRIPTIONS))
    model_name = st.sidebar.selectbox("Model", list(MODEL_DESCRIPTIONS))
    model = select_model(model_name)

    x, y = load_selected_dataset(dataset_name)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    train_model(model, x_train, y_train)
    train_predictions = predict_model(model, x_train)
    test_predictions = predict_model(model, x_test)
    train_metrics = evaluate_regression(y_train, train_predictions)
    test_metrics = evaluate_regression(y_test, test_predictions)

    st.subheader("Ý tưởng đang minh họa")
    st.write(f"**Dataset:** {DATASET_DESCRIPTIONS[dataset_name]}")
    st.write(f"**Model:** {MODEL_DESCRIPTIONS[model_name]}")
    st.write(
        "**Cách đọc kết quả:** RMSE và MAE càng thấp càng tốt; R2 càng gần 1 càng tốt."
    )

    st.subheader("Metrics trên tập test")
    metrics_frame = pd.DataFrame([test_metrics], index=["test"]).round(4)
    st.dataframe(metrics_frame, use_container_width=True)
    st.caption(f"Train R2: {train_metrics['r2']:.4f} | Test R2: {test_metrics['r2']:.4f}")

    if x.shape[1] == 1:
        prediction_figure = plot_predictions_1d(
            model, x_train, y_train, x_test, y_test, f"{model_name} - {dataset_name}"
        )
        st.pyplot(prediction_figure)
        plt.close(prediction_figure)

    if hasattr(get_estimator(model), "coef_"):
        coefficient_figure = plot_coefficients(
            model, list(x.columns), f"Hệ số của {model_name}"
        )
        st.pyplot(coefficient_figure)
        plt.close(coefficient_figure)

    show_automatic_notes(
        model_name, dataset_name, model, train_metrics["r2"], test_metrics["r2"], x
    )


if __name__ == "__main__":
    main()
