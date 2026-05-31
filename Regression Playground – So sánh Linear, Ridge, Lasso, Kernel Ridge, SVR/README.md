# Regression Playground - Chapter 7

Mini-lab Python minh họa chương **Linear Models for Regression** dành cho người mới học Machine Learning. Project không tải dữ liệu từ internet: các ví dụ chính được tự sinh và ví dụ dữ liệu thật dùng `load_diabetes` có sẵn trong scikit-learn.

## Mục tiêu

Project giúp bạn quan sát trực tiếp:

- Linear Regression hoạt động tốt khi quan hệ thật gần tuyến tính.
- Overfitting xảy ra khi model học quá sát tập train nhưng dự đoán kém trên tập test.
- Ridge và Lasso dùng regularization để kiểm soát độ phức tạp.
- Lasso có thể đưa hệ số về `0`, từ đó thực hiện feature selection.
- Kernel Ridge và SVR phù hợp hơn với quan hệ phi tuyến.
- MSE, RMSE, MAE và R2 giúp so sánh model bằng số liệu.

## Project tổng hợp chương 7 như thế nào?

| Thí nghiệm | Dataset | Điều cần quan sát |
| --- | --- | --- |
| Linear | Synthetic tuyến tính | Least Squares tạo đường dự đoán phù hợp dữ liệu |
| Nonlinear | Synthetic dạng sin | Linear model yếu hơn Kernel Ridge và SVR khi dữ liệu cong |
| Sparse | Synthetic nhiều feature | Lasso đưa nhiều coefficient không quan trọng về `0` |
| Diabetes | Dataset thật built-in | So sánh toàn bộ model trên một bài toán thực tế |

## Cấu trúc thư mục

```text
regression-playground-chapter-7/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── app.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── experiment.py
├── notebooks/
│   └── chapter_7_regression_walkthrough.ipynb
├── outputs/
│   ├── figures/
│   └── reports/
└── tests/
    ├── test_data.py
    ├── test_models.py
    └── test_evaluation.py
```

## Cài đặt

Python yêu cầu phiên bản `3.11+`.

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Chạy các thí nghiệm

```bash
python -m src.experiment
```

Kết quả được lưu trong:

- `outputs/reports/`: bảng CSV chứa metric và thời gian xử lý.
- `outputs/figures/`: biểu đồ dự đoán, so sánh model và coefficient.

## Chạy Streamlit app

```bash
streamlit run app.py
```

Sidebar cho phép chọn dataset, model và hyperparameter. Dashboard hiển thị metric, biểu đồ phù hợp và nhận xét tự động.

## Chạy tests

```bash
pytest
```

## Các model

- **Linear Regression:** mô hình nền tảng, dễ hiểu. Model tìm hệ số giảm tổng bình phương sai số.
- **Ridge:** giảm overfitting bằng cách phạt hệ số lớn với regularization L2.
- **Lasso:** vừa regularization vừa feature selection nhờ regularization L1.
- **Kernel Ridge:** dùng kernel để học quan hệ phi tuyến.
- **SVR:** bỏ qua lỗi nhỏ trong epsilon margin và xử lý phi tuyến bằng kernel.

## Cần nhớ gì từ chương 7?

| Chủ đề | Mức độ ưu tiên | Ghi chú |
| --- | --- | --- |
| Regression | Bắt buộc nhớ | Dự đoán một giá trị liên tục |
| RSS / Least Squares | Bắt buộc nhớ | Nền tảng của Linear Regression |
| Ridge / Lasso | Bắt buộc nhớ | Hai cách regularization quan trọng |
| Kernel / SVR | Hiểu ý tưởng | Hữu ích khi quan hệ không còn tuyến tính |
| ADMM / LARS / Dantzig | Đọc lướt lúc đầu | Chưa cần đào sâu khi mới học |

## Gợi ý học

1. Chạy `python -m src.experiment` và xem bảng CSV.
2. Mở dashboard, đổi dataset và model.
3. Trên dataset nonlinear, so sánh Linear Regression với SVR.
4. Trên dataset sparse, đổi `alpha` của Lasso và quan sát số hệ số bằng `0`.
5. Mở notebook để học theo thứ tự từng bước.

