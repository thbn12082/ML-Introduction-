# density-lab-bias-variance

Project này giúp người mới học hiểu các ý tưởng trong density estimation: ta có dữ liệu mẫu, nhưng muốn ước lượng đường cong phân phối thật phía sau dữ liệu.

## Project này dạy gì?

- Density estimation là gì
- Maximum Likelihood Estimation cho Gaussian
- Kernel Density Estimation
- Bandwidth / sigma ảnh hưởng thế nào
- Bias và variance
- Gaussian Mixture Model
- Method of sieves ở mức trực giác
- Curse of dimensionality

## Cài đặt

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Cài dependencies:

```bash
pip install -r requirements.txt
```

## Chạy experiments

```bash
python -m density_lab.experiments
```

Lệnh này tạo hình trong `figures/` và report trong `reports/density_lab_summary.md`.

## Mở notebook

```bash
jupyter notebook notebooks/01_density_lab.ipynb
```

## Chạy Streamlit app

```bash
streamlit run app.py
```

## Nên đọc project theo thứ tự nào?

1. Đọc `README.md`
2. Chạy experiments
3. Mở thư mục `figures/`
4. Đọc `reports/density_lab_summary.md`
5. Inspect notebook `notebooks/01_density_lab.ipynb`
6. Chạy Streamlit app

## Các concept được giải thích

- Density estimation
- MLE
- KDE
- Bandwidth
- Bias
- Variance
- GMM
- Curse of dimensionality

## Method of sieves trực giác

Method of sieves có thể hiểu đơn giản là: thay vì cho mô hình phức tạp vô hạn ngay từ đầu, ta dùng một họ mô hình có độ phức tạp tăng dần. Ví dụ với GMM, ta có thể bắt đầu từ 1 component, rồi 2 components, rồi nhiều hơn. Khi dữ liệu nhiều hơn, ta có thể cho phép mô hình linh hoạt hơn, nhưng vẫn phải kiểm soát để tránh overfitting.
