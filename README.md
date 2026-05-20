# Dự Báo Phụ Tải Điện Năng bằng Mạng Nơ-ron LSTM
(Power Consumption Forecasting using LSTM)

Dự án này sử dụng mạng nơ-ron học sâu LSTM (Long Short-Term Memory) để dự báo lượng điện năng tiêu thụ hàng giờ của khu vực PJM East (Megawatts). Mô hình học từ dữ liệu quá khứ (sliding window 24 giờ) để dự đoán nhu cầu điện năng cho giờ tiếp theo.

## 📊 Về Tập Dữ Liệu
* **Nguồn dữ liệu**: Dữ liệu tiêu thụ điện hàng giờ `PJME_hourly.csv`.
* **Đặc trưng**: Cột thời gian (Index) và cột `PJME_MW` (Điện năng tiêu thụ thực tế).
* **Tiền xử lý**: Đã loại bỏ các giá trị trùng lặp do thay đổi giờ quy ước (Daylight Saving Time), nội suy/loại bỏ dữ liệu rỗng và chuẩn hóa Min-Max (0-1).

## 🛠️ Cài Đặt 
Dự án sử dụng [uv](https://github.com/astral-sh/uv) và `pyproject.toml` để quản lý môi trường và thư viện một cách nhanh chóng, đồng nhất. Đảm bảo bạn đã cài đặt Python >= 3.12.

**1. Cài đặt `uv` (nếu chưa có):**
* Trên macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
* Trên Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

**2. Đồng bộ và cài đặt môi trường:**
Mở terminal tại thư mục gốc của dự án (nơi chứa file `pyproject.toml`) và chạy lệnh sau:


```bash
uv sync
```

## 📈 Chạy Mã Nguồn

uv run lstm_model.py

## Luồng Xử Lý Của Code
Mã nguồn được chia thành 6 bước chính:

1. Tải và Tiền xử lý dữ liệu: Xử lý nhiễu, chuẩn hóa dữ liệu về khoảng [0, 1].

2. Chia tập dữ liệu: Phân chia theo trình tự thời gian với tỷ lệ 80% Train - 20% Test. Áp dụng kỹ thuật Cửa sổ trượt (Sliding Window) với time_steps = 24 (dùng 24h trước đó để dự báo 1h tiếp theo).

3. Xây dựng kiến trúc LSTM:

- Lớp LSTM thứ nhất (50 units) + Dropout(0.2)

- Lớp LSTM thứ hai (50 units) + Dropout(0.2)

- Lớp Output Dense (1 unit)

- Huấn luyện mô hình: Sử dụng thuật toán tối ưu Adam và hàm mất mát MSE. Huấn luyện trong 50 Epochs với validation_split=0.1.

- Đánh giá mô hình: Tính toán các độ đo chuẩn mực cho chuỗi thời gian: RMSE (Root Mean Squared Error) và MAPE (Mean Absolute Percentage Error).

- Trực quan hóa: Vẽ biểu đồ Learning Curve và biểu đồ so sánh Thực tế vs. Dự báo cho 200 giờ cuối cùng của tập Test.