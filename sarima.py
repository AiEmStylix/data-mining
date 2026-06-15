import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pmdarima as pm
import matplotlib

warnings.filterwarnings("ignore")
matplotlib.use('WebAgg')

# ==========================================
# 1. PREPROCESSING
# ==========================================
print("1. Đang tải và xử lý dữ liệu PJM East...")
df = pd.read_csv('dataset/PJME_hourly.csv', index_col=[0], parse_dates=[0])
df = df.sort_index()

# Lọc bỏ các dòng trùng lặp và nội suy dữ liệu khuyết
df = df[~df.index.duplicated(keep='first')]
df = df.asfreq('h')
df = df.interpolate(method='linear')

# GIỚI HẠN DỮ LIỆU XUỐNG CÒN 1 TÁNG (30 ngàHy) ĐỂ AUTO ARIMA CHẠY NHANH
subset_hours = 24 * 30 
df_subset = df.tail(subset_hours)
data = df_subset['PJME_MW']

# Để lại 7 ngày cuối (168 giờ) làm Test
test_hours = 24 * 7
train_size = len(data) - test_hours 
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

print(f"Số lượng mẫu Train: {len(train_data)} giờ (23 ngày)")
print(f"Số lượng mẫu Test: {len(test_data)} giờ (7 ngày)")

# ==========================================
# 2. XÂY DỰNG & HUẤN LUYỆN BẰNG AUTO ARIMA
# ==========================================
print("\n" + "="*70)
print("2. BẮT ĐẦU TÌM KIẾM THAM SỐ VỚI AUTO ARIMA (TẬP DỮ LIỆU 1 THÁNG)")
print("Hệ thống sẽ thử nghiệm nhiều tổ hợp (p,d,q). Vui lòng xem log...")
print("="*70 + "\n")
start_time = time.time()

auto_model = pm.auto_arima(
    train_data, 
    start_p=0, start_q=0,       
    max_p=2, max_q=2,           # Giảm max_p, max_q xuống 2 để tăng tốc
    m=24,                       # Chu kỳ mùa vụ s=24
    start_P=0, start_Q=0,
    max_P=1, max_Q=1,           # Giới hạn tham số mùa vụ tối đa là 1
    seasonal=True,              
    d=1, D=1,                   
    trace=True,                 # Hiển thị log
    error_action='ignore',  
    suppress_warnings=True, 
    stepwise=True               
)

print(f"\n[HOÀN TẤT] Thời gian dò tìm và huấn luyện: {(time.time() - start_time) / 60:.2f} phút")

print("\n" + "="*50)
print("BỘ THAM SỐ TỐT NHẤT (BEST MODEL SUMMARY)")
print("="*50)
print(auto_model.summary())
print("\n")

# ==========================================
# 3. DỰ BÁO VÀ ĐÁNH GIÁ SAI SỐ
# ==========================================
print("3. Đang dự báo trên tập Test và tính toán sai số...")

predictions = auto_model.predict(n_periods=len(test_data))
predictions.index = test_data.index

rmse = np.sqrt(mean_squared_error(test_data, predictions))
mape = mean_absolute_percentage_error(test_data, predictions) * 100

print(f"--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH AUTO ARIMA ---")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.2f}%\n")

# ==========================================
# 4. VISUALIZATION
# ==========================================
print("4. Đang hiển thị biểu đồ...")

plt.figure(figsize=(14, 6))
plt.plot(test_data.index, test_data.values, color='blue', label='Lượng điện Thực tế (Actual)', linewidth=2)
plt.plot(test_data.index, predictions, color='red', label='Dự báo của Auto ARIMA', linestyle='dashed', linewidth=2)

plt.title('Auto ARIMA Baseline: Thực tế vs Dự báo (7 ngày cuối)', fontsize=14, fontweight='bold')
plt.xlabel('Thời gian', fontsize=12)
plt.ylabel('Điện năng tiêu thụ (Megawatt)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.gcf().autofmt_xdate()

print("\nHoàn tất! Bấm Ctrl+C trên terminal để dừng WebAgg server khi xem xong đồ thị.")
plt.show()