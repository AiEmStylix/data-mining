import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
# PREPROCESSING
print("Đang tải và xử lý dữ liệu PJM East...")

df = pd.read_csv('dataset/PJME_hourly.csv', index_col=[0], parse_dates=[0])
df = df.sort_index()

# Remove duplicate by (Daylight Saving Time - Bẫy dữ liệu)
# Keep the first record
df = df[~df.index.duplicated(keep='first')]

df = df.dropna()

data = df['PJME_MW'].values.reshape(-1, 1)

# Chuẩn hóa dữ liệu về khoảng [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)



print("2. Đang chia tập dữ liệu và tạo cửa sổ trượt (Sliding Window)...")

# Chia 80% Train - 20% Test theo trình tự thời gian
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size:, :]

def create_lstm_dataset(dataset, time_steps=24):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), 0])  # Input: 24 giờ quá khứ
        y.append(dataset[i + time_steps, 0])      # Target: Giờ tiếp theo
    return np.array(X), np.array(y)

# Chọn Time_Steps = 24 để nắm bắt chu kỳ ngày đêm
time_steps = 24 

X_train, y_train = create_lstm_dataset(train_data, time_steps)
X_test, y_test = create_lstm_dataset(test_data, time_steps)

# Reshape thành Tensor 3D cho LSTM: [Samples, Time_Steps, Features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ==========================================
# BƯỚC 3: XÂY DỰNG KIẾN TRÚC MẠNG LSTM
# ==========================================
print("3. Đang khởi tạo mạng lưới LSTM...")
model = Sequential()

# Layer 1 (Stacked Layer)
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2)) # Tắt 20% nơ-ron để chống Overfitting

# 2nd LSTM layer (Lớp ẩn sâu)
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Output
model.add(Dense(units=1))

# Adam và hàm mất mát MSE
model.compile(optimizer='adam', loss='mean_squared_error')
# ==========================================

# ==========================================
print("\n4. BẮT ĐẦU HUẤN LUYỆN (Vui lòng đợi vài phút)...\n")
start_time = time.time()

# 50 vòng lặp (epochs)
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=64, 
    validation_split=0.1, 
    verbose=1,
)

print(f"\n[HOÀN TẤT] Thời gian huấn luyện: {(time.time() - start_time) / 60:.2f} phút")

# ==========================================
print("\n5. Đang dự báo trên tập Test và tính toán sai số...")

# Dự báo trên tập Test
predictions_scaled = model.predict(X_test)

# Giải chuẩn hóa
predictions = scaler.inverse_transform(predictions_scaled)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Tính toán các độ đo sai số chuẩn mực
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mape = mean_absolute_percentage_error(y_test_actual, predictions) * 100

print(f"--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH LSTM ---")
print(f"RMSE: {rmse:.2f} MW (Sai lệch trung bình)")
print(f"MAPE: {mape:.2f}% (Sai số phần trăm)\n")


# ==========================================
# 6. VISUALIZATION (GOM TẤT CẢ BIỂU ĐỒ VÀO ĐÂY)
# ==========================================
print("6. Đang hiển thị các biểu đồ...")

# Lấy tổng số epoch thực tế mà mô hình đã chạy
actual_epochs = len(history.history['loss'])

# --- BIỂU ĐỒ 1: LEARNING CURVE CÓ ĐÁNH DẤU ---
plt.figure(figsize=(10, 5))
train_loss = history.history['loss']
val_loss = history.history['val_loss']

min_train_loss = np.min(train_loss)
best_epoch_train = np.argmin(train_loss)
min_val_loss = np.min(val_loss)
best_epoch_val = np.argmin(val_loss)

plt.plot(train_loss, label='Train MSE', color='blue', marker='o', markersize=4)
plt.plot(val_loss, label='Validation MSE', color='red', marker='o', markersize=4)

plt.scatter(best_epoch_train, min_train_loss, color='cyan', s=100, zorder=5, edgecolor='black', 
            label=f'Min Train MSE: {min_train_loss:.5f} (Epoch {best_epoch_train + 1})')
plt.scatter(best_epoch_val, min_val_loss, color='gold', marker='*', s=200, zorder=5, edgecolor='black',
            label=f'Min Val MSE: {min_val_loss:.5f} (Epoch {best_epoch_val + 1})')
plt.axvline(x=best_epoch_val, color='gray', linestyle='--', alpha=0.6)

plt.title(f'Đồ thị suy giảm sai số (Learning Curve) hoàn chỉnh sau {actual_epochs} Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Vòng lặp (Epoch)', fontsize=12)
plt.ylabel('Sai số (MSE)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)


step_size = max(1, actual_epochs // 10)
plt.xticks(np.arange(0, actual_epochs, step=step_size), np.arange(1, actual_epochs + 1, step=step_size)) 

# --- BIỂU ĐỒ 2: ACTUAL VS PREDICTED ---
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual[-200:], color='blue', label='Lượng điện Thực tế (Actual)', linewidth=2)
plt.plot(predictions[-200:], color='red', label='Dự báo của LSTM (Predicted)', linestyle='dashed', linewidth=2)
plt.title('So sánh Lượng điện tiêu thụ: Thực tế vs Dự báo (200 giờ cuối của tập Test)', fontsize=14, fontweight='bold')
plt.xlabel('Thời gian (Giờ)', fontsize=12)
plt.ylabel('Điện năng tiêu thụ (Megawatt)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

print("\nHoàn tất quá trình mô phỏng dữ liệu! (Vui lòng xem các cửa sổ đồ thị)")

plt.show()