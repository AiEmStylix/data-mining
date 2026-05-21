import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Vẽ biểu đồ
plt.plot(x, y, marker='o', color='blue')
plt.title("Test Matplotlib với TkAgg")
plt.grid(True)

# Hiển thị cửa sổ
plt.show()