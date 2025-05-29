import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 데이터 불러오기
file_path = 'C:/Users/USER/Desktop/전체/0529_암반추가.csv'  # 경로 수정하세요
df = pd.read_csv(file_path)

# 4열, 5열 추출 (0-based index)
x = df.iloc[:, 3].values
y = df.iloc[:, 4].values

# 동일한 y 값에 대해 역가중치 계산
unique_y, counts = np.unique(y, return_counts=True)
weight_map = {val: 1/count for val, count in zip(unique_y, counts)}
weights = np.array([weight_map[val] for val in y])

# 멱급수 역함수 모델 정의: y = a / (x + c)^b
def inverse_power_func(x, a, b, c):
    return a / (x + c) ** b

# 회귀 피팅
popt, _ = curve_fit(inverse_power_func, x, y, sigma=weights, absolute_sigma=False, maxfev=10000)
a, b, c = popt

# 예측값 계산
y_pred = inverse_power_func(x, a, b, c)

# R² 계산
r2 = r2_score(y, y_pred)
print(f"🌟 피팅 결과: y = {a:.4f} / (x + {c:.4f})^{b:.4f}")
print(f"🌟 R² = {r2:.4f}")

# 예측 곡선용 데이터
x_line = np.linspace(x.min(), x.max(), 200)
y_line = inverse_power_func(x_line, a, b, c)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='실제 데이터', color='blue', alpha=0.6)
plt.plot(x_line, y_line, color='red', label=f'회귀: y={a:.2f}/(x+{c:.2f})^{b:.2f}\nR²={r2:.4f}')
plt.xlabel('입력 데이터 (4열)')
plt.ylabel('정답 데이터 (5열)')
plt.title('멱급수 역함수 회귀 (가중치 적용)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
