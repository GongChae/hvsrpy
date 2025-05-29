import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.font_manager as fm

# 한글 폰트 설정 (맑은 고딕)
font_path = 'C:/Windows/Fonts/malgun.ttf'
fontprop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=fontprop.get_name())
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
file_path = 'C:/Users/USER/Desktop/전체/0529_암반추가_조작.csv'  # 경로 수정하세요
df = pd.read_csv(file_path, encoding='cp949')

# 4열, 5열 추출
x_raw = df.iloc[:, 3]
y_raw = df.iloc[:, 4]

# 같은 y값 그룹으로 묶어서 x값 평균 계산
grouped = df.groupby(df.iloc[:, 4])
x_means = grouped.apply(lambda g: g.iloc[:, 3].mean()).values
y_unique = grouped.apply(lambda g: g.iloc[:, 4].iloc[0]).values

# numpy 배열
x = np.array(x_means)
y = np.array(y_unique)

# x > 0만 사용
mask = x > 0
x_filtered = x[mask]
y_filtered = y[mask]

# 역함수 모델: y = a / (x + c)^b
def inverse_power_func(x, a, b, c):
    return a / np.power(x + c, b)

# 초기값
initial_guess = [1.0, 1.0, 1.0]

# bounds: a ≥ 0, b > 0, c 자유
param_bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

try:
    popt, _ = curve_fit(
        inverse_power_func,
        x_filtered, y_filtered,
        p0=initial_guess,
        bounds=param_bounds,
        maxfev=50000
    )
    a, b, c = popt

    # 예측값
    y_pred = inverse_power_func(x_filtered, a, b, c)

    # R²
    r2 = r2_score(y_filtered, y_pred)
    print(f"🌟 피팅 결과 (역함수형): y = {a:.4f} / (x + {c:.4f})^{b:.4f}")
    print(f"🌟 R² = {r2:.4f}")

    # 예측 곡선
    x_line = np.linspace(x_filtered.min(), x_filtered.max(), 200)
    y_line = inverse_power_func(x_line, a, b, c)

    # 그래프
    plt.figure(figsize=(8, 6))
    plt.scatter(x_filtered, y_filtered, label='평균 데이터 (그룹당 하나)', color='blue', alpha=0.8)
    plt.plot(x_line, y_line, color='red', label=f'회귀: y={a:.2f}/(x{c:.2f})^{b:.2f}\nR²={r2:.4f}')
    plt.xlabel('입력 데이터 (4열 평균)')
    plt.ylabel('정답 데이터 (5열)')
    plt.title('역함수형 회귀 (그룹 평균값 사용)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except RuntimeError as e:
    print(f"⚠️ 피팅 실패: {e}")
