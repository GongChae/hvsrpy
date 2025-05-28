import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import matplotlib.font_manager as fm

# 한글 폰트 설정 (맑은 고딕)
font_path = 'C:/Windows/Fonts/malgun.ttf'
fontprop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=fontprop.get_name())
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
file_path = Path("C:/Users/USER/Downloads/제발되라.csv")
df = pd.read_csv(file_path, encoding='cp949')

# 유효 데이터 필터링
valid_mask = (df.iloc[:, 4] > 0) & (df.iloc[:, 3] > 2)
df_valid = df.loc[valid_mask]

# 데이터 추출
x = df_valid.iloc[:, 3].values  # 4열: peak_frequency
y = df_valid.iloc[:, 4].values  # 5열: 기반암심도

# 역함수 모델 정의: y = a / (x + c)
def inverse_func(x, a, c):
    return a / (x + c)

try:
    # curve_fit으로 a, c 추정
    popt, _ = curve_fit(inverse_func, x, y, maxfev=10000)
    a, c = popt

    # 예측값
    y_pred = inverse_func(x, a, c)

    # R² 계산
    r2 = r2_score(y, y_pred)

    print(f"🌟 전체 데이터 역함수 모델: y = {a:.4f} / (x + {c:.4f}), R² = {r2:.4f}")

    # 예측 곡선용 데이터
    x_pred = np.linspace(x.min(), x.max(), 100)
    y_pred_curve = inverse_func(x_pred, a, c)

    # 그래프 출력
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='실제 데이터', color='blue')
    plt.plot(x_pred, y_pred_curve, color='red', label=f'모델: y = {a:.2f}/(x + {c:.2f})')
    plt.xlabel('Peak Frequency (입력)')
    plt.ylabel('기반암심도 (정답)')
    plt.title(f"전체 데이터 역함수 모델\n모델: y = {a:.2f} / (x + {c:.2f}),  R² = {r2:.4f}")
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"⚠️ 전체 데이터에서 모델 피팅 실패: {e}")
