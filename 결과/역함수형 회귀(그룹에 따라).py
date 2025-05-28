import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 경로 설정 (예: Windows용 나눔고딕)
font_path = 'C:/Windows/Fonts/malgunsl.ttf'
fontprop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=fontprop.get_name())

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
file_path = Path("C:/Users/USER/Downloads/제발되라.csv")
df = pd.read_csv(file_path, encoding='cp949')

# 역함수 모델 정의: y = a / (x + c)
def inverse_func(x, a, c):
    return a / (x + c)

# 필터링
valid_mask = (df.iloc[:, 4] > 0) & (df.iloc[:, 3] > 2) & (~df.iloc[:, 5].isna())
df_valid = df.loc[valid_mask]

# 6열 값 기준 그룹화
groups_6th = df_valid.groupby(df_valid.columns[5])

# 그룹별 처리
for group_name, group_df in groups_6th:
    x = group_df.iloc[:, 3].values  # 4열: peak_frequency
    y = group_df.iloc[:, 4].values  # 5열: 기반암심도

    try:
        popt, _ = curve_fit(inverse_func, x, y, maxfev=10000)
        a, c = popt

        y_pred = inverse_func(x, a, c)
        r2 = r2_score(y, y_pred)

        print(f"🌟 그룹 '{group_name}' 역함수 모델: y = {a:.4f} / (x + {c:.4f}), R² = {r2:.4f}")

        x_pred = np.linspace(x.min(), x.max(), 100)
        y_pred_curve = inverse_func(x_pred, a, c)

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, label='실제 데이터', color='blue')
        plt.plot(x_pred, y_pred_curve, color='red', label=f'모델: y = {a:.2f}/(x + {c:.2f}), R² = {r2:.2f}')
        plt.xlabel('Peak Frequency (입력)')
        plt.ylabel('기반암심도 (정답)')
        plt.title(f"그룹 '{group_name}' 역함수 모델")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"⚠️ 그룹 '{group_name}'에서 모델 피팅 실패: {e}")