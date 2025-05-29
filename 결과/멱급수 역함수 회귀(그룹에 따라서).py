import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.font_manager as fm

# 한글 폰트 설정 (맑은 고딕)
font_path = 'C:/Windows/Fonts/malgun.ttf'
fontprop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=fontprop.get_name())
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
file_path = Path("C:/Users/USER/Desktop/전체/0529_암반추가.csv")
df = pd.read_csv(file_path, encoding='cp949')

# 유효 데이터 필터링
valid_mask = (df.iloc[:, 4] > 0) & (df.iloc[:, 3] > 2) & (~df.iloc[:, 5].isna())
df_valid = df.loc[valid_mask]

# 6열 값 기준 그룹화
groups_6th = df_valid.groupby(df_valid.columns[5])

# 그룹별 처리
for group_name, group_df in groups_6th:
    x = group_df.iloc[:, 3].values  # 4열: peak_frequency
    y = group_df.iloc[:, 4].values  # 5열: 기반암심도

    try:
        # 로그 변환 (ln x, ln y)
        log_x = np.log(x).reshape(-1, 1)
        log_y = np.log(y)

        # 선형 회귀 (역비례 멱급수: y = a * x^{-b})
        model = LinearRegression()
        model.fit(log_x, log_y)

        # 계수 추출
        b = -model.coef_[0]
        ln_a = model.intercept_
        a = np.exp(ln_a)

        # 예측값
        y_pred = a * (x ** (-b))

        # R² 계산
        r2 = r2_score(y, y_pred)

        print(f"🌟 그룹 '{group_name}' 멱급수 역비례 모델: y = {a:.4f} * x^(-{b:.4f}), R² = {r2:.4f}")

        # 예측 곡선용 데이터
        x_pred = np.linspace(x.min(), x.max(), 100)
        y_pred_curve = a * (x_pred ** (-b))

        # 그래프 출력
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, label='실제 데이터', color='blue')
        plt.plot(x_pred, y_pred_curve, color='red', label=f'모델: y = {a:.2f} * x^(-{b:.2f}),  R² = {r2:.4f}')
        plt.xlabel('Peak Frequency (입력)')
        plt.ylabel('기반암심도 (정답)')
        plt.title(f"그룹 '{group_name}'")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"⚠️ 그룹 '{group_name}'에서 모델 피팅 실패: {e}")
