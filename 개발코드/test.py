import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (맑은 고딕)
font_path = 'C:/Windows/Fonts/malgun.ttf'
fontprop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=fontprop.get_name())
plt.rcParams['axes.unicode_minus'] = False

# === 1️⃣ merged_result.xlsx 불러오기
file_path = 'C:/SOLODATA/boxplot/merged_result.xlsx'
df = pd.read_excel(file_path)

# === 박스플롯 먼저 ===

# X축 레이블 준비
x_labels = df.iloc[:, 0].astype(str)

# 값 데이터 (행 기준 박스플롯, transpose 필요)
data = df.iloc[:, 1:]
data_t = data.transpose()

# 숫자형 강제 변환
data_numeric = data_t.apply(pd.to_numeric, errors='coerce')

# 왼쪽 두 범주 제외
x_labels_for_plot = x_labels[2:].reset_index(drop=True)
data_for_plot = data_numeric.iloc[:, 2:]

# === X축 범주 및 데이터 반전 ===
x_labels_reversed = x_labels_for_plot[::-1].reset_index(drop=True)
data_reversed = data_for_plot.iloc[:, ::-1]  # 열 순서 반대로

plt.figure(figsize=(14, 6))
bp = plt.boxplot(data_reversed.values, tick_labels=x_labels_reversed, vert=True, patch_artist=True, showmeans=True,
                 showfliers=True,  # 이상치 표시
                 meanprops={'markerfacecolor': 'red', 'markeredgecolor': 'red', 'marker': '^'})

for patch in bp['boxes']:
    patch.set_facecolor('skyblue')

plt.xlabel('Record Duration', fontsize=14)
plt.ylabel('값 분포', fontsize=14)
plt.title('Record Duration별 박스플롯 (왼쪽 두 범주 제외, X축 반전)', fontsize=16)
plt.xticks(rotation=45)
plt.ylim(0,0.2)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === 2️⃣ IQR 계산 및 저장 ===
iqr_list = []
for idx, row in df.iterrows():
    row_values = pd.to_numeric(row[1:], errors='coerce').dropna().values  # 첫 열(X값) 제외
    record_duration = row.iloc[0]

    if len(row_values) >= 4:
        q1 = np.percentile(row_values, 25)
        q3 = np.percentile(row_values, 75)
        iqr = q3 - q1
        iqr_list.append({'Record Duration': record_duration, 'Q1': q1, 'Q3': q3, 'IQR': iqr})
        print(
            f"✅ Record Duration '{record_duration}' → Count: {len(row_values)}, Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
    else:
        iqr_list.append({'Record Duration': record_duration, 'Q1': None, 'Q3': None, 'IQR': None})
        print(f"⚠️ Record Duration '{record_duration}' → 데이터 수 부족 (Count: {len(row_values)}), IQR 계산 생략")

iqr_df = pd.DataFrame(iqr_list)
output_iqr_path = 'C:/SOLODATA/boxplot/iqr_values_by_row.xlsx'
iqr_df.to_excel(output_iqr_path, index=False)
print(f"✅ 행별 IQR 값이 {output_iqr_path} 에 저장되었습니다.")
