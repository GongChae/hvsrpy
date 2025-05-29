import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === 경로 설정 ===
input_dir = Path("C:/SOLODATA/MAPE")
output_dir = Path("C:/SOLODATA/boxplot")
output_dir.mkdir(parents=True, exist_ok=True)

# === 모든 .xlsx 파일 목록 수집
files = sorted(input_dir.glob("*.xlsx"))
if not files:
    raise FileNotFoundError("📁 .xlsx 파일이 존재하지 않습니다.")

# === C열 데이터 모으기 (concat 방식)
col_list = []
for idx, file in enumerate(files):
    df = pd.read_excel(file, header=None)
    if df.shape[1] < 3:
        print(f"⚠️ {file.name} - C열 없음, 건너뜀")
        continue
    col_list.append(df.iloc[:, 2].rename(f"File{idx+1}"))

# === X축 값 추출 + 합치기
df0 = pd.read_excel(files[0], header=None)
x_values = df0.iloc[:, 0].rename("X").reset_index(drop=True)
result_df = pd.concat([x_values] + col_list, axis=1)

# === 병합 결과 저장
merged_path = output_dir / "merged_result.xlsx"
result_df.to_excel(merged_path, index=False)
print(f"✅ 병합 완료: {merged_path}")

# === IQR 계산 (행 기준)
iqr_list = []
for idx, row in result_df.iterrows():
    row_values = pd.to_numeric(row[1:], errors='coerce').dropna().values  # 첫 열(X값) 제외
    if len(row_values) >= 4:
        q1 = np.percentile(row_values, 25)
        q3 = np.percentile(row_values, 75)
        iqr = q3 - q1
        iqr_list.append({'X': row['X'], 'Q1': q1, 'Q3': q3, 'IQR': iqr})
        print(f"✅ X '{row['X']}' → Count: {len(row_values)}, Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
    else:
        iqr_list.append({'X': row['X'], 'Q1': None, 'Q3': None, 'IQR': None})
        print(f"⚠️ X '{row['X']}' → 데이터 수 부족 (Count: {len(row_values)}), IQR 계산 생략")

# === IQR 결과 DataFrame
iqr_df = pd.DataFrame(iqr_list)

# === 결과 저장
iqr_output_path = output_dir / "iqr_values_by_row.xlsx"
iqr_df.to_excel(iqr_output_path, index=False)
print(f"✅ 행별 IQR 값이 {iqr_output_path} 에 저장되었습니다.")

# === 행 기준 박스플롯 그리기 (X값 기준)
data_for_plot = result_df.drop(columns='X').transpose()  # 첫 열(X) 제거 후 전치

plt.figure(figsize=(14, 6))
plt.boxplot(data_for_plot.values, labels=result_df['X'], vert=True, patch_artist=True)

# 스타일
box_color = "skyblue"
for patch in plt.gca().artists:
    patch.set_facecolor(box_color)

plt.xlabel('X 값', fontsize=14)
plt.ylabel('값 분포', fontsize=14)
plt.title('각 X 값별 박스플롯 (행 기준)', fontsize=16)
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
