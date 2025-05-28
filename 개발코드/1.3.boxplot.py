import pandas as pd
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

# === 병합된 파일에서 1~3행 제거 후 불러오기
df = pd.read_excel(merged_path, skiprows=3)

# === X축 값과 데이터 분리
x_labels = df.iloc[:, 0].reset_index(drop=True)
data = df.iloc[:, 1:]
data_t = data.transpose()

# === IQR 계산 (숫자 강제 변환 + 적은 데이터 체크)
iqr_records = []

for idx, col in enumerate(data.columns):
    col_data = pd.to_numeric(data[col], errors='coerce').dropna()
    unique_vals = col_data.unique()
    count = len(col_data)

    if count >= 4 and idx < len(x_labels):  # 최소 4개 이상 있어야 분위 계산 의미 있음
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        x_label_value = x_labels.iloc[idx]
        iqr_records.append({"X": x_label_value, "IQR": iqr})
        print(f"✅ X값 '{x_label_value}' → Count: {count}, Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
    else:
        x_label_value = x_labels.iloc[idx]
        iqr_records.append({"X": x_label_value, "IQR": None})
        print(f"⚠️ X값 '{x_label_value}' → 데이터 수 부족 (Count: {count}), IQR 계산 생략")

# === IQR 결과 저장
iqr_df = pd.DataFrame(iqr_records)
iqr_output_path = output_dir / "iqr_results.xlsx"
iqr_df.to_excel(iqr_output_path, index=False)
print(f"✅ IQR 결과 저장 완료: {iqr_output_path}")

# === 박스플롯 그리기
if not data_t.empty:
    plt.figure(figsize=(14, 6))
    bp = plt.boxplot(data_t.values, tick_labels=x_labels, vert=True, patch_artist=True)
    box_color = "skyblue"
    for patch in bp['boxes']:
        patch.set_facecolor(box_color)

    plt.title("HVSR MAPE by Record Duration (Row-wise Boxplot)")
    plt.xlabel("Record Duration (X values)")
    plt.ylabel("MAPE")
    plt.xticks(rotation=45)
    plt.ylim(0, 0.2)
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("❌ boxplot을 그릴 수 있는 유효한 숫자 데이터가 없습니다.")
