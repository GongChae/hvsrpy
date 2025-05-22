import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === 경로 설정 ===
input_dir = Path("C:/SOLODATA/MAPE")      # 원본 .xlsx 폴더
output_dir = Path("C:/SOLODATA/boxplot") # 결과 저장 폴더
output_dir.mkdir(parents=True, exist_ok=True)

# === 모든 .xlsx 파일 목록 수집 ===
files = sorted(input_dir.glob("*.xlsx"))
if not files:
    raise FileNotFoundError("📁 .xlsx 파일이 존재하지 않습니다.")

# === 첫 번째 파일에서 x축 (A열) 추출
df0 = pd.read_excel(files[0], header=None)
x_values = df0.iloc[:, 0]  # 첫 번째 열
result_df = pd.DataFrame()
result_df["X"] = x_values  # 첫 열: X축

# === 나머지 파일에서 3열(C열) 값 모으기
for idx, file in enumerate(files):
    df = pd.read_excel(file, header=None)

    if df.shape[1] < 3:
        print(f"⚠️ {file.name} - C열 없음, 건너뜀")
        continue

    col_name = f"File{idx+1}"
    result_df[col_name] = df.iloc[:, 2]  # C열 추가

# === 결과 저장
merged_path = output_dir / "merged_result.xlsx"
result_df.to_excel(merged_path, index=False)
print(f"✅ 병합 완료: {merged_path}")

# === Boxplot 그리기 (X축: 첫 번째 열, 데이터: 나머지 열)
plt.figure(figsize=(12, 6))
result_df.iloc[:, 1:].boxplot()
plt.title("HVSR MAPE by Record duration")
plt.xlabel("Record duration")
plt.ylabel("Boxplot")
plt.grid(True)
plt.tight_layout()
plt.show()
