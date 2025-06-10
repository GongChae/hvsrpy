import pandas as pd
from pathlib import Path

# === 경로 설정 ===
input_dir = Path("C:/SOLODATA/MAPE")  # 여기 폴더에 1번 형식 엑셀들 모아놓음
output_file = Path("C:/SOLODATA/boxplot/mape_merged.xlsx")  # 최종 결과 저장 경로

# === 파일 수집 ===
files = sorted(input_dir.glob("*.xlsx"))
if not files:
    raise FileNotFoundError("❌ .xlsx 파일이 폴더에 없습니다.")

# === 첫 번째 파일에서 첫 번째 열 추출
df_first = pd.read_excel(files[0], header=None)
first_column = df_first.iloc[:, 0].reset_index(drop=True)
merged_df = pd.DataFrame({"X": first_column})

# === 각 파일의 세 번째 열만 추가
for idx, file in enumerate(files):
    df = pd.read_excel(file, header=None)
    if df.shape[1] < 3:
        print(f"⚠️ {file.name}: 3열이 없어 건너뜀")
        continue
    col_data = df.iloc[:, 2].reset_index(drop=True)
    merged_df[f"File{idx+1}"] = col_data

# === 결과 저장
merged_df.to_excel(output_file, index=False)
print(f"✅ 병합 완료: {output_file}")
