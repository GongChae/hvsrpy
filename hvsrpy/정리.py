import pandas as pd
from pathlib import Path

# === 경로 설정 ===
input_folder = Path("C:/Users/USER/Desktop/전체/모으기")  # XLSX 파일들이 있는 폴더
output_file = Path("C:/Users/USER/Desktop/전체/merged.xlsx")  # 저장할 파일 경로

# === 폴더 안의 모든 .xlsx 파일 찾기 ===
xlsx_files = sorted(input_folder.glob("*.xlsx"))
if not xlsx_files:
    raise FileNotFoundError("❌ 폴더에 .xlsx 파일이 없습니다.")

# === 첫 번째 파일 기준으로 초기 데이터프레임 생성 ===
base_df = pd.read_excel(xlsx_files[0], header=None)

# 1행 전체 + 2행부터 1~4열만 추출
header_row = base_df.iloc[0:1, :]
base_data = base_df.iloc[1:, 0:4].copy()

# 8열 값(= 7번째 index, 0 기반)부터 붙이기
for idx, file in enumerate(xlsx_files):
    df = pd.read_excel(file, header=None)

    if df.shape[1] < 8:
        print(f"⚠️ {file.name}: 8열이 없어 건너뜀")
        continue

    col_data = df.iloc[1:, 7]  # 2행부터 8열 (index 7)
    base_data[f"File{idx+1}"] = col_data.values

# === 최종 데이터프레임 구성 ===
final_df = pd.concat([header_row, base_data], ignore_index=True)

# === 결과 저장 ===
final_df.to_excel(output_file, index=False, header=False)
print(f"✅ 병합 완료: {output_file}")
