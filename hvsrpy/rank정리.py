import pandas as pd
from pathlib import Path

# === 경로 설정 ===
input_file = Path("C:/Users/USER/Desktop/전체/merged_pass개수.xlsx")  # 입력 파일

# === 파일 읽기 ===
df = pd.read_excel(input_file)

# === 5열 값 기준으로 상위 5개 행 추출 ===
top5_rows = df.nlargest(5, df.columns[4])  # 5열 = 인덱스 4

# === 1~10열 선택 ===
output_df = top5_rows.iloc[:, 0:4]

# === 출력 ===
print("✅ 상위 5개 행 (1~10열):")
print(output_df)
