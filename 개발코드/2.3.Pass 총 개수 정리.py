import pandas as pd
from pathlib import Path

# === 경로 설정 ===
input_file = Path("C:/Users/USER/Desktop/전체/merged.xlsx")   # 입력 파일
output_file = Path("C:/Users/USER/Desktop/전체/merged_pass개수.xlsx")  # 출력 파일

# === 파일 읽기 ===
df = pd.read_excel(input_file)

# === 각 행에서 'Pass' 개수 세기 ===
pass_counts = df.apply(lambda row: (row.astype(str) == "Pass").sum(), axis=1)
df['Pass_Count'] = pass_counts

# === 원하는 열만 선택 ===
# 1~4열 + 마지막 열만 남김
trimmed_df = pd.concat([df.iloc[:, 0:4], df.iloc[:, -1]], axis=1)

# === 결과 저장 ===
trimmed_df.to_excel(output_file, index=False)
print(f"✅ 처리 완료: {output_file}")
