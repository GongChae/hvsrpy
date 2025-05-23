import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# === 경로 설정 ===
input_dir = Path("C:/SOLODATA/MAPE")      # 원본 .xlsx 폴더
output_dir = Path("C:/SOLODATA/boxplot")  # 결과 저장 폴더
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

# === 병합된 파일에서 1~3행 제거하고 불러오기
df = pd.read_excel(merged_path, skiprows=3)

# === X축 값과 데이터 분리
x_labels = df.iloc[:, 0]             # 첫 번째 열 → X축 레이블
data = df.iloc[:, 1:]                # 나머지 열 → Boxplot 데이터

# === 행 기준 boxplot을 위해 전치 (transpose)
data_t = data.transpose()

# === 박스플롯 그리기
if not data_t.empty:
    plt.figure(figsize=(14, 6))

    # boxplot 그리기 및 반환 객체 저장
    bp = plt.boxplot(data_t.values, tick_labels=x_labels, vert=True, patch_artist=True)

    # ✅ 모든 박스를 하나의 색상으로 지정
    box_color = "skyblue"  # 원하는 색상: 예) "lightcoral", "#66c2a5", "lightgray" 등
    for patch in bp['boxes']:
        patch.set_facecolor(box_color)

    # === 기타 설정
    plt.title("HVSR MAPE by Record Duration (Row-wise Boxplot)")
    plt.xlabel("Record Duration (X values)")
    plt.ylabel("MAPE")
    plt.xticks(rotation=45)
    plt.ylim(0, 0.5)
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("❌ boxplot을 그릴 수 있는 유효한 숫자 데이터가 없습니다.")