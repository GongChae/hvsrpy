import pandas as pd
from pathlib import Path

# === 경로 설정 ===
input_dir = Path("C:/SOLODATA/zonghap/hvsr_try/36dB")      # 원본 .xlsx 폴더
output_dir = Path("C:/SOLODATA/MAPE") # 결과 저장 폴더
output_dir.mkdir(parents=True, exist_ok=True)

for file in input_dir.glob("*.xlsx"):
    try:
        df = pd.read_excel(file)

        # B열(2번째 열)이 존재하는지 확인
        if df.shape[1] < 2:
            print(f"❌ 열 개수 부족 (B열 없음): {file.name} → 건너뜀")
            continue

        # B열 값이 모두 숫자인지 확인
        if not pd.to_numeric(df.iloc[:, 1], errors='coerce').notna().all():
            print(f"❌ B열에 숫자가 아닌 값 존재: {file.name} → 건너뜀")
            continue

        # B2 기준값
        b2 = df.iat[1, 1]
        if not isinstance(b2, (int, float)) or b2 == 0:
            print(f"❌ B2 값이 유효하지 않음: {file.name} → 건너뜀")
            continue

        # C열(3번째 열) 생성 또는 갱신
        while df.shape[1] < 3:
            df[df.shape[1]] = None  # C열이 없으면 추가
            df.columns.values[2] = "C"  # 3번째 열 이름을 'C'로 지정

        # B3부터 C3 계산
        for i in range(1, len(df)):  # B2부터!

            bn = df.iat[i, 1]
            if isinstance(bn, (int, float)):
                result = abs((bn - b2) / b2)
                df.iat[i, 2] = result  # C열

        # 저장
        output_path = output_dir / file.name
        df.to_excel(output_path, index=False)
        print(f"✅ 처리 완료: {file.name}")

    except Exception as e:
        print(f"❌ 처리 실패: {file.name} → {e}")