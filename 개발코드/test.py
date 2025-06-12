import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from scipy.stats import iqr
import hvsrpy

# === 폰트 설정 (한글 깨짐 방지) ===
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 환경
plt.rcParams['axes.unicode_minus'] = False

# === 설정 ===
data_root = Path("C:/SOLODATA/zonghap/3month/종합설계")
save_root = Path("C:/SOLODATA/MAPE_Results")
save_root.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

duration_sec = list(range(300, 3600, 300))

preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
preprocessing_settings.detrend = "linear"
preprocessing_settings.window_length_in_seconds = 30
preprocessing_settings.orient_to_degrees_from_north = 0.0
preprocessing_settings.filter_corner_frequencies_in_hz = (0.1, 20)

processing_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
processing_settings.window_type_and_width = ("tukey", 0.1)
processing_settings.smoothing = dict(
    operator="konno_and_ohmachi",
    bandwidth=40,
    center_frequencies_in_hz=np.geomspace(0.5, 20, 200)
)
processing_settings.method_to_combine_horizontals = "total_horizontal_energy"
processing_settings.handle_dissimilar_time_steps_by = "frequency_domain_resampling"

def compute_mape(true_val, predictions):
    preds = np.array(predictions, dtype=float)
    return np.mean(np.abs((true_val - preds) / true_val)) * 100

# === 📦 모든 .sac 파일에서 Z/N/E 세트 구성 ===
grouped_files = defaultdict(dict)

for sac_file in data_root.rglob("*.sac"):
    parts = sac_file.stem.split(".")
    if len(parts) < 2:
        continue
    base_id = ".".join(parts[:-1])
    comp = parts[-1].upper()
    grouped_files[base_id][comp] = str(sac_file)

fname_sets = []
for base, comps in grouped_files.items():
    if all(k in comps for k in ["Z", "N", "E"]):
        fname_sets.append((base, [comps["Z"], comps["N"], comps["E"]]))

if not fname_sets:
    print("❌ E/N/Z 세트를 찾을 수 없습니다.")
    exit()

print(f"✅ 총 유효한 세트 수: {len(fname_sets)}")

# === 분석 수행 ===
all_mape = defaultdict(list)

for base_id, fnames in fname_sets:
    print(f"\n🔍 {base_id} 처리 중...")
    try:
        srecords = hvsrpy.read([fnames])
        ts_sample = getattr(srecords[0], "vt")
        end_time = ts_sample.time()[-1]

        # 기준값 (60분) 계산
        srecords_60 = deepcopy(srecords)
        for rec in srecords_60:
            for comp in ("ns", "ew", "vt"):
                getattr(rec, comp).trim(end_time - 3600, end_time)
        srecords_60 = hvsrpy.preprocess(srecords_60, preprocessing_settings)
        hvsr_60 = hvsrpy.process(srecords_60, processing_settings)
        true_fn = hvsr_60.mean_fn_frequency()
        print(f" 기준 fn: {true_fn:.3f} Hz")

        # duration 별 처리
        for dur in duration_sec:
            s_copy = deepcopy(srecords)
            for rec in s_copy:
                for comp in ("ns", "ew", "vt"):
                    getattr(rec, comp).trim(end_time - dur, end_time)
            try:
                s_pre = hvsrpy.preprocess(s_copy, preprocessing_settings)
                hvsr_d = hvsrpy.process(s_pre, processing_settings)
                pred_fn = hvsr_d.mean_fn_frequency()
                mape = compute_mape(true_fn, [pred_fn])
                all_mape[dur].append(mape)
                print(f"  - {dur}초: fn={pred_fn:.3f} Hz → MAPE={mape:.2f}%")
            except Exception as e:
                print(f"  - {dur}초 실패: {e}")
    except Exception as e:
        print(f"❌ 처리 실패: {base_id} → {e}")

# === 결과 저장 및 시각화 ===
mape_df = pd.DataFrame({f"{k}초": v for k, v in all_mape.items()})
mape_df = mape_df.dropna(axis=1, how='all')
mape_df = mape_df.select_dtypes(include=[np.number])

if not mape_df.empty:
    plt.figure(figsize=(12, 6))
    mape_df.boxplot()
    plt.title("측정 기간 별 Boxplot 그래프",fontsize=20)
    plt.ylabel("MAPE (%)",fontsize=16)
    plt.xlabel("측정 기간 (초)",fontsize=16)
    plt.ylim(0, 80)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_root / "mape_boxplot.png")
    plt.show()

    iqr_series = mape_df.apply(iqr, axis=0)
    iqr_series.to_csv(save_root / "MAPE_IQR.csv", header=["IQR"])
    mape_df.to_csv(save_root / "MAPE_raw_values.csv", index=False)
    print("\n✅ 모든 작업이 완료되었습니다. 결과는 다음 폴더에 저장되었습니다:")
    print(save_root)
else:
    print("\n❗ 유효한 MAPE 데이터가 없어 결과를 저장하지 않았습니다.")