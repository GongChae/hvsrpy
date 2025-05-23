import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import hvsrpy
from hvsrpy import sesame
import pandas as pd  # XLSX 저장용

plt.style.use(hvsrpy.HVSRPY_MPL_STYLE)

# === 입력 및 출력 폴더 설정 ===
data_dir = Path("C:/SOLODATA/zonghap/3month/36dB")  # .sac 파일들이 들어있는 폴더
save_root = Path("C:/SOLODATA/zonghap/hvsr_try")
output_dir = save_root / data_dir.name
output_dir.mkdir(parents=True, exist_ok=True)

# === .sac 파일 그룹화 (Z/N/E) ===
sac_files = list(data_dir.glob("*.sac"))
grouped_files = defaultdict(dict)
for file in sac_files:
    parts = file.stem.split(".")
    if len(parts) < 2:
        continue
    base_id = ".".join(parts[:-1])
    comp = parts[-1].upper()
    grouped_files[base_id][comp] = str(file)

# === 유효한 세트만 추출 ===
fname_sets = []
for base, comps in grouped_files.items():
    if all(k in comps for k in ["E", "N", "Z"]):
        fname_sets.append((base, [comps["Z"], comps["N"], comps["E"]]))

if not fname_sets:
    print("❌ E/N/Z 세트를 찾을 수 없습니다.")
    exit()

# === HVSR 설정 ===
preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
preprocessing_settings.detrend = "linear"
preprocessing_settings.window_length_in_seconds = 30
preprocessing_settings.orient_to_degrees_from_north = 0.0
preprocessing_settings.filter_corner_frequencies_in_hz = (0.1, 20)
preprocessing_settings.ignore_dissimilar_time_step_warning = False

processing_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
processing_settings.window_type_and_width = ("tukey", 0.1)
processing_settings.smoothing = dict(
    operator="konno_and_ohmachi",
    bandwidth=40,
    center_frequencies_in_hz=np.geomspace(0.1, 20, 200)
)
processing_settings.method_to_combine_horizontals = "total_horizontal_energy"
processing_settings.handle_dissimilar_time_steps_by = "frequency_domain_resampling"

# === 각 세트 처리 ===
for base_id, fnames in fname_sets:
    print(f"✅ 처리 중: {base_id}")

    try:
        srecords = hvsrpy.read([fnames])
        ts_sample = getattr(srecords[0], "vt")
        end_time = ts_sample.time()[-1]
        intervals = list(range(3600, 299, -60))

        results = []

        for interval in intervals:
            start_time = end_time - interval
            srecords_copy = deepcopy(srecords)

            try:
                for rec in srecords_copy:
                    for comp in ("ns", "ew", "vt"):
                        ts = getattr(rec, comp)
                        ts.trim(start_time, end_time)

                srecords_pre = hvsrpy.preprocess(srecords_copy, preprocessing_settings)

                passing = hvsrpy.sta_lta_window_rejection(
                    srecords_pre,
                    sta_seconds=1,
                    lta_seconds=30,
                    min_sta_lta_ratio=0.1,
                    max_sta_lta_ratio=2.5,
                    hvsr=None
                )

                hvsr = hvsrpy.process(passing, processing_settings)

                if hvsr and hasattr(hvsr, 'mean_curve'):
                    mean_curve = hvsr.mean_curve()
                    f0 = hvsr.frequency[np.argmax(mean_curve)]
                    results.append([interval, f"{f0:.5f}"])
                else:
                    results.append([interval, "N/A"])

            except Exception as e:
                results.append([interval, f"error: {str(e)}"])

        # 결과 저장: 선택 – CSV 또는 XLSX
        save_as_xlsx = True  # ← True: xlsx로 저장, False: csv로 저장

        output_file = output_dir / f"{base_id}_hvsr_peaks"
        if save_as_xlsx:
            df = pd.DataFrame(results, columns=["time_interval_sec", "peak_frequency_Hz"])
            df.to_excel(output_file.with_suffix(".xlsx"), index=False)
            print(f"📁 XLSX 저장 완료: {output_file.with_suffix('.xlsx')}")
        else:
            with open(output_file.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time_interval_sec", "peak_frequency_Hz"])
                writer.writerows(results)
            print(f"📁 CSV 저장 완료: {output_file.with_suffix('.csv')}")

    except Exception as e:
        print(f"❌ 처리 실패: {base_id} → {e}")
