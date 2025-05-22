import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from pathlib import Path

import hvsrpy
from hvsrpy import sesame

plt.style.use(hvsrpy.HVSRPY_MPL_STYLE)

# === 입력 SAC 경로 ===
input_folder = Path("C:/SOLODATA/try")
output_folder = Path("C:/SOLODATA/hvsr_parameter_sweep")
output_folder.mkdir(parents=True, exist_ok=True)

# === E/N/Z 파일 찾기 ===
def find_enz_files(folder_path):
    e_file = glob.glob(str(folder_path / "*.E.sac"))
    n_file = glob.glob(str(folder_path / "*.N.sac"))
    z_file = glob.glob(str(folder_path / "*.Z.sac"))
    if e_file and n_file and z_file:
        return sorted([e_file[0], n_file[0], z_file[0]])
    else:
        return None

enz_files = find_enz_files(input_folder)
if not enz_files:
    raise FileNotFoundError("E/N/Z 파일을 모두 찾을 수 없습니다.")

# === HVSR 설정값 반복 범위 ===
window_lengths = range(5, 61, 5)
taper_ratios = np.round(np.arange(0.0, 1.01, 0.1), 2)
bandwidths = range(10, 61, 10)

# === 결과 저장용 리스트 ===
results = []

# === 데이터 로드 ===
srecords = hvsrpy.read([enz_files])

# === 파라미터 조합 반복 ===
for win_len in window_lengths:
    for taper in taper_ratios:
        for bw in bandwidths:
            try:
                # ✅ 전처리 설정
                preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
                preprocessing_settings.detrend = "linear"
                preprocessing_settings.window_length_in_seconds = win_len
                preprocessing_settings.orient_to_degrees_from_north = 0.0
                preprocessing_settings.filter_corner_frequencies_in_hz = (0.5, 20)
                preprocessing_settings.ignore_dissimilar_time_step_warning = False

                # ✅ 처리 설정
                processing_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
                processing_settings.window_type_and_width = ("tukey", taper)
                processing_settings.smoothing = dict(
                    operator="konno_and_ohmachi",
                    bandwidth=bw,
                    center_frequencies_in_hz=np.geomspace(0.2, 50, 200)
                )
                processing_settings.method_to_combine_horizontals = "total_horizontal_energy"
                processing_settings.handle_dissimilar_time_steps_by = "frequency_domain_resampling"

                # ✅ 전처리 & 처리
                s_copy = deepcopy(srecords)
                s_pre = hvsrpy.preprocess(s_copy, preprocessing_settings)
                passing = hvsrpy.sta_lta_window_rejection(
                    s_pre,
                    sta_seconds=1,
                    lta_seconds=30,
                    min_sta_lta_ratio=0.2,
                    max_sta_lta_ratio=2.5,
                    hvsr=None
                )
                hvsr = hvsrpy.process(passing, processing_settings)

                # ✅ peak frequency 추출
                if hvsr and hasattr(hvsr, 'mean_curve'):
                    mean_curve = hvsr.mean_curve()
                    peak_frequency = hvsr.frequency[np.argmax(mean_curve)]
                    peak_str = f"{peak_frequency:.5f}"
                else:
                    peak_str = "N/A"

                # ✅ SESAME 평가
                search_range_in_hz = (None, None)
                hvsr.update_peaks_bounded(search_range_in_hz=search_range_in_hz)

                clarity = sesame.clarity(
                    frequency=hvsr.frequency,
                    mean_curve=hvsr.mean_curve(distribution="lognormal"),
                    std_curve=hvsr.std_curve(distribution="lognormal"),
                    fn_std=hvsr.std_fn_frequency(distribution="normal"),
                    search_range_in_hz=search_range_in_hz,
                    verbose=0
                )

                reliability = sesame.reliability(
                    windowlength=win_len,
                    passing_window_count=np.sum(hvsr.valid_window_boolean_mask),
                    frequency=hvsr.frequency,
                    mean_curve=hvsr.mean_curve(distribution="lognormal"),
                    std_curve=hvsr.std_curve(distribution="lognormal"),
                    search_range_in_hz=search_range_in_hz,
                    verbose=0
                )

                results.append([
                    win_len,
                    taper,
                    bw,
                    peak_str,
                    "Pass" if clarity else "Fail",
                    "Pass" if reliability else "Fail"
                ])

            except Exception as e:
                results.append([win_len, taper, bw, "error", "error", "error"])

# === 결과 저장 ===
df = pd.DataFrame(results, columns=[
    "window_length", "taper_ratio", "bandwidth",
    "peak_frequency", "clarity_pass", "reliability_pass"
])

xlsx_path = output_folder / "hvsr_parameter_sweep_results.xlsx"
df.to_excel(xlsx_path, index=False)
print(f"\n✅ 결과 저장 완료: {xlsx_path}")
