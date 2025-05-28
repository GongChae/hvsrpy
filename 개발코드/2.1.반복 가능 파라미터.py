import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import hvsrpy
from hvsrpy import sesame
import pandas as pd

plt.style.use(hvsrpy.HVSRPY_MPL_STYLE)

# === 입력 및 출력 경로 설정 ===
data_dir = Path("C:/Users/USER/Desktop/그룹2")
save_root = Path("C:/SOLODATA/parameter_frequency_domain")
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

# === 파라미터 설정 ===
#window_lengths = range(5, 61, 5)
window_lengths = range(10, 51, 5)
taper_ratios = np.round(np.arange(0.0, 1.01, 0.1), 2)
bandwidths = range(10, 61, 10)
methods = [
    "arithmetic_mean", "squared_average", "quadratic_mean", "root_mean_square",
    "effective_amplitude_spectrum", "geometric_mean",
    "total_horizontal_energy", "vector_summation", "maximum_horizontal_value"
]

# === 각 세트 처리 ===
for base_id, fnames in fname_sets:
    print(f"✅ 처리 중: {base_id}")
    results = []

    try:
        srecords = hvsrpy.read([fnames])
        ts_sample = getattr(srecords[0], "vt")
        end_time = ts_sample.time()[-1]

        # 전체 구간 중 마지막 1500초만 사용
        for rec in srecords:
            for comp in ("ns", "ew", "vt"):
                ts = getattr(rec, comp)
                ts.trim(end_time - 1500, end_time)

        # === 파라미터 반복 실행 ===
        for win_len in window_lengths:
            for taper in taper_ratios:
                for bw in bandwidths:
                    for method in methods:
                        try:
                            # ── 전처리 설정 ──
                            preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
                            preprocessing_settings.detrend = "linear"
                            preprocessing_settings.window_length_in_seconds = win_len
                            preprocessing_settings.orient_to_degrees_from_north = 0.0
                            preprocessing_settings.filter_corner_frequencies_in_hz = (1, 20)
                            preprocessing_settings.ignore_dissimilar_time_step_warning = False

                            # ── 처리 설정 ──
                            processing_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
                            processing_settings.window_type_and_width = ("tukey", taper)
                            processing_settings.smoothing = dict(
                                operator="konno_and_ohmachi",
                                bandwidth=bw,
                                center_frequencies_in_hz=np.geomspace(1, 20, 200)
                            )
                            processing_settings.method_to_combine_horizontals = method
                            processing_settings.handle_dissimilar_time_steps_by = "frequency_domain_resampling"

                            # ── 전처리 및 STA-LTA 제거 ──
                            s_copy = deepcopy(srecords)
                            s_pre = hvsrpy.preprocess(s_copy, preprocessing_settings)
                            hvsr = hvsrpy.process(s_pre, processing_settings)
                            n=2
                            search_range_in_hz = (1, 20)
                            _ = hvsrpy.frequency_domain_window_rejection(hvsr, n=n, search_range_in_hz=search_range_in_hz)

                            # ── 고유주파수 추출 ──
                            if hvsr and hasattr(hvsr, 'mean_curve'):
                                mean_curve = hvsr.mean_curve()
                                peak_frequency = hvsr.frequency[np.argmax(mean_curve)]
                                peak_str = f"{peak_frequency:.5f}"
                            else:
                                peak_str = "N/A"

                            # ── SESAME 평가 ──
                            hvsr.update_peaks_bounded(search_range_in_hz=(None, None))

                            reliability = sesame.reliability(
                                windowlength=win_len,
                                passing_window_count=np.sum(hvsr.valid_window_boolean_mask),
                                frequency=hvsr.frequency,
                                mean_curve=hvsr.mean_curve(distribution="lognormal"),
                                std_curve=hvsr.std_curve(distribution="lognormal"),
                                search_range_in_hz=(None, None),
                                verbose=0
                            )
                            reliability_pass = bool(np.all(reliability))

                            clarity_flags = sesame.clarity(
                                frequency=hvsr.frequency,
                                mean_curve=hvsr.mean_curve(distribution="lognormal"),
                                std_curve=hvsr.std_curve(distribution="lognormal"),
                                fn_std=hvsr.std_fn_frequency(distribution="normal"),
                                search_range_in_hz=(None, None),
                                verbose=0
                            )
                            clarity_pass = np.sum(clarity_flags) >= 5

                            # ── 결과 저장 ──
                            results.append([
                                win_len, taper, bw, method,
                                peak_str,
                                "Pass" if clarity_pass else "Fail",
                                "Pass" if reliability_pass else "Fail",
                                "Pass" if clarity_pass and reliability_pass else "Fail"
                            ])

                        except Exception as e:
                            results.append([win_len, taper, bw, method, "error", "error", "error", "error"])

        # === 세트별 결과 Excel로 저장 ===
        df = pd.DataFrame(results, columns=[
            "window_length", "taper_ratio", "bandwidth", "method",
            "peak_frequency", "clarity_pass", "reliability_pass", "total_pass"
        ])
        xlsx_path = output_dir / f"{base_id}_hvsr_parameter_results.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"📁 결과 저장 완료: {xlsx_path}")

    except Exception as e:
        print(f"❌ {base_id} 처리 실패: {e}")
