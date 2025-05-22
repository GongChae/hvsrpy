import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hvsrpy
from hvsrpy.timeseries import TimeSeries

# ----------------------------------
# 1. 파일 찾기: 접두어 기반 E/N/Z 묶기
# ----------------------------------
def find_one_enz_group(folder_path):
    z_files = sorted(glob.glob(os.path.join(folder_path, "*.Z.sac")))
    for z_path in z_files:
        prefix = z_path[:-6]
        e_path = prefix + ".E.sac"
        n_path = prefix + ".N.sac"
        if os.path.exists(e_path) and os.path.exists(n_path):
            return [(e_path, n_path, z_path)]
    return None

input_folder = "C:/SOLODATA/try"
enz_sets = find_one_enz_group(input_folder)
if not enz_sets:
    raise FileNotFoundError("❌ E/N/Z 3성분 파일 세트를 찾을 수 없습니다.")

# ----------------------------------
# 2. SAC 로드
# ----------------------------------
srecords = hvsrpy.read(enz_sets)
print(f"✅ 레코드 수: {len(srecords)}")

# ----------------------------------
# 3. trim: 끝에서 1800초 사용
# ----------------------------------
for rec in srecords:
    for comp in ["ns", "ew", "vt"]:
        ts = getattr(rec, comp)
        end_time = ts.time()[-1]
        start_time = max(0, end_time - 1800)
        ts.trim(start_time, end_time)

# 윈도우 길이 정확히 설정
window_length_for_rejection = end_time - start_time

# ----------------------------------
# 4. 첫 번째 preprocess: STA/LTA rejection용 (trim 길이만큼 하나의 윈도우 생성)
# ----------------------------------
pre_settings_rejection = hvsrpy.settings.HvsrPreProcessingSettings()
pre_settings_rejection.detrend = None
pre_settings_rejection.window_length_in_seconds = window_length_for_rejection
pre_settings_rejection.filter_corner_frequencies_in_hz = (None,None)

srecords_for_rejection = hvsrpy.preprocess(srecords, pre_settings_rejection)
# ----------------------------------
# 5. STA/LTA 윈도우 rejection
# ----------------------------------
srecords_passed = hvsrpy.sta_lta_window_rejection(
    records=srecords_for_rejection,
    sta_seconds=1,
    lta_seconds=30,
    min_sta_lta_ratio=0.2,
    max_sta_lta_ratio=2.5,
    components=("ns", "ew", "vt")
)

if not srecords_passed:
    raise RuntimeError("❌ STA/LTA 조건을 만족하는 데이터가 없습니다.")

print(f"✅ STA/LTA 통과 레코드 수: {len(srecords_passed)}")

# ----------------------------------
# 6. 두 번째 preprocess: 분석용 (윈도우 나눔, detrend, filter)
# ----------------------------------
pre_settings_analysis = hvsrpy.settings.HvsrPreProcessingSettings()
pre_settings_analysis.detrend = "linear"
pre_settings_analysis.window_length_in_seconds = 30
pre_settings_analysis.filter_corner_frequencies_in_hz = (0.5, 20)

srecords_analysis = hvsrpy.preprocess(srecords_passed, pre_settings_analysis)

# ----------------------------------
# 7. HVSR 처리 설정 및 실행
# ----------------------------------
proc_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
proc_settings.window_type_and_width = ("tukey", 0.1)
proc_settings.smoothing = {
    "operator": "konno_and_ohmachi",
    "bandwidth": 40,
    "center_frequencies_in_hz": np.geomspace(0.5, 20, 200)
}
proc_settings.method_to_combine_horizontals = "total_horizontal_energy"

hvsr = hvsrpy.process(srecords_analysis, proc_settings)

# ----------------------------------
# 8. 시각화: HVSR + STA/LTA rejection 결과
# ----------------------------------
def compare_sta_lta_rejection(pre_records, hvsr_result, title="STA/LTA 적용 전후 비교"):
    if not pre_records or not hvsr_result:
        print("❌ 유효한 입력 없음")
        return
    try:
        mfig, axs = hvsrpy.plot_pre_and_post_rejection(pre_records, hvsr_result)
        mfig.suptitle(title, fontsize=14)
        plt.show()
    except Exception as e:
        print(f"❌ 시각화 실패: {e}")

compare_sta_lta_rejection(srecords_analysis, hvsr)

# ----------------------------------
# 9. HVSR 곡선 및 피크 주파수 출력
# ----------------------------------
fig, ax = hvsrpy.plot_single_panel_hvsr_curves(hvsr)
plt.title("HVSR Curve (After STA/LTA Rejection)")
plt.show()

if hvsr and hasattr(hvsr, 'mean_curve'):
    mean_curve = hvsr.mean_curve()
    peak_freq = hvsr.frequency[np.argmax(mean_curve)]
    print(f"📌 피크 주파수: {peak_freq:.3f} Hz")
else:
    print("❌ HVSR 결과가 없습니다.")
