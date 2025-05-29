import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from obspy import read
import hvsrpy
from hvsrpy import sesame
from copy import deepcopy
from collections import defaultdict

# === 경로 설정 ===
data_dir = Path("C:/SOLODATA/zonghap/3month/종합설계")
output_dir = Path('C:/Users/user/Desktop/전체')
output_dir.mkdir(parents=True, exist_ok=True)

# === .sac 파일 그룹화 (Z/N/E) ===
sac_files = list(data_dir.glob('*.sac'))
grouped_files = defaultdict(dict)
coordinates = {}

for file in sac_files:
    parts = file.stem.split('.')
    if len(parts) < 2:
        continue
    base_id = '.'.join(parts[:-1])
    comp = parts[-1].upper()
    grouped_files[base_id][comp] = str(file)

    # 좌표 추출 (Z 파일에서만)
    if comp == 'Z':
        try:
            st = read(str(file))
            sac_header = st[0].stats.sac
            lon = sac_header.stlo
            lat = sac_header.stla
            if lon is not None and lat is not None:
                coordinates[base_id] = (lon, lat)
            else:
                print(f'⚠️ 좌표 없음: {file}')
        except Exception as e:
            print(f'❌ 파일 읽기 오류: {file}, {e}')

# === 유효한 세트만 추출 ===
fname_sets = []
for base, comps in grouped_files.items():
    if all(k in comps for k in ['E', 'N', 'Z']):
        fname_sets.append((base, [comps['Z'], comps['N'], comps['E']]))

if not fname_sets:
    print('❌ E/N/Z 세트를 찾을 수 없습니다.')
    exit()

# === 결과 수집 ===
results = []

for base_id, fnames in fname_sets:
    print(f'✅ 처리 중: {base_id}')
    try:
        srecords = hvsrpy.read([fnames])
        ts_sample = getattr(srecords[0], 'vt')
        end_time = ts_sample.time()[-1]

        # 마지막 1500초만 사용
        for rec in srecords:
            for comp in ('ns', 'ew', 'vt'):
                ts = getattr(rec, comp)
                ts.trim(end_time - 1500, end_time)

        preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
        preprocessing_settings.detrend = 'linear'
        preprocessing_settings.window_length_in_seconds = 15  # 예: 30초
        preprocessing_settings.orient_to_degrees_from_north = 0.0
        preprocessing_settings.filter_corner_frequencies_in_hz = (1, 20)

        processing_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
        processing_settings.window_type_and_width = ('tukey', 0)
        processing_settings.smoothing = dict(
            operator='konno_and_ohmachi',
            bandwidth=20,
            center_frequencies_in_hz=np.geomspace(1, 20, 200)
        )
        processing_settings.method_to_combine_horizontals = 'geometric_mean'
        processing_settings.handle_dissimilar_time_steps_by = 'frequency_domain_resampling'

        s_copy = deepcopy(srecords)
        s_pre = hvsrpy.preprocess(s_copy, preprocessing_settings)
        hvsr = hvsrpy.process(s_pre, processing_settings)
        n=2
        search_range_in_hz=(1,20)
        _ = hvsrpy.frequency_domain_window_rejection(hvsr,n=n,search_range_in_hz=search_range_in_hz)

        if hvsr and hasattr(hvsr, 'mean_curve'):
            mean_curve = hvsr.mean_curve()
            peak_frequency = hvsr.frequency[np.argmax(mean_curve)]
            peak_str = f'{peak_frequency:.5f}'
        else:
            peak_str = 'N/A'

        lon, lat = coordinates.get(base_id, (None, None))
        results.append([base_id, lon, lat, peak_str])

    except Exception as e:
        print(f'❌ {base_id} 처리 실패: {e}')
        results.append([base_id, 'error', 'error', 'error'])

# === CSV로 저장 ===
df = pd.DataFrame(results, columns=['file_name', 'longitude', 'latitude', 'peak_frequency'])
csv_path = output_dir / 'hvsr_summary.csv'
df.to_csv(csv_path, index=False)
print(f'📁 결과 CSV 저장 완료: {csv_path}')
