import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 폰트 설정 (한글 깨짐 방지)
font_path = "C:/Windows/Fonts/malgun.ttf"
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
# 데이터 불러오기
file_path = 'C:/Users/USER/Desktop/전체/0529_암반추가_조작.csv'  # 경로 수정하세요
df = pd.read_csv(file_path, encoding='cp949')
# CSV 불러오기
df = pd.read_csv("MAPE_IQR.csv")
df.columns = ["time", "IQR"]
df["time"] = df["time"].str.replace("초", "")

# 꺾은선 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(df["time"], df["IQR"], marker='o')
plt.axhline(y=10, color='grey', linestyle='--', linewidth=2)
plt.title("녹화 시간에 따른 IQR 변화율", fontsize=20)
plt.xlabel("녹화 시간 (초)", fontsize=16)
plt.ylabel("MAPE (%)", fontsize=16)
plt.grid(False)
plt.show()