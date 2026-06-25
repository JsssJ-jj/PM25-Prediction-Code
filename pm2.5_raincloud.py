import os
import numpy as np
import pandas as pd
import matplotlib as mpl

# 使用非交互式后端
mpl.use('Agg')

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ================================
# 1. 路径和基础设置
# ================================
OUT_DIR = r"C:\Users\焦焦\Desktop\修改1\pm2.5箱线图\云雨图"
os.makedirs(OUT_DIR, exist_ok=True)

# SCI 顶刊字体和线宽设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 2. 数据读取与划分 (总体、建模集、验证集)
# ================================
print("[INFO] 读取原始数据...")
EXCEL_PATH = r"C:\Users\焦焦\Desktop\修改1\pm2.5箱线图\云雨图\PM2.5修改后.xlsx"
df = pd.read_excel(EXCEL_PATH)

# 提取 PM2.5 数据并剔除空值
pm25 = df['PM2.5'].dropna().astype(float)
total_data = pm25.values

# 按照 7:3 进行 Calibration(建模) 和 Validation(验证) 集划分
print("[INFO] 提取总体数据，并按照 7:3 划分 Calibration 与 Validation...")
train_data, test_data = train_test_split(pm25, test_size=0.3, random_state=42)

data_list = [train_data.values, test_data.values, total_data]
labels = ['Calibration Set\n(70%)', 'Validation Set\n(30%)', 'Total Data\n(100%)']

# 颜色也对应调整
colors = ['#EEA885', '#B1D3A5', '#D08E9D']

# ================================
# 3. 绘制 Raincloud (雨云图)
# ================================
print("[INFO] 开始绘制 Raincloud 并排多重图...")
fig, ax = plt.subplots(figsize=(8, 6))

positions = np.array([1, 2, 3])

# (1) 绘制 Half-Violin (云 - 右半边的概率密度)
# 加入 bw_method=0.2 强制平滑，去掉大数据导致的小毛刺，让概率云看起来更像丝滑的水滴形
vp = ax.violinplot(data_list, positions=positions + 0.15, showmeans=False, showmedians=False, showextrema=False, widths=0.45, bw_method=0.2)
for i, pc in enumerate(vp['bodies']):
    # 截取提琴图的右半边，形成“云”
    m = np.mean(pc.get_paths()[0].vertices[:, 0])
    pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_linewidth(0.8)
    pc.set_alpha(0.6) # 半透明更显高级

# (2) 绘制 Boxplot (伞 - 中间的标准箱线)
bp = ax.boxplot(data_list, positions=positions, widths=0.08, patch_artist=True,
                showfliers=False,  # 隐藏异常值，因为左侧的散点会自动展示真实分布
                medianprops=dict(color='black', linewidth=1.8),
                boxprops=dict(facecolor='white', color='black', linewidth=1.2),
                whiskerprops=dict(color='black', linewidth=1.2),
                capprops=dict(color='black', linewidth=1.2))
for i, box in enumerate(bp['boxes']):
    box.set_facecolor(colors[i])
    box.set_alpha(0.8)

# (3) 绘制 Scatter Jitter (雨 - 左半边的原始散点抖动)
for i, d in enumerate(data_list):
    # 使用正交分布抖动(Gaussian Jitter)，让散点边缘自然羽化过渡，代替死板的矩阵矩形
    jitter = np.random.normal(loc=-0.15, scale=0.04, size=len(d))
    jitter = np.clip(jitter, -0.28, -0.04) # 限制边界使其不越过箱线图中心
    ax.scatter(positions[i] + jitter, d, s=12, color=colors[i], alpha=0.45, edgecolors='white', linewidths=0.25, zorder=0)

# ================================
# 4. 图表美化与学术排版
# ================================

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontweight='bold')
ax.set_xlabel('Data Set', fontweight='bold')
ax.set_ylabel(r'PM2.5 Concentration ($\mu g/m^3$)', fontweight='bold')
# 去掉标题，不再设置 ax.set_title

# 去除顶部和右侧外边框 (SCI Tufte 极简风格)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 添加水平暗网格作为辅助线
ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray')

plt.tight_layout()

# 保存输出
out_path = os.path.join(OUT_DIR, "PM25_Split_Raincloud.tif")
plt.savefig(out_path, dpi=300, format="tiff", bbox_inches="tight")
plt.close()

print(f"[INFO] 恭喜！高水准 Raincloud 图已成功生成并保存在: {out_path}")

# ================================
# 5. 计算并保存统计指标
# ================================
print("[INFO] 计算并保存统计指标...")
stats_data = []
for i, d in enumerate(data_list):
    stats = {
        'Dataset': labels[i].replace('\n', ' '),
        'Min': np.min(d),
        'Max': np.max(d),
        'Mean': np.mean(d),
        'Std Dev': np.std(d),
        'CV (%)': (np.std(d) / np.mean(d)) * 100 if np.mean(d) != 0 else 0
    }
    stats_data.append(stats)

stats_df = pd.DataFrame(stats_data)
stats_df = stats_df.round(2) # 所有数据保留两位小数

# 保存为 Excel
stats_out_path = os.path.join(OUT_DIR, "PM25_Data_Statistics.xlsx")
stats_df.to_excel(stats_out_path, index=False)

print(f"[INFO] 统计指标已计算并保存到: {stats_out_path}")
