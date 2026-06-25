import os
import numpy as np
import pandas as pd
import matplotlib as mpl

# 使用非交互式后端，防止在无图形界面环境下出错
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
from catboost import CatBoostRegressor
import shap

# 当前脚本所在目录，用于固定输出路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 统一输出目录（按你的要求）
OUTPUT_DIR = r"C:\Users\焦焦\Desktop\修改1\三级shap图"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# 1. 字体和样式设置（参考 pr 学姐代码）
# ================================
size_cm = 8  # 图宽高 8 cm
size_inch = size_cm / 2.54

times_path = "C:/Windows/Fonts/times.ttf"
if os.path.exists(times_path):
    font_en_65 = FontProperties(fname=times_path, size=6.5)
else:
    print(f"警告: 未找到字体 {times_path}，使用默认字体")
    font_en_65 = FontProperties(size=6.5)

plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['axes.unicode_minus'] = False

# 防止 OpenMP 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================================
# 2. 数据读取（与 Stacking 代码保持一致）
# ================================
EXCEL_PATH = r"C:\Users\焦焦\Desktop\论文数据.xlsx"
TARGET_COL = "PM2.5"
SEQ_COLS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "DEWP",
    "SurfacePressure_mean",
    "RH_mean",
]
TAB_COLS = ["NO2", "SO2", "CO", "O3_8h"]  # 移除PM10

print("[INFO] 正在读取数据并预处理……")
df = pd.read_excel(EXCEL_PATH)
df = df.dropna(subset=[TARGET_COL] + SEQ_COLS + TAB_COLS).reset_index(drop=True)


# 这里只用 Stacking 中第一层 CatBoost 的输入特征 TAB_COLS 做 SHAP 分析（已去除PM10）
X = df[TAB_COLS].astype(float)
y = df[TARGET_COL].astype(float)

print(f"[INFO] 样本数: {len(df)}, 特征数: {X.shape[1]}")

# ================================
# 3. 训练 CatBoost 模型（对应 Stacking 第 1 层 cb1）
# ================================
print("[INFO] 正在训练 CatBoost 模型用于 SHAP 分析……")
cb = CatBoostRegressor(
    loss_function="RMSE",
    depth=6,
    learning_rate=0.04,
    n_estimators=400,
    random_seed=42,
    verbose=0,
)
cb.fit(X, y)

# ================================
# 4. 计算 SHAP 值
# ================================
print("[INFO] 正在计算 SHAP 值……")
explainer = shap.TreeExplainer(cb)
shap_vals = explainer.shap_values(X, check_additivity=False)

# 取绝对值平均，得到全特征的重要性
mean_abs_all = np.abs(shap_vals).mean(0)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Abs_SHAP": mean_abs_all,
})
importance_df = importance_df.sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

# 保存特征重要性表格到指定输出文件夹
out_table = os.path.join(OUTPUT_DIR, "Stacking_SHAP_Importance_Table.xlsx")
importance_df.to_excel(out_table, index=False)
print(f"[INFO] 已保存特征重要性表格: {out_table}")

# ================================
# 5. 绘制 SHAP 反演特征贡献度分析图
#    风格参考 pr 学姐代码，但配色略作升级
# ================================
print("[INFO] 正在绘制 SHAP 特征贡献度分析图……")

# 如果特征不多，就全部画出
top_n = min(15, X.shape[1])

# 排序后取前 top_n 个特征
idx_sorted = np.argsort(mean_abs_all)[::-1]  # 从大到小
idx_top = idx_sorted[:top_n]

top_feats = X.columns[idx_top]
top_vals = mean_abs_all[idx_top]

fig, ax_lower = plt.subplots(figsize=(size_inch, size_inch))
plt.subplots_adjust(left=0.32, right=0.86, top=0.78, bottom=0.25)

ax_upper = ax_lower.twiny()
ax_lower.set_box_aspect(1)
y_pos = np.arange(top_n)

# ----- 条形图：高级一点的紫色系配色 -----
# 由浅到深的紫色，用于 Mean |SHAP|
cmap_bars = plt.cm.Purples(np.linspace(0.35, 0.9, top_n))
ax_upper.barh(y_pos, top_vals, color=cmap_bars, alpha=0.8, height=0.5)

ax_upper.text(
    0.5,
    1.15,
    "Mean absolute SHAP value",
    transform=ax_upper.transAxes,
    fontproperties=font_en_65,
    ha="center",
    va="bottom",
)

max_val = float(np.max(top_vals)) if len(top_vals) > 0 else 0.1
upper_limit = np.ceil(max_val * 110) / 100 if max_val > 0 else 0.1
ax_upper.set_xlim(0, upper_limit)
ax_upper.xaxis.set_major_locator(ticker.LinearLocator(5))
ax_upper.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
for label in ax_upper.get_xticklabels():
    label.set_fontproperties(font_en_65)
ax_upper.tick_params(axis="x", length=2, pad=1, width=0.5)

# ----- 点图：蓝-紫-橙 渐变配色，表示特征值大小 -----
from matplotlib.colors import LinearSegmentedColormap

cmap_dots = LinearSegmentedColormap.from_list(
    "shap_advanced",
    ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"],
)

X_subset = X[top_feats]

# 归一化特征值到 [0,1]，用于着色
X_range = X_subset.max() - X_subset.min()
X_range[X_range == 0] = 1
X_std = (X_subset - X_subset.min()) / X_range

for j, f in enumerate(top_feats):
    feat_idx = X.columns.get_loc(f)
    jitter = np.random.normal(0, 0.1, len(y))
    ax_lower.scatter(
        shap_vals[:, feat_idx],
        j + jitter,
        c=cmap_dots(X_std[f].values.astype(float)),
        s=4,
        alpha=0.8,
        edgecolors="none",
        zorder=10,
    )

ax_lower.text(
    0.5,
    -0.12,
    "SHAP value",
    transform=ax_lower.transAxes,
    fontproperties=font_en_65,
    ha="center",
    va="top",
)

# 对称设置 x 轴范围
curr_lim = float(np.max(np.abs(ax_lower.get_xlim()))) if len(top_vals) > 0 else 0.1
ax_lower.set_xlim(-curr_lim, curr_lim)
ax_lower.xaxis.set_major_locator(ticker.LinearLocator(5))
ax_lower.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
for label in ax_lower.get_xticklabels():
    label.set_fontproperties(font_en_65)
ax_lower.tick_params(axis="x", length=2, pad=1, width=0.5)

# 在右上角添加 (a) 标记
ax_lower.text(
    0.95, 0.95, '(a)',
    transform=ax_lower.transAxes,
    fontsize=5,
    ha='right',
    va='top',
)

def _feature_display_name(feature_name: str) -> str:
    name = str(feature_name)
    if name in {"O3_8h", "O3-8h", "O3_8H", "O3-8H"}:
        return "O3"
    return name

ax_lower.set_yticks(y_pos)
ax_lower.set_yticklabels([_feature_display_name(f) for f in top_feats])
for label in ax_lower.get_yticklabels():
    label.set_fontproperties(font_en_65)
ax_lower.tick_params(axis="y", length=0, pad=3)

ax_lower.axvline(0, color="black", lw=0.5, alpha=0.6, zorder=2)

# 右侧色条，表示特征值从 Low 到 High
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ax_ins = inset_axes(
    ax_lower,
    width="5%",
    height="100%",
    loc="lower left",
    bbox_to_anchor=(1.12, 0, 1, 1),
    bbox_transform=ax_lower.transAxes,
    borderpad=0,
)
cb = mpl.colorbar.ColorbarBase(ax_ins, cmap=cmap_dots, orientation="vertical")
cb.outline.set_linewidth(0.5)
cb.set_ticks([])
cb.ax.text(
    0.5,
    1.03,
    "High",
    transform=cb.ax.transAxes,
    ha="center",
    va="bottom",
    fontproperties=font_en_65,
)
cb.ax.text(
    0.5,
    -0.03,
    "Low",
    transform=cb.ax.transAxes,
    ha="center",
    va="top",
    fontproperties=font_en_65,
)

# 输出图片按要求保存：TIFF、dpi=300、固定文件名
output_base = os.path.join(OUTPUT_DIR, "CatBoost特征重要性图")
plt.savefig(f"{output_base}.tiff", dpi=300, format="tiff", bbox_inches="tight")
plt.close()

print(f"[INFO] 已完成特征重要性图绘制: {output_base}.tiff")
