import os
import numpy as np
import pandas as pd
import matplotlib as mpl

# 使用非交互式后端，防止在无图形界面环境下出错
mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import shap
import tensorflow as tf

# ================================
# 0. 输出路径（按你的要求）
# ================================
OUTPUT_DIR = r"C:\Users\焦焦\Desktop\修改1\三级shap图"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# 0.1 SHAP 计算提速参数（如需更精细结果可适当调大）
# ================================
BACKGROUND_MAX_ROWS = 1000
BACKGROUND_KMEANS_K = 15
SHAP_SAMPLE_SIZE = 200
KERNEL_NSAMPLES = 200

# ================================
# 1. 字体和样式设置（与 CatBoost 图保持一致）
# ================================
size_cm = 8  # 图宽高 8 cm
size_inch = size_cm / 2.54

times_path = "C:/Windows/Fonts/times.ttf"
if os.path.exists(times_path):
    font_en_65 = FontProperties(fname=times_path, size=6.5)
else:
    print(f"警告: 未找到字体 {times_path}，使用默认字体")
    font_en_65 = FontProperties(size=6.5)

plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["axes.unicode_minus"] = False

# 防止 OpenMP 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================================
# 2. 数据读取（与 Stacking 三级融合脚本保持一致）
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

print("[INFO] 正在读取数据并预处理……")
df = pd.read_excel(EXCEL_PATH)
df = df.dropna(subset=[TARGET_COL] + SEQ_COLS).reset_index(drop=True)

y = df[TARGET_COL].astype(float).values
X_seq_2d = df[SEQ_COLS].astype(float)
X_seq_3d = X_seq_2d.values.reshape(-1, len(SEQ_COLS), 1)

print(f"[INFO] 样本数: {len(df)}, 特征数: {X_seq_2d.shape[1]}")

# ================================
# 3. 训练 LSTM（结构参考你给的脚本）
# ================================
print("[INFO] 正在训练 LSTM 模型用于 SHAP 分析……")
np.random.seed(42)

lstm_model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(24, input_shape=(len(SEQ_COLS), 1)),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

lstm_model.compile(optimizer="adam", loss="mse")
es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
lstm_model.fit(X_seq_3d, y, epochs=50, batch_size=32, callbacks=[es], verbose=0)

# ================================
# 4. 计算 LSTM 的 SHAP 值（KernelExplainer + 采样提速）
# ================================
print("[INFO] 正在计算 LSTM 的 SHAP 值（KernelExplainer）……")


def _lstm_predict_wrapper(X_2d_arr: np.ndarray) -> np.ndarray:
    X_3d_arr = np.asarray(X_2d_arr, dtype=float).reshape(-1, len(SEQ_COLS), 1)
    return lstm_model.predict(X_3d_arr, verbose=0).flatten()


# 背景数据：kmeans 浓缩（参考你给的 Level1_SHAP_Combined.py）
background_pool = X_seq_2d.values[: min(BACKGROUND_MAX_ROWS, len(X_seq_2d))]
background = shap.kmeans(background_pool, BACKGROUND_KMEANS_K)
explainer = shap.KernelExplainer(_lstm_predict_wrapper, background)

# SHAP 采样：取前 sample_size 个样本（减少运行时间）
sample_size = min(SHAP_SAMPLE_SIZE, len(X_seq_2d))
X_sample = X_seq_2d.iloc[:sample_size].copy()

try:
    shap_vals = explainer.shap_values(X_sample.values, nsamples=KERNEL_NSAMPLES, silent=True)
except TypeError:
    try:
        shap_vals = explainer.shap_values(X_sample.values, nsamples=KERNEL_NSAMPLES)
    except TypeError:
        shap_vals = explainer.shap_values(X_sample.values)

# 兼容不同 shap 版本返回值
if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]
shap_vals = np.asarray(shap_vals, dtype=float)

# ================================
# 5. 保存特征重要性表格（与 CatBoost 同风格输出 Excel）
# ================================
mean_abs_all = np.abs(shap_vals).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": X_seq_2d.columns,
    "Mean_Abs_SHAP": mean_abs_all,
}).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

out_table = os.path.join(OUTPUT_DIR, "LSTM_SHAP_Importance_Table.xlsx")
importance_df.to_excel(out_table, index=False)
print(f"[INFO] 已保存 LSTM 特征重要性表格: {out_table}")

# ================================
# 6. 绘制：与 CatBoost 特征重要性图一致的“条形图 + 散点 + 颜色条”
# ================================
print("[INFO] 正在绘制 LSTM SHAP 特征重要性图……")

feature_names = list(X_seq_2d.columns)
top_n = min(15, len(feature_names))
idx_sorted = np.argsort(mean_abs_all)[::-1]
idx_top = idx_sorted[:top_n]

top_feats = X_seq_2d.columns[idx_top]
top_vals = mean_abs_all[idx_top]

fig, ax_lower = plt.subplots(figsize=(size_inch, size_inch))
plt.subplots_adjust(left=0.32, right=0.86, top=0.78, bottom=0.25)

ax_upper = ax_lower.twiny()
ax_lower.set_box_aspect(1)
y_pos = np.arange(top_n)

# 条形图：紫色系
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

# 散点：蓝-紫-橙-红 渐变，表示特征值大小
cmap_dots = LinearSegmentedColormap.from_list(
    "shap_advanced",
    ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"],
)

X_subset = X_sample[top_feats]
X_range = X_subset.max() - X_subset.min()
X_range[X_range == 0] = 1
X_std = (X_subset - X_subset.min()) / X_range


def _feature_display_name(feature_name: str) -> str:
    name = str(feature_name)
    mapping = {
        "temperature_2m_max": "Tmax",
        "temperature_2m_min": "Tmin",
        "precipitation_sum": "PRE",
        "wind_speed_10m_max": "WSmax",
        "DEWP": "DEWP",
        "SurfacePressure_mean": "SP",
        "RH_mean": "RH",
    }
    return mapping.get(name, name)

for j, f in enumerate(top_feats):
    feat_idx = feature_names.index(str(f))
    jitter = np.random.normal(0, 0.1, shap_vals.shape[0])
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

curr_lim = float(np.max(np.abs(ax_lower.get_xlim()))) if len(top_vals) > 0 else 0.1
ax_lower.set_xlim(-curr_lim, curr_lim)
ax_lower.xaxis.set_major_locator(ticker.LinearLocator(5))
ax_lower.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
for label in ax_lower.get_xticklabels():
    label.set_fontproperties(font_en_65)
ax_lower.tick_params(axis="x", length=2, pad=1, width=0.5)

ax_lower.set_yticks(y_pos)
ax_lower.set_yticklabels([_feature_display_name(f) for f in top_feats])
for label in ax_lower.get_yticklabels():
    label.set_fontproperties(font_en_65)
ax_lower.tick_params(axis="y", length=0, pad=3)

ax_lower.axvline(0, color="black", lw=0.5, alpha=0.6, zorder=2)

# 右侧色条：Low/High
ax_ins = inset_axes(
    ax_lower,
    width="5%",
    height="100%",
    loc="lower left",
    bbox_to_anchor=(1.12, 0, 1, 1),
    bbox_transform=ax_lower.transAxes,
    borderpad=0,
)
colorbar = mpl.colorbar.ColorbarBase(ax_ins, cmap=cmap_dots, orientation="vertical")
colorbar.outline.set_linewidth(0.5)
colorbar.set_ticks([])
colorbar.ax.text(
    0.5,
    1.03,
    "High",
    transform=colorbar.ax.transAxes,
    ha="center",
    va="bottom",
    fontproperties=font_en_65,
)
colorbar.ax.text(
    0.5,
    -0.03,
    "Low",
    transform=colorbar.ax.transAxes,
    ha="center",
    va="top",
    fontproperties=font_en_65,
)

# 在右上角添加 (b) 标记
ax_lower.text(
    0.95, 0.95, '(b)',
    transform=ax_lower.transAxes,
    fontsize=5,
    ha='right',
    va='top',
)



# 输出：TIFF、dpi=300、固定文件名
output_base = os.path.join(OUTPUT_DIR, "LSTM特征重要性图")
plt.savefig(f"{output_base}.tiff", dpi=300, format="tiff", bbox_inches="tight")
plt.close()

print(f"[INFO] 已完成 LSTM 特征重要性图绘制: {output_base}.tiff")
