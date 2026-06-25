import os
import numpy as np
import pandas as pd
import matplotlib as mpl

# 使用非交互式后端
mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import shap
import tensorflow as tf
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# ================================
# 0. 输出与路径设置
# ================================
OUTPUT_DIR = r"C:\Users\焦焦\Desktop\修改1\三级shap图"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXCEL_PATH = r"C:\Users\焦焦\Desktop\论文数据.xlsx"

# 防止 OpenMP 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置随机种子以保证结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# ================================
# 0.1 SHAP 计算提速参数 (与 LSTM 脚本保持一致)
# ================================
BACKGROUND_MAX_ROWS = 1000
BACKGROUND_KMEANS_K = 15
SHAP_SAMPLE_SIZE = 200
KERNEL_NSAMPLES = 200

# ================================
# 1. 字体和样式设置 (与你提供的脚本完全一致)
# ================================
size_cm = 8
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

# ================================
# 2. 数据读取与特征定义 (与 Stacking 脚本一致，并移除 PM10)
# ================================
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
TAB_COLS = ["NO2", "SO2", "CO", "O3_8h"]  # 已移除 PM10

print("[INFO] 正在读取数据...")
df = pd.read_excel(EXCEL_PATH)
df = df.dropna(subset=[TARGET_COL] + SEQ_COLS + TAB_COLS).reset_index(drop=True)

# 合并所有原始特征用于 SHAP 分析
X_all_features = df[TAB_COLS + SEQ_COLS].astype(float)
y = df[TARGET_COL].astype(float)

print(f"[INFO] 样本数: {len(df)}, 总特征数: {X_all_features.shape[1]}")

# ================================
# 3. 完整重现 Stacking 模型训练过程
# ================================
print("[INFO] 正在重新训练完整 Stacking 模型...")

# 3.1 数据准备
X_seq_3d = df[SEQ_COLS].values.astype(float).reshape(-1, len(SEQ_COLS), 1)
X_tab = df[TAB_COLS].values.astype(float)

# 3.2 Level 1 模型训练
print("[INFO] 训练 Level 1 模型 (CatBoost & LSTM)...")
cb1 = CatBoostRegressor(
    loss_function='RMSE', depth=6, learning_rate=0.04, n_estimators=300,
    l2_leaf_reg=4, random_seed=42, verbose=0
)
cb1.fit(X_tab, y)

lstm1 = tf.keras.Sequential([
    tf.keras.layers.LSTM(24, input_shape=(len(SEQ_COLS), 1)),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(1)
])
lstm1.compile(optimizer='adam', loss='mse')
es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
# 使用全部数据训练，因为 SHAP 需要一个固定的、已训练好的模型
lstm1.fit(X_seq_3d, y, epochs=120, batch_size=32, callbacks=[es], verbose=0)

# 3.3 Level 2 模型训练
print("[INFO] 训练 Level 2 模型...")
pred_cb1_full = cb1.predict(X_tab)
pred_lstm1_full = lstm1.predict(X_seq_3d, verbose=0).ravel()
Z1_full = np.column_stack([pred_cb1_full, pred_lstm1_full])

cb2 = CatBoostRegressor(
    loss_function='RMSE', depth=5, learning_rate=0.03, n_estimators=200,
    l2_leaf_reg=6, random_seed=42, verbose=0
)
cb2.fit(Z1_full, y)

# 3.4 Level 3 模型训练
print("[INFO] 训练 Level 3 模型...")
pred_cb2_full = cb2.predict(Z1_full)
Z2_full = np.column_stack([pred_cb1_full, pred_lstm1_full, pred_cb2_full])

cb3 = CatBoostRegressor(
    loss_function='RMSE', depth=4, learning_rate=0.025, n_estimators=120,
    l2_leaf_reg=8, random_seed=42, verbose=0
)
cb3.fit(Z2_full, y)

print("[INFO] Stacking 模型训练完成。")

# ================================
# 4. 构建 Stacking 模型的 SHAP 包装器
# ================================
print("[INFO] 构建 SHAP 包装器...")

def _stacking_predict_wrapper(X_2d_arr: np.ndarray) -> np.ndarray:
    """
    接收原始特征(2D Numpy)，在内部完成整个3级stacking预测流程。
    """
    # 确保输入是 DataFrame，以便按列名分离特征
    X_df = pd.DataFrame(X_2d_arr, columns=TAB_COLS + SEQ_COLS)
    
    # 分离 tabular 和 sequential 特征
    x_tab_input = X_df[TAB_COLS].values.astype(float)
    x_seq_input = X_df[SEQ_COLS].values.astype(float)
    x_seq_3d_input = x_seq_input.reshape(-1, len(SEQ_COLS), 1)

    # Level 1 预测
    pred_cb1 = cb1.predict(x_tab_input)
    pred_lstm1 = lstm1.predict(x_seq_3d_input, verbose=0).ravel()

    # Level 2 预测
    Z1 = np.column_stack([pred_cb1, pred_lstm1])
    pred_cb2 = cb2.predict(Z1)

    # Level 3 预测
    Z2 = np.column_stack([pred_cb1, pred_lstm1, pred_cb2])
    pred_final = cb3.predict(Z2)
    
    return pred_final

# ================================
# 5. 计算 Stacking 模型的 SHAP 值 (KernelExplainer)
# ================================
print("[INFO] 正在计算 Stacking 模型的 SHAP 值 (这可能需要几分钟)...")

# 背景数据：从所有原始特征中进行 K-Means 浓缩
background_pool = X_all_features.values[:min(BACKGROUND_MAX_ROWS, len(X_all_features))]
background = shap.kmeans(background_pool, BACKGROUND_KMEANS_K)

# 创建 KernelExplainer
explainer = shap.KernelExplainer(_stacking_predict_wrapper, background)

# 对一小部分样本计算 SHAP 值以节省时间
sample_size = min(SHAP_SAMPLE_SIZE, len(X_all_features))
X_sample = X_all_features.iloc[:sample_size]

# 计算 SHAP 值
shap_vals = explainer.shap_values(X_sample.values, nsamples=KERNEL_NSAMPLES, silent=True)

# 兼容不同 shap 版本返回值
if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]
shap_vals = np.asarray(shap_vals, dtype=float)

# ================================
# 6. 保存与绘图 (代码结构与你的脚本完全一致)
# ================================
print("[INFO] 正在保存表格并绘制 SHAP 总览图...")

# 6.1 保存特征重要性表格
mean_abs_all = np.abs(shap_vals).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": X_all_features.columns,
    "Mean_Abs_SHAP": mean_abs_all,
}).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

out_table = os.path.join(OUTPUT_DIR, "Stacking_Overall_SHAP_Importance.xlsx")
importance_df.to_excel(out_table, index=False)
print(f"[INFO] 已保存 Stacking 模型总特征重要性表格: {out_table}")

# 6.2 绘图
feature_names = list(X_all_features.columns)
top_n = min(15, len(feature_names))
idx_sorted = np.argsort(mean_abs_all)[::-1]
idx_top = idx_sorted[:top_n]

top_feats = X_all_features.columns[idx_top]
top_vals = mean_abs_all[idx_top]

fig, ax_lower = plt.subplots(figsize=(size_inch, size_inch))
plt.subplots_adjust(left=0.32, right=0.86, top=0.78, bottom=0.25)

ax_upper = ax_lower.twiny()
ax_lower.set_box_aspect(1)
y_pos = np.arange(top_n)

# 条形图
cmap_bars = plt.cm.Purples(np.linspace(0.35, 0.9, top_n))
ax_upper.barh(y_pos, top_vals, color=cmap_bars, alpha=0.8, height=0.5)
ax_upper.text(0.5, 1.15, "Mean absolute SHAP value", transform=ax_upper.transAxes, fontproperties=font_en_65, ha="center", va="bottom")
max_val = float(np.max(top_vals)) if len(top_vals) > 0 else 0.1
upper_limit = np.ceil(max_val * 110) / 100 if max_val > 0 else 0.1
ax_upper.set_xlim(0, upper_limit)
ax_upper.xaxis.set_major_locator(ticker.LinearLocator(5))
ax_upper.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
for label in ax_upper.get_xticklabels():
    label.set_fontproperties(font_en_65)
ax_upper.tick_params(axis="x", length=2, pad=1, width=0.5)

# 散点图
cmap_dots = LinearSegmentedColormap.from_list("shap_advanced", ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"])
X_subset = X_sample[top_feats]
X_range = X_subset.max() - X_subset.min()
X_range[X_range == 0] = 1
X_std = (X_subset - X_subset.min()) / X_range

def _feature_display_name(feature_name: str) -> str:
    name = str(feature_name)
    mapping = {
        "temperature_2m_max": "Tmax", "temperature_2m_min": "Tmin",
        "precipitation_sum": "PRE", "wind_speed_10m_max": "WSmax",
        "DEWP": "DEWP", "SurfacePressure_mean": "SP", "RH_mean": "RH",
        "O3_8h": "O3"
    }
    return mapping.get(name, name)

for j, f in enumerate(top_feats):
    feat_idx = feature_names.index(str(f))
    jitter = np.random.normal(0, 0.1, shap_vals.shape[0])
    ax_lower.scatter(
        shap_vals[:, feat_idx], j + jitter, c=cmap_dots(X_std[f].values.astype(float)),
        s=4, alpha=0.8, edgecolors="none", zorder=10
    )

ax_lower.text(0.5, -0.12, "SHAP value", transform=ax_lower.transAxes, fontproperties=font_en_65, ha="center", va="top")
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

# 色条
ax_ins = inset_axes(
    ax_lower, width="5%", height="100%", loc="lower left",
    bbox_to_anchor=(1.12, 0, 1, 1), bbox_transform=ax_lower.transAxes, borderpad=0
)
colorbar = mpl.colorbar.ColorbarBase(ax_ins, cmap=cmap_dots, orientation="vertical")
colorbar.outline.set_linewidth(0.5)
colorbar.set_ticks([])
colorbar.ax.text(0.5, 1.03, "High", transform=colorbar.ax.transAxes, ha="center", va="bottom", fontproperties=font_en_65)
colorbar.ax.text(0.5, -0.03, "Low", transform=colorbar.ax.transAxes, ha="center", va="top", fontproperties=font_en_65)

# 在右上角添加 (c) 标记
ax_lower.text(0.95, 0.95, '(c)', transform=ax_lower.transAxes, fontsize=5, ha='right', va='top')

# 输出
output_base = os.path.join(OUTPUT_DIR, "Stacking_Overall_SHAP_Summary")
plt.savefig(f"{output_base}.tiff", dpi=300, format="tiff", bbox_inches="tight")
plt.close()

print(f"[INFO] 已完成 Stacking 融合模型 SHAP 总览图绘制: {output_base}.tiff")
