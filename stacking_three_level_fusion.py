import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 环境变量控制是否弹出图窗：默认弹出；批量运行可设 FUSION_SHOW=0
SHOW_PLOTS = os.environ.get("FUSION_SHOW", "1") != "0"

# ========== 1. 数据读取与预处理 ==========
EXCEL_PATH = r"C:\Users\焦焦\Desktop\论文数据.xlsx"
TARGET_COL = "PM2.5"
SEQ_COLS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "DEWP", "SurfacePressure_mean", "RH_mean"]
TAB_COLS = ["PM10", "NO2", "SO2", "CO", "O3_8h"]  # 除时间和目标、序列特征外的表格特征

df = pd.read_excel(EXCEL_PATH)
df = df.dropna(subset=[TARGET_COL] + SEQ_COLS + TAB_COLS).reset_index(drop=True)

y = df[TARGET_COL].values.astype(float)
X_seq = df[SEQ_COLS].values.astype(float)
X_tab = df[TAB_COLS].values.astype(float)

# LSTM输入格式：(样本数, 时间步, 特征数)
# 这里假设每个样本的序列特征就是一行（如气象变量），如需多步序列请调整
X_seq = X_seq.reshape(-1, len(SEQ_COLS), 1)

# ========== 2. 7:3 划分 ==========
X_tab_tr, X_tab_va, X_seq_tr, X_seq_va, y_tr, y_va = train_test_split(
    X_tab, X_seq, y, test_size=0.3, random_state=42
)

# ========== 3. LSTM模型 ==========
def build_lstm(input_shape):
    # 适度增强 LSTM 容量，使拟合能力略高一些
    model = Sequential()
    model.add(LSTM(24, input_shape=input_shape))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(X_tr, y_tr, X_va, y_va):
    model = build_lstm(X_tr.shape[1:])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=120, batch_size=32, callbacks=[es], verbose=0)
    return model

# ========== 4. 三级 Stacking ==========
def stacking_3level(X_tab_tr, X_tab_va, X_seq_tr, X_seq_va, y_tr, y_va):
    # Level 1
    # 第一层 CatBoost：在之前基础上适度增强
    cb1 = CatBoostRegressor(
        loss_function='RMSE',
        depth=6,
        learning_rate=0.04,
        n_estimators=300,
        l2_leaf_reg=4,
        random_seed=42,
        verbose=0,
    )
    cb1.fit(X_tab_tr, y_tr)
    lstm1 = train_lstm(X_seq_tr, y_tr, X_seq_va, y_va)
    pred_cb1_tr = cb1.predict(X_tab_tr)
    pred_cb1_va = cb1.predict(X_tab_va)
    pred_lstm1_tr = lstm1.predict(X_seq_tr).ravel()
    pred_lstm1_va = lstm1.predict(X_seq_va).ravel()

    # Level 2
    Z1_tr = np.column_stack([pred_cb1_tr, pred_lstm1_tr])
    Z1_va = np.column_stack([pred_cb1_va, pred_lstm1_va])
    # 第二层元学习器：介于之前“瘦身版”和“加强版”之间
    cb2 = CatBoostRegressor(
        loss_function='RMSE',
        depth=5,
        learning_rate=0.03,
        n_estimators=200,
        l2_leaf_reg=6,
        random_seed=42,
        verbose=0,
    )
    cb2.fit(Z1_tr, y_tr)
    pred_cb2_tr = cb2.predict(Z1_tr)
    pred_cb2_va = cb2.predict(Z1_va)

    # Level 3
    Z2_tr = np.column_stack([pred_cb1_tr, pred_lstm1_tr, pred_cb2_tr])
    Z2_va = np.column_stack([pred_cb1_va, pred_lstm1_va, pred_cb2_va])
    # 第三层元学习器：略小于第二层，防止过拟合
    cb3 = CatBoostRegressor(
        loss_function='RMSE',
        depth=4,
        learning_rate=0.025,
        n_estimators=120,
        l2_leaf_reg=8,
        random_seed=42,
        verbose=0,
    )
    cb3.fit(Z2_tr, y_tr)
    pred_final_tr = cb3.predict(Z2_tr)
    pred_final_va = cb3.predict(Z2_va)
    return pred_final_tr, pred_final_va

# ========== 5. 指标计算 ==========
def calc_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true, ddof=1) / rmse
    return r2, rmse, rpd

# ========== 6. 画1:1图 ==========
def plot_1to1(y_tr, y_pred_tr, y_va, y_pred_va, metrics_tr, metrics_va,
              save_path=r"C:\\Users\\焦焦\\Desktop\\修改1\\融合模型\\stacking_3level_result.tif"):
    plt.figure(figsize=(5,5))
    gs = GridSpec(4,4)
    ax_scatter = plt.subplot(gs[1:4,0:3])
    ax_histx = plt.subplot(gs[0,0:3], sharex=ax_scatter)
    ax_histy = plt.subplot(gs[1:4,3], sharey=ax_scatter)

    # 散点：分别使用指定颜色
    ax_scatter.scatter(
        y_tr,
        y_pred_tr,
        marker="o",
        facecolors="none",
        edgecolors="#7FACD6",
        s=30,
        label="Calibration",
    )
    ax_scatter.scatter(
        y_va,
        y_pred_va,
        marker="^",
        facecolors="none",
        edgecolors="#A5678E",
        s=30,
        label="Validation",
    )

    # 1:1线
    minv = min(np.min(y_tr), np.min(y_va), np.min(y_pred_tr), np.min(y_pred_va))
    maxv = max(np.max(y_tr), np.max(y_va), np.max(y_pred_tr), np.max(y_pred_va))
    ax_scatter.plot([minv, maxv], [minv, maxv], 'k-', lw=1, label="1:1 Line")

    # 拟合线
    if len(y_tr) > 1:
        coef = np.polyfit(y_tr, y_pred_tr, 1)
        ax_scatter.plot([minv, maxv], np.polyval(coef, [minv, maxv]), 'k-.', lw=1, label="Training Fit")
    if len(y_va) > 1:
        coef = np.polyfit(y_va, y_pred_va, 1)
        ax_scatter.plot([minv, maxv], np.polyval(coef, [minv, maxv]), 'k:', lw=1, label="Validation Fit")

    ax_scatter.set_xlabel("Measured values (μg/m³)")
    ax_scatter.set_ylabel("Predicted values (μg/m³)")
    ax_scatter.legend(loc="lower right", frameon=False)

    # 指标
    ax_scatter.text(0.05, 0.92, "Calibration", transform=ax_scatter.transAxes, fontsize=9)
    ax_scatter.text(0.05, 0.75, f"R²={metrics_tr[0]:.2f}\nRMSE={metrics_tr[1]:.2f}\nRPD={metrics_tr[2]:.2f}", transform=ax_scatter.transAxes, fontsize=9)
    # 将验证集文字整体进一步左移，避免与 1:1 线重叠
    ax_scatter.text(0.52, 0.92, "Validation", transform=ax_scatter.transAxes, fontsize=9)
    ax_scatter.text(0.52, 0.75, f"R²={metrics_va[0]:.2f}\nRMSE={metrics_va[1]:.2f}\nRPD={metrics_va[2]:.2f}", transform=ax_scatter.transAxes, fontsize=9)

    # 面板标注：放在顶部直方图右上角（与参考图一致）
    ax_histx.text(
        0.98, 0.92, "(a)",
        transform=ax_histx.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    # 直方图
    bins = 20
    # 顶部直方图：改用新的填充颜色，并保留外边框
    ax_histx.hist(
        np.concatenate([y_tr, y_va]),
        bins=bins,
        color="#F3CCDB",
        edgecolor="black",
        density=True,
    )
    x_line = np.linspace(minv, maxv, 100)
    ax_histx.plot(x_line, norm.pdf(x_line, np.mean(np.concatenate([y_tr, y_va])), np.std(np.concatenate([y_tr, y_va]))), "r--", lw=1)
    # 去掉刻度和数字，但保留边框
    ax_histx.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, length=0)

    # 右侧直方图：同样使用新的填充颜色，并保留外边框
    ax_histy.hist(
        np.concatenate([y_pred_tr, y_pred_va]),
        bins=bins,
        color="#F3CCDB",
        edgecolor="black",
        density=True,
        orientation="horizontal",
    )
    y_line = np.linspace(minv, maxv, 100)
    ax_histy.plot(norm.pdf(y_line, np.mean(np.concatenate([y_pred_tr, y_pred_va])), np.std(np.concatenate([y_pred_tr, y_pred_va]))), y_line, "r--", lw=1)
    ax_histy.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, length=0)

    plt.tight_layout()
    # 保存为 tiff，dpi=300
    plt.savefig(save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# ========== 7. 主流程 ==========
if __name__ == "__main__":
    y_pred_tr, y_pred_va = stacking_3level(X_tab_tr, X_tab_va, X_seq_tr, X_seq_va, y_tr, y_va)
    metrics_tr = calc_metrics(y_tr, y_pred_tr)
    metrics_va = calc_metrics(y_va, y_pred_va)
    print("训练集: R²={:.3f}, RMSE={:.3f}, RPD={:.3f}".format(*metrics_tr))
    print("验证集: R²={:.3f}, RMSE={:.3f}, RPD={:.3f}".format(*metrics_va))
    # 保存为 tiff 格式
    plot_1to1(y_tr, y_pred_tr, y_va, y_pred_va, metrics_tr, metrics_va)