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
SEQ_COLS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
            "wind_speed_10m_max", "DEWP", "SurfacePressure_mean", "RH_mean"]
TAB_COLS = ["PM10", "NO2", "SO2", "CO", "O3_8h"]  # 表格特征

df = pd.read_excel(EXCEL_PATH)
df = df.dropna(subset=[TARGET_COL] + SEQ_COLS + TAB_COLS).reset_index(drop=True)

y = df[TARGET_COL].values.astype(float)
X_seq = df[SEQ_COLS].values.astype(float)
X_tab = df[TAB_COLS].values.astype(float)

# LSTM 输入格式：(样本数, 时间步, 特征数)
X_seq = X_seq.reshape(-1, len(SEQ_COLS), 1)

# ========== 2. 7:3 划分 ==========
X_tab_tr, X_tab_va, X_seq_tr, X_seq_va, y_tr, y_va = train_test_split(
    X_tab, X_seq, y, test_size=0.3, random_state=42
)

# ========== 3. LSTM 模型 ==========
def build_lstm(input_shape):
    # 略微减小 LSTM 容量，降低一点拟合能力
    model = Sequential()
    model.add(LSTM(16, input_shape=input_shape))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(X_tr, y_tr, X_va, y_va):
    model = build_lstm(X_tr.shape[1:])
    es = EarlyStopping(monitor='val_loss', patience=10,
                       restore_best_weights=True, verbose=0)
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=80,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )
    return model

# ========== 4. 动态门控融合 ==========
def build_gate():
    # 输入为 [pred_cb, pred_lstm] 两个值
    m = Sequential()
    m.add(Dense(4, activation='relu', input_shape=(2,)))
    m.add(Dense(1, activation='sigmoid'))  # 输出 w(x) ∈ (0,1)
    m.compile(optimizer='adam', loss='mse')
    return m

def dynamic_gate_fusion(X_tab_tr, X_tab_va, X_seq_tr, X_seq_va, y_tr, y_va):
    # 1) 训练 CatBoost（表格特征）
    cb = CatBoostRegressor(
        loss_function='RMSE',
        depth=4,
        learning_rate=0.03,
        n_estimators=200,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=0
    )
    cb.fit(X_tab_tr, y_tr)
    cb_tr = cb.predict(X_tab_tr)
    cb_va = cb.predict(X_tab_va)

    # 2) 训练 LSTM（序列特征）
    lstm = train_lstm(X_seq_tr, y_tr, X_seq_va, y_va)
    lstm_tr = lstm.predict(X_seq_tr).ravel()
    lstm_va = lstm.predict(X_seq_va).ravel()

    # 3) 训练门控网络，根据 [cb_pred, lstm_pred] 学习 w(x)
    Xg_tr = np.column_stack([cb_tr, lstm_tr])
    Xg_va = np.column_stack([cb_va, lstm_va])

    gate = build_gate()
    es = EarlyStopping(monitor='val_loss', patience=10,
                       restore_best_weights=True, verbose=0)
    gate.fit(
        Xg_tr, y_tr,
        validation_data=(Xg_va, y_va),
        epochs=120,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    w_tr = gate.predict(Xg_tr).ravel()
    w_va = gate.predict(Xg_va).ravel()

    # 最终预测：ŷ = w * y_cb + (1-w) * y_lstm
    y_pred_tr = w_tr * cb_tr + (1.0 - w_tr) * lstm_tr
    y_pred_va = w_va * cb_va + (1.0 - w_va) * lstm_va

    return y_pred_tr, y_pred_va

# ========== 5. 指标计算 ==========
def calc_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true, ddof=1) / rmse
    return r2, rmse, rpd

# ========== 6. 画 1:1 图（配色与前两个一致） ==========
def plot_1to1(
    y_tr, y_pred_tr, y_va, y_pred_va,
    metrics_tr, metrics_va,
    save_path=r"C:\Users\焦焦\Desktop\修改1\融合模型\dynamic_gate_result.tif"
):
    plt.figure(figsize=(5, 5))
    gs = GridSpec(4, 4)
    ax_scatter = plt.subplot(gs[1:4, 0:3])
    ax_histx = plt.subplot(gs[0, 0:3], sharex=ax_scatter)
    ax_histy = plt.subplot(gs[1:4, 3], sharey=ax_scatter)

    # 散点：只彩色边框，无填充
    ax_scatter.scatter(
        y_tr, y_pred_tr,
        marker="o",
        facecolors="none",
        edgecolors="#7FACD6",
        s=30,
        label="Calibration",
    )
    ax_scatter.scatter(
        y_va, y_pred_va,
        marker="^",
        facecolors="none",
        edgecolors="#A5678E",
        s=30,
        label="Validation",
    )

    # 1:1 线
    minv = min(np.min(y_tr), np.min(y_va),
               np.min(y_pred_tr), np.min(y_pred_va))
    maxv = max(np.max(y_tr), np.max(y_va),
               np.max(y_pred_tr), np.max(y_pred_va))
    ax_scatter.plot([minv, maxv], [minv, maxv],
                    'k-', lw=1, label="1:1 Line")

    # 拟合线
    if len(y_tr) > 1:
        coef_tr = np.polyfit(y_tr, y_pred_tr, 1)
        ax_scatter.plot(
            [minv, maxv],
            np.polyval(coef_tr, [minv, maxv]),
            'k-.', lw=1, label="Training Fit"
        )
    if len(y_va) > 1:
        coef_va = np.polyfit(y_va, y_pred_va, 1)
        ax_scatter.plot(
            [minv, maxv],
            np.polyval(coef_va, [minv, maxv]),
            'k:', lw=1, label="Validation Fit"
        )

    ax_scatter.set_xlabel("Measured values (μg/m³)")
    ax_scatter.set_ylabel("Predicted values (μg/m³)")
    ax_scatter.legend(loc="lower right", frameon=False)

    # 指标文字（位置与 Stacking 图保持一致，字号与其完全相同）
    ax_scatter.text(
        0.05, 0.92, "Calibration",
        transform=ax_scatter.transAxes, fontsize=9
    )
    ax_scatter.text(
        0.05, 0.75,
        f"R²={metrics_tr[0]:.2f}\nRMSE={metrics_tr[1]:.2f}\nRPD={metrics_tr[2]:.2f}",
        transform=ax_scatter.transAxes, fontsize=9
    )

    ax_scatter.text(
        0.52, 0.92, "Validation",
        transform=ax_scatter.transAxes, fontsize=9
    )

    # 面板标注：放在顶部直方图右上角（与参考图一致）
    ax_histx.text(
        0.98, 0.92, "(b)",
        transform=ax_histx.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax_scatter.text(
        0.52, 0.75,
        f"R²={metrics_va[0]:.2f}\nRMSE={metrics_va[1]:.2f}\nRPD={metrics_va[2]:.2f}",
        transform=ax_scatter.transAxes, fontsize=9
    )

    # 直方图：填充色 #F3CCDB
    bins = 20
    all_y = np.concatenate([y_tr, y_va])
    all_pred = np.concatenate([y_pred_tr, y_pred_va])

    ax_histx.hist(
        all_y, bins=bins,
        color="#F3CCDB",
        edgecolor="black",
        density=True,
    )
    x_line = np.linspace(minv, maxv, 100)
    ax_histx.plot(
        x_line,
        norm.pdf(x_line, np.mean(all_y), np.std(all_y)),
        "r--", lw=1
    )
    ax_histx.tick_params(
        axis="both", which="both",
        labelbottom=False, labelleft=False, length=0
    )

    ax_histy.hist(
        all_pred, bins=bins,
        color="#F3CCDB",
        edgecolor="black",
        density=True,
        orientation="horizontal",
    )
    y_line = np.linspace(minv, maxv, 100)
    ax_histy.plot(
        norm.pdf(y_line, np.mean(all_pred), np.std(all_pred)),
        y_line, "r--", lw=1
    )
    ax_histy.tick_params(
        axis="both", which="both",
        labelbottom=False, labelleft=False, length=0
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# ========== 7. 主流程 ==========
if __name__ == "__main__":
    y_pred_tr, y_pred_va = dynamic_gate_fusion(
        X_tab_tr, X_tab_va, X_seq_tr, X_seq_va, y_tr, y_va
    )
    metrics_tr = calc_metrics(y_tr, y_pred_tr)
    metrics_va = calc_metrics(y_va, y_pred_va)

    print("【动态门控融合】")
    print("训练集: R²={:.3f}, RMSE={:.3f}, RPD={:.3f}".format(*metrics_tr))
    print("验证集: R²={:.3f}, RMSE={:.3f}, RPD={:.3f}".format(*metrics_va))

    plot_1to1(
        y_tr, y_pred_tr,
        y_va, y_pred_va,
        metrics_tr, metrics_va
    )