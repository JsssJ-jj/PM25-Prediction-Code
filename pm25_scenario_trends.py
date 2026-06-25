import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ========== 1. 模型定义与数据处理函数 ==========
TARGET_COL = "PM2.5"
SEQ_COLS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "DEWP", "SurfacePressure_mean", "RH_mean"]
TAB_COLS = ["PM10", "NO2", "SO2", "CO", "O3_8h"]

def preprocess_data(df, include_y=True):
    required_cols = SEQ_COLS + TAB_COLS
    if include_y:
        required_cols += [TARGET_COL]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    X_seq = df[SEQ_COLS].values.astype(float)
    X_tab = df[TAB_COLS].values.astype(float)
    X_seq = X_seq.reshape(-1, len(SEQ_COLS), 1)
    if include_y:
        y = df[TARGET_COL].values.astype(float)
        return X_tab, X_seq, y
    else:
        return X_tab, X_seq

def build_lstm(input_shape):
    model = Sequential([LSTM(24, input_shape=input_shape), Dense(12, activation='relu'), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(X_tr, y_tr, X_va, y_va):
    model = build_lstm(X_tr.shape[1:])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=120, batch_size=32, callbacks=[es], verbose=0)
    return model

class StackingModel:
    def __init__(self):
        self.cb1 = CatBoostRegressor(loss_function='RMSE', depth=6, learning_rate=0.04, n_estimators=300, l2_leaf_reg=4, random_seed=42, verbose=0)
        self.lstm1 = None
        self.cb2 = CatBoostRegressor(loss_function='RMSE', depth=5, learning_rate=0.03, n_estimators=200, l2_leaf_reg=6, random_seed=42, verbose=0)
        self.cb3 = CatBoostRegressor(loss_function='RMSE', depth=4, learning_rate=0.025, n_estimators=120, l2_leaf_reg=8, random_seed=42, verbose=0)

    def fit(self, X_tab_tr, X_seq_tr, y_tr, X_tab_va, X_seq_va, y_va):
        print("开始训练Stacking模型...")
        self.cb1.fit(X_tab_tr, y_tr)
        self.lstm1 = train_lstm(X_seq_tr, y_tr, X_seq_va, y_va)
        pred_cb1_tr = self.cb1.predict(X_tab_tr)
        pred_lstm1_tr = self.lstm1.predict(X_seq_tr).ravel()
        Z1_tr = np.column_stack([pred_cb1_tr, pred_lstm1_tr])
        self.cb2.fit(Z1_tr, y_tr)
        pred_cb2_tr = self.cb2.predict(Z1_tr)
        Z2_tr = np.column_stack([pred_cb1_tr, pred_lstm1_tr, pred_cb2_tr])
        self.cb3.fit(Z2_tr, y_tr)
        print("模型训练完成！")

    def predict(self, X_tab, X_seq):
        pred_cb1 = self.cb1.predict(X_tab)
        pred_lstm1 = self.lstm1.predict(X_seq).ravel()
        Z1 = np.column_stack([pred_cb1, pred_lstm1])
        pred_cb2 = self.cb2.predict(Z1)
        # ！！！这里修复了两个变量名错误！！！
        Z2 = np.column_stack([pred_cb1, pred_lstm1, pred_cb2])
        pred_final = self.cb3.predict(Z2)
        return pred_final

# ========== 2. 情景定义与文件路径 ==========
BASE_PATH = r"C:\Users\焦焦\Desktop"
SCENARIOS = {
    "S0": {"label": "S0", "path": os.path.join(BASE_PATH, "论文数据.xlsx"), "color": "#030A0C", "marker": "o"},
    "S1": {"label": "S1", "path": os.path.join(BASE_PATH, "修改1", "情景应用", "情景应用数据  温度.xlsx"), "color": "#E41A1C", "marker": "^", "linestyle": "--"},
    "S2": {"label": "S2", "path": os.path.join(BASE_PATH, "修改1", "CO", "CO+温度调整.xlsx"), "color": "#086837", "marker": "s", "linestyle": "-."},
    "S3": {"label": "S3", "path": os.path.join(BASE_PATH, "修改1", "CO", "CO+温度调整.xlsx"), "color": "#B97817", "marker": "D", "linestyle": ":"}
}

# ========== 3. 主流程：训练、预测与绘图 ==========
if __name__ == "__main__":
    # --- 步骤 1: 训练模型 ---
    print("加载并训练模型...")
    s0_df = pd.read_excel(SCENARIOS["S0"]["path"])
    DATE_COLUMN = 'time'
    s0_df[DATE_COLUMN] = pd.to_datetime(s0_df[DATE_COLUMN])
    X_tab_s0, X_seq_s0, y_s0 = preprocess_data(s0_df, include_y=True)
    X_tab_tr, X_tab_va, X_seq_tr, X_seq_va, y_tr, y_va = train_test_split(X_tab_s0, X_seq_s0, y_s0, test_size=0.3, random_state=42)
    stacking_model = StackingModel()
    stacking_model.fit(X_tab_tr, X_seq_tr, y_tr, X_tab_va, X_seq_va, y_va)

    # --- 步骤 2: 情景模拟预测 ---
    predictions_for_future = {}
    BASE_PERIOD_START = '2020-01-01'
    BASE_PERIOD_END = '2020-03-31'
    print(f"\n开始进行情景模拟，使用 {BASE_PERIOD_START} 到 {BASE_PERIOD_END} 的数据作为输入...")
    for name, config in SCENARIOS.items():
        print(f"  - 正在处理情景: {name}")
        df_scenario_source = pd.read_excel(config["path"])
        df_scenario_source[DATE_COLUMN] = pd.to_datetime(df_scenario_source[DATE_COLUMN])
        mask = (df_scenario_source[DATE_COLUMN] >= BASE_PERIOD_START) & (df_scenario_source[DATE_COLUMN] <= BASE_PERIOD_END)
        input_data = df_scenario_source.loc[mask]
        if input_data.empty:
            print(f"    警告: 在情景 {name} 的文件中找不到基准时段的数据。")
            continue
        X_tab_input, X_seq_input = preprocess_data(input_data, include_y=False)
        predictions_for_future[name] = stacking_model.predict(X_tab_input, X_seq_input)

    # --- 步骤 3: 绘制SCI高清图 ---
    print("\n正在绘制最终布局的图表...")
    plt.style.use('seaborn-v0_8-ticks')
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap('Paired').colors
    SCENARIOS["S0"]["color"] = colors[1]
    SCENARIOS["S1"]["color"] = colors[5]
    SCENARIOS["S2"]["color"] = colors[3]
    SCENARIOS["S3"]["color"] = colors[7]

    if predictions_for_future:
        num_days = len(next(iter(predictions_for_future.values())))
    else:
        num_days = 90

    # 十天间隔
    ten_day_ticks = np.arange(1, num_days + 1, 10)
    month_starts = [1, 32, 61]
    month_labels = ['Jan', 'Feb', 'Mar']

    for name, pred_values in predictions_for_future.items():
        config = SCENARIOS[name]
        days = np.arange(1, len(pred_values) + 1)
        ax.plot(
            days, pred_values,
            color=config["color"],
            marker=config.get("marker", "o"),
            linestyle=config.get("linestyle", "-"),
            label=config["label"],
            markersize=5,
            markeredgewidth=1.5,
            linewidth=1.5,
            alpha=0.9
        )

    # 预警线
    ax.axhline(y=75, color='red', linestyle='-', linewidth=2.5, alpha=0.85, label='Pollution Warning Line (75μg/m³)')

    # 十天间隔竖线
    for tick in ten_day_ticks:
        ax.axvline(x=tick, color='gray', linestyle='--', linewidth=1, alpha=0.2, zorder=0)

    ax.set_xticks(ten_day_ticks)
    ax.set_xticklabels([str(tick) for tick in ten_day_ticks], fontsize=11)

    # 月份标注
    month_ranges = [(1, 31), (32, 60), (61, num_days)]
    month_midpoints = [int((start + end) / 2) for start, end in month_ranges]
    for idx, mid in enumerate(month_midpoints):
        ax.annotate(month_labels[idx], xy=(mid, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, 30), textcoords='offset points',
                    ha='center', va='top', fontsize=15, fontweight='bold', color='black',
                    annotation_clip=False)

    ax.set_xlabel("Simulated Days (Jan 2050 - Mar 2050)", fontsize=12)
    ax.set_ylabel("Predicted PM2.5 (μg/m³)", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=13, direction='in', width=1)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    ax.legend(loc='upper right', frameon=False, fontsize=12)

    if predictions_for_future:
        ax.set_xlim(0, num_days + 1)
        max_y = max(np.max(p) for p in predictions_for_future.values() if p.size > 0)
        ax.set_ylim(0, max_y * 1.15)
        xticks = ax.get_xticks()
        ax.set_xticks([tick for tick in xticks if tick != 0])

    plt.tight_layout(pad=0.5)
    save_path = r"C:\Users\焦焦\Desktop\修改1\CO\simulated_pm25_trend_sci_style_CO.tiff"
    plt.savefig(save_path, dpi=300, format='tiff', bbox_inches='tight')
    print(f"最终布局的图表已保存至: {save_path}")
    plt.show()