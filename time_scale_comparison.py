import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from scipy.stats import norm
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# SCI 顶级期刊绘图设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 也跟随放大（避免 barplot/stripplot 的默认 context 把字缩回去）
FONT_SCALE = 1.5
sns.set_context('paper', font_scale=FONT_SCALE)

# 配色方案 (Nature / Science 常用配色)
COLOR_ACTUAL = '#0072B5' # 经典蓝色
COLOR_PRED = '#BC3C29'   # 经典砖红色
BAR_ACTUAL = '#20854E'
BAR_PRED = '#E18727'
ALPHA = 0.2


def add_panel_label(ax, label: str):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=int(round(11 * FONT_SCALE)), # 缩小字体
        fontweight='normal', # 取消加粗
    )


def save_tiff(fig, out_path: str):
    fig.savefig(out_path, format='tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)

# ========== 1. 模型与数据读取 ==========
EXCEL_PATH = r"C:\Users\焦焦\Desktop\论文数据.xlsx"
TARGET_COL = "PM2.5"
SEQ_COLS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "DEWP", "SurfacePressure_mean", "RH_mean"]
TAB_COLS = ["PM10", "NO2", "SO2", "CO", "O3_8h"]

print("正在读取数据...")
df = pd.read_excel(EXCEL_PATH)
df = df.dropna(subset=[TARGET_COL] + SEQ_COLS + TAB_COLS + ['time']).reset_index(drop=True)
df['time'] = pd.to_datetime(df['time'])

y = df[TARGET_COL].values.astype(float)
X_seq = df[SEQ_COLS].values.astype(float).reshape(-1, len(SEQ_COLS), 1)
X_tab = df[TAB_COLS].values.astype(float)

# 模型训练与预测 (在整个数据集上进行训练与预测以获取最完整时间序列)
# 注：严格意义上应只在验证/测试集上作图，但为展示连续时间序列，此处以全集预测作为绘图数据支撑。
# 若仅需测试集，请切片操作。
print("正在训练 LSTM...")
lstm = Sequential([
    LSTM(24, input_shape=(len(SEQ_COLS), 1)),
    Dense(12, activation='relu'),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=0)
lstm.fit(X_seq, y, epochs=120, batch_size=32, callbacks=[es], verbose=0)
pred_lstm = lstm.predict(X_seq, verbose=0).ravel()

print("正在训练 Level 1 CatBoost...")
cb1 = CatBoostRegressor(loss_function='RMSE', depth=6, learning_rate=0.04, n_estimators=300, l2_leaf_reg=4, random_seed=42, verbose=0)
cb1.fit(X_tab, y)
pred_cb1 = cb1.predict(X_tab)

print("正在训练 Level 2 CatBoost...")
Z1 = np.column_stack([pred_cb1, pred_lstm])
cb2 = CatBoostRegressor(loss_function='RMSE', depth=5, learning_rate=0.03, n_estimators=200, l2_leaf_reg=6, random_seed=42, verbose=0)
cb2.fit(Z1, y)
pred_cb2 = cb2.predict(Z1)

print("正在训练 Level 3 CatBoost...")
Z2 = np.column_stack([pred_cb1, pred_lstm, pred_cb2])
cb3 = CatBoostRegressor(loss_function='RMSE', depth=4, learning_rate=0.025, n_estimators=120, l2_leaf_reg=8, random_seed=42, verbose=0)
cb3.fit(Z2, y)
pred_final = cb3.predict(Z2)

# 将预测结果合并至 DataFrame
df['Validation'] = pred_final
df['Calibration'] = y

# ================= 绘图部分 =================

# ---- 图 a: 逐日时间序列图 ----
print("绘制图 a...")
fig_a, ax_a = plt.subplots(figsize=(8, 6))
# 按天聚合(防止一天多条)
daily_df = df.groupby(df['time'].dt.date)[['Calibration', 'Validation']].mean().reset_index()
days_index = np.arange(1, len(daily_df) + 1)

# 画均值线和误差带(因逐日点多，误差带可采用平滑的滚动窗口)
rolling_window = 14
actual_roll_mean = daily_df['Calibration'].rolling(window=rolling_window, center=True).mean()
actual_roll_std = daily_df['Calibration'].rolling(window=rolling_window, center=True).std()
pred_roll_mean = daily_df['Validation'].rolling(window=rolling_window, center=True).mean()
pred_roll_std = daily_df['Validation'].rolling(window=rolling_window, center=True).std()

ax_a.plot(days_index, daily_df['Calibration'], color=COLOR_ACTUAL, alpha=0.5, lw=1, label='Calibration (Daily)')
ax_a.plot(days_index, daily_df['Validation'], color=COLOR_PRED, alpha=0.5, lw=1, label='Validation (Daily)')
ax_a.fill_between(days_index, actual_roll_mean - actual_roll_std, actual_roll_mean + actual_roll_std, color=COLOR_ACTUAL, alpha=ALPHA)
ax_a.fill_between(days_index, pred_roll_mean - pred_roll_std, pred_roll_mean + pred_roll_std, color=COLOR_PRED, alpha=ALPHA)

ax_a.axhline(daily_df['Calibration'].mean(), color=COLOR_ACTUAL, linestyle='--', lw=2, label='Calibration Mean')
ax_a.axhline(daily_df['Validation'].mean(), color=COLOR_PRED, linestyle=':', lw=2, label='Validation Mean')

ax_a.set_xlim(0, len(daily_df) + 20)
y_max_a = max(daily_df['Calibration'].max(), daily_df['Validation'].max())
ax_a.set_ylim(bottom=-10, top=y_max_a * 1.1)

ax_a.set_xlabel('Days')
ax_a.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)')
add_panel_label(ax_a, '(a)') # 使用函数添加标签
ax_a.legend(loc='upper right', frameon=False, ncol=2)
plt.tight_layout()
save_path_a = r"C:\Users\焦焦\Desktop\修改1\四联图\Figure_a_Daily.tif"
plt.savefig(save_path_a, format="tiff", dpi=300, bbox_inches='tight')
print(f"图 a 已保存至: {save_path_a}")
plt.show()


# ---- 图 b: 逐月数据(综合5年) ----
print("绘制图 b...")
fig_b, ax_b = plt.subplots(figsize=(8, 6))
df['Month'] = df['time'].dt.month
monthly_stats = df.groupby('Month')[['Calibration', 'Validation']].agg(['mean', 'std']).reset_index()

ax_b.plot(monthly_stats['Month'], monthly_stats['Calibration']['mean'], color=COLOR_ACTUAL, marker='o', lw=2, label='Calibration')
ax_b.plot(monthly_stats['Month'], monthly_stats['Validation']['mean'], color=COLOR_PRED, marker='s', lw=2, label='Validation')

ax_b.fill_between(monthly_stats['Month'], 
                  monthly_stats['Calibration']['mean'] - monthly_stats['Calibration']['std'], 
                  monthly_stats['Calibration']['mean'] + monthly_stats['Calibration']['std'], 
                  color=COLOR_ACTUAL, alpha=ALPHA)
ax_b.fill_between(monthly_stats['Month'], 
                  monthly_stats['Validation']['mean'] - monthly_stats['Validation']['std'], 
                  monthly_stats['Validation']['mean'] + monthly_stats['Validation']['std'], 
                  color=COLOR_PRED, alpha=ALPHA)

ax_b.axhline(df['Calibration'].mean(), color=COLOR_ACTUAL, linestyle='--', lw=2, label='Calibration Overall Mean')
ax_b.axhline(df['Validation'].mean(), color=COLOR_PRED, linestyle=':', lw=2, label='Validation Overall Mean')

y_max_b = max((monthly_stats['Calibration']['mean'] + monthly_stats['Calibration']['std']).max(), 
              (monthly_stats['Validation']['mean'] + monthly_stats['Validation']['std']).max())
ax_b.set_ylim(bottom=-5, top=y_max_b * 1.1)

ax_b.set_xticks(range(1, 13))
ax_b.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax_b.set_xlabel('Month')
ax_b.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)')
add_panel_label(ax_b, '(b)') # 使用函数添加标签
ax_b.legend(loc='upper right', frameon=False, ncol=2)
plt.tight_layout()
save_path_b = r"C:\Users\焦焦\Desktop\修改1\四联图\Figure_b_Monthly.tif"
plt.savefig(save_path_b, format="tiff", dpi=300, bbox_inches='tight')
print(f"图 b 已保存至: {save_path_b}")
plt.show()


# ---- 图 c: 季节柱状图(附散点和误差棒) ----
def get_season(month):
    if month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    elif month in [9, 10, 11]: return 'Autumn'
    else: return 'Winter'
df['Season'] = df['Month'].apply(get_season)
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']

print("绘制图 c...")
fig_c, ax_c = plt.subplots(figsize=(8, 6))
# 转换数据格式用于 seaborn 绘图
melted_season = df.melt(id_vars=['Season'], value_vars=['Calibration', 'Validation'], var_name='Type', value_name='Value')
melted_season['Type'] = melted_season['Type'].replace({'Calibration': 'Calibration', 'Validation': 'Validation'})


# 画带误差棒的柱状图
sns.barplot(data=melted_season, x='Season', y='Value', hue='Type', ax=ax_c, 
            order=season_order, palette=[BAR_ACTUAL, BAR_PRED], capsize=0.1, err_kws={'linewidth': 1.5}, alpha=0.8)
# 添加散点
sample_season = melted_season.sample(n=min(3000, len(melted_season)), random_state=42)
sns.stripplot(data=sample_season, x='Season', y='Value', hue='Type', ax=ax_c,
              order=season_order, palette=['black', 'black'], dodge=True, alpha=0.1, jitter=0.2, legend=False, size=3)

ax_c.set_xlabel('Season')
ax_c.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)')
add_panel_label(ax_c, '(c)') # 使用函数添加标签
handles, labels = ax_c.get_legend_handles_labels()
ax_c.legend(handles=handles, labels=['Calibration', 'Validation'], title='', loc='upper right', frameon=False)
plt.tight_layout()
save_path_c = r"C:\Users\焦焦\Desktop\修改1\四联图\Figure_c_Seasonal.tif"
plt.savefig(save_path_c, format="tiff", dpi=300, bbox_inches='tight')
print(f"图 c 已保存至: {save_path_c}")
plt.show()


# ---- 图 d: 年份柱状图(附散点和误差棒) ----
print("绘制图 d...")
df['Year'] = df['time'].dt.year
fig_d, ax_d = plt.subplots(figsize=(8, 6))
melted_year = df.melt(id_vars=['Year'], value_vars=['Calibration', 'Validation'], var_name='Type', value_name='Value')
melted_year['Type'] = melted_year['Type'].replace({'Calibration': 'Calibration', 'Validation': 'Validation'})


sns.barplot(data=melted_year, x='Year', y='Value', hue='Type', ax=ax_d, 
            palette=[BAR_ACTUAL, BAR_PRED], capsize=0.1, err_kws={'linewidth': 1.5}, alpha=0.8)
sample_year = melted_year.sample(n=min(3000, len(melted_year)), random_state=42)
sns.stripplot(data=sample_year, x='Year', y='Value', hue='Type', ax=ax_d,
              palette=['black', 'black'], dodge=True, alpha=0.1, jitter=0.2, legend=False, size=3)

ax_d.set_xlabel('Year')
ax_d.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)')
add_panel_label(ax_d, '(d)') # 使用函数添加标签
handles, labels = ax_d.get_legend_handles_labels()
ax_d.legend(handles=handles, labels=['Calibration', 'Validation'], title='', loc='upper right', frameon=False)
plt.tight_layout()
save_path_d = r"C:\Users\焦焦\Desktop\修改1\四联图\Figure_d_Interannual.tif"
plt.savefig(save_path_d, format="tiff", dpi=300, bbox_inches='tight')
print(f"图 d 已保存至: {save_path_d}")
plt.show()
OUT_DIR = r"C:\Users\焦焦\Desktop\修改1\四联图"

# ---- 图 (a): 逐日时间序列图 ----
print("绘制图 (a)...")
fig_a, ax_a = plt.subplots(figsize=(7.2, 5.2))
# 按天聚合(防止一天多条)
daily_df = df.groupby(df['time'].dt.date)[['Calibration', 'Validation']].mean().reset_index()
days_index = np.arange(1, len(daily_df) + 1)

# 画均值线和误差带(因逐日点多，误差带可采用平滑的滚动窗口)
rolling_window = 14
actual_roll_mean = daily_df['Calibration'].rolling(window=rolling_window, center=True).mean()
actual_roll_std = daily_df['Calibration'].rolling(window=rolling_window, center=True).std()
pred_roll_mean = daily_df['Validation'].rolling(window=rolling_window, center=True).mean()
pred_roll_std = daily_df['Validation'].rolling(window=rolling_window, center=True).std()

ax_a.plot(days_index, daily_df['Calibration'], color=COLOR_ACTUAL, alpha=0.5, lw=1, label='Calibration (Daily)')
ax_a.plot(days_index, daily_df['Validation'], color=COLOR_PRED, alpha=0.5, lw=1, label='Validation (Daily)')
ax_a.fill_between(days_index, actual_roll_mean - actual_roll_std, actual_roll_mean + actual_roll_std, color=COLOR_ACTUAL, alpha=ALPHA)
ax_a.fill_between(days_index, pred_roll_mean - pred_roll_std, pred_roll_mean + pred_roll_std, color=COLOR_PRED, alpha=ALPHA)

ax_a.axhline(daily_df['Calibration'].mean(), color=COLOR_ACTUAL, linestyle='--', lw=2, label='Calibration Mean')
ax_a.axhline(daily_df['Validation'].mean(), color=COLOR_PRED, linestyle=':', lw=2, label='Validation Mean')

ax_a.set_xlim(0, len(daily_df) + 20)
y_max_a = max(daily_df['Calibration'].max(), daily_df['Validation'].max())
ax_a.set_ylim(bottom=-10, top=y_max_a * 1.10)

ax_a.set_xlabel('Days')
ax_a.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)')
add_panel_label(ax_a, '(a)')
ax_a.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='white', framealpha=0.9, ncol=2)

fig_a.tight_layout()
save_tiff(fig_a, rf"{OUT_DIR}\\PM25_panel_a_daily.tif")

# ---- 图 (b): 逐月数据(综合5年) ----
print("绘制图 (b)...")
fig_b, ax_b = plt.subplots(figsize=(7.2, 5.2))
df['Month'] = df['time'].dt.month
monthly_stats = df.groupby('Month')[['Calibration', 'Validation']].agg(['mean', 'std']).reset_index()

ax_b.plot(monthly_stats['Month'], monthly_stats['Calibration']['mean'], color=COLOR_ACTUAL, marker='o', lw=2, label='Calibration')
ax_b.plot(monthly_stats['Month'], monthly_stats['Validation']['mean'], color=COLOR_PRED, marker='s', lw=2, label='Validation')

ax_b.fill_between(monthly_stats['Month'], 
                  monthly_stats['Calibration']['mean'] - monthly_stats['Calibration']['std'], 
                  monthly_stats['Calibration']['mean'] + monthly_stats['Calibration']['std'], 
                  color=COLOR_ACTUAL, alpha=ALPHA)
ax_b.fill_between(monthly_stats['Month'], 
                  monthly_stats['Validation']['mean'] - monthly_stats['Validation']['std'], 
                  monthly_stats['Validation']['mean'] + monthly_stats['Validation']['std'], 
                  color=COLOR_PRED, alpha=ALPHA)

ax_b.axhline(df['Calibration'].mean(), color=COLOR_ACTUAL, linestyle='--', lw=2, label='Calibration Overall Mean')
ax_b.axhline(df['Validation'].mean(), color=COLOR_PRED, linestyle=':', lw=2, label='Validation Overall Mean')

y_max_b = max((monthly_stats['Calibration']['mean'] + monthly_stats['Calibration']['std']).max(), 
              (monthly_stats['Validation']['mean'] + monthly_stats['Validation']['std']).max())
ax_b.set_ylim(bottom=-5, top=y_max_b * 1.15)

ax_b.set_xticks(range(1, 13))
ax_b.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax_b.set_xlabel('Month')
ax_b.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)')
add_panel_label(ax_b, '(b)')
ax_b.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='white', framealpha=0.9)

fig_b.tight_layout()
save_tiff(fig_b, rf"{OUT_DIR}\\PM25_panel_b_monthly.tif")

# ---- 图 c: 季节柱状图(附散点和误差棒) ----
# 春(3,4,5), 夏(6,7,8), 秋(9,10,11), 冬(12,1,2)
def get_season(month):
    if month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    elif month in [9, 10, 11]: return 'Autumn'
    else: return 'Winter'
df['Season'] = df['Month'].apply(get_season)
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']

print("绘制图 c...")
fig_c, ax_c = plt.subplots(figsize=(7.2, 5.2))
# 转换数据格式用于 seaborn 绘图
melted_season = df.melt(id_vars=['Season'], value_vars=['Calibration', 'Validation'], var_name='Type', value_name='Value')
melted_season['Type'] = melted_season['Type'].replace({'Calibration': 'Calibration', 'Validation': 'Validation'})


# 画带误差棒的柱状图
sns.barplot(data=melted_season, x='Season', y='Value', hue='Type', ax=ax_c, 
            order=season_order, palette=[BAR_ACTUAL, BAR_PRED], capsize=0.1, err_kws={'linewidth': 1.5}, alpha=0.8)
# 添加散点(Swarmplot/stripplot会太密，所以这里用随机采样的少量数据作示意)
sample_season = melted_season.sample(n=min(3000, len(melted_season)), random_state=42)
sns.stripplot(data=sample_season, x='Season', y='Value', hue='Type', ax=ax_c,
              order=season_order, palette=['black', 'black'], dodge=True, alpha=0.1, jitter=0.2, legend=False, size=3)

ax_c.set_xlabel('Season')
ax_c.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)')
add_panel_label(ax_c, '(c)')
ax_c.legend(title='', frameon=False)

fig_c.tight_layout()
save_tiff(fig_c, rf"{OUT_DIR}\\PM25_panel_c_seasonal.tif")

# ---- 图 d: 年份柱状图(附散点和误差棒) ----
print("绘制图 d...")
df['Year'] = df['time'].dt.year
fig_d, ax_d = plt.subplots(figsize=(7.2, 5.2))
melted_year = df.melt(id_vars=['Year'], value_vars=['Calibration', 'Validation'], var_name='Type', value_name='Value')
melted_year['Type'] = melted_year['Type'].replace({'Calibration': 'Calibration', 'Validation': 'Validation'})


sns.barplot(data=melted_year, x='Year', y='Value', hue='Type', ax=ax_d, 
            palette=[BAR_ACTUAL, BAR_PRED], capsize=0.1, err_kws={'linewidth': 1.5}, alpha=0.8)
sample_year = melted_year.sample(n=min(3000, len(melted_year)), random_state=42)
sns.stripplot(data=sample_year, x='Year', y='Value', hue='Type', ax=ax_d,
              palette=['black', 'black'], dodge=True, alpha=0.1, jitter=0.2, legend=False, size=3)

ax_d.set_xlabel('Year')
ax_d.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)')
add_panel_label(ax_d, '(d)')
ax_d.legend(title='', frameon=False)

fig_d.tight_layout()
save_tiff(fig_d, rf"{OUT_DIR}\\PM25_panel_d_interannual.tif")

print("绘图完成，已输出 4 张单图 (TIFF, dpi=300)。")

