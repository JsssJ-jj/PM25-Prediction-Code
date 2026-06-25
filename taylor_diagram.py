# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
# 仅保存图片，不弹窗；避免某些环境下 Qt 后端初始化卡住
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import math
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.font_manager as fm

# ===================== 1. 全局样式设置 =====================
# 修复中文方框问题：加入 SimSun（宋体）和 Microsoft YaHei 支持中文显示，英文优先用 Times New Roman
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun', 'Microsoft YaHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

FONT_SIZE = 15
LINE_WIDTH = 0.8
MARKER_LW = 0.6

# 强制将这些配置应用到刻度
mpl.rcParams['xtick.major.width'] = LINE_WIDTH
mpl.rcParams['ytick.major.width'] = LINE_WIDTH

# ===================== 2. 配置路径和颜色、形状 =====================
# 你的表格绝对路径
DATA_FILE = r"C:\Users\焦焦\Desktop\修改1\单个模型\8个单个模型指标.xlsx"

# 图片保存目录（按你的要求固定保存到此文件夹）
OUTPUT_DIR = r"C:\Users\焦焦\Desktop\修改1\单个模型\八个单独模型代码"

# 数据集的颜色：绿色=Calibration，红色=Validation
DATASET_COLORS = {
    'Calibration': (0.2, 0.6, 0.2),
    'Validation': (0.8, 0.2, 0.2),
}

# 图上8个模型的图案（如果不喜欢可以自己把 's' 改成 'o' 等）
MODEL_MARKERS = {
    'CatBoost': 'o',      # 圆形
    'LSTM': 's',          # 正方形
    'RandomForest': '^',  # 正三角形
    'RandomRorest': '^',  # 兼容拼写（用于读入/映射）
    'XGBoost': 'D',       # 菱形
    'SVM': 'v',           # 倒三角形
    'MLP': 'p',           # 五边形
    'KNN': '*',           # 星型
    'LightBGM': 'h',      # 六边形 (你表里写的是这个)
    'LightGBM': 'h',      # 兼容另一种拼写
}

# 统一模型命名（用于图例显示/marker 取值）
MODEL_ALIASES = {
    'RandomRorest': 'RandomForest',
    'LightBGM': 'LightGBM',
}

def canonical_model_name(name: str) -> str:
    name = str(name).strip()
    return MODEL_ALIASES.get(name, name)

def display_model_name(name: str) -> str:
    canon = canonical_model_name(name)
    if canon == 'RandomForest':
        return 'RF'
    return canon

# ===================== 3. 计算标准差的核心数学逻辑 =====================
def compute_std_from_metrics(r2, rmse, rpd):
    """
    通过实际数据的公式反推预测标准差：
    已知: E^2 = std_ref^2 + std_mod^2 - 2*std_ref*std_mod*r
    且知道: std_ref = RPD * RMSE
    """
    r2 = max(float(r2), 0.0)
    rmse = float(rmse)
    rpd = float(rpd)

    # 1. 估算出真实的样本标准差 (Reference 标准差)
    std_ref = rmse * rpd
    if std_ref <= 0:
        return std_ref, std_ref

    r = np.sqrt(r2)

    # 2. 解一元二次方程找 prediction 的标准差
    a = 1.0
    b = -2.0 * std_ref * r
    c = std_ref ** 2 - rmse ** 2

    disc = b * b - 4 * a * c
    if disc < 0: 
        disc = 0.0  # 防止浮点数误差导致负数开根号报错

    sqrt_disc = np.sqrt(disc)
    s1 = (-b + sqrt_disc) / (2 * a)
    s2 = (-b - sqrt_disc) / (2 * a)

    # 优先选稍微小一点/更合理的那一个正数根
    candidates = [s for s in (s1, s2) if s > 0]
    if not candidates:
        std_mod = std_ref
    else:
        std_mod = min(candidates, key=lambda x: abs(x - std_ref))

    return std_ref, std_mod

def get_nice_ticks(max_val):
    """根据真实值的最高刻度，自动算出一个顺眼的网格坐标间隔"""
    if max_val <= 0:
        return [1], "{:.1f}"
    order = 10 ** math.floor(math.log10(max_val))
    normalized = max_val / order
    
    if normalized <= 1.5:  step = 0.2 * order
    elif normalized <= 3:  step = 0.5 * order
    elif normalized <= 6:  step = 1.0 * order
    else:                  step = 2.0 * order
        
    ticks = np.arange(step, max_val + step * 0.1, step)
    fmt = "{:.0f}" if max_val >= 10 else "{:.1f}"
    return ticks[ticks <= max_val], fmt

# ===================== 4. 画主泰勒图 =====================
def create_taylor_diagram(records_df):
    fig = plt.figure(figsize=(7.2, 7.0))
    # 压缩底部留白：主图下移一点，图例区变矮
    ax = fig.add_axes([0.12, 0.33, 0.80, 0.62], polar=True)
    ax_leg = fig.add_axes([0.05, 0.05, 0.90, 0.20])
    ax_leg.axis('off')

    # 整个数据集只有一个真实的参考原点 (用所有计算结果中的中位数最为稳妥)
    ref_point_val = records_df['std_ref'].median()

    # 计算出轴的上限：只看“点会落到的半径”(std_mod) + 参考点。
    # 注意：不要用 std_ref 的最大值作为上限，它可能有极端值，会把坐标轴硬拉到 60+，导致中间看起来很拥挤。
    max_std = max(records_df['std_mod'].max(), ref_point_val)
    std_max_limit = max_std * 1.05

    # 如果只是因为留白导致刚好跨过 50、刻度跳到 60，就压回 50（前提是不裁剪数据点）
    if max_std <= 50 and std_max_limit > 50:
        std_max_limit = 50

    # 1. 拿到漂亮的刻度间隔
    std_ticks, tick_fmt = get_nice_ticks(std_max_limit)

    # 画背景标准差圆弧 & 轴标签
    for s in std_ticks:
        ax.plot(np.linspace(0, np.pi / 2, 100), np.full(100, s),
                linestyle=':', linewidth=0.6, color='gray', alpha=0.6)
        
        # 底部的横向标度 (在 s 处向下偏移一点距离)
        ax.text(0, s - std_max_limit * 0.04, f"{tick_fmt.format(s)}", ha='center', va='top', size=12)
        # 左侧竖向标度 (在 s 处向左偏移一点距离)
        ax.text(np.pi/2, s, f" {tick_fmt.format(s)}", ha='right', va='center', size=12)

    # 2. R² (决定系数) 的角度放射线刻度
    r2_ticks = [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    r2_angles = [np.arccos(np.sqrt(r)) for r in r2_ticks]
    ax.set_xticks(r2_angles)
    ax.set_xticklabels([])  # 屏蔽默认配置，我们将手动加
    for t, r in zip(r2_angles, r2_ticks):
        ax.plot([0, t], [0, std_max_limit], linestyle='-', linewidth=0.4, color='lightgray', alpha=0.5, zorder=0)

        label = f"{r:.2f}".rstrip('0').rstrip('.')
        # 0.90/0.95/0.99 角度很密，标签容易重叠；将高相关处的标签“分层”外移
        if r >= 0.99:
            r_label = std_max_limit * 1.08
            fs = 11
        elif r >= 0.95:
            r_label = std_max_limit * 1.05
            fs = 11
        elif r >= 0.90:
            r_label = std_max_limit * 1.02
            fs = 12
        else:
            r_label = std_max_limit * 1.02
            fs = 12

        ax.text(
            t, r_label, label,
            size=fs,
            ha='center',
            va='bottom',
            clip_on=False,
        )

    # ----- 绘制 RMSE 等值曲线背景 -----
    
    rmse_ticks, rmse_fmt = get_nice_ticks(records_df['rmse'].max() * 1.2)
    for e in rmse_ticks:
        phi = np.linspace(0, 2 * np.pi, 1000)
        x = ref_point_val + e * np.cos(phi)
        y = e * np.sin(phi)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # 只保留落在第一象限和边界内的点
        mask = (theta >= 0) & (theta <= np.pi / 2) & (r <= std_max_limit)
        if np.any(mask):
            ax.plot(theta[mask], r[mask], linestyle='--', linewidth=0.6, color='#555555', alpha=0.4)
            valid_idx = np.where(mask)[0]
            if len(valid_idx) > 20: 
                idx = valid_idx[int(len(valid_idx) * 0.4)]
                ax.text(theta[idx], r[idx] + std_max_limit*0.015, f"{rmse_fmt.format(e)}", 
                        size=11, color='gray', ha='center')

    # ----- 把数据点画上去 -----
    
    # 画出那个“观测参考中心点” (缩小星星大小 s=150 -> s=90)
    ax.scatter(0, ref_point_val, marker='*', s=90, color='black', zorder=20, 
               edgecolors='white', linewidth=0.8, clip_on=False)
    ax.plot(np.linspace(0, np.pi / 2, 100), np.full(100, ref_point_val),
            linewidth=1.0, color='black', linestyle='-', zorder=1)

    for _, row in records_df.iterrows():
        dataset = row['dataset']
        model_name = row['model']
        
        angle = np.arccos(np.sqrt(row['r2']))
        radius = row['std_mod']
        
        c = DATASET_COLORS.get(dataset, 'blue')
        m = MODEL_MARKERS.get(canonical_model_name(model_name), 'o')
        
        # 缩小散点大小 s=70 -> s=35, linewidth=1.5 -> 1.0，防拥挤
        ax.scatter(angle, radius, s=35, color='white', marker=m, 
                   edgecolors=c, linewidth=1.0, zorder=15)

    # 设置极限和文字
    ax.set_rlim(0, std_max_limit)
    ax.set_yticks([])
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(LINE_WIDTH)

    # 外围标签（贴着主图，距离参考图一致；底部图例已独立分区，不会再重叠）
    ax.text(
        -0.10, 0.5, "Standard deviation",
        transform=ax.transAxes,
        rotation=90,
        size=FONT_SIZE,
        ha='center',
        va='center'
    )
    ax.text(
        0.5, -0.10, "Standard deviation",
        transform=ax.transAxes,
        size=FONT_SIZE,
        ha='center',
        va='center'
    )
    ax.text(np.deg2rad(45), std_max_limit * 1.18, "Correlation coefficient (R²)", 
            size=FONT_SIZE, rotation=-45, ha='center', va='center')

    # ================= 底部图例（分类排版） =================
    legend_font = fm.FontProperties(size=12)

    # 1) Calibration / Validation（单列）
    dataset_handles = [
        Patch(facecolor=DATASET_COLORS['Calibration'], edgecolor='none', label='Calibration'),
        Patch(facecolor=DATASET_COLORS['Validation'], edgecolor='none', label='Validation'),
    ]

    # 2) 8 个模型（两列）
    model_order = ['CatBoost', 'LSTM', 'RandomForest', 'XGBoost', 'SVM', 'MLP', 'KNN', 'LightGBM']
    present_models = {canonical_model_name(m) for m in records_df['model'].unique()}
    model_handles = []
    for canon in model_order:
        if canon not in present_models:
            continue
        marker = MODEL_MARKERS.get(canon, 'o')
        model_handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                color='white',
                label=display_model_name(canon),
                markerfacecolor='white',
                markeredgecolor='black',
                markersize=6,
                markeredgewidth=1.0,
                linestyle='None',
            )
        )

    # 3) Observed / RMSE（单列）
    other_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10, label='Observed (Reference)'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1, label='RMSE Contours'),
    ]

    # 通过 3 个 legend 实现“按类别分块”的布局（放在 ax_leg 上）
    leg_ds = ax_leg.legend(
        handles=dataset_handles,
        loc='upper left',
        bbox_to_anchor=(0.00, 0.98),
        ncol=1,
        prop=legend_font,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.6,
    )
    ax_leg.add_artist(leg_ds)

    leg_models = ax_leg.legend(
        handles=model_handles,
        loc='upper left',
        bbox_to_anchor=(0.30, 0.98),
        ncol=2,
        prop=legend_font,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    ax_leg.add_artist(leg_models)

    ax_leg.legend(
        handles=other_handles,
        loc='upper left',
        bbox_to_anchor=(0.74, 0.98),
        ncol=1,
        prop=legend_font,
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.2,
        handletextpad=0.6,
    )

    return fig

# ===================== 5. 表格精确读取与执行入口 =====================
def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Could not find data file at:\n{DATA_FILE}")
        return

    print("Reading Excel data...")
    # 根据你的截图，第 1、2 行是合并表头。我们让 pandas 跳过第一行，把第2行认作为原始列。
    # header=1代表跳过0行(模型|训练集|测试集)，读取下面指标一行。
    df = pd.read_excel(DATA_FILE, header=1)

    # 强制将这些列命名为好处理的英文（严格按你表格从左到右的位置映射：列A~列G）
    df.columns = ['Model', 'Train_R2', 'Train_RMSE', 'Train_RPD', 
                           'Test_R2', 'Test_RMSE', 'Test_RPD']

    # 过滤掉空的冗余行 (如果A列 Model 列没有名字的话就跳过)
    df = df[df['Model'].notna()].copy()

    records = []
    for _, row in df.iterrows():
        m_name = str(row['Model']).strip()
        if not m_name or m_name.lower() == 'nan':
            continue
            
        # -- 读取训练集并推演标准差 --
        tr_r2 = row['Train_R2']
        tr_rmse = row['Train_RMSE']
        tr_rpd = row['Train_RPD']
        tr_sref, tr_smod = compute_std_from_metrics(tr_r2, tr_rmse, tr_rpd)
        records.append({
            'dataset': 'Calibration', 'model': m_name,
            'r2': tr_r2, 'rmse': tr_rmse, 'std_ref': tr_sref, 'std_mod': tr_smod
        })
        
        # -- 读取测试集并推演标准差 --
        te_r2 = row['Test_R2']
        te_rmse = row['Test_RMSE']
        te_rpd = row['Test_RPD']
        te_sref, te_smod = compute_std_from_metrics(te_r2, te_rmse, te_rpd)
        records.append({
            'dataset': 'Validation', 'model': m_name,
            'r2': te_r2, 'rmse': te_rmse, 'std_ref': te_sref, 'std_mod': te_smod
        })

    compiled_df = pd.DataFrame(records)
    print(compiled_df)  # 可以在控制台看看数据提取是否完美成功

    if compiled_df.empty:
        print("Error: Failed to parse data, no valid data extracted.")
        return

    # Generate Taylor diagram
    fig = create_taylor_diagram(compiled_df)
    
    # Save to specified output directory, in TIFF format, 300 dpi
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "Result_Taylor_Diagram.tiff")
    fig.savefig(save_path, dpi=300, format='tiff', bbox_inches='tight')
    
    print(f"\nTaylor diagram successfully created and saved to: \n{save_path}")
    # plt.show()  # 注释掉以避免弹窗阻塞

if __name__ == "__main__":
    main()