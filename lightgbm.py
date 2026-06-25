import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor

# ========== 1. 基本配置 ==========
EXCEL_PATH = r"C:\Users\焦焦\Desktop\修改1\pm2.5箱线图\PM2.5.xlsx"
SHEET_NAME = "Sheet1"      # 如果不是 Sheet1，请改成实际名称
TARGET_COL = "PM2.5"       # 改成你 Excel 里真实的 PM2.5 列名

TEST_SIZE = 0.3            # 训练:验证 = 7:3
RANDOM_STATE = 42


# ========== 2. 指标计算函数 ==========
def calc_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true, ddof=1) / rmse
    return r2, rmse, rpd


# ========== 3. 读取数据、构造特征 ==========
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

if TARGET_COL not in df.columns:
    raise ValueError(f"目标列 {TARGET_COL} 不在表格列名中，请检查 TARGET_COL 是否写对。"
                     f"\n当前列名为: {list(df.columns)}")

# 去掉目标为空的行
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# 特征列 = 除目标列以外的所有列（不管是数值还是字符串）
feature_cols = [c for c in df.columns if c != TARGET_COL]

if len(feature_cols) == 0:
    raise ValueError("除了目标列之外没有任何其他列，无法建模。"
                     "请确认 Excel 中是否有可作为输入的特征列。")

X_raw = df[feature_cols].copy()
y = df[TARGET_COL].astype(float).values

# 对非数值列做 one-hot 编码
X = pd.get_dummies(X_raw, drop_first=True)  # 自动处理字符串/类别列

if X.shape[1] == 0:
    raise ValueError("one-hot 编码后仍然没有特征列，请检查数据格式。")

X = X.values

# ========== 4. 7:3 划分训练集 / 验证集 ==========
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# ========== 5. LightGBM 建模 ==========
model = LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# ========== 6. 训练集 / 验证集指标 ==========
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)

r2_tr, rmse_tr, rpd_tr = calc_metrics(y_train, y_train_pred)
r2_va, rmse_va, rpd_va = calc_metrics(y_valid, y_valid_pred)

print("=== LightGBM 模型性能（7:3 划分）===")
print(f"训练集: R2 = {r2_tr:.4f}, RMSE = {rmse_tr:.6f}, RPD = {rpd_tr:.3f}")
print(f"验证集: R2 = {r2_va:.4f}, RMSE = {rmse_va:.6f}, RPD = {rpd_va:.3f}")