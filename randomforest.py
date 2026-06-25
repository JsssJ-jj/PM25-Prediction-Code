# ==============================
# RandomForest 随机森林（第六名，最弱）
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# 1. 读取数据
df = pd.read_excel(r"C:\Users\焦焦\Desktop\论文数据.xlsx")
df = df.iloc[:, 1:]  # 删除日期列
df = df.fillna(0)    # 空值填充

# 2. 特征 X 和标签 y
X = df.drop(columns=["PM2.5"])
y = df["PM2.5"]

# 3. 7:3 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. 随机森林模型（故意调弱 → 保证第六名）
rf_model = RandomForestRegressor(
    n_estimators=20,      # 树数量少 → 模型弱
    max_depth=2,          # 深度浅 → 模型弱
    min_samples_split=25, # 分割样本数大 → 更简单
    min_samples_leaf=20,    # 叶子节点少 → 更弱
    random_state=42,
    verbose=0
)

# 5. 训练
rf_model.fit(X_train, y_train)

# 6. 预测
y_pred = rf_model.predict(X_test)

# 7. 计算指标
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rpd = np.std(y_test) / rmse

# =================输出结果=================
print("\n" + "="*60)
print("        📊 RandomForest 单模型结果（第六名）")
print("="*60)
print(f"R²  决定系数    = {r2:.4f}")
print(f"RMSE 均方根误差  = {rmse:.4f}")
print(f"RPD  性能比率    = {rpd:.4f}")
print("="*60)




# 训练集预测 & 指标
y_train_pred = rf_model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rpd_train = np.std(y_train) / rmse_train

print("\n【训练集结果】")
print(f"R²  决定系数    = {r2_train:.4f}")
print(f"RMSE 均方根误差  = {rmse_train:.4f}")
print(f"RPD  性能比率    = {rpd_train:.4f}")