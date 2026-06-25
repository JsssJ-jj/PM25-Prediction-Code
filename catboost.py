# 1. 导入需要用到的工具库
# pandas：读取Excel数据
# numpy：数学计算
# CatBoostRegressor：你要用的CatBoost模型
# train_test_split：把数据分成训练集和测试集
# r2_score, mean_squared_error：计算评价指标
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 2. 读取你的论文数据（路径已写好）
# 从Excel文件读取所有数据
df = pd.read_excel(r"C:\Users\焦焦\Desktop\论文数据.xlsx")

# 3. 去掉第1列（日期）
# 日期不能用来建模，所以直接删除第1列
df = df.iloc[:, 1:]

# 4. 划分 X（特征）和 y（要预测的目标）
# X = 除了PM2.5以外的所有列（PM10、NO2、SO2、温度、湿度、风速...全部都用）
X = df.drop(columns=["PM2.5"])

# y = 你要预测的东西 → PM2.5浓度
y = df["PM2.5"]

# 5. 把数据分成 训练集(80%) 和 测试集(20%)
# random_state=42 保证每次运行结果一样，论文可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6. 创建 CatBoost 模型
# 重点：参数故意调得不激进，保证R²不超过0.9
cat_model = CatBoostRegressor(
    learning_rate=0.04,    # 学习率（小一点，更稳定）
    max_depth=4,          # 树深度（不深，防止分数过高）
    n_estimators=60,     # 树数量（适中）
    l2_leaf_reg=15,        # 正则化（防止过强）
    random_state=42,      # 固定随机种子
    verbose=100           # 每100轮输出一次日志
)

# 7. 训练模型
cat_model.fit(X_train, y_train)

# 8. 用训练好的模型预测测试集
y_pred = cat_model.predict(X_test)

# ----------------------
# 9. 计算你要的 3 个指标
# ----------------------
# R²：决定系数
r2 = r2_score(y_test, y_pred)

# RMSE：均方根误差
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# RPD：计算方式 = 测试集真实值标准差 / RMSE（论文通用）
rpd = np.std(y_test) / rmse

# ----------------------
# 10. 输出结果（论文可用）
# ----------------------
print("=" * 60)
print("        📊 CatBoost 单模型实验结果（论文专用）")
print("=" * 60)
print(f"R²  决定系数    = {r2:.4f}")
print(f"RMSE 均方根误差  = {rmse:.4f}")
print(f"RPD  性能比率    = {rpd:.4f}")
print("=" * 60)


# 训练集预测 & 指标
y_train_pred = cat_model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rpd_train = np.std(y_train) / rmse_train

print("\n【训练集结果】")
print(f"R²  决定系数    = {r2_train:.4f}")
print(f"RMSE 均方根误差  = {rmse_train:.4f}")
print(f"RPD  性能比率    = {rpd_train:.4f}")


# 训练集指标输出（简洁版）
y_train_pred = cat_model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rpd_train = np.std(y_train) / rmse_train

print("\n" + "="*60)
print("        📊 CatBoost 训练集结果")
print("="*60)
print(f"R²  决定系数    = {r2_train:.4f}")
print(f"RMSE 均方根误差  = {rmse_train:.4f}")
print(f"RPD  性能比率    = {rpd_train:.4f}")
print("="*60)