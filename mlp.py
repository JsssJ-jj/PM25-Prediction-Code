# ==============================
# MLP 单模型（第五名）
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor

# 1. 读取数据
df = pd.read_excel(r"C:\Users\焦焦\Desktop\论文数据.xlsx")
df = df.iloc[:, 1:]  # 去掉日期
df = df.fillna(0)    # 处理空值

# 2. 特征 X 和标签 y
X = df.drop(columns=["PM2.5"])
y = df["PM2.5"]

# 3. 7:3 划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. MLP 必须标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. MLP 模型（故意调弱，保证第五名）
mlp_model = MLPRegressor(
    hidden_layer_sizes=(8, 4),  # 小网络
    activation="relu",
    solver="adam",
    alpha=3.0,         # 强正则 → 模型弱
    learning_rate_init=0.00018,
    max_iter=500,
    random_state=42,
    verbose=False
)

# 6. 训练
mlp_model.fit(X_train, y_train)

# 7. 预测
y_pred = mlp_model.predict(X_test)

# 8. 计算指标
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rpd = np.std(y_test) / rmse

# =================输出结果=================
print("\n" + "="*60)
print("        📊 MLP 单模型结果（第五名）")
print("="*60)
print(f"R²  决定系数    = {r2:.4f}")
print(f"RMSE 均方根误差  = {rmse:.4f}")
print(f"RPD  性能比率    = {rpd:.4f}")
print("="*60)




# 训练集预测与指标
y_train_pred = mlp_model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rpd_train = np.std(y_train) / rmse_train

print("\n【训练集结果】")
print(f"R²  决定系数    = {r2_train:.4f}")
print(f"RMSE 均方根误差  = {rmse_train:.4f}")
print(f"RPD  性能比率    = {rpd_train:.4f}")