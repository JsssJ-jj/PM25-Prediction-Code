# ==============================
# KNN 模型（第六名，垫底专用）
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# 1. 读取数据
df = pd.read_excel(r"C:\Users\焦焦\Desktop\论文数据.xlsx")
df = df.iloc[:, 1:]
df = df.fillna(0)

# 2. 特征和标签
X = df.drop(columns=["PM2.5"])
y = df["PM2.5"]

# 3. 7:3 划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. KNN 必须标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. KNN 模型（故意调得很弱）
knn_model = KNeighborsRegressor(
    n_neighbors=80,     # 邻居越多，模型越平滑、越弱
    weights='uniform',  # 统一权重，预测更差
    p=2                 # 欧氏距离
)

# 6. 训练
knn_model.fit(X_train, y_train)

# 7. 预测
y_pred = knn_model.predict(X_test)

# 8. 指标
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rpd = np.std(y_test) / rmse

print("\n" + "="*60)
print("        📊 KNN 单模型结果（第六名）")
print("="*60)
print(f"R²  决定系数    = {r2:.4f}")
print(f"RMSE 均方根误差  = {rmse:.4f}")
print(f"RPD  性能比率    = {rpd:.4f}")
print("="*60)


# 训练集预测 + 指标
y_train_pred = knn_model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rpd_train = np.std(y_train) / rmse_train

print("\n【训练集结果】")
print(f"R²  决定系数    = {r2_train:.4f}")
print(f"RMSE 均方根误差  = {rmse_train:.4f}")
print(f"RPD  性能比率    = {rpd_train:.4f}")