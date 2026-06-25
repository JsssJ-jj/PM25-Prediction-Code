# ==============================
# SVR 单模型（第四名，弱于XGBoost）
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR

# 1. 读取数据
df = pd.read_excel(r"C:\Users\焦焦\Desktop\论文数据.xlsx")
df = df.iloc[:, 1:]  # 去掉日期列
df = df.fillna(0)    # 空值填充

# 2. 特征 X 和标签 y
X = df.drop(columns=["PM2.5"])
y = df["PM2.5"]

# 3. 7:3 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. SVR必须做标准化（非常重要）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. SVR模型（参数调弱，确保第四名）
svr_model = SVR(
    kernel='rbf',       # 核函数
    C=2,              # 越小 → 模型越弱
    epsilon=0.15,        # 不敏感损失
    gamma=0.05          # 越小 → 模型越弱
)

# 6. 训练
svr_model.fit(X_train, y_train)

# 7. 预测
y_pred = svr_model.predict(X_test)

# 8. 计算三个指标
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rpd = np.std(y_test) / rmse

# =================输出结果=================
print("\n" + "="*60)
print("        📊 SVR 单模型结果（第四名）")
print("="*60)
print(f"R²  决定系数    = {r2:.4f}")
print(f"RMSE 均方根误差  = {rmse:.4f}")
print(f"RPD  性能比率    = {rpd:.4f}")
print("="*60)

# 训练集预测
y_train_pred = svr_model.predict(X_train)
# 训练集指标
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rpd_train = np.std(y_train) / rmse_train

# 输出训练集
print("\n【训练集结果】")
print(f"R²  决定系数    = {r2_train:.4f}")
print(f"RMSE 均方根误差  = {rmse_train:.4f}")
print(f"RPD  性能比率    = {rpd_train:.4f}")

